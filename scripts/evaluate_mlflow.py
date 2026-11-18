import argparse
import os
import sys
import time
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.callbacks import PokemonCallback
from src.config import CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE, log
from src.env import PokemonEnv


def evaluate(num_episodes: int = 3, max_eval_steps: int = 250) -> dict:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pokemon_RL_FireRed")

    venv = DummyVecEnv([lambda: PokemonEnv(env_id=99)])
    vecnorm_path = CFG.model_path + "_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False

    model_zip = CFG.model_path + ".zip"
    if not os.path.exists(model_zip):
        venv.close()
        raise FileNotFoundError(f"Modelo não encontrado para avaliação: {model_zip}")

    model = RecurrentPPO.load(CFG.model_path, env=venv)
    raw_env: PokemonEnv = venv.envs[0]

    ep_rewards: list[float] = []
    ep_steps: list[int] = []
    ep_tiles: list[int] = []
    ep_maps: list[int] = []
    ep_scores: list[float] = []
    ep_strat_queries: list[int] = []
    ep_strat_rew: list[float] = []
    ep_advisor_repeats: list[int] = []

    t_start = time.time()
    try:
        for _ in range(num_episodes):
            obs = venv.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            total_rew = 0.0
            steps = 0
            last_info = {}

            for _ in range(max_eval_steps):
                action_idx, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, dones, infos = venv.step(action_idx)
                last_info = infos[0] if infos else {}
                total_rew += float(rewards[0])
                steps += 1
                episode_starts[0] = bool(dones[0])
                if dones[0]:
                    break

            em = last_info.get("episode_metrics")
            if em is not None:
                ep_rewards.append(float(em["total_reward"]))
                ep_steps.append(int(em["steps"]))
                ep_tiles.append(int(em["unique_tiles"]))
                ep_maps.append(int(em["maps_visited"]))
                ep_strat_queries.append(int(em["strategist_queries_count"]))
                ep_strat_rew.append(float(em["reward_strategist"]))
                ep_advisor_repeats.append(int(em["advisor_repeat_penalties"]))
                ep_metrics = {
                    "best_milestone": em["best_milestone"],
                    "badges": em["badges"],
                    "party_level": em["party_level"],
                    "victories": em["victories"],
                    "maps_visited": em["maps_visited"],
                }
            else:
                ep_rewards.append(total_rew)
                ep_steps.append(steps)
                ep_tiles.append(len(raw_env._stats.tile_visits))
                ep_maps.append(len(raw_env._stats.map_visits))
                ep_strat_queries.append(raw_env._strategist.queries_count)
                ep_strat_rew.append(raw_env._rewards.strategist)
                ep_advisor_repeats.append(raw_env._advisor.repeat_penalties)
                ep_metrics = {
                    "best_milestone": raw_env._progress._best_flags_ever,
                    "badges": raw_env._stats.last_badges,
                    "party_level": raw_env._stats.max_level,
                    "victories": raw_env._stats.victories,
                    "maps_visited": len(raw_env._stats.map_visits),
                }
            ep_scores.append(PokemonCallback._progress_score(ep_metrics))

        elapsed = max(time.time() - t_start, 1e-6)
        total_eval_steps = sum(ep_steps)
        prog_stats = raw_env._progress.stats_dict()
        summary = {
            "eval/episodes": float(num_episodes),
            "eval/model_num_timesteps": float(model.num_timesteps),
            "eval/mean_reward": float(np.mean(ep_rewards)),
            "eval/mean_steps": float(np.mean(ep_steps)),
            "eval/mean_unique_tiles": float(np.mean(ep_tiles)),
            "eval/mean_maps_visited": float(np.mean(ep_maps)),
            "eval/mean_strategist_queries": float(np.mean(ep_strat_queries)),
            "eval/mean_strategist_reward": float(np.mean(ep_strat_rew)),
            "eval/mean_advisor_repeats": float(np.mean(ep_advisor_repeats)),
            "eval/fps": float(total_eval_steps / elapsed),
            "progress/score": float(np.mean(ep_scores)),
            "progress/best_flags_ever": float(prog_stats["progress/best_flags_ever"]),
            "progress/frontier_ratio": float(prog_stats["progress/frontier_ratio"]),
            "reward/total": float(np.mean(ep_rewards)),
            "reward/from_strategist": float(np.mean(ep_strat_rew)),
            "strategist/queries_count": float(np.mean(ep_strat_queries)),
        }
    finally:
        venv.close()

    with mlflow.start_run(run_name=f"{CFG.config_stamp}_eval"):
        params = {
            k: getattr(CFG, k)
            for k in dir(CFG)
            if not k.startswith("_") and not callable(getattr(CFG, k))
        }
        params["RAM_SIZE"] = RAM_SIZE
        params["GRID_SIZE"] = GRID_SIZE
        params["GOAL_SIZE"] = GOAL_SIZE
        mlflow.log_params(params)
        mlflow.set_tags({
            "stage": "evaluation",
            "surgery_verified": "true",
            "ollama_model": CFG.ollama_model,
            "sampling_strategy": "frontier_bias_50_30_20",
        })
        mlflow.log_metrics(summary, step=int(model.num_timesteps))

    client = MlflowClient("sqlite:///mlflow.db")
    exp = client.get_experiment_by_name("Pokemon_RL_FireRed")
    prev_runs = client.search_runs(
        exp.experiment_id,
        filter_string="tags.mlflow.runName = 'v29_gen6_holy_grail'",
        max_results=1,
    )
    if prev_runs:
        prev_score = prev_runs[0].data.metrics.get("progress/score")
        log.info(
            f"[MLflow Eval] Baseline v29 progress/score={prev_score} vs "
            f"v30 eval progress/score={summary['progress/score']:.2f}"
        )

    log.info(f"[MLflow Eval] Summary: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Pokemon FireRed model and log to MLflow")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    evaluate(num_episodes=args.episodes, max_eval_steps=args.steps)
