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
from src.mlflow_utils import (
    EXPERIMENT_TRAIN,
    MLFLOW_TRACKING_URI,
    TRAIN_EXPERIMENT_DESCRIPTION,
    build_run_name,
    ensure_experiment_metadata,
    standard_run_tags,
)


def evaluate(
    num_episodes: int = 3,
    max_eval_steps: int = 250,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    from_start: bool = False,
) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    ensure_experiment_metadata(
        client,
        EXPERIMENT_TRAIN,
        TRAIN_EXPERIMENT_DESCRIPTION,
        {
            "game": "Pokemon-FireRed-GBA",
            "architecture": "RecurrentPPO-LSTM+HRL",
            "llm_planner": f"Ollama-{CFG.ollama_model.replace(':', '-')}",
            "curriculum": "FrontierBias-50-30-20",
            "repo": "pkmn_fire_red",
        },
    )
    mlflow.set_experiment(EXPERIMENT_TRAIN)

    fixed_state = CFG.state_file if from_start else None
    venv = DummyVecEnv([
        lambda: PokemonEnv(env_id=99, fixed_state_file=fixed_state, enable_stochasticity=False)
    ])
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
    ep_left_bedroom: list[float] = []
    ep_left_house: list[float] = []
    ep_entered_lab: list[float] = []
    ep_got_starter: list[float] = []
    ep_reached_route1: list[float] = []
    ep_beat_rival: list[float] = []

    t_start = time.time()
    try:
        for _ in range(num_episodes):
            obs = venv.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            total_rew = 0.0
            steps = 0
            last_info = {}
            ep_maps_seen: set[tuple[int, int]] = set(raw_env._stats.map_visits)
            ep_max_lvl: int = int(raw_env._stats.max_level)

            for _ in range(max_eval_steps):
                action_idx, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, rewards, dones, infos = venv.step(action_idx)
                last_info = infos[0] if infos else {}
                if "map_bank" in last_info and "map_id" in last_info:
                    ep_maps_seen.add((int(last_info["map_bank"]), int(last_info["map_id"])))
                ep_max_lvl = max(ep_max_lvl, int(last_info.get("p1_lvl", 0)))
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
                ep_max_lvl = max(ep_max_lvl, int(em["party_level"]))
                ep_metrics = {
                    "best_milestone": em["best_milestone"],
                    "badges": em["badges"],
                    "party_level": em["party_level"],
                    "victories": em["victories"],
                    "maps_visited": em["maps_visited"],
                }
            else:
                ep_maps_seen |= raw_env._stats.map_visits
                ep_max_lvl = max(ep_max_lvl, int(raw_env._stats.max_level))
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
            ep_max_lvl = max(ep_max_lvl, int(ep_metrics.get("party_level", 0)))
            ep_victories = int(ep_metrics.get("victories", 0))
            ep_scores.append(PokemonCallback._progress_score(ep_metrics))
            ep_left_bedroom.append(float(any(m != (4, 1) for m in ep_maps_seen)))
            ep_left_house.append(float((3, 0) in ep_maps_seen or any(b != 4 for b, _ in ep_maps_seen)))
            ep_entered_lab.append(float((4, 3) in ep_maps_seen))
            ep_got_starter.append(float(ep_max_lvl > 0))
            ep_reached_route1.append(float((3, 19) in ep_maps_seen))
            ep_beat_rival.append(float(ep_max_lvl >= 6 or ep_victories > 0))

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
            "progress/genesis_ratio": float(prog_stats["progress/genesis_ratio"]),
            "progress/start_state_ratio": float(prog_stats["progress/start_state_ratio"]),
            "reward/total": float(np.mean(ep_rewards)),
            "reward/from_strategist": float(np.mean(ep_strat_rew)),
            "strategist/queries_count": float(np.mean(ep_strat_queries)),
        }
        if from_start:
            summary.update({
                "eval/left_bedroom_rate": float(np.mean(ep_left_bedroom)),
                "eval/left_house_rate": float(np.mean(ep_left_house)),
                "eval/entered_oaks_lab_rate": float(np.mean(ep_entered_lab)),
                "eval/got_starter_rate": float(np.mean(ep_got_starter)),
                "eval/reached_route1_rate": float(np.mean(ep_reached_route1)),
                "eval/beat_rival_rate": float(np.mean(ep_beat_rival)),
            })
    finally:
        venv.close()

    run_suffix = "eval_from_start" if from_start else "eval"
    with mlflow.start_run(run_name=build_run_name(suffix=run_suffix)):
        params = {
            k: getattr(CFG, k)
            for k in dir(CFG)
            if not k.startswith("_") and not callable(getattr(CFG, k))
        }
        params["RAM_SIZE"] = RAM_SIZE
        params["GRID_SIZE"] = GRID_SIZE
        params["GOAL_SIZE"] = GOAL_SIZE
        mlflow.log_params(params)
        tags = standard_run_tags("eval")
        tags["stage"] = "evaluation"
        tags["eval_mode"] = "from_start" if from_start else "curriculum"
        tags["surgery_verified"] = "true"
        mlflow.set_tags(tags)
        mlflow.log_metrics(summary, step=int(model.num_timesteps))

    exp = client.get_experiment_by_name(EXPERIMENT_TRAIN)
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
    parser.add_argument(
        "--from-start",
        action="store_true",
        help="Avalia retenção de início de jogo a partir de Start.state.gz",
    )
    args = parser.parse_args()
    evaluate(num_episodes=args.episodes, max_eval_steps=args.steps, from_start=args.from_start)
