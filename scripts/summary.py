from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.mlflow_utils import (
    EXPERIMENT_TRAIN,
    EXPERIMENT_TUNING,
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
)


def print_experiment_summary(client: MlflowClient, exp_name: str, limit: int = 15) -> None:
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"Experiment '{exp_name}' not found.")
        return

    print(f"\n=== MLflow Experiment: {exp_name} (id={exp.experiment_id}) ===")
    runs = client.search_runs(exp.experiment_id, order_by=["start_time DESC"], max_results=limit)
    for r in reversed(runs):
        name = r.info.run_name or r.data.tags.get("mlflow.runName", "Unknown")
        status = r.info.status
        m = r.data.metrics
        step = int(m.get("time/total_timesteps", m.get("eval/model_num_timesteps", 0)))
        score = m.get("progress/score", 0.0)
        rew = m.get("trend/reward_mean20", m.get("eval/mean_reward", m.get("reward/total", 0.0)))
        tiles = m.get("trend/tiles_mean20", m.get("eval/mean_unique_tiles", 0.0))
        expl = m.get("trend/expl_net_mean20", 0.0)
        lvl = m.get("trend/level_mean20", 0.0)
        stuck = m.get("trend/stuck_ratio_mean20", 0.0)
        strat_q = m.get("strategist/queries_count", m.get("eval/mean_strategist_queries", 0.0))
        frontier = m.get("progress/frontier_ratio", 0.0)
        print(
            f"[{name}] ({status}) Steps={step:,} | Score={score:.1f} | "
            f"Rew={rew:.2f} | Tiles={tiles:.1f} | Expl={expl:.2f} | "
            f"Lvl={lvl:.1f} | Stuck={stuck:.2f} | StratQ={strat_q:.1f} | Frontier={frontier:.2f}"
        )


def print_registry_summary(client: MlflowClient, model_name: str = REGISTERED_MODEL_NAME) -> None:
    print(f"\n=== MLflow Model Registry: {model_name} ===")
    try:
        rm = client.get_registered_model(model_name)
    except MlflowException:
        print(f"Registered model '{model_name}' not found.")
        return

    alias_str = ", ".join(f"@{k}=v{v}" for k, v in sorted(rm.aliases.items())) or "none"
    print(f"Aliases: {alias_str}")
    versions = sorted(
        client.search_model_versions(f"name='{model_name}'"),
        key=lambda item: int(item.version),
    )
    for mv in versions:
        t = mv.tags or {}
        print(
            f"  v{mv.version} (run={mv.run_id[:8]}) | steps={t.get('timesteps', '-')} | "
            f"rew20={t.get('reward_mean20', '-')} | score={t.get('score', '-')} | "
            f"commit={t.get('git_commit', '-')} | ollama={t.get('ollama_model', '-')} | "
            f"{mv.description or ''}"
        )


def main() -> None:
    client = MlflowClient(MLFLOW_TRACKING_URI)
    for exp_name in (EXPERIMENT_TRAIN, EXPERIMENT_TUNING):
        print_experiment_summary(client, exp_name)
    print_registry_summary(client)


if __name__ == "__main__":
    main()
