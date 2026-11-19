from __future__ import annotations

import os
import subprocess
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE, log

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_TRAIN = "Pokemon_RL_FireRed"
EXPERIMENT_TUNING = "Pokemon_RL_Tuning"
REGISTERED_MODEL_NAME = "PokemonFireRed-RecurrentPPO"

TRAIN_EXPERIMENT_DESCRIPTION = (
    "Agente HRL autônomo para zerar Pokémon FireRed (GBA): RecurrentPPO + LSTM "
    "com leitura direta de RAM/EWRAM, matriz de colisão 9x9, Frontier Bias (50/30/20) "
    "em progress_states/ e planejador assíncrono Strategist via Ollama local."
)

TUNING_EXPERIMENT_DESCRIPTION = (
    "Sweeps controlados de hiperparâmetros e reward shaping (Gen4-Gen6) no Pokémon FireRed."
)

REGISTERED_MODEL_DESCRIPTION = (
    "Política RecurrentPPO MultiInputLstmPolicy (CNN 84x84x4 + RAM 23 + Collision Grid 9x9 + "
    "Goal Embedding 6 + VecNormalize) para Pokémon FireRed (USA v1.0)."
)


def resolve_git_short() -> str:
    """Retorna o hash curto do commit HEAD atual ou 'nogit' se indisponível."""
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "nogit"
    except Exception:
        return "nogit"


def build_run_name(suffix: Optional[str] = None) -> str:
    """Gera run_name único e legível ou respeita MLFLOW_RUN_NAME."""
    override = os.getenv("MLFLOW_RUN_NAME")
    if override and override.strip():
        base = override.strip()
    else:
        model_slug = CFG.ollama_model.replace(":", "-")
        base = f"{CFG.config_stamp}_{resolve_git_short()}_{model_slug}"
    if suffix:
        return f"{base}_{suffix}"
    return base


def standard_run_tags(
    run_type: str,
    notes: Optional[str] = None,
) -> dict[str, str]:
    """Tags padronizadas para qualquer run de treino, avaliação ou tuning."""
    tags: dict[str, str] = {
        "run_type": run_type,
        "config_stamp": CFG.config_stamp,
        "git_commit": resolve_git_short(),
        "ollama_model": CFG.ollama_model,
        "llm_enabled": "true",
        "sampling_strategy": "frontier_bias_50_30_20",
        "obs_spec": f"RAM{RAM_SIZE}_GRID{GRID_SIZE}x{GRID_SIZE}_GOAL{GOAL_SIZE}",
    }
    note_val = notes if notes is not None else os.getenv("MLFLOW_RUN_NOTES", "")
    if note_val and note_val.strip():
        tags["mlflow.note.content"] = note_val.strip()
    return tags


def ensure_experiment_metadata(
    client: MlflowClient,
    experiment_name: str,
    description: str,
    tags: dict[str, str],
) -> str:
    """Cria ou recupera o experimento e garante descrição e tags atualizadas."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        exp_id = client.create_experiment(
            experiment_name,
            tags={**tags, "mlflow.note.content": description},
        )
    else:
        exp_id = exp.experiment_id
        client.set_experiment_tag(exp_id, "mlflow.note.content", description)
        for k, v in tags.items():
            client.set_experiment_tag(exp_id, k, str(v))
    return exp_id


def ensure_registered_model(
    client: MlflowClient,
    model_name: str = REGISTERED_MODEL_NAME,
    description: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
) -> None:
    """Garante que o modelo exista no MLflow Model Registry com descrição e tags."""
    try:
        client.create_registered_model(
            name=model_name,
            tags=tags or {},
            description=description,
        )
    except MlflowException:
        if description is not None:
            client.update_registered_model(name=model_name, description=description)
        if tags:
            for k, v in tags.items():
                client.set_registered_model_tag(model_name, k, str(v))


def _maybe_promote_champion(
    client: MlflowClient,
    model_name: str,
    version: str,
    candidate_metric: float,
) -> None:
    """Promove @champion somente se a métrica candidata superar o campeão atual."""
    current_best: Optional[float] = None
    try:
        cur = client.get_model_version_by_alias(model_name, "champion")
        raw = cur.tags.get("reward_mean20")
        if raw is not None:
            current_best = float(raw)
    except (MlflowException, ValueError, TypeError):
        current_best = None

    if current_best is None or candidate_metric > current_best:
        client.set_registered_model_alias(model_name, "champion", version)


def register_checkpoint_version(
    client: MlflowClient,
    run_id: str,
    description: str = "",
    tags: Optional[dict[str, str]] = None,
    aliases: tuple[str, ...] = (),
    model_name: str = REGISTERED_MODEL_NAME,
    promote_champion_metric: Optional[float] = None,
    allow_multiple_per_run: bool = False,
    model_description: Optional[str] = REGISTERED_MODEL_DESCRIPTION,
) -> Optional[str]:
    """Registra ou atualiza uma versão de checkpoint do run no Model Registry."""
    ensure_registered_model(
        client,
        model_name=model_name,
        description=model_description,
        tags={
            "framework": "sb3-contrib-RecurrentPPO",
            "policy": "MultiInputLstmPolicy",
            "game": "Pokemon-FireRed-GBA",
        },
    )

    run = client.get_run(run_id)
    source = run.info.artifact_uri
    str_tags = {str(k): str(v) for k, v in (tags or {}).items()}
    if promote_champion_metric is not None and "reward_mean20" not in str_tags:
        str_tags["reward_mean20"] = f"{float(promote_champion_metric):.2f}"

    version_str: Optional[str] = None
    if not allow_multiple_per_run:
        existing = client.search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
        if existing:
            mv = max(existing, key=lambda item: int(item.version))
            version_str = str(mv.version)
            if description:
                client.update_model_version(name=model_name, version=version_str, description=description)
            for k, v in str_tags.items():
                client.set_model_version_tag(model_name, version_str, k, v)

    if version_str is None:
        mv = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
            tags=str_tags,
            description=description,
        )
        version_str = str(mv.version)

    for alias in aliases:
        client.set_registered_model_alias(model_name, alias, version_str)

    if promote_champion_metric is not None:
        _maybe_promote_champion(client, model_name, version_str, float(promote_champion_metric))

    log.info(
        f"[MLflow Registry] {model_name} v{version_str} vinculado ao run {run_id[:8]}"
    )
    return version_str


def close_orphan_runs(
    client: MlflowClient,
    experiment_name: str,
    keep_run_ids: tuple[str, ...] = (),
) -> list[str]:
    """Encerra runs antigos presos com status RUNNING fora de keep_run_ids como KILLED."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return []

    running = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'RUNNING'",
        max_results=200,
    )
    closed: list[str] = []
    keep_set = set(keep_run_ids)
    for run in running:
        rid = run.info.run_id
        if rid in keep_set:
            continue
        client.set_terminated(rid, status="KILLED")
        closed.append(rid)
    return closed
