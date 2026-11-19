import os
import signal
import sys
import threading

# Certifica-se de que a pasta raiz do projeto está no path para as importações de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
from mlflow.tracking import MlflowClient

from src.callbacks import Trainer
from src.config import CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE
from src.mlflow_utils import (
    EXPERIMENT_TRAIN,
    MLFLOW_TRACKING_URI,
    TRAIN_EXPERIMENT_DESCRIPTION,
    build_run_name,
    ensure_experiment_metadata,
    standard_run_tags,
)


def _install_sigterm_handler() -> None:
    """Converte SIGTERM em KeyboardInterrupt para salvar checkpoint e fechar o run MLflow."""
    if threading.current_thread() is not threading.main_thread():
        return

    def _handle_sigterm(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)


def main(reset_training: bool = False, tracking_uri: str = MLFLOW_TRACKING_URI) -> None:
    _install_sigterm_handler()
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

    with mlflow.start_run(run_name=build_run_name()):
        params = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith('_') and not callable(getattr(CFG, k))}
        params["RAM_SIZE"] = RAM_SIZE
        params["GRID_SIZE"] = GRID_SIZE
        params["GOAL_SIZE"] = GOAL_SIZE
        mlflow.log_params(params)
        mlflow.set_tags(standard_run_tags("train"))

        try:
            Trainer(reset_training=reset_training).train()
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    # v14: RAM_SIZE=20, VecNormalize, first_strike_bonus, BugFix A+B.
    # Checkpoint v13 é incompatível — stamp diferente força reset automático.
    # Para forçar reset manual: main(reset_training=True)
    main(reset_training=False)