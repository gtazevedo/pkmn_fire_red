import sys
import os

# Certifica-se de que a pasta raiz do projeto está no path para as importações de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.callbacks import Trainer
from src.env import PokemonEnv
from src.config import CFG, ActionSpace

import mlflow

def main(reset_training: bool = False) -> None:
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment("Pokemon_RL_FireRed")
    
    with mlflow.start_run(run_name=CFG.config_stamp):
        params = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith('_') and not callable(getattr(CFG, k))}
        mlflow.log_params(params)
        
        Trainer(reset_training=reset_training).train()

if __name__ == "__main__":
    # v14: RAM_SIZE=20, VecNormalize, first_strike_bonus, BugFix A+B.
    # Checkpoint v13 é incompatível — stamp diferente força reset automático.
    # Para forçar reset manual: main(reset_training=True)
    main(reset_training=False)