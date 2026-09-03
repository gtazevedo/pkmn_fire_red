import sys
import os
import shutil

# Certifica-se de que a pasta raiz do projeto está no path para as importações de src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.callbacks import Trainer
from src.config import CFG
import mlflow

def run_experiment(name: str, ent_coef: float, stuck_pen: float, explore_weight: float, steps: int = 200_000):
    print(f"\n=======================================================")
    print(f" Iniciando Tuning: {name}")
    print(f" ent_coef={ent_coef} | stuck_pen={stuck_pen} | explore_weight={explore_weight}")
    print(f"=======================================================\n")
    
    # Sobrescreve as configurações para este teste (burlando o FrozenInstanceError)
    object.__setattr__(CFG, 'ent_coef', ent_coef)
    object.__setattr__(CFG, 'stuck_pen', stuck_pen)
    object.__setattr__(CFG, 'explore_weight', explore_weight)
    
    # Redireciona o checkpoint para não sobrescrever nosso modelo principal!
    original_model_path = CFG.model_path
    object.__setattr__(CFG, 'model_path', f"{CFG.model_path}_tune_{name}")
    object.__setattr__(CFG, 'config_stamp', f"tune_{name}")
    object.__setattr__(CFG, 'lr_total_budget', steps) # força o término em 100k
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pokemon_RL_Tuning")
    
    with mlflow.start_run(run_name=CFG.config_stamp):
        params = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith('_') and not callable(getattr(CFG, k))}
        mlflow.log_params(params)
        
        # Inicia do zero para vermos a velocidade de aprendizado puro
        try:
            Trainer(reset_training=True).train()
        except KeyboardInterrupt:
            print("Tuning abortado pelo usuário.")
            
    # Restaura o caminho (caso vá rodar outro em sequência no mesmo processo)
    object.__setattr__(CFG, 'model_path', original_model_path)

def main():
    # Bateria de testes
    tests = [
        {"name": "Baseline",       "ent": 0.01, "pen": -0.01, "expl": 3.0},
        {"name": "High_Entropy",   "ent": 0.05, "pen": -0.01, "expl": 3.0},
        {"name": "Harsh_Penalty",  "ent": 0.01, "pen": -0.10, "expl": 3.0},
        {"name": "High_Explorer",  "ent": 0.01, "pen": -0.01, "expl": 5.0},
    ]
    
    for t in tests:
        run_experiment(
            name=t["name"], 
            ent_coef=t["ent"], 
            stuck_pen=t["pen"], 
            explore_weight=t["expl"],
            steps=100_000 # 100k steps é +- 5 a 10 min
        )
        
if __name__ == "__main__":
    main()
