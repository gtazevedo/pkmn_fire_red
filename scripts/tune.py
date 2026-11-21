import sys
import os
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.callbacks import Trainer
from src.config import CFG
from src.mlflow_utils import EXPERIMENT_TUNING, MLFLOW_TRACKING_URI
import mlflow

def run_experiment(name: str, config_updates: dict, steps: int = 100_000):
    print(f"\n=======================================================")
    print(f" Iniciando Tuning Geração 5: {name}")
    for k, v in config_updates.items():
        print(f"   {k}: {v}")
    print(f"=======================================================\n")
    
    defaults = {
        "max_steps_p3": 4000, "ent_coef": 0.01, "stuck_penalty": 0.0, "explore_weight": 3.0,
        "victory_bonus": 25.0, "first_strike_bonus": 12.0, "explore_weight_outdoor_bonus": 1.5,
        "heal_reward_scale": 8.0, "new_map_bonus": 0.0, "battle_idle_penalty": 0.0,
        "battle_a_bonus": 0.03, "battle_a_cap": 50, "whiteout_penalty": 0.0,
        "time_penalty": 0.0, "stuck_grace": 300, "north_bonus_per_tile": 0.0,
        "map_explore_cap_exterior": 300.0, "map_explore_cap_interior": 10.0,
        "levelup_weight": 40.0, "near_ko_bonus": 5.0, "battle_sustain_bonus": 0.0,
        "hp_pct_scale": 15.0, "indoor_step_penalty": 0.0, "stale_map_penalty": 0.0,
        "battle_b_penalty": 0.0, "battle_b_threshold": 999, "new_map_bonus_route": 0.0,
        "outdoor_sustain_bonus": 0.0
    }
    for k, v in defaults.items():
        object.__setattr__(CFG, k, v)
        
    for k, v in config_updates.items():
        object.__setattr__(CFG, k, v)
    
    original_model_path = CFG.model_path
    object.__setattr__(CFG, 'model_path', f"{CFG.model_path}_tune_Gen5_{name}")
    object.__setattr__(CFG, 'config_stamp', f"tune_Gen5_{name}")
    object.__setattr__(CFG, 'lr_total_budget', steps)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_TUNING)
    
    with mlflow.start_run(run_name=CFG.config_stamp):
        params = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith('_') and not callable(getattr(CFG, k))}
        mlflow.log_params(params)
        
        try:
            Trainer(reset_training=True).train()
        except KeyboardInterrupt:
            print("Tuning abortado pelo usuário.")
            
    object.__setattr__(CFG, 'model_path', original_model_path)

def main():
    # Base campeã da Geração 4: Ultimate HighEnt
    base_ultimate = {
        "max_steps_p3": 20000, "ent_coef": 0.08, "explore_weight": 3.0,
        "victory_bonus": 25.0, "first_strike_bonus": 12.0, "new_map_bonus": 50.0, 
        "battle_a_bonus": 0.1, "battle_a_cap": 100, "time_penalty": -0.01, 
        "explore_weight_outdoor_bonus": 5.0, "stuck_penalty": 0.00
    }

    tests = [
        {"name": "Real_Punisher", "updates": {**base_ultimate, "stuck_penalty": -0.1}},
        {"name": "North_Star", "updates": {**base_ultimate, "north_bonus_per_tile": 2.0}},
        {"name": "Infinite_MapCap", "updates": {**base_ultimate, "map_explore_cap_exterior": 5000.0, "map_explore_cap_interior": 100.0}},
        {"name": "Level_Glutton", "updates": {**base_ultimate, "levelup_weight": 200.0, "near_ko_bonus": 20.0}},
        {"name": "Combat_Sustain", "updates": {**base_ultimate, "battle_sustain_bonus": 0.5, "hp_pct_scale": 50.0}},
        {"name": "Evict_Indoor", "updates": {**base_ultimate, "indoor_step_penalty": -0.05, "stale_map_penalty": -0.05}},
        {"name": "Anti_B_Spam", "updates": {**base_ultimate, "battle_b_penalty": -0.1, "battle_b_threshold": 5}},
        {"name": "Route_Lover", "updates": {**base_ultimate, "new_map_bonus_route": 100.0, "outdoor_sustain_bonus": 0.5}},
        
        {"name": "Total_Anarchy", "updates": {**base_ultimate, "map_explore_cap_exterior": 5000.0, "north_bonus_per_tile": 2.0, "battle_sustain_bonus": 0.5, "levelup_weight": 200.0}},
        {"name": "Dictatorship", "updates": {**base_ultimate, "stuck_penalty": -0.1, "indoor_step_penalty": -0.05, "battle_b_penalty": -0.1, "battle_b_threshold": 5, "stale_map_penalty": -0.05}}
    ]
    
    for t in tests:
        run_experiment(
            name=t["name"], 
            config_updates=t["updates"],
            steps=100_000 
        )
        
if __name__ == "__main__":
    main()
