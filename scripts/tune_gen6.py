import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.callbacks import Trainer
from src.config import CFG
import mlflow

def run_experiment(name: str, config_updates: dict, steps: int = 300_000):
    print("\n=======================================================")
    print(f" Iniciando Tuning Geracao 6 (300k steps): {name}")
    for k, v in config_updates.items():
        print(f"   {k}: {v}")
    print("=======================================================\n")
    
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
    object.__setattr__(CFG, 'model_path', f"{CFG.model_path}_tune_Gen6_{name}")
    object.__setattr__(CFG, 'config_stamp', f"tune_Gen6_{name}")
    object.__setattr__(CFG, 'lr_total_budget', steps)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Pokemon_RL_Tuning")
    
    with mlflow.start_run(run_name=f"tune_Gen6_{name}"):
        params = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith('_') and not callable(getattr(CFG, k))}
        mlflow.log_params(params)
        
        try:
            Trainer(reset_training=True).train()
        except KeyboardInterrupt:
            print("Tuning Gen 6 abortado pelo usuario.")
            
    object.__setattr__(CFG, 'model_path', original_model_path)

def main():
    base_gen5_hero = {
        "max_steps_p3": 20000,
        "ent_coef": 0.08,
        "explore_weight": 3.0,
        "victory_bonus": 25.0,
        "first_strike_bonus": 12.0,
        "new_map_bonus": 50.0,
        "battle_a_bonus": 0.1,
        "battle_a_cap": 100,
        "time_penalty": -0.01,
        "explore_weight_outdoor_bonus": 5.0,
        "stuck_penalty": -0.05,
        "levelup_weight": 200.0,
        "near_ko_bonus": 20.0,
        "map_explore_cap_exterior": 5000.0,
        "map_explore_cap_interior": 50.0
    }

    tests = [
        {"name": "1_Grand_Glutton_Long", "updates": {**base_gen5_hero}},
        {"name": "2_Quest_Speedrunner", "updates": {**base_gen5_hero, "new_map_bonus_route": 150.0, "north_bonus_per_tile": 3.0}},
        {"name": "3_Hardcore_Punisher", "updates": {**base_gen5_hero, "stuck_penalty": -0.15, "indoor_step_penalty": -0.08, "time_penalty": -0.02}},
        {"name": "4_Apex_Warrior", "updates": {**base_gen5_hero, "victory_bonus": 50.0, "first_strike_bonus": 25.0, "battle_sustain_bonus": 1.0}},
        {"name": "5_Holy_Grail_Combined", "updates": {**base_gen5_hero, "new_map_bonus_route": 100.0, "north_bonus_per_tile": 2.0, "victory_bonus": 40.0, "stuck_penalty": -0.08}}
    ]
    
    for t in tests:
        run_experiment(
            name=t["name"], 
            config_updates=t["updates"],
            steps=300_000
        )
        
if __name__ == "__main__":
    main()
