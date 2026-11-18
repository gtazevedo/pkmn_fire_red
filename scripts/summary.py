import mlflow
from mlflow.tracking import MlflowClient
import os

client = MlflowClient('sqlite:///mlflow.db')
exp = client.get_experiment_by_name('Pokemon_RL_Tuning')
if exp is None:
    print("Experiment not found.")
else:
    runs = client.search_runs(exp.experiment_id, order_by=["start_time ASC"])
    for r in runs:
        name = r.data.tags.get('mlflow.runName', 'Unknown')
        stuck = r.data.metrics.get('trend/stuck_ratio_mean20', 0)
        expl = r.data.metrics.get('trend/expl_net_mean20', 0)
        lvl = r.data.metrics.get('trend/level_mean20', 0)
        step = r.data.metrics.get('time/total_timesteps', 0)
        print(f"[{name}] Steps: {step} | Stuck: {stuck:.2f} | Explore: {expl:.2f} | Lvl: {lvl:.2f}")
