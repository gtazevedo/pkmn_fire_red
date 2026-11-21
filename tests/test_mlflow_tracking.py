import mlflow
from mlflow.tracking import MlflowClient

import scripts.train as train_script
from src.config import CFG
from src.mlflow_utils import (
    EXPERIMENT_TRAIN,
    REGISTERED_MODEL_NAME,
    build_run_name,
    close_orphan_runs,
    ensure_experiment_metadata,
    register_checkpoint_version,
)


def test_ensure_experiment_metadata_and_build_run_name(monkeypatch, tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri)
    exp_id = ensure_experiment_metadata(
        client,
        EXPERIMENT_TRAIN,
        "Test FireRed HRL experiment",
        {"game": "Pokemon-FireRed-GBA", "curriculum": "FrontierBias"},
    )
    exp = client.get_experiment(exp_id)
    assert exp.tags.get("mlflow.note.content") == "Test FireRed HRL experiment"
    assert exp.tags.get("game") == "Pokemon-FireRed-GBA"
    assert exp.tags.get("curriculum") == "FrontierBias"

    monkeypatch.delenv("MLFLOW_RUN_NAME", raising=False)
    monkeypatch.setattr("src.mlflow_utils.resolve_git_short", lambda: "abc1234")
    auto_name = build_run_name()
    assert CFG.config_stamp in auto_name
    assert "abc1234" in auto_name
    assert CFG.ollama_model.replace(":", "-") in auto_name
    assert build_run_name(suffix="eval").endswith("_eval")

    monkeypatch.setenv("MLFLOW_RUN_NAME", "custom_batch_run")
    assert build_run_name() == "custom_batch_run"


def test_registry_versioning_and_champion_promotion(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(uri)
    exp_id = client.create_experiment(
        EXPERIMENT_TRAIN,
        artifact_location=str(tmp_path / "mlruns"),
    )

    def _make_run(name: str):
        with mlflow.start_run(experiment_id=exp_id, run_name=name) as run:
            dummy = tmp_path / f"{name}.txt"
            dummy.write_text("ckpt")
            mlflow.log_artifact(str(dummy))
            return run.info.run_id

    r1 = _make_run("run1")
    r2 = _make_run("run2")
    r3 = _make_run("run3")

    v1 = register_checkpoint_version(
        client,
        run_id=r1,
        description="v1",
        tags={"timesteps": "1000"},
        aliases=("current",),
        promote_champion_metric=120.0,
    )
    assert str(v1) == "1"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion").version) == "1"

    # Métrica maior deve promover @champion para v2
    v2 = register_checkpoint_version(
        client,
        run_id=r2,
        description="v2",
        tags={"timesteps": "2000"},
        aliases=("current",),
        promote_champion_metric=340.5,
    )
    assert str(v2) == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion").version) == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "current").version) == "2"

    # Métrica menor não deve derrubar @champion de v2
    v3 = register_checkpoint_version(
        client,
        run_id=r3,
        description="v3",
        tags={"timesteps": "3000"},
        aliases=("current",),
        promote_champion_metric=210.0,
    )
    assert str(v3) == "3"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion").version) == "2"
    assert str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "current").version) == "3"


def test_close_orphan_runs_marks_killed_preserving_active(tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(uri)
    exp_id = client.create_experiment(EXPERIMENT_TRAIN)

    stale = client.create_run(exp_id, run_name="stale_orphan")
    active = client.create_run(exp_id, run_name="still_running")

    closed = close_orphan_runs(client, EXPERIMENT_TRAIN, keep_run_ids=(active.info.run_id,))
    assert closed == [stale.info.run_id]
    assert client.get_run(stale.info.run_id).info.status == "KILLED"
    assert client.get_run(active.info.run_id).info.status == "RUNNING"


def test_train_main_logs_tags_params_and_finishes_on_keyboard_interrupt(monkeypatch, tmp_path):
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    class FakeTrainer:
        def __init__(self, reset_training: bool = False):
            self.reset_training = reset_training

        def train(self) -> None:
            raise KeyboardInterrupt("simulated SIGTERM/SIGINT shutdown")

    monkeypatch.setattr(train_script, "Trainer", FakeTrainer)
    monkeypatch.setattr(train_script, "_install_sigterm_handler", lambda: None)

    try:
        train_script.main(reset_training=False, tracking_uri=uri)
    except KeyboardInterrupt:
        pass

    client = MlflowClient(uri)
    exp = client.get_experiment_by_name(EXPERIMENT_TRAIN)
    assert exp is not None
    assert "RecurrentPPO" in exp.tags.get("mlflow.note.content", "")

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert run.data.tags.get("run_type") == "train"
    assert run.data.tags.get("config_stamp") == CFG.config_stamp
    assert run.data.tags.get("ollama_model") == CFG.ollama_model
    assert run.data.tags.get("llm_enabled") == "true"
    assert run.data.params.get("config_stamp") == CFG.config_stamp
    assert "RAM_SIZE" in run.data.params


def test_all_training_storage_paths_live_under_base_save_dir():
    import os
    from src.mlflow_utils import MLFLOW_ARTIFACT_ROOT, MLFLOW_TRACKING_URI

    base = os.path.abspath(CFG.base_save_dir)
    paths = [
        CFG.model_path,
        CFG.checkpoint_dir,
        CFG.progress_dir,
        CFG.pallet_exterior_state_file,
        CFG.tensorboard_log_dir,
        CFG.log_file,
        CFG.whiteout_frame_dir,
        CFG.screenshot_dir,
        CFG.mlflow_dir,
        CFG.mlflow_db_path,
        CFG.mlflow_artifact_dir,
        CFG.steps_state_file,
    ]
    for p in paths:
        abs_p = os.path.abspath(p)
        assert abs_p == base or abs_p.startswith(base + os.sep), (
            f"Path {p!r} -> {abs_p!r} fora de CFG.base_save_dir ({base!r})"
        )

    assert MLFLOW_TRACKING_URI == f"sqlite:///{os.path.abspath(CFG.mlflow_db_path)}"
    assert MLFLOW_ARTIFACT_ROOT == os.path.abspath(CFG.mlflow_artifact_dir)
