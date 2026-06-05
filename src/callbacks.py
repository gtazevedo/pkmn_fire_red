import os
import time
import shutil
from collections import deque
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from src.config import CFG, log, RAM_SIZE, FRAMES_PER_STEP
from src.env import PokemonEnv
from src.progress import ProgressManager

class PokemonCallback(BaseCallback):
    """
    Logging melhorado com métricas de combate detalhadas para diagnóstico.
    """

    _WINDOW    = 20
    _TRAIN_WIN = 10

    def __init__(self, total_steps_ref: list[int], verbose: int = 0):
        super().__init__(verbose)
        self._total_steps_ref = total_steps_ref

        self._hist = {k: deque(maxlen=self._WINDOW) for k in [
            "tiles", "level", "reward", "damage", "battles", "victories",
            "badges", "maps", "hp_lost", "stuck_ratio", "win_rate",
            "kill_eff", "idle_pct", "combat_net", "exploration_net",
        ]}
        self._train_hist = {k: deque(maxlen=self._TRAIN_WIN) for k in [
            "value_loss", "explained_variance", "entropy_loss",
        ]}

        self._total_episodes  = 0
        self._last_log_step   = 0
        self._training_start: float = None

    def _on_training_start(self) -> None:
        self._training_start = time.time()
        log.info("Training started.")

    def _current_phase(self) -> int:
        t = self._total_steps_ref[0]
        if t < CFG.phase1_steps:  return 1
        if t < CFG.phase2_steps:  return 2
        return 3

    @staticmethod
    def _progress_score(m: dict) -> float:
        milestone_pct = max(m["best_milestone"] + 1, 0) / 10.0
        badge_pct     = m["badges"] / 8.0
        level_pct     = min(m["party_level"] / 200.0, 1.0)
        victory_pct   = min(m["victories"] / 20.0, 1.0)
        maps_pct      = min(m["maps_visited"] / 10.0, 1.0)

        return round(
            milestone_pct * 40 +
            badge_pct     * 25 +
            level_pct     * 20 +
            victory_pct   * 10 +
            maps_pct      * 5,
            2
        )

    @staticmethod
    def _combat_net(m: dict) -> float:
        return (m["reward_entry"] + m["reward_victory"] +
                m["reward_damage"] + m["reward_idle_pen"])

    @staticmethod
    def _exploration_net(m: dict) -> float:
        return m["reward_explore"] + m["reward_map_disc"] + m["reward_milestone"]

    @staticmethod
    def _kill_efficiency(m: dict) -> float:
        b_steps = max(m["advisor_battle_steps"], 1) if m["battles_fought"] > 0 else 1
        if m["battles_fought"] == 0:
            return 0.0
        return m["damage"] / b_steps

    @staticmethod
    def _idle_pct(m: dict) -> float:
        b_steps = m["advisor_battle_steps"]
        if b_steps == 0:
            return 0.0
        idle_pen_steps = abs(m["reward_idle_pen"]) / abs(CFG.battle_idle_penalty) if CFG.battle_idle_penalty != 0 else 0
        return min(idle_pen_steps / max(b_steps, 1), 1.0)

    def _record_progress(self, m: dict) -> None:
        score = self._progress_score(m)
        self.logger.record("progress/score",           score)
        self.logger.record("progress/milestone_pct",   max(m["best_milestone"] + 1, 0) / 9.0 * 100)
        self.logger.record("progress/badges",          m["badges"])
        self.logger.record("progress/party_level",     m["party_level"])
        self.logger.record("progress/total_maps_ever", m["total_maps_ever"])
        self.logger.record("progress/total_tiles_ever",m["total_tiles_ever"])

    def _record_battle(self, m: dict) -> None:
        win_rate = m["victories"] / max(m["battles_fought"], 1)
        kill_eff = self._kill_efficiency(m)
        idle_p   = self._idle_pct(m)
        hp_eff   = m["damage"] / max(m["hp_lost"], 0.01)

        self.logger.record("battle/battles_fought",    m["battles_fought"])
        self.logger.record("battle/victories",         m["victories"])
        self.logger.record("battle/win_rate",          win_rate)
        self.logger.record("battle/kill_efficiency",   kill_eff)
        self.logger.record("battle/idle_pct",          idle_p)
        self.logger.record("battle/hp_efficiency",     hp_eff)
        self.logger.record("battle/damage_dealt",      m["damage"])
        self.logger.record("battle/hp_lost",           m["hp_lost"])
        self.logger.record("battle/whiteouts",         m["whiteouts"])
        self.logger.record("battle/battle_ratio",      m["advisor_battle_steps"] / max(m["steps"], 1))
        self.logger.record("battle/move_select_steps", m.get("advisor_move_select_steps", 0))
        combat_net = self._combat_net(m)
        battles    = max(m["battles_fought"], 1)
        self.logger.record("battle/combat_net_per_battle", combat_net / battles)
        self.logger.record("battle/farm_detected",     m.get("farm_detected", 0))
        self.logger.record("battle/farm_detections",   m.get("farm_detections", 0))

    def _record_reward(self, m: dict) -> None:
        combat_net = self._combat_net(m)
        expl_net   = self._exploration_net(m)

        self.logger.record("reward/total",           m["total_reward"])
        self.logger.record("reward/combat_net",      combat_net)
        self.logger.record("reward/exploration_net", expl_net)
        self.logger.record("reward/from_explore",    m["reward_explore"])
        self.logger.record("reward/from_map_disc",   m["reward_map_disc"])
        self.logger.record("reward/from_damage",     m["reward_damage"])
        self.logger.record("reward/from_entry",      m["reward_entry"])
        self.logger.record("reward/from_victory",    m["reward_victory"])
        self.logger.record("reward/from_idle_pen",   m["reward_idle_pen"])
        self.logger.record("reward/from_levelup",    m["reward_levelup"])
        self.logger.record("reward/from_badge",      m["reward_badge"])
        self.logger.record("reward/from_milestone",  m["reward_milestone"])
        self.logger.record("reward/from_stuck_pen",  m["reward_stuck"])
        self.logger.record("reward/from_time_pen",   m["reward_time_pen"])
        self.logger.record("reward/from_text",       m["reward_text"])

    def _record_game(self, m: dict) -> None:
        self.logger.record("game/unique_tiles",      m["unique_tiles"])
        self.logger.record("game/maps_visited",      m["maps_visited"])
        self.logger.record("game/levels_gained",     m["levels_gained"])
        self.logger.record("game/episode_steps",     m["steps"])
        self.logger.record("game/stuck_steps_ratio", m["stuck_steps_ratio"])
        self.logger.record("game/best_milestone",    m["best_milestone"])

    def _record_trends(self) -> None:
        if len(self._hist["reward"]) < 5:
            return
        for tb_key, hist_key in [
            ("trend/reward_mean20",      "reward"),
            ("trend/tiles_mean20",       "tiles"),
            ("trend/level_mean20",       "level"),
            ("trend/maps_mean20",        "maps"),
            ("trend/battles_mean20",     "battles"),
            ("trend/victories_mean20",   "victories"),
            ("trend/win_rate_mean20",    "win_rate"),
            ("trend/kill_eff_mean20",    "kill_eff"),
            ("trend/idle_pct_mean20",    "idle_pct"),
            ("trend/damage_mean20",      "damage"),
            ("trend/badges_mean20",      "badges"),
            ("trend/combat_net_mean20",  "combat_net"),
            ("trend/expl_net_mean20",    "exploration_net"),
            ("trend/stuck_ratio_mean20", "stuck_ratio"),
        ]:
            vals = list(self._hist[hist_key])
            if vals:
                self.logger.record(tb_key, float(np.mean(vals)))

    def _record_training(self, ep: int) -> None:
        if not self._training_start:
            return
        elapsed = time.time() - self._training_start
        self.logger.record("training/episodes_total",   ep)
        self.logger.record("training/curriculum_phase", self._current_phase())
        self.logger.record("training/steps_per_sec",    self.num_timesteps / max(elapsed, 1))
        self.logger.record("training/hours_elapsed",    elapsed / 3600)

    def _record_health(self) -> None:
        try:
            ntv = self.logger.name_to_value
            for k, hk in [("train/value_loss", "value_loss"),
                           ("train/explained_variance", "explained_variance"),
                           ("train/entropy_loss", "entropy_loss")]:
                if k in ntv:
                    self._train_hist[hk].append(ntv[k])
        except Exception:
            pass

        if len(self._train_hist["value_loss"]) >= 3:
            vl_ma  = float(np.mean(self._train_hist["value_loss"]))
            ev_ma  = float(np.mean(self._train_hist["explained_variance"]))
            ent_ma = float(np.mean(self._train_hist["entropy_loss"]))
            self.logger.record("health/value_loss_ma10",  vl_ma)
            self.logger.record("health/ev_ma10",          ev_ma)
            self.logger.record("health/policy_entropy",   -ent_ma)

    def _update_histories(self, m: dict) -> None:
        self._hist["tiles"].append(m["unique_tiles"])
        self._hist["level"].append(m["party_level"])
        self._hist["reward"].append(m["total_reward"])
        self._hist["damage"].append(m["damage"])
        self._hist["battles"].append(m["battles_fought"])
        self._hist["victories"].append(m["victories"])
        self._hist["badges"].append(m["badges"])
        self._hist["maps"].append(m["maps_visited"])
        self._hist["hp_lost"].append(m["hp_lost"])
        self._hist["stuck_ratio"].append(m["stuck_steps_ratio"])
        self._hist["win_rate"].append(m["victories"] / max(m["battles_fought"], 1))
        self._hist["kill_eff"].append(self._kill_efficiency(m))
        self._hist["idle_pct"].append(self._idle_pct(m))
        self._hist["combat_net"].append(self._combat_net(m))
        self._hist["exploration_net"].append(self._exploration_net(m))

    def _log_console(self, ep: int, m: dict) -> None:
        event_record = m.get("best_milestone", 0)
        score      = self._progress_score(m)
        win_rate   = m["victories"] / max(m["battles_fought"], 1)
        combat_net = self._combat_net(m)
        farm_flag  = " 🚨FARM" if m.get("farm_detected", 0) else ""
        log.info(
            f"[Ep {ep:>5}|Ph{self._current_phase()}] "
            f"score={score:>5.1f}  "
            f"events={event_record:>4}  "
            f"lvl={m['party_level']:>3}  "
            f"bat={m['battles_fought']:>2} win={win_rate:.0%}  "
            f"cnet={combat_net:>7.1f}  "
            f"idle={m['reward_idle_pen']:>7.1f}  "
            f"tiles={m['unique_tiles']:>4}(Σ{m['total_tiles_ever']:>5})  "
            f"maps={m['maps_visited']:>2}(Σ{m['total_maps_ever']:>2})  "
            f"R={m['total_reward']:>7.1f}{farm_flag}"
        )

    def _on_step(self) -> bool:
        self._total_steps_ref[0] = self.num_timesteps

        if self.num_timesteps % 1000 < self.training_env.num_envs:
            try:
                with open(CFG.steps_state_file, "w") as f:
                    f.write(str(self.num_timesteps))
            except Exception:
                pass

        for idx, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            m = self.locals["infos"][idx].get("episode_metrics")
            if m is None:
                continue

            self._total_episodes += 1
            ep = self._total_episodes

            self._update_histories(m)
            self._record_progress(m)
            self._record_battle(m)
            self._record_reward(m)
            self._record_game(m)
            self._record_trends()
            self._record_training(ep)
            self._record_health()
            self._log_console(ep, m)

        since = self.num_timesteps - self._last_log_step
        if since >= CFG.log_freq_steps * self.training_env.num_envs:
            self._last_log_step = self.num_timesteps
            avg_r = float(np.mean(self._hist["reward"])) if self._hist["reward"] else 0.0
            avg_s = float(np.mean(list(self._hist["win_rate"]))) if self._hist["win_rate"] else 0.0
            log.debug(
                f"[Step {self.num_timesteps:>9,}]  "
                f"phase={self._current_phase()}  eps={self._total_episodes}  "
                f"reward20={avg_r:.1f}  winrate20={avg_s:.1%}"
            )

        return True


def make_lr_schedule() -> callable:
    lr = CFG.lr_start
    def _constant_lr(progress_remaining: float) -> float:
        return lr
    return _constant_lr


class Trainer:
    def __init__(self, reset_training: bool = False):
        self._model_file      = CFG.model_path + ".zip"
        self._total_steps_ref = [0]

        if reset_training:
            self._clear_artefacts()

        os.makedirs(CFG.checkpoint_dir, exist_ok=True)
        os.makedirs(CFG.whiteout_frame_dir, exist_ok=True)
        self._env       = self._make_env()
        self._callbacks = self._make_callbacks()
        self._model     = self._load_or_create_model()

    @staticmethod
    def _clear_artefacts() -> None:
        log.info("reset_training=True — clearing previous run artefacts")
        for path in [
            CFG.model_path + ".zip",
            CFG.model_path + "_vecnorm.pkl",
            CFG.tensorboard_log_dir,
            CFG.checkpoint_dir,
        ]:
            if not os.path.exists(path):
                continue
            (shutil.rmtree if os.path.isdir(path) else os.remove)(path)

    def _make_env(self) -> VecNormalize:
        log.info(f"Spawning {CFG.num_cpu} SubprocVecEnv workers …")
        ref = self._total_steps_ref

        def _make(i: int):
            def _init():
                env = PokemonEnv(env_id=i)
                env._total_steps_ref = ref
                return env
            return _init

        venv = SubprocVecEnv([_make(i) for i in range(CFG.num_cpu)])

        venv = VecNormalize(
            venv,
            norm_obs=False,
            norm_reward=True,
            clip_reward=10.0,
            gamma=CFG.gamma,
        )
        log.info("VecNormalize wrapping applied (norm_reward=True, clip_reward=10.0)")
        return venv

    def _make_callbacks(self) -> CallbackList:
        checkpoint_cb = CheckpointCallback(
            save_freq   = max(CFG.checkpoint_freq // CFG.num_cpu, 1),
            save_path   = CFG.checkpoint_dir,
            name_prefix = "pokemon_lstm",
        )
        pokemon_cb = PokemonCallback(total_steps_ref=self._total_steps_ref)
        return CallbackList([pokemon_cb, checkpoint_cb])

    def _load_or_create_model(self) -> RecurrentPPO:
        lr_schedule = make_lr_schedule()

        if os.path.exists(self._model_file):
            stamp_file  = self._model_file.replace(".zip", ".stamp")
            saved_stamp = ""
            if os.path.exists(stamp_file):
                with open(stamp_file) as f:
                    saved_stamp = f.read().strip()

            if saved_stamp != CFG.config_stamp:
                log.warning(
                    f"Stale checkpoint detected!\n"
                    f"  Saved stamp  : '{saved_stamp}'\n"
                    f"  Current stamp: '{CFG.config_stamp}'\n"
                    f"RAM_SIZE=20, VecNormalize, first_strike_bonus, BugFix A+B (v14).\n"
                    f"Deletando checkpoint stale e iniciando do zero."
                )
                os.remove(self._model_file)
                vecnorm_path = CFG.model_path + "_vecnorm.pkl"
                if os.path.exists(vecnorm_path):
                    os.remove(vecnorm_path)
                if os.path.exists(stamp_file):
                    os.remove(stamp_file)
            else:
                log.info(f"Resuming from {self._model_file} (stamp OK: {saved_stamp})")
                model = RecurrentPPO.load(CFG.model_path, env=self._env)
                self._total_steps_ref[0] = model.num_timesteps

                vecnorm_path = CFG.model_path + "_vecnorm.pkl"
                if os.path.exists(vecnorm_path):
                    self._env = VecNormalize.load(vecnorm_path, self._env.venv)
                    self._env.training = True
                    log.info(f"VecNormalize stats restored from {vecnorm_path}")
                else:
                    log.warning(
                        "VecNormalize stats not found — normalizador recomeça do zero."
                    )

                model.set_env(self._env)
                model.learning_rate = lr_schedule
                model.batch_size    = CFG.batch_size
                model.n_epochs      = CFG.n_epochs
                log.info(
                    f"Resumed at step {model.num_timesteps:,} → "
                    f"phase {self._current_phase()}  "
                    f"lr={CFG.lr_start}  batch={CFG.batch_size}  "
                    f"epochs={CFG.n_epochs}  ent_coef={CFG.ent_coef}"
                )
                return model

        log.info("Creating new RecurrentPPO from scratch")
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            self._env,
            verbose=1,
            n_steps=CFG.n_steps,
            batch_size=CFG.batch_size,
            n_epochs=CFG.n_epochs,
            learning_rate=lr_schedule,
            gamma=CFG.gamma,
            gae_lambda=CFG.gae_lambda,
            ent_coef=CFG.ent_coef,
            tensorboard_log=CFG.tensorboard_log_dir,
        )
        return model

    def _current_phase(self) -> int:
        t = self._total_steps_ref[0]
        if t < CFG.phase1_steps:  return 1
        if t < CFG.phase2_steps:  return 2
        return 3

    def train(self) -> None:
        log.info("Training started. Launch TensorBoard with:")
        log.info(f"  tensorboard --logdir {CFG.tensorboard_log_dir}")
        log.info(f"Live log: {CFG.log_file}")

        learn_iter     = 0
        training_start = time.time()

        try:
            while True:
                learn_iter += 1
                iter_start  = time.time()

                self._model.learn(
                    total_timesteps=CFG.learn_batch,
                    reset_num_timesteps=False,
                    callback=self._callbacks,
                )
                self._model.save(CFG.model_path)
                self._env.save(CFG.model_path + "_vecnorm.pkl")

                self._total_steps_ref[0] = self._model.num_timesteps
                stamp_file = CFG.model_path + ".stamp"
                with open(stamp_file, "w") as f:
                    f.write(CFG.config_stamp)
                    
                # [USER REQUEST] Salva um print do ambiente a cada checkpoint para inspeção do Agente "Juiz"
                try:
                    import cv2
                    frames = self._env.get_attr("_last_raw_frame")
                    if frames and frames[0] is not None:
                        img_path = "/mnt/c/Users/guilh/.gemini/antigravity/brain/7277d84d-02ea-4c17-85c5-47df9833bc22/checkpoint_latest.png"
                        cv2.imwrite(img_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
                except Exception as e:
                    log.error(f"Failed to save checkpoint screenshot: {e}")

                elapsed_total = time.time() - training_start
                elapsed_iter  = time.time() - iter_start
                fps = CFG.learn_batch / elapsed_iter
                log.info(
                    f"[Iter {learn_iter:>4}|Ph{self._current_phase()}] "
                    f"total_steps={self._model.num_timesteps:>9,}  "
                    f"iter_time={elapsed_iter:.1f}s  "
                    f"total_time={elapsed_total / 3600:.2f}h  "
                    f"steps/s={fps:.0f}  "
                    f"gba_fps={fps * FRAMES_PER_STEP:.0f}  "
                    f"model saved ✓"
                )

        except KeyboardInterrupt:
            log.info("KeyboardInterrupt — saving model …")
            self._model.save(CFG.model_path)
            self._env.save(CFG.model_path + "_vecnorm.pkl")
            stamp_file = CFG.model_path + ".stamp"
            with open(stamp_file, "w") as f:
                f.write(CFG.config_stamp)
            log.info("Done.")

        finally:
            self._env.close()
