import os
import random
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import retro
import cv2

from src.config import CFG, RAM_SIZE, log, ActionSpace
from src.reader import RamReader
from src.advisor import RamAdvisor
from src.progress import EpisodeStats, RewardAccumulator, get_progress_manager

class PokemonEnv(gym.Env):
    """
    Gymnasium wrapper around stable-retro PokemonFireRed.
    """

    def __init__(self, env_id: int = 0):
        super().__init__()
        self.env_id = env_id
        self.env    = retro.make(game="PokemonFireRed", state=None)
        self._total_steps_ref: list[int] = [0]
        self._cached_total_steps: int    = 0

        self.action_space = spaces.Discrete(ActionSpace.size())
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
            "ram":   spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
        })

        self._stats    = EpisodeStats()
        self._rewards  = RewardAccumulator()
        self._advisor  = RamAdvisor()
        self._progress = get_progress_manager()

        self._episode_num: int = 0
        self._last_raw_frame: Optional[np.ndarray] = None

        # [FIX v13-A] Farm threshold randomizado — inicializado no primeiro reset()
        self._farm_ratio_threshold: float = CFG.farm_ratio_threshold_max
        # Propaga para _stats imediatamente (antes do primeiro reset)
        self._stats._farm_ratio_threshold = self._farm_ratio_threshold

        # [FIX v15-Stairs] Controle de loop de escadas/transição de mapa (A -> B -> A)
        self._map_history = []
        self._last_map_transition_step = 0

        # [FIX v17] Rastreamento de incentivos positivos
        self._explore_reward_per_map: dict = {}   # cap diferenciado por mapa
        self._prev_map_key: tuple = ()            # para detectar transições
        self._ever_left_interior: bool = False    # para early termination interior-only
        self._steps_interior_only: int = 0        # contador de steps só em bank=4

    @property
    def max_steps(self) -> int:
        t = self._cached_total_steps
        if t < CFG.phase1_steps:  return CFG.max_steps_p1
        if t < CFG.phase2_steps:  return CFG.max_steps_p2
        return CFG.max_steps_p3

    def _read_total_steps(self) -> int:
        try:
            with open(CFG.steps_state_file, "r") as f:
                return int(f.read().strip())
        except Exception:
            return 0

    @staticmethod
    def _to_gray_84(img: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, -1)

    def _build_obs(self, raw_frame: np.ndarray, info: dict) -> dict:
        ms = self.max_steps
        base_ram = np.array([
            RamReader.coord(info, "player_x")  / CFG.max_coord,
            RamReader.coord(info, "player_y")  / CFG.max_coord,
            RamReader.map_id(info)             / CFG.max_map_id,
            RamReader.map_bank(info)           / CFG.max_map_id,
            float(RamReader.in_battle(info)),
            float(RamReader.script_lock(info)),
            float(RamReader.player_moving(info)),
            RamReader.enemy_hp(info)           / CFG.max_enemy_hp,
            RamReader.party_hp(info)           / CFG.max_party_hp,
            RamReader.party_level(info)        / CFG.max_party_level_sum,
            RamReader.badges(info)             / 8.0,
            self._stats.steps                  / ms,
        ], dtype=np.float32)

        text_ram = self._advisor.get_embedding()

        # [FIX 12] battle_idle_ratio
        idle_ratio = np.array([
            min(self._stats.battle_idle_steps, 100) / 100.0,
        ], dtype=np.float32)

        # [FIX v11-J] battle_outcome_potential: progresso HP% da batalha atual
        outcome_potential = np.array([
            self._stats.battle_outcome_potential,
        ], dtype=np.float32)

        # [FIX v13-F] steps_in_battle_current normalizado
        # Permite ao LSTM distinguir "acabei de entrar" de "estou aqui há 300 steps"
        steps_in_battle_norm = np.array([
            min(self._stats.steps_in_battle_current, CFG.steps_in_battle_norm_cap)
            / CFG.steps_in_battle_norm_cap,
        ], dtype=np.float32)

        return {
            "image": self._to_gray_84(raw_frame),
            "ram":   np.concatenate([base_ram, text_ram, idle_ratio,
                                     outcome_potential, steps_in_battle_norm]),
        }

    def _warm_up(self, n: int = 10) -> tuple:
        obs, info = None, {}
        for _ in range(n):
            obs, _, _, _, info = self.env.step(ActionSpace.NO_OP)
        return obs, info

    def _run_action(self, action: np.ndarray) -> tuple:
        total   = CFG.frames_held + CFG.frames_noop
        frames  = []
        raw_rgb = None

        for i in range(total):
            a = action if i < CFG.frames_held else ActionSpace.NO_OP
            obs, _, _, _, info = self.env.step(a)
            if i == CFG.frames_held - 1:
                raw_rgb = obs
            if CFG.pool_start <= i < CFG.pool_end:
                frames.append(obs)

        pooled = np.max(np.stack(frames), axis=0)
        return raw_rgb, pooled, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self._episode_num += 1

        self._cached_total_steps = self._read_total_steps()

        # [FIX v13-A] Sorteia novo farm_ratio_threshold para este episódio.
        self._farm_ratio_threshold = random.uniform(
            CFG.farm_ratio_threshold_min,
            CFG.farm_ratio_threshold_max,
        )
        
        self._map_history.clear()
        self._last_map_transition_step = 0

        state_file = self._progress.current_state_file
        log.info(f"[Env {self.env_id}] Loading GBA state file: {state_file}")
        with open(state_file, "rb") as f:
            self.env.em.set_state(f.read())

        # [FIX v11-H] Warmup estável: exige 3 leituras consecutivas iguais de level>0
        raw_frame, info = self._warm_up()
        prev_lvl  = 0
        stable    = 0
        for _ in range(20):
            lvl = RamReader.party_level(info)
            if lvl > 0 and lvl == prev_lvl:
                stable += 1
                if stable >= 2:
                    break
            else:
                stable = 0
            prev_lvl  = lvl
            raw_frame, info = self._warm_up(n=5)

        RamReader.debug_dump(info, self.env_id, "POST-RESET")
        self._stats.reset(info, ram_array=self.env.get_ram())
        # [FIX v13-A] Propaga o threshold sorteado para _stats (usado em update_battle)
        self._stats._farm_ratio_threshold = self._farm_ratio_threshold
        self._rewards.reset()
        self._advisor.reset()
        self._last_raw_frame = raw_frame

        # [FIX v15-Stairs] Reset do controle de loop
        self._map_history.clear()
        self._last_map_transition_step = 0
        self._stairs_loop_count = 0

        # [FIX v17] Reset incentivos positivos
        self._explore_reward_per_map  = {}
        self._prev_map_key            = ()
        self._ever_left_interior      = False
        self._steps_interior_only     = 0

        log.debug(
            f"[Env {self.env_id}] Reset ep={self._episode_num} — "
            f"level={self._stats.max_level}  "
            f"level_baseline_set={self._stats._level_baseline_set}  "
            f"party_hp={self._stats.last_party_hp}  "
            f"badges={self._stats.last_badges}  "
            f"max_steps={self.max_steps}  "
            f"farm_threshold={self._farm_ratio_threshold:.2f}"
        )

        return self._build_obs(raw_frame, info), {}

    def step(self, action_idx: int):
        action                   = ActionSpace.get(action_idx)
        raw_frame, pooled, info  = self._run_action(action)
        self._last_raw_frame     = raw_frame

        _in_battle_now  = RamReader.in_battle(info)
        _script_lock    = bool(RamReader.script_lock(info))
        
        step_reward = 0.0
        # [FIX v18.2] Suspensão da punição de tempo durante diálogos
        if not _script_lock:
            step_reward += self._rewards.add("time_pen", CFG.time_penalty)
        ram_array       = self.env.get_ram()
        text_bonus      = self._advisor.update(_in_battle_now, _script_lock, action_idx, ram_array)
        step_reward    += self._rewards.add("text", text_bonus)

        # [FIX v18.3] Novas recompensas Whiddy-style
        op_lvl_bonus = self._stats.update_max_op_level(info)
        step_reward += self._rewards.add("op_lvl", op_lvl_bonus)
        
        heal_bonus = self._stats.update_heal_reward(info)
        step_reward += self._rewards.add("heal", heal_bonus)

        event_bonus, total_flags = self._stats.update_events(ram_array)
        if event_bonus > 0:
            step_reward += self._rewards.add("milestone", event_bonus)
            self._progress.check_and_save(self.env, total_flags, self.env_id)

        x        = RamReader.coord(info, "player_x")
        y        = RamReader.coord(info, "player_y")
        map_id   = RamReader.map_id(info)
        map_bank = RamReader.map_bank(info)
        tile     = (map_bank, map_id, x, y)
        map_key  = (map_bank, map_id)

        # [FIX v18.1] Salva dinamicamente o state exterior apenas quando ele está
        # a uma distância segura da porta (y >= 10), para evitar que nasça pisando
        # no warp e volte para dentro de casa acidentalmente.
        if map_bank == 3 and map_id == 0 and y >= 10:
            if not getattr(self, '_saved_exterior_safe', False):
                if not os.path.exists(CFG.pallet_exterior_state_file):
                    state_data = self.env.em.get_state()
                    os.makedirs(os.path.dirname(CFG.pallet_exterior_state_file), exist_ok=True)
                    with open(CFG.pallet_exterior_state_file, "wb") as f:
                        f.write(state_data)
                    log.info(f"[Env {self.env_id}] 🎉 SALVO STATE EXTERIOR A UMA DISTANCIA SEGURA DA PORTA (y={y})!")
                self._saved_exterior_safe = True

        # [FIX v15-Stairs] Detecção de loop de transição de mapas (A -> B -> A)
        # Se o mapa mudou em relação ao último mapa registrado
        if not self._map_history or map_key != self._map_history[-1]:
            # Se já temos histórico e voltamos para o mapa de antes do último (A -> B -> A)
            if len(self._map_history) >= 2 and map_key == self._map_history[-2]:
                steps_since_transition = self._stats.steps - self._last_map_transition_step
                
                # [FIX v18] Só aplica punição se o loop inteiro for dentro de mapas internos (bank=4).
                # Isso permite transições casa <-> rua sem punição acidental.
                all_indoor = (map_key[0] == 4 and self._map_history[-1][0] == 4 and self._map_history[-2][0] == 4)
                
                if steps_since_transition < 150 and all_indoor:
                    self._stairs_loop_count += 1
                    if self._stairs_loop_count >= 3:
                        penalty = -15.0
                        step_reward += self._rewards.add("stuck", penalty)
                        terminated = True
                        log.info(f"[Env {self.env_id}] 🚨 STAIRS LOOP FATAL! 3x repetido. Terminating (penalty {penalty:.1f})")
                    else:
                        penalty = -3.0
                        step_reward += self._rewards.add("stuck", penalty)
                        log.info(f"[Env {self.env_id}] 🚨 STAIRS LOOP DETECTED (A->B->A inside)! Penalty {penalty:.1f} applied.")
            else:
                # Se não for um loop de A->B->A, reseta o contador de loop de escadas consecutivo
                self._stairs_loop_count = 0
            
            # Adiciona ao histórico e mantém limite de tamanho 3
            self._map_history.append(map_key)
            if len(self._map_history) > 3:
                self._map_history.pop(0)
                
            self._last_map_transition_step = self._stats.steps

        # Não recompensa exploração durante batalha (evita reward espúrio de teleporte pós-batalha)
        if not _in_battle_now:

            is_interior = (map_bank == 4)

            # [FIX v17-A] Early termination se nunca saiu de mapas interiores
            if is_interior:
                self._steps_interior_only += 1
                # [FIX v18] Penalidade calibrada para forçar saída (-0.005/step)
                step_reward += self._rewards.add("stuck", CFG.indoor_step_penalty)
            else:
                self._ever_left_interior   = True
                self._steps_interior_only  = 0

            # [FIX v17-B] Outdoor sustain bonus: incentivo positivo por estar lá fora
            if not is_interior:
                step_reward += self._rewards.add("milestone", CFG.outdoor_sustain_bonus)

            explore_r, is_new_map = self._stats.update_exploration(tile, map_key)

            # [FIX v17-C] Cap diferenciado: interior esgota rápido, exterior é abundante
            cap = CFG.map_explore_cap_interior if is_interior else CFG.map_explore_cap_exterior
            already_earned = self._explore_reward_per_map.get(map_key, 0.0)
            if already_earned >= cap:
                explore_r = 0.0
            else:
                explore_r = min(explore_r, cap - already_earned)
                self._explore_reward_per_map[map_key] = already_earned + explore_r

            step_reward += self._rewards.add("explore", explore_r)
            
            if is_new_map:
                # Reset y_min_episode to current y so we don't get a massive north bonus just by changing map
                self._stats.y_min_episode = y

            # North bonus: y decresce ao ir para norte no FireRed
            if y < self._stats.y_min_episode:
                north_steps = self._stats.y_min_episode - y
                north_r = north_steps * CFG.north_bonus_per_tile
                self._stats.y_min_episode = y
                step_reward += self._rewards.add("milestone", north_r)
                log.debug(f"[Env {self.env_id}] North progress! y={y} (+{north_r:.1f})")
        else:
            is_new_map = False

        if is_new_map:
            self._rewards.add("map_disc", CFG.new_map_bonus if map_bank != 4 else CFG.new_map_bonus_route)
            log.info(f"[Env {self.env_id}] New map: bank={map_key[0]} id={map_key[1]}")

        step_reward += self._rewards.add(
            "stuck",
            self._stats.update_stuck(in_battle=_in_battle_now, script_lock=_script_lock)
        )

        dmg_reward, idle_pen, vic_bonus, is_new_battle, whiteout_pen, farm_terminate = self._stats.update_battle(
            _in_battle_now,
            RamReader.enemy_hp(info),
            RamReader.party_hp(info),
            script_lock=_script_lock,
            last_raw_frame=self._last_raw_frame,
            env_id=self.env_id,
            episode_num=self._episode_num,
            current_tile=tile,
        )
        step_reward += self._rewards.add("damage", dmg_reward)

        if idle_pen != 0.0:
            step_reward += self._rewards.add("idle_pen", idle_pen)

        if vic_bonus > 0.0:
            step_reward += self._rewards.add("victory", vic_bonus)

        if whiteout_pen != 0.0:
            step_reward += self._rewards.add("stuck", whiteout_pen)
            log.info(f"[Env {self.env_id}] WHITEOUT! Penalty {whiteout_pen:.1f}")

        if is_new_battle:
            if tile in self._stats.whiteout_tile_cooldown and self._stats.whiteout_tile_cooldown[tile] > 0:
                actual_entry = CFG.entry_bonus_revisit  # 0.0
                log.debug(f"[Env {self.env_id}] Entry suppressed (whiteout cooldown)")
            else:
                actual_entry = CFG.entry_bonus
            step_reward += self._rewards.add("entry", actual_entry)
            log.debug(f"[Env {self.env_id}] Battle entered! +{actual_entry:.1f} entry bonus")

        if dmg_reward > 0.0 and not self._stats.first_strike_paid:
            self._stats.first_strike_paid = True
            step_reward += self._rewards.add("damage", CFG.first_strike_bonus)
            log.debug(f"[Env {self.env_id}] FIRST STRIKE! +{CFG.first_strike_bonus:.1f}")

        if _in_battle_now:
            step_reward += self._rewards.add("text", CFG.battle_sustain_bonus)

        # [FIX v11-D] A bonus em batalha com cap aumentado
        if _in_battle_now and not _script_lock and action_idx == ActionSpace.NAMES.index("A"):
            self._stats.battle_total_a += 1
            if self._stats.battle_a_presses < CFG.battle_a_cap:
                self._stats.battle_a_presses += 1
                step_reward += self._rewards.add("text", CFG.battle_a_bonus)

        # [FIX v11-B] B penalty só após battle_b_threshold A presses
        if (_in_battle_now and not _script_lock and
                action_idx == ActionSpace.NAMES.index("B") and
                self._stats.battle_total_a >= CFG.battle_b_threshold):
            step_reward += self._rewards.add("idle_pen", CFG.battle_b_penalty)

        step_reward += self._rewards.add(
            "levelup",
            self._stats.update_level(RamReader.party_level(info), self.env_id),
        )

        step_reward += self._rewards.add(
            "badge",
            self._stats.update_badges(RamReader.badges(info), self.env_id),
        )

        if farm_terminate:
            step_reward += CFG.farm_episode_penalty
            self._rewards.add("stuck", CFG.farm_episode_penalty)
            log.info(f"[Env {self.env_id}] 🚨 FARM PENALTY applied: {CFG.farm_episode_penalty}")

        self._stats.steps += 1
        ms = self.max_steps

        over_limit = self._stats.steps >= ms
        hard_cap   = self._stats.steps >= int(ms * 1.1)

        # [FIX v17-A] Early termination: se agente nunca saiu de bank=4 e já passou
        # o threshold, encerra o episódio sem penalidade extra. Episódio curto = poucos
        # pontos. Episódio lá fora = longo e lucrativo. Cria diferença de oportunidade.
        interior_only_terminate = (
            not self._ever_left_interior and
            not _in_battle_now and
            self._steps_interior_only >= CFG.interior_only_terminate_steps
        )
        if interior_only_terminate:
            log.debug(f"[Env {self.env_id}] Interior-only early termination at step {self._stats.steps}")

        in_post_battle_grace = self._stats.post_battle_grace_remaining > 0
        done = (hard_cap or farm_terminate or interior_only_terminate or
                (over_limit and not _in_battle_now and not in_post_battle_grace))

        obs_dict = self._build_obs(raw_frame, info)

        if done:
            advisor_stats = self._advisor.stats_dict()
            info["episode_metrics"] = {
                **self._stats.to_dict(ms),
                **self._rewards.to_dict(),
                "max_steps_used":        ms,
                "advisor_dialog_steps":  advisor_stats["advisor/dialog_steps"],
                "advisor_battle_steps":  advisor_stats["advisor/battle_steps"],
                "advisor_move_select_steps": advisor_stats["advisor/move_select_steps"],
                "advisor_hints_match":   advisor_stats["advisor/hints_matched"],
                "advisor_text_bonus":    advisor_stats["advisor/total_bonus"],
                "total_tiles_ever":      len(self._stats.persistent_tile_visits),
                "total_maps_ever":       len(self._stats.all_maps_visited),
                "best_milestone":        self._progress._best_flags_ever,
                "farm_detected":         int(self._stats.farm_detected),
                "farm_detections":       self._stats.farm_detections,
            }

        return obs_dict, step_reward, done, False, info
