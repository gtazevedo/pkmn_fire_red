import os
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np
import cv2
from src.config import CFG, log
from src.reader import RamReader

class ProgressManager:
    """
    Substitui os Milestones estáticos por um currículo dinâmico baseado em Event Flags.
    Sempre que a soma das flags aumenta em 5 (ou 10), salvamos um novo state.
    Ao iniciar um episódio, sorteamos um state dentre os já descobertos.
    """
    def __init__(self):
        os.makedirs(CFG.progress_dir, exist_ok=True)
        self._best_flags_ever: int = 0
        self._saved_states: list[str] = [CFG.state_file]
        if os.path.exists(CFG.pallet_exterior_state_file):
            self._saved_states.append(CFG.pallet_exterior_state_file)
        self._load_best_from_disk()

    @property
    def current_state_file(self) -> str:
        import random
        # Sorteia um state do pool descoberto. Dá leve preferência pros states mais avançados?
        # Para simplificar: sorteio uniforme para manter exploração ampla.
        return random.choice(self._saved_states)

    def check_and_save(self, emulator, total_flags: int, env_id: int) -> float:
        # Se quebrou o recorde histórico absoluto...
        if total_flags > self._best_flags_ever:
            # Salvamos um novo checkpoint a cada nova flag (Curriculum Learning orgânico)
            state_path = self._state_path(total_flags)
            try:
                # [FIX v18.5] Race condition protection: só salva se o arquivo não existir
                if not os.path.exists(state_path):
                    state_data = emulator.em.get_state()
                    with open(state_path, "wb") as f:
                        f.write(state_data)
                
                if state_path not in self._saved_states:
                    self._saved_states.append(state_path)
                    
                log.info(
                    f"[Env {env_id}] ★ NEW EVENT RECORD: {total_flags} flags! "
                    f"Saved/Loaded → {state_path}"
                )
                self._best_flags_ever = total_flags
            except Exception as e:
                log.warning(f"[Env {env_id}] Falha ao salvar state de flag: {e}")
        return 0.0

    def _state_path(self, total_flags: int) -> str:
        return os.path.join(CFG.progress_dir, f"events_{total_flags:03d}.state")

    def _load_best_from_disk(self) -> None:
        # Carrega states existentes do disco para o pool
        import glob
        pattern = os.path.join(CFG.progress_dir, "events_*.state")
        files = glob.glob(pattern)
        max_flag = 0
        for f in files:
            if f not in self._saved_states:
                self._saved_states.append(f)
            # Extrai o numero de flags do nome
            basename = os.path.basename(f)
            try:
                num = int(basename.replace("events_", "").replace(".state", ""))
                if num > max_flag:
                    max_flag = num
            except ValueError:
                pass
        
        self._best_flags_ever = max_flag
        if max_flag > 0:
            log.info(f"[ProgressManager] Currículo dinâmico: {len(files)} states carregados, recorde = {max_flag} flags.")
        else:
            log.info("[ProgressManager] Sem progresso salvo — iniciando do começo.")


_progress_manager: Optional[ProgressManager] = None

def get_progress_manager() -> ProgressManager:
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager


@dataclass
class RewardAccumulator:
    total:     float = 0.0
    explore:   float = 0.0
    map_disc:  float = 0.0
    damage:    float = 0.0
    levelup:   float = 0.0
    badge:     float = 0.0
    stuck:     float = 0.0
    time_pen:  float = 0.0
    text:      float = 0.0
    entry:     float = 0.0
    milestone: float = 0.0
    victory:   float = 0.0
    idle_pen:  float = 0.0
    op_lvl:    float = 0.0
    heal:      float = 0.0

    def add(self, component: str, value: float) -> float:
        setattr(self, component, getattr(self, component) + value)
        self.total += value
        return value

    def reset(self) -> None:
        self.total = self.explore = self.map_disc = self.damage = 0.0
        self.levelup = self.badge = self.stuck = self.time_pen = self.text = self.entry = self.milestone = 0.0
        self.victory = self.idle_pen = self.op_lvl = self.heal = 0.0

    def to_dict(self) -> dict:
        return {
            "total_reward":     self.total,
            "reward_explore":   self.explore,
            "reward_map_disc":  self.map_disc,
            "reward_damage":    self.damage,
            "reward_levelup":   self.levelup,
            "reward_badge":     self.badge,
            "reward_stuck":     self.stuck,
            "reward_time_pen":  self.time_pen,
            "reward_text":      self.text,
            "reward_entry":     self.entry,
            "reward_milestone": self.milestone,
            "reward_victory":   self.victory,
            "reward_idle_pen":  self.idle_pen,
            "reward_op_lvl":    self.op_lvl,
            "reward_heal":      self.heal,
        }


@dataclass
class EpisodeStats:
    # per-episode
    tile_visits:       defaultdict = field(default_factory=lambda: defaultdict(int))
    map_visits:        set         = field(default_factory=set)
    steps_on_cur_tile: int   = 0
    total_stuck_steps: int   = 0
    last_tile:         tuple = field(default_factory=tuple)

    steps:             int  = 0
    damage:            int  = 0
    hp_lost:           int  = 0
    whiteouts:         int  = 0
    max_level:         int  = 0
    levels_gained:     int  = 0
    battles_fought:    int  = 0
    victories:         int  = 0
    global_battles_fought: int = 0
    global_victories:      int = 0
    last_enemy_hp:     int  = 0
    # [FIX v11-C] HP inicial do inimigo para calcular reward por HP%
    initial_enemy_hp:  int  = 0
    # [FIX v11-E] near_ko_bonus já pago nessa batalha?
    near_ko_paid:      bool = False
    last_party_hp:     int  = 0
    was_in_battle:     bool = False
    last_badges:       int  = 0
    battle_idle_steps: int  = 0
    y_min_episode:     int  = 255
    # [FIX v11-D] Conta presses de A recompensados na batalha atual
    battle_a_presses:  int  = 0
    # [FIX v11-B] Conta presses de A totais na batalha (para threshold de B penalty)
    battle_total_a:    int  = 0
    post_battle_grace_remaining: int = 0

    # [FIX v11-J] Progresso da batalha atual para obs
    # = HP removido / HP inicial do inimigo (0.0 a 1.0)
    battle_outcome_potential: float = 0.0

    # [FIX v13-A] Farm ratio threshold — sorteado por episódio em PokemonEnv.reset()
    _farm_ratio_threshold: float = 0.55
    battle_steps_total:    int   = 0
    battle_damage_total:   int   = 0
    farm_detected:         bool  = False
    farm_detections:       int   = 0

    # [FIX v14-B] First strike tracking (resetado ao entrar em nova batalha)
    first_strike_paid:     bool  = False

    # [FIX v13-F] steps dentro da batalha atual
    steps_in_battle_current: int = 0

    # [FIX v12-C] Controle de cooldown pós-whiteout por tile
    whiteout_tile_cooldown: dict = field(default_factory=dict)

    # [FIX v18.3] Healing and max enemy level tracking
    total_healing_rew: float = 0.0
    party_size: int = 0
    max_event_sum: int = 0

    # [FIX 6] Sentinel para level baseline
    _level_baseline_set: bool = False

    # persistent across episodes
    persistent_tile_visits: defaultdict = field(
        default_factory=lambda: defaultdict(int)
    )
    all_maps_visited:       set = field(default_factory=set)
    furthest_map_count:     int = 0
    max_enemy_lvl_ever:     int = 0

    def global_win_rate(self) -> float:
        if self.global_battles_fought == 0:
            return 0.0
        return self.global_victories / self.global_battles_fought

    def reset(self, info: dict, ram_array=None) -> None:
        self.tile_visits       = defaultdict(int)
        self.map_visits        = set()
        self.steps_on_cur_tile = 0
        self.total_stuck_steps = 0
        self.last_tile         = ()
        self.steps             = 0
        self.damage            = 0
        self.hp_lost           = 0
        self.whiteouts         = 0
        self.levels_gained     = 0
        self.battles_fought    = 0
        self.victories         = 0
        self.last_enemy_hp     = 0
        self.initial_enemy_hp  = 0
        self.near_ko_paid      = False
        self.last_party_hp     = RamReader.party_hp(info)
        self.party_size        = RamReader.party_size(info)
        self.total_healing_rew = 0.0
        self.was_in_battle     = False
        self.last_badges       = RamReader.badges(info)
        self.battle_idle_steps = 0
        self.y_min_episode     = 255
        self.battle_a_presses  = 0
        self.battle_total_a    = 0
        self.post_battle_grace_remaining = 0
        self.battle_outcome_potential    = 0.0
        self.battle_steps_total          = 0
        self.battle_damage_total         = 0
        self.farm_detected               = False
        self.farm_detections             = 0
        self.first_strike_paid           = False
        self.steps_in_battle_current     = 0
        
        self.max_event_sum = RamReader.event_flags_sum(ram_array)

        # Não reseta whiteout_tile_cooldown — persiste entre episódios
        self.whiteout_tile_cooldown = {
            t: max(0, c - 1)
            for t, c in self.whiteout_tile_cooldown.items()
            if c > 1
        }

        # [FIX v11-H] Warmup estável: exige leituras estáveis de level
        initial_level = RamReader.party_level(info)
        if initial_level > 0:
            self.max_level           = initial_level
            self._level_baseline_set = True
        else:
            self.max_level           = 0
            self._level_baseline_set = False

    def update_events(self, ram_array) -> tuple[float, int]:
        total_flags = RamReader.event_flags_sum(ram_array)
        if total_flags > self.max_event_sum:
            diff = total_flags - self.max_event_sum
            self.max_event_sum = total_flags
            # Peso de 20.0 por cada flag nova descoberta
            return diff * 20.0, total_flags
        return 0.0, total_flags

    def update_exploration(self, tile: tuple, map_key: tuple) -> tuple[float, bool]:
        if map_key != getattr(self, '_last_map_for_stuck', None):
            self._last_map_for_stuck = map_key
            self.steps_on_cur_tile   = 0
            self.last_tile           = tile
        elif tile == self.last_tile:
            self.steps_on_cur_tile += 1
        else:
            self.steps_on_cur_tile = 0
            self.last_tile = tile

        self.tile_visits[tile] += 1
        self.persistent_tile_visits[tile] += 1
        n = self.persistent_tile_visits[tile]

        is_new_map      = map_key not in self.map_visits
        is_new_map_ever = map_key not in self.all_maps_visited
        self.map_visits.add(map_key)
        self.all_maps_visited.add(map_key)

        if not hasattr(self, 'explore_rewards_per_map'):
            self.explore_rewards_per_map = defaultdict(float)

        # [FIX v18.2] Regras estritas de exploração
        # 1. Bônus apenas na PRIMEIRA vez que pisa no tile (naquele episódio)
        if self.tile_visits[tile] == 1:
            base_r = CFG.explore_weight
        else:
            base_r = 0.0

        # Adiciona ao contador do mapa
        if base_r > 0:
            self.explore_rewards_per_map[map_key] += base_r

        # 2. Define o cap de exploração baseado no tipo de mapa
        map_bank = map_key[0]
        explore_cap = CFG.map_explore_cap_interior if map_bank == 4 else CFG.map_explore_cap_exterior

        # 3. Se o mapa bateu no limite de exploração
        if self.explore_rewards_per_map[map_key] > explore_cap:
            self.explore_rewards_per_map[map_key] = explore_cap
            base_r = 0.0

        # PWhiddy philosophy: No stale_map_penalty here. If fully explored, just gives 0 points.

        if is_new_map_ever:
            # [FIX v11-I] new_map_bonus diferenciado por bank
            map_bank = map_key[0]
            bonus = CFG.new_map_bonus if map_bank != 4 else CFG.new_map_bonus_route
            base_r += bonus

        return base_r, is_new_map

    def update_stuck(self, in_battle: bool = False, script_lock: bool = False) -> float:
        if self.steps_on_cur_tile >= CFG.stuck_threshold:
            self.total_stuck_steps += 1
            if in_battle or script_lock:
                return 0.0
            return CFG.stuck_penalty
        return 0.0

    def update_battle(self, in_battle: bool, enemy_hp: int, party_hp: int,
                       script_lock: bool,
                       last_raw_frame: Optional[np.ndarray], env_id: int,
                       episode_num: int,
                       current_tile: tuple,
                       party_level: int) -> tuple:
        reward         = 0.0
        idle_pen       = 0.0
        vic_bonus      = 0.0
        is_new_battle  = False
        farm_terminate = False

        if self.post_battle_grace_remaining > 0:
            self.post_battle_grace_remaining -= 1

        if in_battle and not self.was_in_battle:
            self.last_enemy_hp           = enemy_hp
            self.initial_enemy_hp        = max(enemy_hp, 1)
            self.near_ko_paid            = False
            self.first_strike_paid       = False   # [FIX v14-B] reset per battle
            self.battles_fought         += 1
            self.global_battles_fought  += 1
            self.battle_idle_steps       = 0
            self.battle_a_presses        = 0
            self.battle_total_a          = 0
            self.battle_outcome_potential = 0.0
            self.steps_in_battle_current  = 0
            is_new_battle = True

        if in_battle:
            self.battle_steps_total      += 1
            self.steps_in_battle_current += 1

            if self.initial_enemy_hp > 0:
                hp_destroyed = max(0, self.initial_enemy_hp - enemy_hp)
                self.battle_outcome_potential = hp_destroyed / self.initial_enemy_hp

            dealt_damage = (enemy_hp < self.last_enemy_hp)
            if dealt_damage:
                dmg = self.last_enemy_hp - enemy_hp
                self.damage              += dmg
                self.battle_damage_total += dmg

                if self.initial_enemy_hp > 0:
                    hp_pct_dealt = dmg / self.initial_enemy_hp
                    reward = hp_pct_dealt * CFG.hp_pct_scale
                else:
                    reward = dmg * CFG.damage_weight

                self.battle_idle_steps = 0

                if (not self.near_ko_paid and
                        self.initial_enemy_hp > 0 and
                        enemy_hp < self.initial_enemy_hp * 0.10 and
                        enemy_hp > 0):
                    vic_bonus += CFG.near_ko_bonus
                    self.near_ko_paid = True

                if enemy_hp == 0 and self.last_enemy_hp > 0:
                    vic_bonus += CFG.victory_bonus
                    self.victories += 1
                    self.global_victories += 1
                    self.post_battle_grace_remaining = CFG.post_battle_grace
                    log.info(
                        f"[Env {env_id}] ★ VICTORY! +{CFG.victory_bonus:.1f}  "
                        f"post_battle_grace={CFG.post_battle_grace} steps"
                    )
            else:
                if not script_lock:
                    self.battle_idle_steps += 1
                    if self.battle_idle_steps > CFG.battle_idle_grace:
                        idle_pen = CFG.battle_idle_penalty

            # [FIX v13-A] Detecção de farming com threshold randomizado
            if (not self.farm_detected and
                    self.battle_steps_total >= CFG.farm_detection_window and
                    self.steps > 0):
                kill_eff     = self.battle_damage_total / max(self.battle_steps_total, 1)
                battle_ratio = self.battle_steps_total / max(self.steps, 1)

                if (battle_ratio > self._farm_ratio_threshold and
                        kill_eff < CFG.farm_kill_threshold):
                    # Perdão dinâmico: se o nível for menor que 15, não pune.
                    if party_level < 15:
                        pass # Silencia o log para não dar spam a cada frame
                    else:
                        self.farm_detected    = True
                        self.farm_detections += 1
                        farm_terminate        = True
                        log.info(
                            f"[Env {env_id}] 🚨 FARM DETECTED! "
                            f"battle_ratio={battle_ratio:.2f} "
                            f"threshold={self._farm_ratio_threshold:.2f} "
                            f"kill_eff={kill_eff:.4f} "
                            f"→ terminating (penalty={CFG.farm_episode_penalty})"
                        )

        elif self.was_in_battle:
            self.battle_idle_steps        = 0
            self.battle_outcome_potential = 0.0
            self.steps_in_battle_current  = 0   # [FIX v13-F]

        whiteout_pen = 0.0
        if party_hp < self.last_party_hp:
            self.hp_lost += self.last_party_hp - party_hp
            if party_hp == 0 and self.last_party_hp > 0:
                self.whiteouts += 1

                # [FIX v12-A] Whiteout penalty escalado por idle acumulado
                idle_over_grace = max(0, self.battle_idle_steps - CFG.battle_idle_grace)
                scale = 1.0 + CFG.whiteout_idle_multiplier * idle_over_grace / 100.0
                whiteout_pen = CFG.whiteout_penalty * scale

                log.info(
                    f"[Env {env_id}] ☠ WHITEOUT! "
                    f"idle_over_grace={idle_over_grace} scale={scale:.2f} "
                    f"penalty={whiteout_pen:.1f}"
                )
                save_whiteout_frame(last_raw_frame, env_id, episode_num, self.steps)

                # [FIX v12-C] Registra tile do whiteout para cooldown de entry_bonus
                if current_tile:
                    self.whiteout_tile_cooldown[current_tile] = CFG.whiteout_entry_cooldown
                    # Também marca tiles vizinhos (±1 em x ou y)
                    bank, mid, x, y = current_tile
                    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                        neighbor = (bank, mid, x+dx, y+dy)
                        self.whiteout_tile_cooldown[neighbor] = CFG.whiteout_entry_cooldown

        self.last_enemy_hp = enemy_hp
        self.last_party_hp = party_hp
        self.was_in_battle = in_battle
        return reward, idle_pen, vic_bonus, is_new_battle, whiteout_pen, farm_terminate

    def update_level(self, current_level: int, env_id: int) -> float:
        if current_level == 0:
            return 0.0

        if not self._level_baseline_set:
            self.max_level           = current_level
            self._level_baseline_set = True
            log.debug(f"[Env {env_id}] Level baseline established at {current_level}")
            return 0.0

        delta = current_level - self.max_level
        if delta <= 0:
            return 0.0
        if delta > CFG.max_level_delta:
            log.warning(
                f"[Env {env_id}] Suspicious level jump "
                f"{self.max_level}→{current_level} (no reward)"
            )
            self.max_level = current_level
            return 0.0
        reward = delta * CFG.levelup_weight
        self.levels_gained += delta
        self.max_level      = current_level
        log.info(f"[Env {env_id}] Level up! +{delta} → total={current_level}")
        return reward

    def update_max_op_level(self, info: dict) -> float:
        current_max = RamReader.max_enemy_level(info)
        if current_max > self.max_enemy_lvl_ever:
            self.max_enemy_lvl_ever = current_max
            return self.max_enemy_lvl_ever * CFG.max_op_level_reward_scale
        return 0.0

    def update_heal_reward(self, info: dict) -> float:
        cur_hp = RamReader.party_hp(info)
        cur_party_size = RamReader.party_size(info)
        max_hp = RamReader.party_max_hp(info)
        reward = 0.0
        
        # Only reward healing if party size hasn't changed
        if cur_hp > self.last_party_hp and cur_party_size == self.party_size:
            if self.last_party_hp > 0 and max_hp > 0:
                heal_pct = (cur_hp - self.last_party_hp) / max(1, max_hp)
                reward = heal_pct * CFG.heal_reward_scale
                self.total_healing_rew += reward
        
        self.party_size = cur_party_size
        return reward

    def update_badges(self, badges: int, env_id: int) -> float:
        if badges <= self.last_badges:
            return 0.0
        reward = (badges - self.last_badges) * CFG.badge_weight
        log.info(f"[Env {env_id}] Badge earned! badges={badges}  bonus={reward:.1f}")
        self.last_badges = badges
        return reward

    def to_dict(self, max_steps: int) -> dict:
        return {
            "unique_tiles":      len(self.tile_visits),
            "maps_visited":      len(self.map_visits),
            "party_level":       self.max_level,
            "levels_gained":     self.levels_gained,
            "badges":            self.last_badges,
            "battles_fought":    self.battles_fought,
            "victories":         self.victories,
            "damage":            self.damage,
            "hp_lost":           self.hp_lost,
            "whiteouts":         self.whiteouts,
            "steps":             self.steps,
            "stuck_steps_ratio": self.total_stuck_steps / max(self.steps, 1),
            "farm_detected":     int(self.farm_detected),
            "farm_detections":   self.farm_detections,
            "battle_ratio":      self.battle_steps_total / max(self.steps, 1),
        }

def save_whiteout_frame(
    frame: Optional[np.ndarray],
    env_id: int,
    episode_num: int,
    step: int,
) -> None:
    if frame is None:
        log.warning(f"[Env {env_id}] Whiteout at step {step} — no frame to save")
        return
    try:
        os.makedirs(CFG.whiteout_frame_dir, exist_ok=True)
        fname = f"whiteout_env{env_id:02d}_ep{episode_num:05d}_step{step:06d}.png"
        path  = os.path.join(CFG.whiteout_frame_dir, fname)
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        log.info(f"[Env {env_id}] ☠  WHITEOUT frame saved → {path}")
    except Exception as e:
        log.warning(f"[Env {env_id}] Could not save whiteout frame: {e}")
