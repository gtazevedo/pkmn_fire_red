import os
import re
import math
import glob
import random
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import numpy as np
import cv2
from src.config import CFG, GRID_SIZE, log
from src.reader import RamReader

class ProgressManager:
    """
    Currículo dinâmico baseado em Event Flags com amostragem priorizada
    (Frontier Bias + Genesis Anchor anti-esquecimento catastrófico).
    Distribuição alvo (com >= 3 states):
      - ~50% nos estados de fronteira (maior contagem de Event Flags / marcos)
      - ~30% em estados intermediários recentes
      - ~20% em estados iniciais, dos quais ~60% (~12% do total) permanecem
        ancorados nos primeiros estados (Start.state.gz e pallet_exterior.state)
        independentemente de quantos marcos existam em progress_states/.
    """
    MAP_PROGRESSION_RANK: dict[tuple[int, int], int] = {
        (3, 19): 1,
        (3, 1): 2,
        (5, 4): 3,
        (5, 3): 4,
        (3, 20): 13,
        (15, 0): 14,
        (1, 0): 15,
        (15, 2): 16,
        (3, 2): 17,
        (6, 2): 18,
        (3, 21): 19,
        (1, 1): 20,
        (3, 22): 21,
        (3, 3): 22,
    }
    PARCEL_PROGRESSION_RANK: dict[tuple[int, int], int] = {
        (5, 3): 5,
        (3, 1): 6,
        (3, 19): 7,
        (3, 0): 8,
        (4, 3): 9,
    }
    POST_POKEDEX_PROGRESSION_RANK: dict[tuple[int, int], int] = {
        (3, 0): 10,
        (3, 19): 11,
        (3, 1): 12,
        (3, 20): 13,
        (15, 0): 14,
        (1, 0): 15,
        (15, 2): 16,
        (3, 2): 17,
        (6, 2): 18,
        (3, 21): 19,
        (1, 1): 20,
        (3, 22): 21,
        (3, 3): 22,
    }

    def __init__(self, progress_dir: Optional[str] = None, state_file: Optional[str] = None):
        self._progress_dir = progress_dir or CFG.progress_dir
        os.makedirs(self._progress_dir, exist_ok=True)
        self._best_flags_ever: int = 0
        self._best_map_rank: int = 0
        self.tier_counts: dict[str, int] = {"frontier": 0, "intermediate": 0, "initial": 0}
        self.genesis_samples: int = 0
        self.start_state_samples: int = 0
        self._saved_states: list[str] = [state_file or CFG.state_file]
        if progress_dir is None and os.path.exists(CFG.pallet_exterior_state_file):
            self._saved_states.append(CFG.pallet_exterior_state_file)
        self._load_best_from_disk()

    def sample_state(self, rng: Optional[random.Random] = None) -> str:
        self._load_best_from_disk(quiet=True)
        chooser = rng if rng is not None else random
        if not self._saved_states:
            return CFG.state_file

        sorted_states = sorted(self._saved_states, key=self._progression_key)
        n = len(sorted_states)

        if n == 1:
            self.tier_counts["initial"] += 1
            self.genesis_samples += 1
            self.start_state_samples += 1
            return sorted_states[0]

        if n == 2:
            total_p = CFG.frontier_prob + CFG.initial_prob
            threshold = (CFG.frontier_prob / total_p) if total_p > 0 else 0.5
            if chooser.random() < threshold:
                self.tier_counts["frontier"] += 1
                return sorted_states[1]
            self.tier_counts["initial"] += 1
            self.genesis_samples += 1
            self.start_state_samples += 1
            return sorted_states[0]

        n_frontier = max(1, min(3, n // 3))
        n_initial = max(1, (n - n_frontier) // 2)
        initial_states = sorted_states[:n_initial]
        frontier_states = sorted_states[-n_frontier:]
        intermediate_states = sorted_states[n_initial:-n_frontier]

        u = chooser.random()
        if u < CFG.frontier_prob:
            self.tier_counts["frontier"] += 1
            if len(frontier_states) > 1 and chooser.random() < CFG.frontier_top_state_share:
                return frontier_states[-1]
            if len(frontier_states) > 1:
                return chooser.choice(frontier_states[:-1])
            return frontier_states[0]
        if u < CFG.frontier_prob + CFG.intermediate_prob:
            self.tier_counts["intermediate"] += 1
            return chooser.choice(intermediate_states)
        self.tier_counts["initial"] += 1
        anchor_k = max(1, min(len(initial_states), CFG.genesis_anchor_count))
        genesis_states = initial_states[:anchor_k]
        early_replay_states = initial_states[anchor_k:]
        if not early_replay_states or chooser.random() < CFG.genesis_anchor_share:
            picked = chooser.choice(genesis_states)
            self.genesis_samples += 1
            if picked == sorted_states[0]:
                self.start_state_samples += 1
            return picked
        return chooser.choice(early_replay_states)

    @property
    def current_state_file(self) -> str:
        return self.sample_state()

    @property
    def _best_state_file(self) -> str:
        if not self._saved_states:
            return CFG.state_file
        return max(self._saved_states, key=self._state_sort_key)

    @property
    def _best_milestone_idx(self) -> int:
        return self._best_flags_ever

    def stats_dict(self) -> dict:
        total = sum(self.tier_counts.values())
        return {
            "progress/saved_states": len(self._saved_states),
            "progress/best_flags_ever": self._best_flags_ever,
            "progress/frontier_samples": self.tier_counts["frontier"],
            "progress/intermediate_samples": self.tier_counts["intermediate"],
            "progress/initial_samples": self.tier_counts["initial"],
            "progress/frontier_ratio": self.tier_counts["frontier"] / max(total, 1),
            "progress/genesis_samples": self.genesis_samples,
            "progress/start_state_samples": self.start_state_samples,
            "progress/genesis_ratio": self.genesis_samples / max(total, 1),
            "progress/start_state_ratio": self.start_state_samples / max(total, 1),
        }

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
                self._best_map_rank = 0
            except Exception as e:
                log.warning(f"[Env {env_id}] Falha ao salvar state de flag: {e}")
        return 0.0

    def check_and_save_map_milestone(
        self,
        emulator,
        total_flags: int,
        map_key: tuple[int, int],
        env_id: int,
        has_parcel: bool = False,
        has_pokedex: bool = False,
    ) -> float:
        if has_pokedex:
            rank = self.POST_POKEDEX_PROGRESSION_RANK.get(map_key, 0)
        elif has_parcel:
            rank = self.PARCEL_PROGRESSION_RANK.get(map_key, 0)
        else:
            rank = self.MAP_PROGRESSION_RANK.get(map_key, 0)
        if rank == 0:
            return 0.0
        if rank > self._best_map_rank or total_flags > self._best_flags_ever:
            state_path = os.path.join(
                self._progress_dir,
                f"events_{total_flags:03d}_m{rank:02d}_{map_key[0]}_{map_key[1]}.state",
            )
            try:
                already_exists = os.path.exists(state_path)
                if not already_exists:
                    state_data = emulator.em.get_state()
                    with open(state_path, "wb") as f:
                        f.write(state_data)
                if state_path not in self._saved_states:
                    self._saved_states.append(state_path)
                self._best_flags_ever = max(self._best_flags_ever, total_flags)
                self._best_map_rank = max(self._best_map_rank, rank)
                if not already_exists:
                    log.info(
                        f"[Env {env_id}] 🗺️ NEW MAP MILESTONE: map={map_key} rank={rank} "
                        f"parcel={has_parcel} pokedex={has_pokedex} flags={total_flags} → {state_path}"
                    )
            except Exception as e:
                log.warning(f"[Env {env_id}] Falha ao salvar map milestone: {e}")
        return 0.0

    @staticmethod
    def _state_flags(path: str) -> int:
        basename = os.path.basename(path)
        match = re.match(r"events_(\d+)", basename)
        if match:
            return int(match.group(1))
        if "pallet_exterior" in basename:
            return 1
        return 0

    @classmethod
    def _progression_key(cls, path: str) -> tuple[int, int, int, float]:
        flags = cls._state_flags(path)
        rank = cls._state_rank(path)
        if rank >= 10 or flags >= 125:
            stage = 3
        elif flags >= 124:
            stage = 2
        elif rank > 0:
            stage = 1
        else:
            stage = 0
        parcel_sub = 1 if (stage == 1 and rank >= 5) else 0
        return (stage, parcel_sub, rank, cls._state_sort_key(path))

    @staticmethod
    def _state_rank(path: str) -> int:
        basename = os.path.basename(path)
        match = re.match(r"events_\d+_m(\d+)(?:_\d+_\d+)?\.state$", basename)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _state_sort_key(path: str) -> float:
        """Extrai o número de flags (e sub-rank geográfico) do nome do arquivo para ordenação."""
        basename = os.path.basename(path)
        match = re.match(r"events_(\d+)(?:_m(\d+)(?:_\d+_\d+)?)?\.state$", basename)
        if match:
            flags = int(match.group(1))
            rank = int(match.group(2)) if match.group(2) else 0
            return flags + rank / 100.0
        # Start.state e pallet_exterior → prioridade baixa
        if "pallet_exterior" in basename:
            return 1.0
        return 0.0

    def _state_path(self, total_flags: int) -> str:
        return os.path.join(self._progress_dir, f"events_{total_flags:03d}.state")

    def _load_best_from_disk(self, quiet: bool = False) -> None:
        # Carrega states existentes do disco para o pool
        pattern = os.path.join(self._progress_dir, "events_*.state")
        files = glob.glob(pattern)
        max_flag = self._best_flags_ever
        max_rank = self._best_map_rank
        for f in files:
            if f not in self._saved_states:
                self._saved_states.append(f)
            # Extrai o numero de flags (e rank geografico) do nome
            basename = os.path.basename(f)
            match = re.match(r"events_(\d+)(?:_m(\d+)(?:_\d+_\d+)?)?\.state$", basename)
            if match:
                num = int(match.group(1))
                rank = int(match.group(2)) if match.group(2) else 0
                max_flag = max(max_flag, num)
                max_rank = max(max_rank, rank)

        self._best_flags_ever = max_flag
        self._best_map_rank = max_rank
        if not quiet:
            if max_flag > 0:
                log.info(
                    f"[ProgressManager] Currículo dinâmico: {len(files)} states carregados, "
                    f"recorde = {max_flag} flags (map_rank={max_rank})."
                )
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
    total:      float = 0.0
    explore:    float = 0.0
    map_disc:   float = 0.0
    damage:     float = 0.0
    levelup:    float = 0.0
    badge:      float = 0.0
    stuck:      float = 0.0
    time_pen:   float = 0.0
    text:       float = 0.0
    entry:      float = 0.0
    milestone:  float = 0.0
    victory:    float = 0.0
    idle_pen:   float = 0.0
    op_lvl:     float = 0.0
    heal:       float = 0.0
    tactical:   float = 0.0
    strategist: float = 0.0

    def add(self, component: str, value: float) -> float:
        setattr(self, component, getattr(self, component) + value)
        self.total += value
        return value

    def reset(self) -> None:
        self.total = self.explore = self.map_disc = self.damage = 0.0
        self.levelup = self.badge = self.stuck = self.time_pen = self.text = self.entry = self.milestone = 0.0
        self.victory = self.idle_pen = self.op_lvl = self.heal = self.tactical = self.strategist = 0.0

    def to_dict(self) -> dict:
        return {
            "total_reward":      self.total,
            "reward_explore":    self.explore,
            "reward_map_disc":   self.map_disc,
            "reward_damage":     self.damage,
            "reward_levelup":    self.levelup,
            "reward_badge":      self.badge,
            "reward_stuck":      self.stuck,
            "reward_time_pen":   self.time_pen,
            "reward_text":       self.text,
            "reward_entry":      self.entry,
            "reward_milestone":  self.milestone,
            "reward_victory":    self.victory,
            "reward_idle_pen":   self.idle_pen,
            "reward_op_lvl":     self.op_lvl,
            "reward_heal":       self.heal,
            "reward_tactical":   self.tactical,
            "reward_strategist": self.strategist,
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
    y_max_episode:     int  = 0
    x_min_episode:     int  = 255
    x_max_episode:     int  = 0
    map_battles:       defaultdict = field(default_factory=lambda: defaultdict(int))
    current_battle_map: tuple = field(default_factory=tuple)
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
    current_battle_damage: int   = 0
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
        if "map_bank" in info or "map_id" in info:
            self.map_visits = {(RamReader.map_bank(info), RamReader.map_id(info))}
        else:
            self.map_visits = set()
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
        if "player_x" in info:
            init_x = RamReader.coord(info, "player_x")
            self.x_min_episode = init_x
            self.x_max_episode = init_x
        else:
            self.x_min_episode = 255
            self.x_max_episode = 0
        if "player_y" in info:
            init_y = RamReader.coord(info, "player_y")
            self.y_min_episode = init_y
            self.y_max_episode = init_y
        else:
            self.y_min_episode = 255
            self.y_max_episode = 0
        self.map_battles        = defaultdict(int)
        self.current_battle_map = ()
        self.battle_a_presses  = 0
        self.battle_total_a    = 0
        self.post_battle_grace_remaining = 0
        self.battle_outcome_potential    = 0.0
        self.battle_steps_total          = 0
        self.battle_damage_total         = 0
        self.current_battle_damage       = 0
        self.farm_detected               = False
        self.farm_detections             = 0
        self.first_strike_paid           = False
        self.steps_in_battle_current     = 0
        self.explore_rewards_per_map     = defaultdict(float)
        
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
            # Peso de 40.0 por cada flag nova descoberta
            return diff * 40.0, total_flags
        return 0.0, total_flags

    def update_exploration(
        self,
        tile: tuple,
        map_key: tuple,
        include_new_map_bonus: bool = True,
        script_lock: bool = False,
    ) -> tuple[float, bool]:
        if script_lock:
            self.steps_on_cur_tile = 0
            self.last_tile = tile
            self._last_map_for_stuck = map_key
        elif map_key != getattr(self, '_last_map_for_stuck', None):
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

        # [FIX v27 / v31] Decaimento limitado com piso estacionário (explore_novelty_floor)
        if self.tile_visits[tile] == 1:
            n_persistent = self.persistent_tile_visits[tile]
            novelty_factor = CFG.explore_novelty_floor + (1.0 - CFG.explore_novelty_floor) / (
                1.0 + math.log1p(min(n_persistent, 50))
            )
            base_r = CFG.explore_weight * novelty_factor
            # Bônus extra para tiles em mapas exteriores (routes com grama)
            if map_key[0] == 3:  # bank=3 = exterior/routes
                base_r += CFG.explore_weight_outdoor_bonus * novelty_factor
        else:
            base_r = 0.0

        # Adiciona ao contador do mapa
        if base_r > 0:
            self.explore_rewards_per_map[map_key] += base_r

        # 2. Define o cap de exploração baseado no tipo de mapa
        map_bank = map_key[0]
        explore_cap = CFG.map_explore_cap_interior if map_bank >= 4 else CFG.map_explore_cap_exterior

        # 3. Se o mapa bateu no limite de exploração
        if self.explore_rewards_per_map[map_key] > explore_cap:
            self.explore_rewards_per_map[map_key] = explore_cap
            base_r = 0.0

        # PWhiddy philosophy: No stale_map_penalty here. If fully explored, just gives 0 points.

        if include_new_map_bonus and is_new_map_ever:
            # [FIX v11-I / v30b] new_map_bonus diferenciado por bank (exterior/route = bank 3)
            map_bank = map_key[0]
            bonus = CFG.new_map_bonus_route if map_bank == 3 else CFG.new_map_bonus
            base_r += bonus

        return base_r, is_new_map

    def get_visited_grid(
        self,
        map_key: tuple[int, int],
        px: int,
        py: int,
        grid_size: int = GRID_SIZE,
    ) -> np.ndarray:
        """Retorna matriz local grid_size x grid_size de visitas no episódio em [0.0, 1.0]."""
        grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        radius = grid_size // 2
        bank, mid = map_key
        for row, dy in enumerate(range(-radius, radius + 1)):
            ty = py + dy
            for col, dx in enumerate(range(-radius, radius + 1)):
                tx = px + dx
                visits = self.tile_visits.get((bank, mid, tx, ty), 0)
                if visits > 0:
                    grid[row, col] = min(visits, 5) / 5.0
        return grid

    def update_stuck(self, in_battle: bool = False, script_lock: bool = False) -> float:
        if self.steps_on_cur_tile >= CFG.stuck_threshold:
            self.total_stuck_steps += 1
            if in_battle or script_lock:
                return 0.0
            return CFG.stuck_penalty
        return 0.0

    def current_battle_fatigue(self, has_parcel: bool = False) -> float:
        n = self.map_battles.get(self.current_battle_map, self.battles_fought)
        is_outdoor_route = bool(self.current_battle_map) and self.current_battle_map[0] == 3
        if is_outdoor_route:
            if has_parcel:
                if n <= 1:
                    return 0.5
                if n == 2:
                    return 0.2
                return 0.05
            if n <= 1:
                return 1.0
            if n == 2:
                return 0.5
            if n == 3:
                return 0.25
            return 0.10

        # Em ginásios, dungeons (bank=1) e batalhas de história (Lab bank=4), nunca zera
        if n <= 1:
            return 1.0
        if n == 2:
            return 0.5
        if n == 3:
            return 0.25
        return 0.1

    def update_battle(self, in_battle: bool, enemy_hp: int, party_hp: int,
                       script_lock: bool,
                       last_raw_frame: Optional[np.ndarray], env_id: int,
                       episode_num: int,
                       current_tile: tuple,
                       party_level: int,
                       has_parcel: bool = False) -> tuple:
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
            self.current_battle_map      = (
                (current_tile[0], current_tile[1])
                if current_tile and len(current_tile) >= 2
                else (-1, -1)
            )
            self.map_battles[self.current_battle_map] += 1
            self.battle_idle_steps       = 0
            self.battle_a_presses        = 0
            self.battle_total_a          = 0
            self.battle_outcome_potential = 0.0
            self.steps_in_battle_current  = 0
            self.current_battle_damage    = 0
            is_new_battle = True

        fatigue = self.current_battle_fatigue(has_parcel=has_parcel)

        if in_battle:
            self.battle_steps_total      += 1
            self.steps_in_battle_current += 1

            if self.initial_enemy_hp > 0:
                hp_destroyed = max(0, self.initial_enemy_hp - enemy_hp)
                self.battle_outcome_potential = hp_destroyed / self.initial_enemy_hp

            dealt_damage = (enemy_hp < self.last_enemy_hp)
            if dealt_damage:
                dmg = self.last_enemy_hp - enemy_hp
                self.damage                += dmg
                self.battle_damage_total   += dmg
                self.current_battle_damage += dmg

                if self.initial_enemy_hp > 0:
                    hp_pct_dealt = dmg / self.initial_enemy_hp
                    reward = hp_pct_dealt * CFG.hp_pct_scale * fatigue
                else:
                    reward = dmg * CFG.damage_weight * fatigue

                self.battle_idle_steps = 0

                if (not self.near_ko_paid and
                        self.initial_enemy_hp > 0 and
                        enemy_hp < self.initial_enemy_hp * 0.10 and
                        enemy_hp > 0):
                    vic_bonus += CFG.near_ko_bonus * fatigue
                    self.near_ko_paid = True

                if enemy_hp == 0 and self.last_enemy_hp > 0:
                    actual_vic_bonus = CFG.victory_bonus * fatigue
                    vic_bonus += actual_vic_bonus
                    self.victories += 1
                    self.global_victories += 1
                    self.post_battle_grace_remaining = CFG.post_battle_grace
                    log.info(
                        f"[Env {env_id}] ★ VICTORY! +{actual_vic_bonus:.1f} (fatigue={fatigue:.2f}) "
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
            if enemy_hp == 0 and self.last_enemy_hp > 0:
                actual_vic_bonus = CFG.victory_bonus * fatigue
                vic_bonus += actual_vic_bonus
                self.victories += 1
                self.global_victories += 1
                self.post_battle_grace_remaining = CFG.post_battle_grace
                log.info(
                    f"[Env {env_id}] ★ LATE VICTORY! +{actual_vic_bonus:.1f} (fatigue={fatigue:.2f}) "
                )

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

    def update_level(
        self,
        current_level: int,
        env_id: int,
        badges: int = 0,
        in_battle: bool = False,
        has_parcel: bool = False,
    ) -> float:
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

        level_cap = CFG.max_rewarded_level_base + max(0, int(badges)) * CFG.max_rewarded_level_per_badge
        rewarded_delta = max(0, min(current_level, level_cap) - self.max_level)
        reward = rewarded_delta * CFG.levelup_weight
        if in_battle and has_parcel:
            reward *= 0.5
        self.levels_gained += delta
        self.max_level      = current_level
        log.info(f"[Env {env_id}] Level up! +{delta} → total={current_level} (rew={reward:.1f}, cap={level_cap})")
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
        
        # Only reward healing if party size hasn't changed and prior HP was genuinely low
        if (
            cur_hp > self.last_party_hp
            and cur_party_size == self.party_size
            and self.last_party_hp > 0
            and max_hp > 0
            and (self.last_party_hp / max_hp) < CFG.heal_min_prior_hp_ratio
            and (cur_hp - self.last_party_hp) >= max(3, int(max_hp * 0.15))
        ):
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
