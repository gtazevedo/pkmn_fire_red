import os
import re
import math
import random
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
try:
    import retro
except ImportError:  # pragma: no cover
    retro = None
import cv2

from src.config import BATTLE_SIZE, CFG, GOAL_SIZE, GRID_SIZE, RAM_SIZE, log, ActionSpace
from src.battle_data import score_moves
from src.reader import RamReader
from src.advisor import RamAdvisor
from src.strategist import Strategist
from src.progress import EpisodeStats, RewardAccumulator, get_progress_manager
from src.vision import VisionEngine, GameState

class PokemonEnv(gym.Env):
    """
    Gymnasium wrapper around stable-retro PokemonFireRed.
    """

    def __init__(
        self,
        env_id: int = 0,
        fixed_state_file: Optional[str] = None,
        enable_stochasticity: bool = True,
    ):
        super().__init__()
        if retro is None:
            raise ImportError("stable-retro é necessário para instanciar PokemonEnv")
        self.env_id = env_id
        self._fixed_state_file = fixed_state_file
        self._enable_stochasticity = bool(enable_stochasticity)
        self.env    = retro.make(game="PokemonFireRed", state=None)
        self._total_steps_ref: list[int] = [0]
        self._cached_total_steps: int    = 0

        self.action_space = spaces.Discrete(ActionSpace.size())
        self.observation_space = spaces.Dict({
            "image":          spaces.Box(0, 255, (84, 84, 1), dtype=np.uint8),
            "ram":            spaces.Box(0.0, 1.0, (RAM_SIZE,), dtype=np.float32),
            "collision_grid": spaces.Box(0.0, 1.0, (GRID_SIZE, GRID_SIZE), dtype=np.float32),
            "visited_grid":   spaces.Box(0.0, 1.0, (GRID_SIZE, GRID_SIZE), dtype=np.float32),
            "goal":           spaces.Box(0.0, 1.0, (GOAL_SIZE,), dtype=np.float32),
            "battle":         spaces.Box(0.0, 1.0, (BATTLE_SIZE,), dtype=np.float32),
        })

        self._stats      = EpisodeStats()
        self._rewards    = RewardAccumulator()
        self._advisor    = RamAdvisor()
        self._strategist = Strategist(advisor=self._advisor)
        self._progress   = get_progress_manager()
        self._vision     = VisionEngine()

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
        self._battle_screenshot_saved: bool = False
        self._had_parcel_in_ep: bool = False
        self._had_teachy_tv_in_ep: bool = False
        self._battle_potion_attempts: int = 0
        self._battle_potion_cooldown: int = 0
        self._battle_capture_attempts: int = 0
        self._battle_capture_cooldown: int = 0
        self._episode_tm_attempts: int = 0
        self._mart_buy_rewarded_in_ep: bool = False
        self._mart_buy_attempts: int = 0
        self._owned_species_in_ep: set[int] = set()
        self._prev_party_size_in_ep: int = 0
        self._last_battle_best_slot: int = 0
        self._last_battle_best_mult: float = 1.0
        self._last_was_trainer_battle: bool = False
        self._battle_captured_this_battle: bool = False
        self._last_overworld_tile: tuple = ()
        self._pending_event_save_flags: int = 0
        self._steps_in_current_map: int = 0
        self._last_overworld_move_action: Optional[int] = None
        self._last_script_lock: bool = False
        self._last_cb2: int = 134571445
        self._battle_switch_cooldown: int = 0
        self._smart_flee_rewarded_since_heal: bool = False

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
        ram = self.env.get_ram()
        RamReader.enrich_info_from_ram(info, ram)
        px_norm = RamReader.coord(info, "player_x") / CFG.max_coord
        py_norm = RamReader.coord(info, "player_y") / CFG.max_coord
        if (
            self._enable_stochasticity
            and CFG.coord_dropout_prob > 0.0
            and random.random() < CFG.coord_dropout_prob
        ):
            px_norm = 0.0
            py_norm = 0.0
        p_hp = RamReader.party_hp(info)
        p_max_hp = RamReader.party_max_hp(info)
        party_hp_pct = max(0.0, min(1.0, p_hp / float(p_max_hp))) if p_max_hp > 0 else 0.0
        potions = RamReader.get_potions_count(ram)
        balls = RamReader.get_pokeballs_count(ram)
        money = RamReader.get_money(ram)
        mk = (RamReader.map_bank(info), RamReader.map_id(info))
        battle_obs = RamReader.get_battle_observation(ram, info)

        base_ram = np.array([
            px_norm,
            py_norm,
            RamReader.map_id(info)             / CFG.max_map_id,
            RamReader.map_bank(info)           / CFG.max_map_id,
            float(RamReader.is_in_battle(info)),
            # Use RAM for script lock if available, but CV overrides later
            float(info.get('script_lock', 0)),
            float(RamReader.player_moving(info)),
            RamReader.enemy_hp(info)           / CFG.max_enemy_hp,
            party_hp_pct,
            RamReader.party_level(info)        / CFG.max_party_level_sum,
            RamReader.badges(info)             / 8.0,
            self._stats.steps                  / ms,
            # [FIX v22] Sixth Sense
            min(potions, 99) / 99.0,
            min(RamReader.get_tms_count(ram), 99) / 99.0,
            float(RamReader.get_p1_status(ram)),
        ], dtype=np.float32)

        cb2_obs = int(info.get("gMain_callback2", 0) or 0)
        in_bat_obs = RamReader.is_in_battle(info) or (
            getattr(self._stats, "was_in_battle", False) and cb2_obs != 134571445 and cb2_obs != 0
        )
        p_mon_obs = RamReader.get_battle_mon(ram, 0)
        active_hp_obs = p_mon_obs["hp"] if p_mon_obs["max_hp"] > 0 else p_hp
        active_max_obs = p_mon_obs["max_hp"] if p_mon_obs["max_hp"] > 0 else max(p_max_hp, 1)
        active_hp_pct_obs = active_hp_obs / float(max(active_max_obs, 1))
        need_switch_obs = in_bat_obs and p_mon_obs["max_hp"] > 0 and p_mon_obs["hp"] == 0 and p_hp > 0
        can_potion_obs = (
            in_bat_obs
            and potions > 0
            and 0 < active_hp_pct_obs < 0.50
            and self._battle_potion_attempts < CFG.max_potion_macro_attempts_per_battle
            and self._battle_potion_cooldown == 0
        )
        can_capture_obs = bool(battle_obs[11] > 0.5)
        can_mart_obs = (
            not in_bat_obs
            and mk in ((5, 3), (6, 0))
            and RamReader.has_pokedex(ram)
            and not RamReader.has_oaks_parcel(ram)
            and ((potions < 2 and money >= 300) or (balls < 3 and money >= 200))
            and self._mart_buy_attempts < 2
        )
        is_trainer_obs = RamReader.is_trainer_battle(ram) if in_bat_obs else False
        should_flee_obs = (
            in_bat_obs
            and (not is_trainer_obs)
            and potions == 0
            and 0 < active_hp_pct_obs < 0.35
            and party_hp_pct < 0.35
        )
        tactical_signal = 1.0 if (need_switch_obs or can_potion_obs or can_capture_obs or can_mart_obs) else (0.5 if should_flee_obs else 0.0)
        text_ram = self._advisor.get_embedding().copy()
        text_ram[4] = tactical_signal

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

        px = RamReader.coord(info, "player_x")
        py = RamReader.coord(info, "player_y")
        return {
            "image":          self._to_gray_84(raw_frame),
            "ram":            np.concatenate([base_ram, text_ram, idle_ratio,
                                              outcome_potential, steps_in_battle_norm]),
            "collision_grid": RamReader.get_collision_grid_normalized(ram, info),
            "visited_grid":   self._stats.get_visited_grid(mk, px, py, GRID_SIZE),
            "goal":           self._strategist.get_goal_embedding(),
            "battle":         battle_obs,
        }

    def _warm_up(self, n: int = 10) -> tuple:
        obs, info = None, {}
        for _ in range(n):
            obs, _, _, _, info = self.env.step(ActionSpace.NO_OP)
        return obs, info

    def _random_safe_walk(self, raw_frame: Optional[np.ndarray], info: dict) -> tuple:
        if not (self._enable_stochasticity and CFG.reset_random_walk_max > 0):
            return raw_frame, info
        ram = self.env.get_ram()
        RamReader.enrich_info_from_ram(info, ram)
        if (
            int(info.get("gMain_callback2", 0) or 0) != 134571445
            or RamReader.is_script_locked(ram)
            or RamReader.is_in_battle(info)
        ):
            return raw_frame, info

        origin_map = (RamReader.map_bank(info), RamReader.map_id(info))
        n_walk = random.randint(0, CFG.reset_random_walk_max)
        center = GRID_SIZE // 2
        cardinal_offsets = (
            (2, center - 1, center),  # UP
            (3, center + 1, center),  # DOWN
            (4, center, center - 1),  # LEFT
            (5, center, center + 1),  # RIGHT
        )
        for _ in range(n_walk):
            grid = RamReader.get_collision_grid(ram, info)
            px_w, py_w = RamReader.get_player_coords(ram, info)
            safe_moves = [
                act_idx
                for act_idx, r_idx, c_idx in cardinal_offsets
                if grid[r_idx, c_idx] == RamReader.TILE_WALKABLE
                and not (origin_map == (3, 20) and py_w >= 75 and act_idx == 3)
            ]
            if not safe_moves:
                break
            chosen = random.choice(safe_moves)
            raw_frame, _, info = self._run_action(ActionSpace.get(chosen))
            ram = self.env.get_ram()
            RamReader.enrich_info_from_ram(info, ram)
            cur_map = (RamReader.map_bank(info), RamReader.map_id(info))
            if (
                cur_map != origin_map
                or RamReader.is_script_locked(ram)
                or RamReader.is_in_battle(info)
                or int(info.get("gMain_callback2", 0) or 0) != 134571445
            ):
                break
        return raw_frame, info

    _MACRO_BTN_IDX = {
        "B": 0, "SELECT": 2, "START": 3,
        "UP": 4, "DOWN": 5, "LEFT": 6, "RIGHT": 7, "A": 8,
    }

    def _macro_step(self, btn: Optional[str], hold: int = 5, delay: int = 15) -> dict:
        vec = np.zeros(12, dtype=np.int8)
        if btn is not None and btn in self._MACRO_BTN_IDX:
            vec[self._MACRO_BTN_IDX[btn]] = 1
        noop = np.zeros(12, dtype=np.int8)
        info = {}
        for _ in range(hold):
            _, _, _, _, info = self.env.step(vec)
        for _ in range(delay):
            _, _, _, _, info = self.env.step(noop)
        return info

    def _open_battle_bag(self) -> bool:
        """Aguarda menu principal de batalha pronto, posiciona em BAG (slot 1) e confirma abertura (0x08107EE1)."""
        info: dict = {}
        battle_txt = ""
        for _ in range(32):
            ram = self.env.get_ram()
            battle_txt = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram))
            if "what will" in battle_txt:
                break
            cb2_now = int(info.get("gMain_callback2", 0) or 0)
            if cb2_now == 134571445:
                return False
            p_mon = RamReader.get_battle_mon(ram, 0)
            if p_mon["max_hp"] > 0 and p_mon["hp"] == 0:
                return False
            info = self._macro_step("B", hold=4, delay=10)
        else:
            ram = self.env.get_ram()
            battle_txt = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram))

        if "what will" not in battle_txt:
            return False
        self._macro_step(None, hold=0, delay=12)

        main_cur, _ = RamReader.get_battle_cursors(ram)
        if main_cur in (2, 3):  # PKMN ou RUN -> subir para linha superior
            self._macro_step("UP", hold=5, delay=8)
            main_cur, _ = RamReader.get_battle_cursors(self.env.get_ram())
        if main_cur == 0:       # FIGHT -> mover para direita (BAG)
            self._macro_step("RIGHT", hold=5, delay=8)
        if RamReader.get_battle_cursors(self.env.get_ram())[0] != 1:
            return False

        info = self._macro_step("A", hold=5, delay=20)
        for _ in range(6):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if cb2 == 0x08107EE1:
                self._macro_step(None, hold=0, delay=45)  # Aguarda fade-in completo da Mochila
                return True
            info = self._macro_step(None, hold=0, delay=10)
        return False

    def _run_potion_macro(self, potion_slot: int = 0) -> None:
        log.info(f"[Env {self.env_id}] Executing Potion Macro (slot={potion_slot})...")
        if not self._open_battle_bag():
            return
        pots_before = RamReader.get_potions_count(self.env.get_ram())
        # Clamping à esquerda garante cair no bolso 0 (ITEMS), mesmo que a Bag lembre outro bolso
        for _ in range(2):
            self._macro_step("LEFT", hold=5, delay=25)
        for _ in range(max(0, min(potion_slot, 15))):
            self._macro_step("DOWN", hold=5, delay=8)
        self._macro_step("A", hold=5, delay=20)   # Seleciona item -> submenu USE/CANCEL
        info = self._macro_step("A", hold=5, delay=15)  # Seleciona USE

        def _in_bag_or_party(cb2_val: int) -> bool:
            return cb2_val == 0x08107EE1 or (0x08110000 <= cb2_val <= 0x0812FFFF)

        # Aguarda transição da Mochila (0x08107EE1) para o Menu da Equipe (0x0811EBD1 -> 0x0811EBA1)
        for _ in range(10):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if cb2 != 0x08107EE1:
                break
            info = self._macro_step(None, hold=0, delay=10)
        for _ in range(8):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if cb2 == 0x0811EBA1 or not _in_bag_or_party(cb2):
                break
            info = self._macro_step(None, hold=0, delay=10)
        if _in_bag_or_party(int(info.get("gMain_callback2", 0) or 0)):
            self._macro_step(None, hold=0, delay=15)

        # Aplica no Pokémon 1 (slot 0) e avança texto de cura ("HP was restored...") até sair do Party Menu
        for _ in range(6):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if not _in_bag_or_party(cb2):
                break
            info = self._macro_step("A", hold=5, delay=30)

        # Fallback: se ainda preso na Bag/Party Menu, termina diálogo (A se já usou Poção, senão B)
        for _ in range(5):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if not _in_bag_or_party(cb2):
                break
            if RamReader.get_potions_count(self.env.get_ram()) < pots_before:
                info = self._macro_step("A", hold=5, delay=25)
            else:
                info = self._macro_step("B", hold=5, delay=18)
        log.info(f"[Env {self.env_id}] Potion Macro completed.")

    def _run_pokeball_macro(self, ball_slot: int = 0) -> None:
        log.info(f"[Env {self.env_id}] Executing Pokeball Macro (slot={ball_slot})...")
        if not self._open_battle_bag():
            return
        # Clamping à direita garante cair no bolso 2 (POKE BALLS), mesmo após abrir bolso 0 (ITEMS)
        for _ in range(2):
            self._macro_step("RIGHT", hold=5, delay=25)
        for _ in range(max(0, min(ball_slot, 8))):
            self._macro_step("DOWN", hold=5, delay=8)
        self._macro_step("A", hold=5, delay=20)    # Seleciona Poké Ball
        info = self._macro_step("A", hold=5, delay=120)  # USE + animação de lançamento
        for step_k in range(12):
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if cb2 == 134571445:
                break
            ram_now = self.env.get_ram()
            battle_txt = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram_now))
            if step_k >= 2 and cb2 != 0x08107EE1 and "what will" in battle_txt:
                break
            info = self._macro_step("B", hold=4, delay=20)
        log.info(f"[Env {self.env_id}] Pokeball Macro completed.")

    def _run_switch_pokemon_macro(self) -> bool:
        """Seleciona o próximo Pokémon vivo mais forte no Party Menu de batalha (0x0811EBD1/0x0811EBA1)."""
        ram = self.env.get_ram()
        target_slot = RamReader.get_best_alive_switch_slot(ram)
        if target_slot <= 0:
            return False
        log.info(f"[Env {self.env_id}] Executing Forced Switch Macro (target_slot={target_slot})...")

        bt0 = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram))
        if "use next pokemon" in bt0:
            for _ in range(8):
                info = self._macro_step("A", hold=5, delay=20)
                cb2 = int(info.get("gMain_callback2", 0) or 0)
                if 0x08110000 <= cb2 <= 0x0812FFFF:
                    break

        for _ in range(10):
            info = self._macro_step(None, hold=0, delay=10)
            cb2 = int(info.get("gMain_callback2", 0) or 0)
            if cb2 == 0x0811EBA1:
                break
        self._macro_step(None, hold=0, delay=15)

        ew = RamAdvisor.decode_ram_text(self.env.get_ram()[138520:138776]).strip()
        if "do what with this" in ew:
            self._macro_step("B", hold=5, delay=15)

        for _ in range(target_slot):
            self._macro_step("DOWN", hold=5, delay=12)
        self._macro_step("A", hold=5, delay=20)   # Abre submenu SHIFT / SUMMARY / CANCEL
        info = self._macro_step("A", hold=5, delay=35)  # Confirma SHIFT

        for _ in range(14):
            cb2_s = int(info.get("gMain_callback2", 0) or 0)
            ram_s = self.env.get_ram()
            p_mon_s = RamReader.get_battle_mon(ram_s, 0)
            if cb2_s == 0x08011101 and p_mon_s["hp"] > 0:
                log.info(
                    f"[Env {self.env_id}] Forced Switch Macro SUCCESS! "
                    f"Sent out species={p_mon_s['species']} (hp={p_mon_s['hp']}/{p_mon_s['max_hp']})"
                )
                return True
            if cb2_s == 134571445:
                return False
            info = self._macro_step("B", hold=4, delay=14)
        return False

    def _run_mart_buy_macro(self, need_potions: int, need_balls: int) -> None:
        log.info(
            f"[Env {self.env_id}] Executing Mart Buy Macro "
            f"(need_potions={need_potions}, need_balls={need_balls})..."
        )
        ram = self.env.get_ram()
        px, py = RamReader.get_player_coords(ram)
        for _ in range(15):
            if abs(px - 4) + abs(py - 3) <= 10 and not RamReader.is_script_locked(ram):
                break
            self._macro_step(None, hold=0, delay=10)
            ram = self.env.get_ram()
            px, py = RamReader.get_player_coords(ram)
        if abs(px - 4) + abs(py - 3) > 10:
            return
        # Navega desviando de prateleiras até (4, 3) ao lado do balcão do atendente
        visits: dict[tuple[int, int], int] = {}
        dirs = (("UP", 3, 4, 0, -1), ("DOWN", 5, 4, 0, 1), ("LEFT", 4, 3, -1, 0), ("RIGHT", 4, 5, 1, 0))
        for _ in range(45):
            ram = self.env.get_ram()
            px, py = RamReader.get_player_coords(ram)
            visits[(px, py)] = visits.get((px, py), 0) + 1
            if px == 4 and py == 3:
                break
            grid = RamReader.get_collision_grid(ram)
            opts = []
            for d_name, r, c, dx, dy in dirs:
                if grid[r, c] in (1, 2, 3):
                    nx, ny = px + dx, py + dy
                    opts.append((visits.get((nx, ny), 0), abs(nx - 4) + abs(ny - 3), d_name))
            if not opts:
                break
            opts.sort()
            self._macro_step(opts[0][2], hold=6, delay=6)

        px_end, py_end = RamReader.get_player_coords(self.env.get_ram())
        if px_end != 4 or py_end != 3:
            return
        self._macro_step(None, hold=0, delay=24)  # Aguarda término da caminhada
        self._macro_step("LEFT", hold=6, delay=12)

        # Fala com atendente e seleciona BUY aguardando callback real da loja (0x0809ADE5)
        buy_open = False
        for _ in range(10):
            info = self._macro_step("A", hold=5, delay=22)
            if int(info.get("gMain_callback2", 0) or 0) in (0x0809AE15, 0x0809ADE5):
                buy_open = True
                break
        if not buy_open:
            for _ in range(4):
                self._macro_step("B", hold=5, delay=15)
            return
        self._macro_step(None, hold=0, delay=40)  # Aguarda fade-in completo da lista BUY

        balls_rem = min(max(0, need_balls), RamReader.get_money(self.env.get_ram()) // 200)
        pots_rem = max(0, need_potions)

        def _confirm_mart_purchase(target_qty: int) -> bool:
            qty_confirmed = False
            for _ in range(20):
                t_now = RamAdvisor.decode_ram_text(self.env.get_ram()[138520:138776]).strip()
                if "enough money" in t_now or "no room" in t_now:
                    self._macro_step("B", hold=5, delay=20)
                    return False
                if "here you are" in t_now or "thank you" in t_now:
                    for _ in range(4):
                        self._macro_step("A", hold=5, delay=28)
                        t_after = RamAdvisor.decode_ram_text(self.env.get_ram()[138520:138776]).strip()
                        if t_after.isdigit() and len(t_after) >= 3:
                            break
                    return True
                if t_now.isdigit() and len(t_now) == 2:
                    cur_q = int(t_now)
                    if not qty_confirmed and cur_q < target_qty:
                        self._macro_step("UP", hold=4, delay=10)
                    else:
                        qty_confirmed = True
                        self._macro_step("A", hold=5, delay=28)
                    continue
                if any(k in t_now for k in ("okay", "how many", "would you like", "certainly", "that will be")):
                    self._macro_step("A", hold=5, delay=28)
                    continue
                # Se já retornou ao saldo numérico (>= 3 dígitos) após confirmar YES
                if t_now.isdigit() and len(t_now) >= 3 and qty_confirmed:
                    return True
                self._macro_step(None, hold=0, delay=15)
            return False

        def _cancel_mart_item() -> None:
            # Aguarda abrir seletor de quantidade ('01') e pressiona B exatamente 1x para voltar à lista BUY
            for _ in range(12):
                t_w = RamAdvisor.decode_ram_text(self.env.get_ram()[138520:138776]).strip()
                if t_w.isdigit() and len(t_w) == 2:
                    self._macro_step("B", hold=5, delay=22)
                    return
                self._macro_step("A", hold=4, delay=14)
            self._macro_step("B", hold=5, delay=22)

        for _ in range(5):
            if balls_rem <= 0 and pots_rem <= 0:
                break
            self._macro_step("A", hold=5, delay=30)
            txt = RamAdvisor.decode_ram_text(self.env.get_ram()[138520:138776]).strip()
            if "poke ball" in txt and balls_rem > 0:
                qty = min(balls_rem, RamReader.get_money(self.env.get_ram()) // 200)
                if qty > 0 and _confirm_mart_purchase(qty):
                    balls_rem = 0
            elif "potion" in txt and "super" not in txt and pots_rem > 0:
                qty = min(pots_rem, RamReader.get_money(self.env.get_ram()) // 300)
                if qty > 0 and _confirm_mart_purchase(qty):
                    pots_rem = 0
            elif any(k in txt for k in ("how many", "would you like", "certainly")) or (
                txt.isdigit() and len(txt) == 2
            ):
                _cancel_mart_item()
            self._macro_step("DOWN", hold=5, delay=15)

        for _ in range(16):
            info = self._macro_step("B", hold=5, delay=20)
            ram_now = self.env.get_ram()
            if (
                int(info.get("gMain_callback2", 0) or 0) == 134571445
                and not RamReader.is_script_locked(ram_now)
                and not bool(info.get("script_lock", 0))
            ):
                break
        log.info(f"[Env {self.env_id}] Mart Buy Macro completed.")

    def _run_tm_macro(self):
        log.info(f"[Env {self.env_id}] Executing TM Macro (Robotic Arm)...")
        self._macro_step("START", hold=5, delay=20)
        for _ in range(5):
            self._macro_step("UP", hold=5, delay=5)
        self._macro_step("DOWN", hold=5, delay=5)
        self._macro_step("DOWN", hold=5, delay=5)
        self._macro_step("A", hold=5, delay=30)
        for _ in range(3):
            self._macro_step("RIGHT", hold=5, delay=15)
        self._macro_step("A", hold=5, delay=20)
        self._macro_step("A", hold=5, delay=20)
        self._macro_step("A", hold=5, delay=40)
        self._macro_step("A", hold=5, delay=60)
        for _ in range(15):
            self._macro_step("A", hold=5, delay=25)
        for _ in range(5):
            self._macro_step("B", hold=5, delay=20)
        log.info(f"[Env {self.env_id}] TM Macro completed.")


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

        state_file = self._fixed_state_file or self._progress.current_state_file
        log.info(f"[Env {self.env_id}] Loading GBA state file: {state_file}")
        with open(state_file, "rb") as f:
            self.env.em.set_state(f.read())

        # [FIX v20] Evitar glitch de RAM: Inicializar max_event_sum a partir do nome do arquivo
        basename = os.path.basename(state_file)
        m_flags = re.match(r"events_(\d+)", basename)
        base_flags = int(m_flags.group(1)) if m_flags else 0

        # [FIX v11-H] Warmup estável: exige 3 leituras consecutivas iguais de level>0
        raw_frame, info = self._warm_up()
        RamReader.enrich_info_from_ram(info, self.env.get_ram())
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
            RamReader.enrich_info_from_ram(info, self.env.get_ram())

        # Desbloquear callbacks presos em diálogo/placa pós-reset (ex: 0x0805674D em events_113/121)
        B_BTN = ActionSpace.get(ActionSpace.NAMES.index("B"))
        for _ in range(10):
            if info.get("gMain_callback2", 0) != 0x0805674D:
                break
            for _ in range(4):
                raw_frame, _, _, _, info = self.env.step(B_BTN)
            for _ in range(4):
                raw_frame, _, _, _, info = self.env.step(ActionSpace.NO_OP)
            RamReader.enrich_info_from_ram(info, self.env.get_ram())

        if self._enable_stochasticity and CFG.reset_random_noop_max > 0:
            jitter = random.randint(0, CFG.reset_random_noop_max)
            if jitter > 0:
                raw_frame, info = self._warm_up(n=jitter)
        raw_frame, info = self._random_safe_walk(raw_frame, info)

        ram_post_reset = self.env.get_ram()
        RamReader.enrich_info_from_ram(info, ram_post_reset)
        RamReader.debug_dump(info, self.env_id, "POST-RESET")
        self._stats.reset(info, ram_array=ram_post_reset)
        # [FIX v20] Override max_event_sum com o valor correto do arquivo para evitar +2000 pontos no step 1
        self._stats.max_event_sum = max(self._stats.max_event_sum, base_flags)
        
        # [FIX v13-A] Propaga o threshold sorteado para _stats (usado em update_battle)
        self._stats._farm_ratio_threshold = self._farm_ratio_threshold
        self._rewards.reset()
        self._advisor.reset()
        self._strategist.reset(info, ram_array=ram_post_reset)
        self._last_raw_frame = raw_frame
        self._battle_screenshot_saved = False
        self._had_parcel_in_ep = RamReader.has_oaks_parcel(ram_post_reset)
        self._had_teachy_tv_in_ep = RamReader.has_teachy_tv(ram_post_reset)
        self._parcel_delivered_in_ep = False
        self._battle_potion_attempts = 0
        self._battle_potion_cooldown = 0
        self._battle_capture_attempts = 0
        self._battle_capture_cooldown = 0
        self._episode_tm_attempts = 0
        self._mart_buy_rewarded_in_ep = False
        self._mart_buy_attempts = 0
        self._owned_species_in_ep = set(RamReader.get_party_species(ram_post_reset))
        self._prev_party_size_in_ep = RamReader.party_size(info)
        self._last_battle_best_slot = 0
        self._last_battle_best_mult = 1.0
        self._last_was_trainer_battle = False
        self._battle_captured_this_battle = False
        self._last_overworld_tile = ()

        # [FIX v15-Stairs] Reset do controle de loop
        self._map_history.clear()
        self._last_map_transition_step = 0
        self._stairs_loop_count = 0

        # [FIX v17] Reset incentivos positivos
        self._explore_reward_per_map  = {}
        self._prev_map_key            = (RamReader.map_bank(info), RamReader.map_id(info))
        self._ever_left_interior      = False
        self._steps_interior_only     = 0
        self._pending_event_save_flags = 0
        self._steps_in_current_map    = 0
        self._last_overworld_move_action = None
        self._last_script_lock        = False
        self._last_cb2                = int(info.get("gMain_callback2", 134571445) or 134571445)
        self._battle_switch_cooldown  = 0
        self._smart_flee_rewarded_since_heal = False

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
        terminated = False
        if (
            self._enable_stochasticity
            and CFG.sticky_action_prob > 0.0
            and not getattr(self._stats, "was_in_battle", False)
            and not self._last_script_lock
        ):
            if (
                action_idx in (2, 3, 4, 5)
                and self._last_overworld_move_action in (2, 3, 4, 5)
                and random.random() < CFG.sticky_action_prob
            ):
                action_idx = self._last_overworld_move_action
            self._last_overworld_move_action = action_idx if action_idx in (2, 3, 4, 5) else None
        else:
            self._last_overworld_move_action = None

        action = ActionSpace.get(action_idx)
        action_name = ActionSpace.NAMES[action_idx]

        was_in_battle_pre = getattr(self._stats, "was_in_battle", False)
        ram_pre = self.env.get_ram() if was_in_battle_pre else None
        last_cb2_pre = getattr(self, "_last_cb2", 134571445)
        in_party_menu_pre = (0x08110000 <= last_cb2_pre <= 0x0812FFFF)
        is_trainer_pre = False
        is_low_hp_wild_pre = False
        active_fainted_pre = False
        pre_active_hp_pct = 1.0
        pre_lvl = 0
        pre_badges = 0
        if was_in_battle_pre and ram_pre is not None:
            is_trainer_pre = RamReader.is_trainer_battle(ram_pre)
            p_pre_mon = RamReader.get_battle_mon(ram_pre, 0)
            pre_info: dict = {}
            RamReader.enrich_info_from_ram(pre_info, ram_pre)
            pre_party_max = RamReader.party_max_hp(pre_info)
            pre_party_hp = RamReader.party_hp(pre_info)
            pre_party_hp_pct = (pre_party_hp / float(pre_party_max)) if pre_party_max > 0 else 1.0
            pre_active_hp = p_pre_mon["hp"] if p_pre_mon["max_hp"] > 0 else pre_party_hp
            pre_active_max = p_pre_mon["max_hp"] if p_pre_mon["max_hp"] > 0 else max(pre_party_max, 1)
            pre_active_hp_pct = pre_active_hp / float(max(pre_active_max, 1))
            pre_lvl = RamReader.party_level(pre_info)
            pre_badges = RamReader.badges(pre_info)
            pre_potions = RamReader.get_potions_count(ram_pre)
            is_low_hp_wild_pre = (
                (not is_trainer_pre)
                and pre_potions == 0
                and pre_active_hp_pct < 0.35
                and pre_party_hp_pct < 0.35
            )
            active_fainted_pre = (p_pre_mon["max_hp"] > 0 and p_pre_mon["hp"] == 0)

        # [FIX v19 / v32 / v32c / v32d] Dynamic Battle Curriculum: libera D-pad em Party Menu, troca, HP baixo ou slot > 0
        if (
            CFG.enable_battle_curriculum
            and was_in_battle_pre
            and not in_party_menu_pre
            and not is_low_hp_wild_pre
            and not active_fainted_pre
        ):
            if self._stats.global_win_rate() < CFG.curriculum_win_rate_threshold:
                if action_name in ["START", "SELECT"]:
                    action = ActionSpace.NO_OP
                elif self._last_battle_best_slot == 0 and action_name in ["UP", "DOWN", "LEFT", "RIGHT"]:
                    mc_c, mvc_c = RamReader.get_battle_cursors(ram_pre) if ram_pre is not None else (0, 0)
                    if mc_c == 0 and mvc_c == 0:
                        action = ActionSpace.NO_OP
                    elif action_name in ["DOWN", "RIGHT"]:
                        action = ActionSpace.NO_OP

        # [FIX v22] TACTICAL_MACRO interception
        if action_name == "TACTICAL_MACRO":
            action = ActionSpace.NO_OP

        battle_guard_pen = 0.0
        if was_in_battle_pre and action_name != "TACTICAL_MACRO" and ram_pre is not None and not in_party_menu_pre:
            main_cur_pre, move_cur_pre = RamReader.get_battle_cursors(ram_pre)
            bt_pre = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram_pre))
            if is_trainer_pre:
                block_run_pre = True
            else:
                block_run_pre = (
                    pre_active_hp_pct >= 0.35
                    and not RamReader.has_oaks_parcel(ram_pre)
                    and pre_badges == 0
                    and 0 < pre_lvl < 16
                )
            if block_run_pre and "what will" in bt_pre:
                cursor_pen = (
                    CFG.flee_trainer_cursor_penalty
                    if is_trainer_pre
                    else CFG.flee_wild_cursor_penalty
                )
                if (main_cur_pre == 1 and action_name == "DOWN") or (
                    main_cur_pre == 2 and action_name == "RIGHT"
                ):
                    action = ActionSpace.NO_OP
                    battle_guard_pen += cursor_pen
                elif main_cur_pre == 3 and action_name == "A":
                    action = ActionSpace.get(ActionSpace.NAMES.index("UP"))
                    battle_guard_pen += CFG.flee_trainer_penalty
            elif bt_pre.startswith("type") and action_name == "A":
                p_mon_pre = RamReader.get_battle_mon(ram_pre, 0)
                e_mon_pre = RamReader.get_battle_mon(ram_pre, 1)
                scores_pre, _, _ = score_moves(
                    (p_mon_pre["type1"], p_mon_pre["type2"]),
                    (e_mon_pre["type1"], e_mon_pre["type2"]),
                    p_mon_pre["moves"],
                    p_mon_pre["pp"],
                )
                if 0 <= move_cur_pre < 4 and scores_pre[move_cur_pre] <= 0.05 and max(scores_pre) > 0.10:
                    battle_guard_pen += CFG.status_move_spam_penalty

        raw_frame, pooled, info  = self._run_action(action)
        self._last_raw_frame     = raw_frame

        # [CV Pipeline + RAM Battle Detection]
        ram_array = self.env.get_ram()
        RamReader.enrich_info_from_ram(info, ram_array)
        cv_state = self._vision.detect_state(raw_frame)
        _cb2 = int(info.get("gMain_callback2", 0) or 0)
        self._last_cb2 = _cb2
        _in_scripted_tutorial = RamReader.is_scripted_tutorial(info)
        _not_in_overworld = (_cb2 != 134571445)
        _is_whiteout   = (RamReader.party_hp(info) == 0 and RamReader.party_max_hp(info) > 0)
        _was_in_battle = getattr(self._stats, "was_in_battle", False)
        _in_battle_now = (
            RamReader.is_in_battle(info)
            or ((cv_state == GameState.BATTLE) and _not_in_overworld)
            or (_was_in_battle and _not_in_overworld and not _is_whiteout and _cb2 != 0)
        )
        _script_lock = (
            bool(info.get('script_lock', 0))
            or (cv_state == GameState.DIALOGUE)
            or (not _in_battle_now and _not_in_overworld and _cb2 != 0)
        )
        self._last_script_lock = bool(_script_lock)

        if not _in_battle_now:
            self._battle_potion_attempts = 0
            self._battle_potion_cooldown = 0
            self._battle_capture_attempts = 0
            self._battle_capture_cooldown = 0
            self._battle_switch_cooldown = 0
            self._last_battle_best_slot = 0
            self._last_battle_best_mult = 1.0
        else:
            if self._battle_potion_cooldown > 0:
                self._battle_potion_cooldown -= 1
            if self._battle_capture_cooldown > 0:
                self._battle_capture_cooldown -= 1
            if getattr(self, "_battle_switch_cooldown", 0) > 0:
                self._battle_switch_cooldown -= 1
            p_mon_now = RamReader.get_battle_mon(ram_array, 0)
            if (
                not _in_scripted_tutorial
                and (0x08110000 <= _cb2 <= 0x0812FFFF)
                and p_mon_now["max_hp"] > 0
                and p_mon_now["hp"] == 0
                and RamReader.party_hp(info) > 0
                and getattr(self, "_battle_switch_cooldown", 0) == 0
            ):
                self._battle_switch_cooldown = 8
                self._run_switch_pokemon_macro()
                ram_array = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_array)
                _cb2 = int(info.get("gMain_callback2", 0) or 0)
                self._last_cb2 = _cb2
                _not_in_overworld = (_cb2 != 134571445)
                _is_whiteout = (RamReader.party_hp(info) == 0 and RamReader.party_max_hp(info) > 0)
                _in_battle_now = (
                    RamReader.is_in_battle(info)
                    or (_was_in_battle and _not_in_overworld and not _is_whiteout and _cb2 != 0)
                )
                p_mon_now = RamReader.get_battle_mon(ram_array, 0)
            e_mon_now = RamReader.get_battle_mon(ram_array, 1)
            _, best_slot_now, best_mult_now = score_moves(
                (p_mon_now["type1"], p_mon_now["type2"]),
                (e_mon_now["type1"], e_mon_now["type2"]),
                p_mon_now["moves"],
                p_mon_now["pp"],
            )
            self._last_battle_best_slot = best_slot_now
            self._last_battle_best_mult = best_mult_now

        # [FIX v22 / v30b / v32 / v32d] Tactical Macro Execution: troca forçada -> poção (HP ativo) -> captura -> Mart -> TM
        macro_reward = 0.0
        if action_name == "TACTICAL_MACRO":
            potions = RamReader.get_potions_count(ram_array)
            balls = RamReader.get_pokeballs_count(ram_array)
            tms = RamReader.get_tms_count(ram_array)
            cur_hp = RamReader.party_hp(info)
            max_hp = max(RamReader.party_max_hp(info), 1)
            p_mon_t = RamReader.get_battle_mon(ram_array, 0)
            active_hp_t = p_mon_t["hp"] if p_mon_t["max_hp"] > 0 else cur_hp
            active_max_t = p_mon_t["max_hp"] if p_mon_t["max_hp"] > 0 else max_hp
            mk_now = (RamReader.map_bank(info), RamReader.map_id(info))
            is_trainer_now = RamReader.is_trainer_battle(ram_array)
            e_mon_t = RamReader.get_battle_mon(ram_array, 1)
            e_hp_t = e_mon_t["hp"] if e_mon_t["max_hp"] > 0 else RamReader.enemy_hp(info)
            e_max_t = e_mon_t["max_hp"] if e_mon_t["max_hp"] > 0 else max(e_hp_t, 1)
            enemy_hp_pct_t = e_hp_t / float(max(e_max_t, 1))
            party_sp_now = set(RamReader.get_party_species(ram_array))
            party_sz_now = RamReader.party_size(info)
            money_now = RamReader.get_money(ram_array)

            if (
                _in_battle_now
                and not _in_scripted_tutorial
                and p_mon_t["max_hp"] > 0
                and p_mon_t["hp"] == 0
                and cur_hp > 0
                and getattr(self, "_battle_switch_cooldown", 0) == 0
            ):
                self._battle_switch_cooldown = 8
                if self._run_switch_pokemon_macro():
                    macro_reward += CFG.forced_switch_bonus
                else:
                    macro_reward += CFG.wasted_macro_penalty
                ram_after = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_after)
                ram_array = ram_after
            elif (
                _in_battle_now
                and not _in_scripted_tutorial
                and potions > 0
                and 0 < active_hp_t < active_max_t * 0.50
                and self._battle_potion_attempts < CFG.max_potion_macro_attempts_per_battle
                and self._battle_potion_cooldown == 0
            ):
                self._battle_potion_attempts += 1
                self._battle_potion_cooldown = CFG.potion_macro_cooldown_steps
                self._run_potion_macro(max(0, RamReader.get_potion_slot(ram_array)))
                ram_after = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_after)
                potions_after = RamReader.get_potions_count(ram_after)
                hp_after = RamReader.party_hp(info)
                if potions_after < potions or hp_after > cur_hp:
                    macro_reward += 100.0
                    log.info(f"[Env {self.env_id}] TACTICAL MACRO SUCCESS: Potion Triggered! ({cur_hp}->{hp_after} HP)")
                else:
                    macro_reward += CFG.wasted_macro_penalty
                ram_array = ram_after
            elif (
                _in_battle_now
                and not _in_scripted_tutorial
                and not is_trainer_now
                and balls > 0
                and 0 < e_hp_t
                and enemy_hp_pct_t < 0.55
                and party_sz_now < 6
                and e_mon_t["species"] > 0
                and e_mon_t["species"] not in party_sp_now
                and self._battle_capture_attempts < CFG.max_capture_macro_attempts_per_battle
                and self._battle_capture_cooldown == 0
            ):
                self._battle_capture_attempts += 1
                self._battle_capture_cooldown = CFG.potion_macro_cooldown_steps
                self._run_pokeball_macro(max(0, RamReader.get_pokeball_slot(ram_array)))
                ram_after = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_after)
                balls_after = RamReader.get_pokeballs_count(ram_after)
                sp_after = set(RamReader.get_party_species(ram_after))
                if balls_after < balls or len(sp_after) > len(party_sp_now):
                    macro_reward += CFG.pokeball_throw_bonus
                    log.info(
                        f"[Env {self.env_id}] TACTICAL MACRO: Pokeball thrown! "
                        f"balls={balls}->{balls_after}, species={e_mon_t['species']}"
                    )
                else:
                    macro_reward += CFG.wasted_macro_penalty
                ram_array = ram_after
            elif (
                not _in_battle_now
                and not _script_lock
                and mk_now in ((5, 3), (6, 0))
                and RamReader.has_pokedex(ram_array)
                and not RamReader.has_oaks_parcel(ram_array)
                and ((potions < 2 and money_now >= 300) or (balls < 3 and money_now >= 200))
                and self._mart_buy_attempts < 2
            ):
                self._mart_buy_attempts += 1
                self._run_mart_buy_macro(max(0, 3 - potions), max(0, 5 - balls))
                ram_after = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_after)
                potions_after = RamReader.get_potions_count(ram_after)
                balls_after = RamReader.get_pokeballs_count(ram_after)
                money_after = RamReader.get_money(ram_after)
                if (potions_after > potions or balls_after > balls) and money_after <= money_now:
                    if not self._mart_buy_rewarded_in_ep:
                        macro_reward += CFG.mart_buy_bonus
                        self._mart_buy_rewarded_in_ep = True
                    log.info(
                        f"[Env {self.env_id}] MART BUY SUCCESS! "
                        f"potions={potions}->{potions_after}, balls={balls}->{balls_after}, money={money_now}->{money_after}"
                    )
                else:
                    macro_reward += CFG.wasted_macro_penalty
                ram_array = ram_after
            elif (
                not _in_battle_now
                and not _script_lock
                and tms > 0
                and self._episode_tm_attempts < CFG.max_tm_macro_attempts_per_episode
            ):
                self._episode_tm_attempts += 1
                self._run_tm_macro()
                ram_after = self.env.get_ram()
                RamReader.enrich_info_from_ram(info, ram_after)
                if RamReader.get_tms_count(ram_after) < tms:
                    macro_reward += 500.0
                    log.info(f"[Env {self.env_id}] TACTICAL MACRO SUCCESS: TM Teaching Triggered!")
                else:
                    macro_reward += CFG.wasted_macro_penalty
                ram_array = ram_after
            else:
                # [FIX v23] Punição para queimar a mão do agente ao spammar o botão fora de hora
                macro_reward += CFG.wasted_macro_penalty

            # Atualiza estado pós-macro para que captura/término de batalha na macro reflitam neste step
            _cb2 = int(info.get("gMain_callback2", 0) or 0)
            self._last_cb2 = _cb2
            _not_in_overworld = (_cb2 != 134571445)
            _is_whiteout = (RamReader.party_hp(info) == 0 and RamReader.party_max_hp(info) > 0)
            _in_battle_now = (
                RamReader.is_in_battle(info)
                or (_was_in_battle and _not_in_overworld and not _is_whiteout and _cb2 != 0)
            )
            _script_lock = (
                bool(info.get("script_lock", 0))
                or (not _in_battle_now and _not_in_overworld and _cb2 != 0)
            )
            self._last_script_lock = bool(_script_lock)
        
        # [USER REQUEST] Salvar print da tela quando iniciar a batalha (após fade-in)
        if not _in_battle_now:
            self._battle_screenshot_saved = False
        elif not self._battle_screenshot_saved and float(np.mean(raw_frame)) > 30.0:
            artifact_dir = CFG.screenshot_dir
            os.makedirs(artifact_dir, exist_ok=True)
            img_path = os.path.join(artifact_dir, f"battle_start_env{self.env_id}.png")
            # raw_frame is RGB, cv2 expects BGR
            cv2.imwrite(img_path, cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR))
            self._battle_screenshot_saved = True
            log.info(f"[Env {self.env_id}] 📸 PRINT DA BATALHA SALVO EM {img_path}")
        
        step_reward = 0.0
        # [FIX v22] Recompensa Massiva do Botão Tático (HRL)
        step_reward += self._rewards.add("tactical", macro_reward)
        
        # [FIX v20] A punição de tempo agora é incondicional.
        # Ficar em diálogos ("Gary wants to battle") gasta tempo e reduz a recompensa.
        # Isso ensina o agente a avançar os textos rapidamente.
        step_reward += self._rewards.add("time_pen", CFG.time_penalty)
        party_max_hp_now = RamReader.party_max_hp(info)
        party_hp_pct_now = (RamReader.party_hp(info) / party_max_hp_now) if party_max_hp_now > 0 else 1.0
        text_bonus      = self._advisor.update(
            _in_battle_now,
            _script_lock,
            action_idx,
            ram_array,
            party_level=RamReader.party_level(info),
            map_key=(RamReader.map_bank(info), RamReader.map_id(info)),
            party_hp_pct=party_hp_pct_now,
        )
        if _in_scripted_tutorial:
            self._advisor.dialogue_stuck_steps = 0
        step_reward    += self._rewards.add("text", text_bonus)

        # [FIX v18.3] Novas recompensas Whiddy-style
        op_lvl_bonus = self._stats.update_max_op_level(info)
        step_reward += self._rewards.add("op_lvl", op_lvl_bonus)
        
        heal_bonus = self._stats.update_heal_reward(info)
        step_reward += self._rewards.add("heal", heal_bonus)

        event_bonus, total_flags = self._stats.update_events(ram_array)
        if event_bonus > 0:
            step_reward += self._rewards.add("milestone", event_bonus)
            self._pending_event_save_flags = max(self._pending_event_save_flags, total_flags)

        strategist_bonus = self._strategist.update(info, ram_array, _in_battle_now, total_flags)
        if strategist_bonus != 0.0:
            step_reward += self._rewards.add("strategist", strategist_bonus)

        x        = RamReader.coord(info, "player_x")
        y        = RamReader.coord(info, "player_y")
        map_id   = RamReader.map_id(info)
        map_bank = RamReader.map_bank(info)
        tile     = (map_bank, map_id, x, y)
        map_key  = (map_bank, map_id)

        if map_key != self._prev_map_key:
            self._steps_in_current_map = 0
            self._prev_map_key = map_key
        elif not _in_battle_now:
            self._steps_in_current_map += 1

        grid_now = RamReader.get_collision_grid(ram_array, info)
        center_is_warp = bool(grid_now[GRID_SIZE // 2, GRID_SIZE // 2] == RamReader.TILE_WARP)
        is_clean_overworld_save_frame = (
            _cb2 == 134571445
            and not _script_lock
            and not _in_battle_now
            and not _is_whiteout
            and not center_is_warp
            and RamReader.party_level(info) > 0
            and RamReader.party_hp(info) >= max(1, math.ceil(RamReader.party_max_hp(info) * 0.50))
        )
        if self._pending_event_save_flags > 0 and is_clean_overworld_save_frame:
            self._progress.check_and_save(self.env, self._pending_event_save_flags, self.env_id)
            self._pending_event_save_flags = 0

        if not _in_battle_now:
            if self._last_overworld_tile and tile != self._last_overworld_tile:
                self._advisor.dialogue_stuck_steps = 0
            self._last_overworld_tile = tile

        has_parcel_now = RamReader.has_oaks_parcel(ram_array)
        has_pokedex_now = RamReader.has_pokedex(ram_array)
        if has_parcel_now and not self._had_parcel_in_ep:
            self._had_parcel_in_ep = True
            step_reward += self._rewards.add("milestone", CFG.parcel_pickup_bonus)
            self._explore_reward_per_map.clear()
            self._stats.tile_visits.clear()
            self._stats.map_visits = {map_key}
            self._stats.x_min_episode = x
            self._stats.x_max_episode = x
            self._stats.y_min_episode = y
            self._stats.y_max_episode = y
            log.info(
                f"[Env {self.env_id}] 📦 Oak's Parcel obtido! +{CFG.parcel_pickup_bonus:.1f} "
                "— resetando exploração para viagem de retorno ao Lab."
            )
        elif (
            self._had_parcel_in_ep
            and not has_parcel_now
            and not self._parcel_delivered_in_ep
            and map_key == (4, 3)
        ):
            self._parcel_delivered_in_ep = True
            step_reward += self._rewards.add("milestone", CFG.parcel_delivery_bonus)
            log.info(
                f"[Env {self.env_id}] 🎉 Oak's Parcel entregue ao Prof. Oak! "
                f"+{CFG.parcel_delivery_bonus:.1f}"
            )

        has_teachy_tv_now = RamReader.has_teachy_tv(ram_array)
        if has_teachy_tv_now and not self._had_teachy_tv_in_ep:
            self._had_teachy_tv_in_ep = True
            step_reward += self._rewards.add("milestone", CFG.teachy_tv_bonus)
            log.info(
                f"[Env {self.env_id}] 📺 Teachy TV obtido (Old Man tutorial concluído)! "
                f"+{CFG.teachy_tv_bonus:.1f}"
            )

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
                        # [FIX v28] Early Stop em vez de punição para proteger a curva de aprendizado
                        log.info(f"[Env {self.env_id}] 🚨 STAIRS LOOP 3x repetido. Early Stop!")
                        terminated = True
                    else:
                        # penalty = -3.0
                        # step_reward += self._rewards.add("stuck", penalty)
                        log.info(f"[Env {self.env_id}] 🚨 STAIRS LOOP DETECTED (A->B->A inside)!")
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

            is_interior = (map_bank >= 4)

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

            explore_r, is_new_map = self._stats.update_exploration(
                tile, map_key, include_new_map_bonus=False, script_lock=_script_lock
            )

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
                if self._stats.steps >= 2:
                    map_disc_bonus = CFG.new_map_bonus_route if map_bank == 3 else CFG.new_map_bonus
                    step_reward += self._rewards.add("map_disc", map_disc_bonus)
                # Reset directional bounds on map transition so entering a map doesn't give a free bonus
                self._stats.x_min_episode = x
                self._stats.x_max_episode = x
                self._stats.y_min_episode = y
                self._stats.y_max_episode = y

            if is_clean_overworld_save_frame and self._steps_in_current_map >= 2:
                self._progress.check_and_save_map_milestone(
                    self.env,
                    total_flags,
                    map_key,
                    self.env_id,
                    has_parcel=has_parcel_now,
                    has_pokedex=has_pokedex_now,
                )

            nav_hint = self._strategist.current_goal.navigation_hint
            if nav_hint == "south" and y > self._stats.y_max_episode:
                delta = y - self._stats.y_max_episode
                self._stats.y_max_episode = y
                dir_r = delta * CFG.north_bonus_per_tile
                step_reward += self._rewards.add("milestone", dir_r)
                log.debug(f"[Env {self.env_id}] South return progress! y={y} (+{dir_r:.1f})")
            elif nav_hint == "east" and x > self._stats.x_max_episode:
                delta = x - self._stats.x_max_episode
                self._stats.x_max_episode = x
                dir_r = delta * CFG.north_bonus_per_tile
                step_reward += self._rewards.add("milestone", dir_r)
                log.debug(f"[Env {self.env_id}] East progress! x={x} (+{dir_r:.1f})")
            elif nav_hint == "west" and x < self._stats.x_min_episode:
                delta = self._stats.x_min_episode - x
                self._stats.x_min_episode = x
                dir_r = delta * CFG.north_bonus_per_tile
                step_reward += self._rewards.add("milestone", dir_r)
                log.debug(f"[Env {self.env_id}] West progress! x={x} (+{dir_r:.1f})")
            elif nav_hint not in ("south", "east", "west") and y < self._stats.y_min_episode:
                # North bonus: y decresce ao ir para norte no FireRed
                delta = self._stats.y_min_episode - y
                self._stats.y_min_episode = y
                dir_r = delta * CFG.north_bonus_per_tile
                step_reward += self._rewards.add("milestone", dir_r)
                log.debug(f"[Env {self.env_id}] North progress! y={y} (+{dir_r:.1f})")
        else:
            is_new_map = False
            self._stats.steps_on_cur_tile = 0

        if is_new_map:
            log.info(f"[Env {self.env_id}] New map: bank={map_key[0]} id={map_key[1]}")

        stuck_pen = self._stats.update_stuck(in_battle=_in_battle_now, script_lock=_script_lock)
        step_reward += self._rewards.add("stuck", stuck_pen)
        
        # [FIX v28] Early Stop por inatividade em vez de ficar drenando a Value Network com punição.
        if self._stats.steps_on_cur_tile >= CFG.stuck_threshold and not (_in_battle_now or _script_lock):
            log.info(f"[Env {self.env_id}] 🚨 STUCK (parado na mesma coordenada). Early Stop!")
            terminated = True
        elif self._advisor.dialogue_stuck_steps >= CFG.dialogue_stuck_terminate and not _in_battle_now:
            log.info(
                f"[Env {self.env_id}] 💬 DIALOGUE FREEZE "
                f"({self._advisor.dialogue_stuck_steps} steps sem avanço). Early Stop!"
            )
            terminated = True

        was_in_battle_before = bool(getattr(self._stats, "was_in_battle", False))
        prev_last_enemy_hp = self._stats.last_enemy_hp

        dmg_reward, idle_pen, vic_bonus, is_new_battle, whiteout_pen, farm_terminate = self._stats.update_battle(
            _in_battle_now,
            RamReader.enemy_hp(info),
            RamReader.party_hp(info),
            script_lock=(_script_lock and not _in_battle_now),
            last_raw_frame=self._last_raw_frame,
            env_id=self.env_id,
            episode_num=self._episode_num,
            current_tile=tile,
            party_level=RamReader.party_level(info),
            has_parcel=has_parcel_now,
            badges=RamReader.badges(info),
        )
        step_reward += self._rewards.add("damage", dmg_reward)

        if idle_pen != 0.0:
            step_reward += self._rewards.add("idle_pen", idle_pen)

        if vic_bonus > 0.0:
            step_reward += self._rewards.add("victory", vic_bonus)

        if whiteout_pen != 0.0:
            step_reward += self._rewards.add("stuck", whiteout_pen)
            log.info(f"[Env {self.env_id}] WHITEOUT! Penalty {whiteout_pen:.1f}")
            if CFG.terminate_on_whiteout:
                terminated = True
                log.info(f"[Env {self.env_id}] WHITEOUT -> encerrando episódio imediatamente.")

        if is_new_battle:
            self._battle_captured_this_battle = False
            if tile in self._stats.whiteout_tile_cooldown and self._stats.whiteout_tile_cooldown[tile] > 0:
                actual_entry = CFG.entry_bonus_revisit  # 0.0
                log.debug(f"[Env {self.env_id}] Entry suppressed (whiteout cooldown)")
            else:
                actual_entry = CFG.entry_bonus
            step_reward += self._rewards.add("entry", actual_entry)
            log.debug(f"[Env {self.env_id}] Battle entered! +{actual_entry:.1f} entry bonus")

            if map_bank == 3:
                n_map_battles = self._stats.map_battles.get(map_key, 0)
                grind_threshold = 2 if has_parcel_now else 3
                if n_map_battles >= grind_threshold:
                    step_reward += self._rewards.add("idle_pen", CFG.wild_grind_penalty)
                    log.info(
                        f"[Env {self.env_id}] WILD GRIND PENALTY {CFG.wild_grind_penalty:.1f} "
                        f"(map={map_key}, battles={n_map_battles}, parcel={has_parcel_now})"
                    )
                cur_hp = RamReader.party_hp(info)
                max_hp = RamReader.party_max_hp(info)
                if (
                    not RamReader.is_trainer_battle(ram_array)
                    and RamReader.get_potions_count(ram_array) == 0
                    and max_hp > 0
                    and (cur_hp / max_hp) < 0.35
                ):
                    step_reward += self._rewards.add("idle_pen", CFG.low_hp_battle_penalty)
                    log.info(
                        f"[Env {self.env_id}] LOW-HP WILD BATTLE PENALTY {CFG.low_hp_battle_penalty:.1f} "
                        f"(hp={cur_hp}/{max_hp})"
                    )

        # [FIX v14-B] First Strike Reward (com fadiga por mapa)
        battle_fatigue = (
            self._stats.current_battle_fatigue(
                has_parcel=has_parcel_now,
                party_level=RamReader.party_level(info),
                badges=RamReader.badges(info),
            )
            if _in_battle_now
            else 1.0
        )
        if _in_battle_now and self._stats.current_battle_damage > 0 and not getattr(self._stats, 'first_strike_paid', False):
            self._stats.first_strike_paid = True
            actual_first_strike = CFG.first_strike_bonus * battle_fatigue
            step_reward += self._rewards.add("damage", actual_first_strike)
            log.info(f"[Env {self.env_id}] First Strike! +{actual_first_strike:.1f} (fatigue={battle_fatigue:.2f})")

        # [FIX v32] Efetividade de tipo (>=2x super-efetivo) & anti-fuga (funciona com script_lock=1 em treinadores)
        if _in_battle_now and dmg_reward > 0.0 and self._last_battle_best_mult >= 2.0:
            se_bonus = CFG.super_effective_bonus * battle_fatigue
            step_reward += self._rewards.add("damage", se_bonus)

        if battle_guard_pen < 0.0:
            step_reward += self._rewards.add("idle_pen", battle_guard_pen)
        elif _in_battle_now and RamReader.is_trainer_battle(ram_array) and action_name == "A":
            bt_now = RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram_array))
            if "no running" in bt_now:
                step_reward += self._rewards.add("idle_pen", CFG.flee_trainer_penalty)

        # [FIX v32] Captura de nova espécie + anti-double-count de levelup + anti-fuga desnecessária
        cur_party_sz_now = RamReader.party_size(info)
        sp_now = set(RamReader.get_party_species(ram_array)) if not _in_scripted_tutorial else set()
        new_sp_caught = set()
        if sp_now:
            if self._owned_species_in_ep:
                new_sp_caught = sp_now - self._owned_species_in_ep
                if new_sp_caught and cur_party_sz_now >= 2:
                    cap_r = CFG.capture_new_species_bonus * len(new_sp_caught)
                    step_reward += self._rewards.add("milestone", cap_r)
                    self._pending_event_save_flags = max(self._pending_event_save_flags, self._stats.max_event_sum)
                    self._battle_captured_this_battle = True
                    log.info(
                        f"[Env {self.env_id}] CAPTURED NEW SPECIES {sorted(new_sp_caught)}! +{cap_r:.1f}"
                    )
                if cur_party_sz_now > self._prev_party_size_in_ep and self._prev_party_size_in_ep >= 1:
                    self._battle_captured_this_battle = True
                    self._stats.max_level = max(self._stats.max_level, RamReader.party_level(info))
            self._owned_species_in_ep.update(sp_now)
        self._prev_party_size_in_ep = cur_party_sz_now

        if party_hp_pct_now >= 0.50:
            self._smart_flee_rewarded_since_heal = False

        if (
            was_in_battle_before
            and not _in_battle_now
            and not self._last_was_trainer_battle
            and not _is_whiteout
            and not _in_scripted_tutorial
            and not new_sp_caught
            and not self._battle_captured_this_battle
            and _cb2 == 134571445
            and vic_bonus == 0.0
            and prev_last_enemy_hp > 0
            and RamReader.enemy_hp(info) > 0
        ):
            if party_hp_pct_now >= 0.45:
                step_reward += self._rewards.add("idle_pen", CFG.unnecessary_flee_penalty)
                log.info(f"[Env {self.env_id}] Unnecessary wild flee! {CFG.unnecessary_flee_penalty:.1f}")
            elif (
                party_hp_pct_now < 0.35
                and RamReader.get_potions_count(ram_array) == 0
                and not getattr(self, "_smart_flee_rewarded_since_heal", False)
            ):
                self._smart_flee_rewarded_since_heal = True
                step_reward += self._rewards.add("heal", CFG.smart_flee_bonus)
                log.info(f"[Env {self.env_id}] Smart low-HP wild flee! +{CFG.smart_flee_bonus:.1f}")

        if _in_battle_now:
            self._last_was_trainer_battle = RamReader.is_trainer_battle(ram_array)
        else:
            self._last_was_trainer_battle = False
            self._battle_captured_this_battle = False

        if _in_battle_now:
            step_reward += self._rewards.add("text", CFG.battle_sustain_bonus)

        # [FIX v11-D] A bonus em batalha escalado pela fadiga por mapa
        if _in_battle_now and action_idx == ActionSpace.NAMES.index("A"):
            self._stats.battle_total_a += 1
            if self._stats.battle_a_presses < CFG.battle_a_cap:
                self._stats.battle_a_presses += 1
                step_reward += self._rewards.add("text", CFG.battle_a_bonus * battle_fatigue)

        # [FIX v11-B] B penalty só após battle_b_threshold A presses
        if (_in_battle_now and
                action_idx == ActionSpace.NAMES.index("B") and
                self._stats.battle_total_a >= CFG.battle_b_threshold):
            step_reward += self._rewards.add("idle_pen", CFG.battle_b_penalty)

        step_reward += self._rewards.add(
            "levelup",
            self._stats.update_level(
                RamReader.party_level(info),
                self.env_id,
                badges=RamReader.badges(info),
                in_battle=_in_battle_now,
                has_parcel=has_parcel_now,
            ),
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
        done = (terminated or hard_cap or farm_terminate or interior_only_terminate or
                (over_limit and not _in_battle_now and not in_post_battle_grace))

        obs_dict = self._build_obs(raw_frame, info)

        if done:
            advisor_stats = self._advisor.stats_dict()
            strategist_stats = self._strategist.stats_dict()
            prog_stats = self._progress.stats_dict()
            info["episode_metrics"] = {
                **self._stats.to_dict(ms),
                **self._rewards.to_dict(),
                "max_steps_used":             ms,
                "advisor_dialog_steps":       advisor_stats["advisor/dialog_steps"],
                "advisor_battle_steps":       advisor_stats["advisor/battle_steps"],
                "advisor_move_select_steps":  advisor_stats["advisor/move_select_steps"],
                "advisor_hints_match":        advisor_stats["advisor/hints_matched"],
                "advisor_text_bonus":         advisor_stats["advisor/total_bonus"],
                "advisor_repeat_penalties":   advisor_stats["advisor/repeat_penalties"],
                "strategist_queries_count":   strategist_stats["strategist/queries_count"],
                "strategist_llm_success":     strategist_stats["strategist/llm_success_count"],
                "strategist_goal_alignments": strategist_stats["strategist/goal_alignments"],
                "strategist_alignment_bonus": strategist_stats["strategist/alignment_bonus"],
                "total_tiles_ever":           len(self._stats.persistent_tile_visits),
                "total_maps_ever":            len(self._stats.all_maps_visited),
                "best_milestone":             self._progress._best_flags_ever,
                "frontier_ratio":             prog_stats["progress/frontier_ratio"],
                "genesis_ratio":              prog_stats["progress/genesis_ratio"],
                "start_state_ratio":          prog_stats["progress/start_state_ratio"],
                "farm_detected":              int(self._stats.farm_detected),
                "farm_detections":            self._stats.farm_detections,
            }

        return obs_dict, step_reward, done, False, info

    def close(self):
        self._strategist.close()
        self.env.close()
