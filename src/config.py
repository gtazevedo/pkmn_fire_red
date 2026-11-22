import os
import logging
import numpy as np
from dataclasses import dataclass

try:
    import retro
except ImportError:  # pragma: no cover - allows unit testing without retro installed
    retro = None

# [FIX v22] RAM obs expandido: +3 para Sixth Sense (potions, tms, p1_status) = 23
RAM_SIZE = 12 + 5 + 3 + 3   # = 23
GRID_SIZE = 9               # Matriz local 9x9 de colisão e warps
GOAL_SIZE = 6               # Embedding do sub-objetivo estratégico (4 priority one-hot + map_id + nav_hint)
BATTLE_SIZE = 12            # Embedding de batalha/economia (4 move scores + best + 2 cursores + wild + balls + money + hp + capture)


def _default_base_save_dir() -> str:
    env_dir = os.getenv("PKMN_SAVE_DIR")
    if env_dir:
        return env_dir
    preferred = "/mnt/d/projects/pkmn_fire_red_saves"
    if os.path.isdir(preferred) or os.path.isdir("/mnt/d"):
        return preferred
    return "./saves"


def _default_state_file() -> str:
    if retro is not None:
        return os.path.join(
            os.path.dirname(retro.__file__),
            "data", "stable", "PokemonFireRed", "Start.state.gz",
        )
    return os.path.join(_default_base_save_dir(), "progress_states", "Start.state.gz")


@dataclass(frozen=True)
class Config:
    """Todos os parâmetros tuneáveis em um lugar. Frozen para evitar mutação."""

    # --- paths ---
    # Caminho base configurável via env var, com fallback seguro fora do WSL/Windows (prioriza D:)
    base_save_dir: str = _default_base_save_dir()
    state_file: str = _default_state_file()
    # [FIX v18] Save state exterior: ponto de partida alternativo já fora da casa
    pallet_exterior_state_file: str = f"{base_save_dir}/progress_states/pallet_exterior.state"
    # Probabilidade de iniciar do exterior (quando o state existir)
    exterior_start_ratio: float = 0.60

    model_path:          str = f"{base_save_dir}/pokemon_lstm_model"
    checkpoint_dir:      str = f"{base_save_dir}/checkpoints/"
    progress_dir:        str = f"{base_save_dir}/progress_states/"
    tensorboard_log_dir: str = f"{base_save_dir}/pokemon_tensorboard/"
    log_file:            str = f"{base_save_dir}/pokemon_training.log"
    whiteout_frame_dir:  str = f"{base_save_dir}/whiteout_frames/"
    screenshot_dir:      str = f"{base_save_dir}/screenshots/"
    mlflow_dir:          str = f"{base_save_dir}/mlflow"
    mlflow_db_path:      str = f"{mlflow_dir}/mlflow.db"
    mlflow_artifact_dir: str = f"{mlflow_dir}/mlruns"

    # [FIX v32] Battle IQ (gBattleMons/tipos/cursores) + Captura + Mart + Anti-Flee + Anti-OOM (4 workers)
    config_stamp:        str = "v32_battle_iq_capture_mart"

    # [FIX 16] steps_state_file para sincronização de currículo entre workers
    steps_state_file:    str = f"{base_save_dir}/steps_state.txt"

    # --- curriculum sampling (Frontier Bias + Genesis Anchor) ---
    frontier_prob:            float = 0.50
    frontier_top_state_share: float = 0.50
    intermediate_prob:        float = 0.30
    initial_prob:             float = 0.20
    genesis_anchor_share:     float = 0.60
    genesis_anchor_count:     int   = 2

    # --- strategist & advisor ---
    ollama_url:                  str   = "http://localhost:11434/api/generate"
    ollama_model:                str   = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout_sec:          float = 15.0
    strategist_stagnation_steps: int   = 500
    strategist_goal_bonus:       float = 15.0
    dialog_repeat_penalty:       float = -0.001

    # --- training ---
    num_cpu:         int   = 4
    learn_batch:     int   = 40_000
    n_steps:         int   = 1024
    batch_size:      int   = 512
    n_epochs:        int   = 3
    gamma:           float = 0.997
    gae_lambda:      float = 0.95
    # [FIX v15 / v31] ent_coef 0.018 (Whiddy v2=0.01) + target_kl=0.025 + batch=512
    ent_coef:        float = 0.018
    lr_start:        float = 6e-5
    lr_end:          float = 3e-5
    lr_total_budget: int   = 50_000_000
    target_kl:       float = 0.025
    checkpoint_freq: int   = 10_000
    log_freq_steps:  int   = 500

    # --- episode length curriculum ---
    phase1_steps: int = 500_000
    phase2_steps: int = 2_000_000
    max_steps_p1: int = 5_000
    max_steps_p2: int = 10_000
    # [FIX v15] max_steps_p3 20_000→10_000: episódios muito longos permitiam
    # farm de exploração e enrolação do agente.
    # [FIX v27] Reduzido de 10k para 4k: episódios longos demais permitem
    # ao agente "enrolar" com R≈80 sem explorar. Mais curtos = mais resets = mais variedade.
    max_steps_p3: int = 8_000

    # --- frame skip ---
    frames_held: int = 6
    frames_noop: int = 2
    pool_start:  int = 4
    pool_end:    int = 8

    # --- anti-memorization (Machado sticky actions + reset jitter + coord dropout) ---
    sticky_action_prob:    float = 0.08
    reset_random_noop_max: int   = 15
    reset_random_walk_max: int   = 3
    coord_dropout_prob:    float = 0.15

    # --- reward weights ---
    # [FIX v18.3] Remoção do medo do tempo (Whiddy-style).
    # Como não perdemos mais tempo passivamente, o limite de passos cuida da inércia.
    # [FIX v18.4] Reativado levemente o time_penalty para causar inércia desconfortável e forçar movimento
    # [FIX v24 / v30b] Zerado o time_penalty (Whiddy-Pure)
    time_penalty:    float = 0.0
    # [FIX v26 / v31] Exploração estacionária com piso de novidade >= 0.50
    explore_weight:  float = 1.0
    explore_weight_outdoor_bonus: float = 0.5
    explore_novelty_floor:        float = 0.50

    # [FIX v18.3] Novas recompensas inspiradas em Peter Whiddy
    max_op_level_reward_scale: float = 0.2
    heal_reward_scale:         float = 50.0
    heal_min_prior_hp_ratio:   float = 0.40
    # Quando o agente esgotar os tiles do mapa, ele ganha uma punição de tempo leve
    max_explore_reward_per_map: float = 150.0
    stale_map_penalty:          float = 0.0

    # [FIX v31] Reduzido para evitar exploit de oscilar na borda entre mapas
    new_map_bonus:        float = 5.0
    new_map_bonus_route:  float = 15.0

    # [FIX v15] battle_idle_penalty: -0.05 → -0.01, grace: 120 → 200
    # Punições muito altas geravam fobia de batalhas e colapso na grama.
    battle_idle_penalty:  float = 0.0
    battle_idle_grace:    int   = 200   # mais carência para lidar com menus e gary

    # [FIX v11-C / v31 / v32c] Recompensas de combate reforçadas para vencer batalhas sem fobia
    hp_pct_scale:         float = 20.0
    damage_weight:        float = 0.05  # fallback

    victory_bonus:        float = 50.0
    near_ko_bonus:        float = 5.0
    post_battle_grace:    int   = 200

    # Zero incentivo gratuito para simplesmente entrar em batalha
    entry_bonus:          float = 0.0

    # [FIX v13-F] steps_in_battle para nova obs feature
    # Normalizado por steps_in_battle_norm_cap para ficar em [0, 1]
    steps_in_battle_norm_cap: int = 500

    battle_sustain_bonus: float = 0.0

    # Sem bônus passivo de spam de A durante batalha
    battle_a_bonus:       float = 0.0
    battle_a_cap:         int   = 0

    # [FIX v11-B] B penalty só depois de battle_b_threshold A presses
    # PWhiddy philosophy: zero punishments for menus/navigation
    battle_b_penalty:     float = 0.0
    battle_b_threshold:   int   = 999  # Disable b penalty
    
    # [FIX v31 / v32d] Whiteout reforçado (-30) e guarda anti-entrada suicida em grama (<35% HP e 0 Poções)
    whiteout_penalty:          float = -30.0
    whiteout_idle_multiplier:  float = 0.0
    terminate_on_whiteout:     bool  = True
    wild_grind_penalty:        float = 0.0
    low_hp_battle_penalty:     float = -15.0

    # [FIX v12-B / v13-A] Anti-farming: threshold RANDOMIZADO por episódio
    # v12 usava 0.60 fixo → agente calibrou farming em exatamente 0.60.
    # v13: sorteado em [farm_ratio_threshold_min, farm_ratio_threshold_max]
    # a cada reset, tornando calibração impossível.
    farm_ratio_threshold_min: float = 0.80
    farm_ratio_threshold_max: float = 0.95
    farm_kill_threshold:      float = 0.003
    farm_detection_window:    int   = 200
    # Soft reset to prevent endless loops, but no harsh punishment
    farm_episode_penalty:     float = 0.0

    # First Strike reforçado e sujeito à fadiga suave por mapa
    first_strike_bonus:       float = 10.0

    # [FIX v19 / v31 / v32e] Dynamic Battle Curriculum: libera D-pad quando win_rate >= 35%
    enable_battle_curriculum:             bool  = True
    curriculum_win_rate_threshold:        float = 0.35
    max_potion_macro_attempts_per_battle: int   = 2
    potion_macro_cooldown_steps:          int   = 12
    max_tm_macro_attempts_per_episode:    int   = 2
    wasted_macro_penalty:                 float = -0.25

    # [FIX v32 / v32e] Captura, Poké Mart, troca forçada, fuga tática e anti-status spam
    capture_new_species_bonus:            float = 120.0
    pokeball_throw_bonus:                 float = 15.0
    max_capture_macro_attempts_per_battle: int  = 2
    mart_buy_bonus:                       float = 80.0
    super_effective_bonus:                float = 3.0
    flee_trainer_penalty:                 float = -5.0
    flee_trainer_cursor_penalty:          float = -1.0
    status_move_spam_penalty:             float = -1.5
    unnecessary_flee_penalty:             float = -25.0
    smart_flee_bonus:                     float = 15.0
    forced_switch_bonus:                  float = 10.0

    # [FIX v13-B] entry_bonus_revisit: 2.0 → 0.0 (zero não dá incentivo)
    entry_bonus_revisit:       float = 0.0    # era 2.0
    whiteout_entry_cooldown:   int   = 500
    levelup_weight:       float = 60.0
    max_rewarded_level_base:      int   = 24
    max_rewarded_level_per_badge: int   = 12
    badge_weight:         float = 100.0
    milestone_weight:     float = 10.0
    parcel_pickup_bonus:   float = 150.0
    parcel_delivery_bonus: float = 300.0
    teachy_tv_bonus:       float = 100.0
    dialogue_advance_bonus:   float = 0.15
    dialogue_stuck_grace:     int   = 25
    dialogue_stuck_penalty:   float = -0.02
    dialogue_stuck_terminate: int   = 450
    stuck_threshold:      int   = 150
    stuck_penalty:        float = -0.08

    # Bússola direcional alinhada ao sub-objetivo N/S/L/O
    north_bonus_per_tile: float = 1.0

    # [FIX v17] Incentivos positivos para sair da casa (substitui v16 punições)
    # Cap diferenciado: mapas interiores (bank=4) esgotam rápido, exterior equilibrado
    map_explore_cap_interior: float = 35.0   # bank=4 (quartos, casas) — esgota rápido
    map_explore_cap_exterior: float = 600.0  # bank!=4 exterior equilibrado vs recompensas de combate

    outdoor_sustain_bonus: float = 0.0

    # Early termination desativado (Whiddy philosophy) para não matar o agente no meio da batalha contra o Gary.
    interior_only_terminate_steps: int = 10000

    # [FIX v17] Removido: map_boredom_penalty (causava -230 quebrando gradiente)
    # [FIX v17] Removido: map_explore_cap (substituído por caps diferenciados)
    # [FIX v17] Removido: map_chain_bonus (north_bonus já dá direção clara)

    # [FIX v18] Penalidade interior calibrada para evitar ociosidade no laboratório:
    # [FIX v28] Early Stopping substitui punição: se não achar a porta, o episódio
    # simplesmente encerra mais cedo. Zero de punição contínua para proteger a curva.
    indoor_step_penalty: float = 0.0

    # --- RAM sanity caps ---
    max_party_level_sum: int = 600
    max_party_hp:        int = 5000
    max_enemy_hp:        int = 800
    max_coord:           int = 255
    max_map_id:          int = 255
    max_level_delta:     int = 6


CFG = Config()
FRAMES_PER_STEP = CFG.frames_held + CFG.frames_noop

def setup_logging() -> logging.Logger:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    try:
        log_dir = os.path.dirname(CFG.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        handlers.insert(0, logging.FileHandler(CFG.log_file, mode="a"))
    except OSError:
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("pokemon_rl")

log = setup_logging()

class ActionSpace:
    NAMES = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "TACTICAL_MACRO"]

    _ACTIONS = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # A
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # B
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # UP
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # DOWN
        np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # LEFT
        np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # RIGHT
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # TACTICAL_MACRO (No-op para o emulador)
    ]

    NO_OP: list = [0] * 12

    @classmethod
    def get(cls, idx: int) -> np.ndarray:
        return cls._ACTIONS[idx]

    @classmethod
    def size(cls) -> int:
        return len(cls._ACTIONS)
