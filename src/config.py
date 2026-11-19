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


def _default_state_file() -> str:
    if retro is not None:
        return os.path.join(
            os.path.dirname(retro.__file__),
            "data", "stable", "PokemonFireRed", "Start.state.gz",
        )
    return "./progress_states/Start.state.gz"


def _default_base_save_dir() -> str:
    env_dir = os.getenv("PKMN_SAVE_DIR")
    if env_dir:
        return env_dir
    preferred = "/mnt/d/projects/pkmn_fire_red_saves"
    if os.path.isdir(preferred) or os.path.isdir("/mnt/d"):
        return preferred
    return "./saves"


@dataclass(frozen=True)
class Config:
    """Todos os parâmetros tuneáveis em um lugar. Frozen para evitar mutação."""

    # --- paths ---
    state_file: str = _default_state_file()
    # [FIX v18] Save state exterior: ponto de partida alternativo já fora da casa
    pallet_exterior_state_file: str = "./progress_states/pallet_exterior.state"
    # Probabilidade de iniciar do exterior (quando o state existir)
    exterior_start_ratio: float = 0.60
    
    # Caminho base configurável via env var, com fallback seguro fora do WSL/Windows
    base_save_dir: str = _default_base_save_dir()
    model_path:          str = f"{base_save_dir}/pokemon_lstm_model"
    checkpoint_dir:      str = f"{base_save_dir}/checkpoints/"
    progress_dir:        str = "./progress_states/"
    tensorboard_log_dir: str = f"{base_save_dir}/pokemon_tensorboard/"
    log_file:            str = f"{base_save_dir}/pokemon_training.log"
    whiteout_frame_dir:  str = f"{base_save_dir}/whiteout_frames/"

    # [FIX v30] Strategist + Frontier Bias + Matriz de Colisão 9x9 (weight surgery compatível)
    config_stamp:        str = "v30_hrl_strategist_grid9x9"

    # [FIX 16] steps_state_file para sincronização de currículo entre workers
    steps_state_file:    str = "./steps_state.txt"

    # --- curriculum sampling (Frontier Bias) ---
    frontier_prob:       float = 0.50
    intermediate_prob:   float = 0.30
    initial_prob:        float = 0.20

    # --- strategist & advisor ---
    ollama_url:                  str   = "http://localhost:11434/api/generate"
    ollama_model:                str   = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    ollama_timeout_sec:          float = 15.0
    strategist_stagnation_steps: int   = 500
    strategist_goal_bonus:       float = 30.0
    dialog_repeat_penalty:       float = -0.001

    # --- training ---
    num_cpu:         int   = 6
    learn_batch:     int   = 40_000
    n_steps:         int   = 1024
    batch_size:      int   = 256
    n_epochs:        int   = 4
    gamma:           float = 0.997
    gae_lambda:      float = 0.95
    # [FIX v15 / v30b] ent_coef 0.08→0.02: permite sequências de batalha FIGHT→ataque
    ent_coef:        float = 0.02
    lr_start:        float = 1e-4
    lr_end:          float = 1e-4
    lr_total_budget: int   = 50_000_000
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

    # --- reward weights ---
    # [FIX v18.3] Remoção do medo do tempo (Whiddy-style).
    # Como não perdemos mais tempo passivamente, o limite de passos cuida da inércia.
    # [FIX v18.4] Reativado levemente o time_penalty para causar inércia desconfortável e forçar movimento
    # [FIX v24 / v30b] Zerado o time_penalty (Whiddy-Pure)
    time_penalty:    float = 0.0
    # [FIX v26] Whiddy-Pure: explore_weight é a ÚNICA fonte primária de dopamina
    explore_weight:  float = 3.0
    # [FIX v27] Bônus extra por tile novo em mapas exteriores (bank=3, routes)
    # Torna o exterior ~50% mais lucrativo que o interior sem punir estar dentro.
    explore_weight_outdoor_bonus: float = 5.0

    # [FIX v18.3] Novas recompensas inspiradas em Peter Whiddy
    max_op_level_reward_scale: float = 0.2
    heal_reward_scale:         float = 8.0
    # Quando o agente esgotar os tiles do mapa, ele ganha uma punição de tempo leve
    max_explore_reward_per_map: float = 150.0
    stale_map_penalty:          float = 0.0

    # [FIX v26] Whiddy-Pure: zerado new_map_bonus (causava exploit de +60 grátis no spawn)
    new_map_bonus:        float = 50.0
    new_map_bonus_route:  float = 100.0

    # [FIX v15] battle_idle_penalty: -0.05 → -0.01, grace: 120 → 200
    # Punições muito altas geravam fobia de batalhas e colapso na grama.
    battle_idle_penalty:  float = 0.0
    battle_idle_grace:    int   = 200   # mais carência para lidar com menus e gary

    # [FIX v11-C] Novo sistema de reward de dano baseado em HP%
    hp_pct_scale:         float = 15.0
    damage_weight:        float = 0.05  # fallback

    # [FIX v18.2] victory_bonus reduzido para 25.0 para nerfar farm de matinho
    victory_bonus:        float = 40.0
    near_ko_bonus:        float = 20.0
    post_battle_grace:    int   = 200

    # [FIX v11-F / v14-B] entry_bonus reduzido para valor simbólico.
    # O lucro real de batalhar vem do first_strike_bonus (12.0) pago ao
    # causar o primeiro dano. entry_bonus sinaliza "você entrou em batalha"
    # sem ser o incentivo principal — isso elimina o ciclo de farming passivo.
    entry_bonus:          float = 1.0

    # [FIX v13-F] steps_in_battle para nova obs feature
    # Normalizado por steps_in_battle_norm_cap para ficar em [0, 1]
    steps_in_battle_norm_cap: int = 500

    battle_sustain_bonus: float = 0.0

    # [FIX v11-D] battle_a_cap 20→50, battle_a_bonus 0.05→0.03
    # Cap de 20 bloqueava recompensa no meio da batalha. Aumentado para cobrir
    # navegação completa: FIGHT→move→confirmar→esperar = ~10-15 A presses.
    battle_a_bonus:       float = 0.1
    battle_a_cap:         int   = 100

    # [FIX v11-B] B penalty só depois de battle_b_threshold A presses
    # PWhiddy philosophy: zero punishments for menus/navigation
    battle_b_penalty:     float = 0.0
    battle_b_threshold:   int   = 999  # Disable b penalty
    
    # PWhiddy philosophy: zero punishment for whiteout
    whiteout_penalty:          float = 0.0
    whiteout_idle_multiplier:  float = 0.0

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

    # [FIX v18.2] First Strike Bonus: reduzido para 12.0 para nerfar farm de matinho
    # O lucro real de batalhar vem do first_strike_bonus.
    first_strike_bonus:       float = 12.0

    # [FIX v19] Dynamic Battle Curriculum (rodinhas de bicicleta)
    # Bloqueia (NO-OP) direcionais, start e select nas batalhas até bater 10% de win rate.
    enable_battle_curriculum:      bool  = True
    curriculum_win_rate_threshold: float = 0.10

    # [FIX v13-B] entry_bonus_revisit: 2.0 → 0.0 (zero não dá incentivo)
    entry_bonus_revisit:       float = 0.0    # era 2.0
    whiteout_entry_cooldown:   int   = 500
    levelup_weight:       float = 200.0
    badge_weight:         float = 50.0
    milestone_weight:     float = 10.0
    stuck_threshold:      int   = 150
    stuck_penalty:        float = -0.08

    # [FIX v30b] Reduzido north_bonus_per_tile (2.0 → 0.5) para priorizar exploração e batalhas
    north_bonus_per_tile: float = 0.5

    # [FIX v17] Incentivos positivos para sair da casa (substitui v16 punições)
    # Cap diferenciado: mapas interiores (bank=4) esgotam rápido, exterior é abundante
    map_explore_cap_interior: float = 50.0   # bank=4 (quartos, casas) — esgota rápido
    map_explore_cap_exterior: float = 5000.0  # bank!=4 exterior expandido para mais exploração

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
