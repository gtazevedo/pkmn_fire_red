import os
import logging
import numpy as np
from dataclasses import dataclass
import retro
# [FIX v13-F] RAM obs expandido: 12 base + 5 situation embedding
# + 1 battle_idle_ratio + 1 battle_outcome_potential + 1 steps_in_battle_norm = 20
RAM_SIZE = 12 + 5 + 3   # = 20

@dataclass(frozen=True)
class Config:
    """Todos os parâmetros tuneáveis em um lugar. Frozen para evitar mutação."""

    # --- paths ---
    state_file: str = os.path.join(
        os.path.dirname(retro.__file__),
        "data", "stable", "PokemonFireRed", "Start.state.gz",
    )
    # [FIX v18] Save state exterior: ponto de partida alternativo já fora da casa
    pallet_exterior_state_file: str = "./progress_states/pallet_exterior.state"
    # Probabilidade de iniciar do exterior (quando o state existir)
    exterior_start_ratio: float = 0.60
    model_path:          str = "pokemon_lstm_model"
    checkpoint_dir:      str = "./checkpoints/"
    progress_dir:        str = "./progress_states/"
    tensorboard_log_dir: str = "./pokemon_tensorboard/"
    log_file:            str = "./pokemon_training.log"
    whiteout_frame_dir:  str = "./whiteout_frames/"

    # [FIX v18] config_stamp bumped — exterior save state + calibrated indoor penalty
    config_stamp:        str = "v18_exterior_state"

    # [FIX 16] steps_state_file para sincronização de currículo entre workers
    steps_state_file:    str = "./steps_state.txt"

    # --- training ---
    num_cpu:         int   = 6
    learn_batch:     int   = 40_000
    n_steps:         int   = 1024
    batch_size:      int   = 256
    n_epochs:        int   = 4
    gamma:           float = 0.997
    gae_lambda:      float = 0.95
    # [FIX v15] ent_coef 0.08→0.015: entropia muito alta (0.08) impedia
    # sequências de botões complexas (FIGHT→ataque). 0.015 é mais equilibrado.
    ent_coef:        float = 0.015
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
    max_steps_p3: int = 10_000

    # --- frame skip ---
    frames_held: int = 6
    frames_noop: int = 2
    pool_start:  int = 4
    pool_end:    int = 8

    # --- reward weights ---
    # [FIX v18.3] Remoção do medo do tempo (Whiddy-style).
    # Como não perdemos mais tempo passivamente, o limite de passos cuida da inércia.
    time_penalty:    float = 0.0
    explore_weight:  float = 1.0

    # [FIX v18.3] Novas recompensas inspiradas em Peter Whiddy
    max_op_level_reward_scale: float = 0.2
    heal_reward_scale:         float = 4.0
    # Quando o agente esgotar os tiles do mapa, ele ganha uma punição de tempo leve
    max_explore_reward_per_map: float = 150.0
    stale_map_penalty:          float = -0.002

    # [FIX v11-I] new_map_bonus 20→30 para cidades/dungeons (bank≠4)
    # bank=4 routes já têm north_bonus contínuo; cidades merecem mais.
    new_map_bonus:        float = 30.0
    new_map_bonus_route:  float = 10.0  # bank=4 (Routes): menos bônus, north_bonus compensa

    # [FIX v15] battle_idle_penalty: -0.05 → -0.01, grace: 120 → 200
    # Punições muito altas geravam fobia de batalhas e colapso na grama.
    battle_idle_penalty:  float = -0.01
    battle_idle_grace:    int   = 200   # mais carência para lidar com menus e gary

    # [FIX v11-C] Novo sistema de reward de dano baseado em HP%
    hp_pct_scale:         float = 15.0
    damage_weight:        float = 0.05  # fallback

    # [FIX v18.2] victory_bonus aumentado para tornar a batalha mais lucrativa
    victory_bonus:        float = 50.0
    near_ko_bonus:        float = 5.0
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
    battle_a_bonus:       float = 0.03
    battle_a_cap:         int   = 50

    # [FIX v11-B] B penalty só depois de battle_b_threshold A presses
    # Antes era pago sempre, punindo navegação legítima de menu.
    battle_b_penalty:     float = -0.10
    battle_b_threshold:   int   = 3  # só pena B após 3 A presses na batalha atual

    # [FIX v18.2] Whiteout penalty bastante reduzido para evitar fobia do rival.
    whiteout_penalty:          float = -2.0
    whiteout_idle_multiplier:  float = 0.0   # [FIX] flat penalty para não punir morte natural

    # [FIX v12-B / v13-A] Anti-farming: threshold RANDOMIZADO por episódio
    # v12 usava 0.60 fixo → agente calibrou farming em exatamente 0.60.
    # v13: sorteado em [farm_ratio_threshold_min, farm_ratio_threshold_max]
    # a cada reset, tornando calibração impossível.
    farm_ratio_threshold_min: float = 0.35
    farm_ratio_threshold_max: float = 0.55
    farm_kill_threshold:      float = 0.003
    farm_detection_window:    int   = 200
    # [FIX v15] farm_episode_penalty: -50.0 → -15.0
    # Evita punições desmedidas quando o agente apenas se perdeu nos menus.
    farm_episode_penalty:     float = -15.0

    # [FIX v18.2] First Strike Bonus: 25.0
    # O lucro real de batalhar vem do first_strike_bonus.
    first_strike_bonus:       float = 25.0

    # [FIX v13-B] entry_bonus_revisit: 2.0 → 0.0 (zero não dá incentivo)
    entry_bonus_revisit:       float = 0.0    # era 2.0
    whiteout_entry_cooldown:   int   = 500
    levelup_weight:       float = 15.0
    badge_weight:         float = 50.0
    milestone_weight:     float = 10.0
    stuck_threshold:      int   = 150
    stuck_penalty:        float = -0.20

    # North bonus (y decresce ao ir para norte no FireRed)
    # [FIX v17] Aumentado de 1.0 -> 2.5: norte é o caminho do lab, deve ser irresistível
    north_bonus_per_tile: float = 2.5

    # [FIX v17] Incentivos positivos para sair da casa (substitui v16 punições)
    # Cap diferenciado: mapas interiores (bank=4) esgotam rápido, exterior é abundante
    map_explore_cap_interior: float = 25.0   # bank=4 (quartos, casas) — esgota rápido
    map_explore_cap_exterior: float = 300.0  # bank!=4 (ruas, rotas) — praticamente ilimitado

    # Bônus sustentado por estar do lado de fora (bank!=4): incentivo positivo constante
    outdoor_sustain_bonus: float = 0.015

    # Early termination: encerra episódio sem penalidade se agente nunca sair de bank=4
    # Episódio curto = poucos pontos. Episódio longo lá fora = muitos pontos.
    interior_only_terminate_steps: int = 600

    # [FIX v17] Removido: map_boredom_penalty (causava -230 quebrando gradiente)
    # [FIX v17] Removido: map_explore_cap (substituído por caps diferenciados)
    # [FIX v17] Removido: map_chain_bonus (north_bonus já dá direção clara)

    # [FIX v18] Penalidade interior muito pequena e calibrada:
    # - 600 steps dentro = -3.0 máximo
    # - Sair para Pallet Town = +30.0 new_map_bonus
    # - Net ao sair: +27.0 positivo mesmo após penalidade máxima
    indoor_step_penalty: float = -0.005

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(CFG.log_file, mode="a"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("pokemon_rl")

log = setup_logging()

class ActionSpace:
    NAMES = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT"]

    _ACTIONS = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),  # A
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # B
        np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),  # UP
        np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),  # DOWN
        np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),  # LEFT
        np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),  # RIGHT
    ]

    NO_OP: list = [0] * 12

    @classmethod
    def get(cls, idx: int) -> np.ndarray:
        return cls._ACTIONS[idx]

    @classmethod
    def size(cls) -> int:
        return len(cls._ACTIONS)
