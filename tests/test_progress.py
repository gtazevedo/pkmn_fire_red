import os
import random
from types import SimpleNamespace

from src.config import CFG
from src.progress import EpisodeStats, ProgressManager, RewardAccumulator


def test_state_sort_key_ordering():
    assert ProgressManager._state_sort_key("/tmp/Start.state.gz") == 0
    assert ProgressManager._state_sort_key("./progress_states/pallet_exterior.state") == 1
    assert ProgressManager._state_sort_key("./progress_states/events_112.state") == 112
    assert ProgressManager._state_sort_key("./progress_states/events_122.state") == 122


def test_frontier_bias_sampling_distribution(tmp_path):
    # Cria 9 states de eventos + Start + pallet_exterior = 11 states
    for flags in [112, 113, 114, 116, 117, 118, 120, 121, 122]:
        (tmp_path / f"events_{flags:03d}.state").write_bytes(b"state")

    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    pm._saved_states.append("./progress_states/pallet_exterior.state")
    assert pm._best_flags_ever == 122
    assert pm._best_milestone_idx == 122
    assert os.path.basename(pm._best_state_file) == "events_122.state"

    rng = random.Random(42)
    n_samples = 10_000
    for _ in range(n_samples):
        pm.sample_state(rng=rng)

    stats = pm.stats_dict()
    frontier_ratio = stats["progress/frontier_samples"] / n_samples
    intermediate_ratio = stats["progress/intermediate_samples"] / n_samples
    initial_ratio = stats["progress/initial_samples"] / n_samples

    assert abs(frontier_ratio - CFG.frontier_prob) < 0.03
    assert abs(intermediate_ratio - CFG.intermediate_prob) < 0.03
    assert abs(initial_ratio - CFG.initial_prob) < 0.03


def test_frontier_bias_small_pools(tmp_path):
    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    rng = random.Random(7)

    # 1 state
    assert pm.sample_state(rng=rng) == "/tmp/Start.state.gz"
    assert pm.tier_counts["initial"] == 1

    # 2 states: deve preferir o state mais avançado (~71.4%)
    adv = str(tmp_path / "events_010.state")
    pm._saved_states.append(adv)
    hits_adv = sum(1 for _ in range(1000) if pm.sample_state(rng=rng) == adv)
    assert 650 < hits_adv < 780


def test_check_and_save_monotonic_record(tmp_path):
    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    fake_emu = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"NEW_STATE_BYTES"))

    pm.check_and_save(fake_emu, 5, env_id=0)
    assert pm._best_flags_ever == 5
    assert (tmp_path / "events_005.state").read_bytes() == b"NEW_STATE_BYTES"

    # Flags menores ou iguais não sobrescrevem nem duplicam
    pm.check_and_save(fake_emu, 3, env_id=0)
    assert pm._best_flags_ever == 5
    assert not (tmp_path / "events_003.state").exists()


def test_episode_stats_monotonic_events_and_reward_accumulator():
    import struct
    ram = bytearray(262144 + 0x6000)
    sb1 = 0x025000
    struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1)
    flags_start = sb1 + 0x0EE0

    stats = EpisodeStats()
    ram[flags_start] = 0b00000011  # 2 flags iniciais
    stats.reset({}, ram_array=ram)
    assert stats.max_event_sum == 2

    # Sobe para 4 flags (+2 novas -> +80.0)
    ram[flags_start] = 0b00001111
    rew, total = stats.update_events(ram)
    assert rew == 80.0 and total == 4

    # Desliga e religa flag: NÃO pode pagar de novo (anti-reward hacking)
    ram[flags_start] = 0b00000111
    rew_drop, _ = stats.update_events(ram)
    assert rew_drop == 0.0

    ram[flags_start] = 0b00001111
    rew_reup, _ = stats.update_events(ram)
    assert rew_reup == 0.0

    # Nova flag real (5 flags) -> +40.0
    ram[flags_start] = 0b00011111
    rew_new, total_new = stats.update_events(ram)
    assert rew_new == 40.0 and total_new == 5

    acc = RewardAccumulator()
    acc.add("explore", 3.0)
    acc.add("strategist", 3.0)
    assert acc.total == 6.0
    assert acc.to_dict()["reward_strategist"] == 3.0
    acc.reset()
    assert acc.total == 0.0
    assert acc.strategist == 0.0


def test_explore_rewards_per_map_resets_between_episodes():
    stats = EpisodeStats()
    stats.reset({})
    map_key = (4, 0)  # interior cap = 35.0
    # Satura o cap de exploração no episódio 1
    for x in range(60):
        stats.update_exploration((4, 0, x, 0), map_key, include_new_map_bonus=False)
    assert stats.explore_rewards_per_map[map_key] == CFG.map_explore_cap_interior

    # No episódio 2, explore_rewards_per_map deve resetar e voltar a pontuar tiles visitados
    stats.reset({})
    assert stats.explore_rewards_per_map == {}
    rew, is_new = stats.update_exploration((4, 0, 99, 99), map_key, include_new_map_bonus=False)
    assert is_new is True
    assert rew > 0.0


def test_current_battle_damage_resets_per_battle():
    stats = EpisodeStats()
    stats.reset({"p1_lvl": 5, "p1_hp": 20, "p1_maxhp": 20})

    # Batalha 1: entra com enemy_hp=20 e causa 7 de dano
    stats.update_battle(True, enemy_hp=20, party_hp=20, script_lock=False,
                        last_raw_frame=None, env_id=0, episode_num=1,
                        current_tile=(3, 19, 5, 5), party_level=5)
    assert stats.current_battle_damage == 0
    stats.update_battle(True, enemy_hp=13, party_hp=20, script_lock=False,
                        last_raw_frame=None, env_id=0, episode_num=1,
                        current_tile=(3, 19, 5, 5), party_level=5)
    assert stats.current_battle_damage == 7
    assert stats.battle_damage_total == 7

    # Sai da batalha
    stats.update_battle(False, enemy_hp=0, party_hp=20, script_lock=False,
                        last_raw_frame=None, env_id=0, episode_num=1,
                        current_tile=(3, 19, 5, 5), party_level=5)

    # Batalha 2: current_battle_damage deve resetar para 0 enquanto battle_damage_total acumula
    stats.update_battle(True, enemy_hp=18, party_hp=20, script_lock=False,
                        last_raw_frame=None, env_id=0, episode_num=1,
                        current_tile=(3, 19, 5, 5), party_level=5)
    assert stats.current_battle_damage == 0
    assert stats.battle_damage_total == 7

    stats.update_battle(True, enemy_hp=10, party_hp=20, script_lock=False,
                        last_raw_frame=None, env_id=0, episode_num=1,
                        current_tile=(3, 19, 5, 5), party_level=5)
    assert stats.current_battle_damage == 8
    assert stats.battle_damage_total == 15


def test_map_milestone_saving_and_sort_key(tmp_path):
    assert ProgressManager._state_sort_key("/tmp/events_122.state") == 122.0
    assert ProgressManager._state_sort_key("/tmp/events_122_m02_3_1.state") == 122.02

    (tmp_path / "events_122.state").write_bytes(b"BASE122")
    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    assert pm._best_flags_ever == 122
    assert pm._best_map_rank == 0

    fake_emu_r1 = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"ROUTE1_STATE"))
    pm.check_and_save_map_milestone(fake_emu_r1, 122, (3, 19), env_id=0)
    p_r1 = tmp_path / "events_122_m01_3_19.state"
    assert p_r1.read_bytes() == b"ROUTE1_STATE"
    assert pm._best_map_rank == 1

    fake_emu_vir = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"VIRIDIAN_STATE"))
    pm.check_and_save_map_milestone(fake_emu_vir, 122, (3, 1), env_id=0)
    p_vir = tmp_path / "events_122_m02_3_1.state"
    assert p_vir.read_bytes() == b"VIRIDIAN_STATE"
    assert pm._best_map_rank == 2

    # Mapa de rank menor ou igual no mesmo flags nao deve duplicar, nem mapa fora do grafo
    pm.check_and_save_map_milestone(fake_emu_r1, 122, (3, 19), env_id=0)
    pm.check_and_save_map_milestone(fake_emu_r1, 122, (4, 0), env_id=0)
    assert not (tmp_path / "events_122_m00_4_0.state").exists()

    # Recarregar do disco deve preservar _best_map_rank=2 e colocar m02 no topo
    pm_reload = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    assert pm_reload._best_flags_ever == 122
    assert pm_reload._best_map_rank == 2
    assert os.path.basename(pm_reload._best_state_file) == "events_122_m02_3_1.state"


def test_episode_stats_seeds_spawn_map_and_y_bounds():
    stats = EpisodeStats()
    stats.reset({"map_bank": 4, "map_id": 0, "player_y": 8})
    assert stats.map_visits == {(4, 0)}
    assert stats.y_min_episode == 8
    assert stats.y_max_episode == 8

    _, is_new_spawn = stats.update_exploration((4, 0, 5, 8), (4, 0), include_new_map_bonus=False)
    assert is_new_spawn is False
    _, is_new_pallet = stats.update_exploration((3, 0, 5, 8), (3, 0), include_new_map_bonus=False)
    assert is_new_pallet is True


def test_parcel_milestones_and_cross_worker_disk_sync(tmp_path):
    (tmp_path / "events_123.state").write_bytes(b"FLAGS123")
    pm_worker_a = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    pm_worker_b = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")

    # Mesmo partindo de 122 flags com recorde global 123, pegar Parcel no Mart (5,3) deve salvar rank=5
    fake_emu = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"PARCEL_MART"))
    pm_worker_a.check_and_save_map_milestone(fake_emu, 122, (5, 3), env_id=0, has_parcel=True)
    parcel_file = tmp_path / "events_122_m05_5_3.state"
    assert parcel_file.read_bytes() == b"PARCEL_MART"
    assert pm_worker_a._best_map_rank == 5

    # Worker B deve descobrir o novo arquivo no disco ao chamar sample_state()
    rng = random.Random(123)
    pm_worker_b.sample_state(rng=rng)
    assert str(parcel_file) in pm_worker_b._saved_states


def test_anti_battle_junkie_fatigue_whiteout_and_level_cap(tmp_path):
    stats = EpisodeStats()
    stats.reset({"p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24})

    # Rota exterior (3,19) sem Parcel: fadiga suave 1.0 -> 0.5 -> 0.25 -> 0.10 (nunca zera totalmente)
    observed = []
    for i in range(4):
        stats.update_battle(True, 20, 24, False, None, 0, 1, (3, 19, 5, 5), 7 + i)
        r_dmg, _, r_ko, _, _, _ = stats.update_battle(True, 1, 24, False, None, 0, 1, (3, 19, 5, 5), 7 + i)
        _, _, r_vic, _, _, _ = stats.update_battle(True, 0, 24, False, None, 0, 1, (3, 19, 5, 5), 7 + i)
        if i < 3:
            stats.update_battle(False, 0, 24, False, None, 0, 1, (3, 19, 5, 5), 7 + i)
        observed.append((stats.current_battle_fatigue(), r_dmg, r_ko, r_vic))

    assert [round(o[0], 2) for o in observed] == [1.0, 0.5, 0.25, 0.1]
    assert observed[2][1] > 0.0 and observed[2][3] > 0.0
    assert observed[3][1] > 0.0 and observed[3][3] > 0.0

    # Sem Oak's Parcel, level up em batalha paga valor cheio (ate level_cap); com Parcel paga 50%
    assert stats.update_level(8, env_id=0, badges=0, in_battle=True, has_parcel=False) == CFG.levelup_weight
    stats.update_battle(False, 0, 24, False, None, 0, 1, (3, 19, 5, 5), 8)

    # Com Oak's Parcel na rota exterior, fadiga eh 0.5 -> 0.2 -> 0.05
    stats.update_battle(True, 20, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)
    assert stats.current_battle_fatigue(has_parcel=True) == 0.5
    stats.update_battle(False, 0, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)
    stats.update_battle(True, 20, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)
    assert stats.current_battle_fatigue(has_parcel=True) == 0.2
    stats.update_battle(False, 0, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)
    stats.update_battle(True, 20, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)
    assert stats.current_battle_fatigue(has_parcel=True) == 0.05
    stats.update_battle(False, 0, 24, False, None, 0, 1, (3, 1, 5, 5), 8, has_parcel=True)

    # Em ginasio/interior (6,2), fadiga decai 1.0 -> 0.5 -> 0.25 -> 0.1 e nunca zera
    gym_fatigue = []
    for _ in range(4):
        stats.update_battle(True, 20, 24, False, None, 0, 1, (6, 2, 5, 5), 8)
        gym_fatigue.append(stats.current_battle_fatigue())
        stats.update_battle(False, 0, 24, False, None, 0, 1, (6, 2, 5, 5), 8)
    assert gym_fatigue == [1.0, 0.5, 0.25, 0.1]
    assert stats.update_level(9, env_id=0, badges=0, in_battle=True, has_parcel=False) == CFG.levelup_weight
    assert stats.update_level(10, env_id=0, badges=0, in_battle=True, has_parcel=True) == CFG.levelup_weight * 0.5
    stats.max_level = 9  # restaura baseline 9 para o teste de teto abaixo (12 - 9 = 3)

    # Whiteout aplica penalidade -30.0 com terminate_on_whiteout=True; entrada em grama <35% sem pocoes pune -15.0
    _, _, _, _, whiteout_pen, _ = stats.update_battle(True, 20, 0, False, None, 0, 1, (6, 2, 5, 5), 9)
    assert whiteout_pen == CFG.whiteout_penalty == -30.0
    assert CFG.terminate_on_whiteout is True
    assert CFG.wild_grind_penalty == 0.0
    assert CFG.low_hp_battle_penalty == -15.0

    # Teto de nivel por insignia fora de batalha: lvl 24 paga, lvl 25 sem insignia paga 0, com 1 insignia volta a pagar
    stats.max_level = 21
    assert stats.update_level(24, env_id=0, badges=0, in_battle=False) == 3 * CFG.levelup_weight
    assert stats.update_level(25, env_id=0, badges=0, in_battle=False) == 0.0
    assert stats.update_level(26, env_id=0, badges=1, in_battle=False) == CFG.levelup_weight

    # Subnivelado na rota exterior (party_level < 14, badges=0): fadiga abrandada para XP
    under_stats = EpisodeStats()
    under_stats.reset({"p1_lvl": 8, "p1_hp": 26, "p1_maxhp": 26})
    under_fatigue = []
    for k in range(7):
        under_stats.update_battle(True, 20, 26, False, None, 0, 1, (3, 19, 5, 5), 8 + (k // 3), badges=0)
        under_fatigue.append(under_stats.current_battle_fatigue(has_parcel=False, party_level=8 + (k // 3), badges=0))
        under_stats.update_battle(False, 0, 26, False, None, 0, 1, (3, 19, 5, 5), 8 + (k // 3), badges=0)
    assert under_fatigue == [1.0, 1.0, 1.0, 1.0, 0.6, 0.6, 0.3]

    # Marco geografico com flags menores que o recorde global ainda salva e entra em frontier
    for fl in range(112, 124):
        (tmp_path / f"events_{fl:03d}.state").write_bytes(b"BASE")
    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    fake_emu = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"PARCEL_116"))
    pm.check_and_save_map_milestone(fake_emu, 116, (5, 3), env_id=0, has_parcel=True)
    parcel_116 = str(tmp_path / "events_116_m05_5_3.state")
    assert os.path.exists(parcel_116)

    rng = random.Random(99)
    parcel_hits = sum(1 for _ in range(200) if pm.sample_state(rng=rng) == parcel_116)
    assert parcel_hits > 15


def test_genesis_anchor_prevents_start_dilution_at_scale(tmp_path):
    # Simula pool grande com 60 estados avancados alem de Start e pallet_exterior (62 total)
    for fl in range(112, 172):
        (tmp_path / f"events_{fl:03d}.state").write_bytes(b"STATE")

    start_path = str(tmp_path / "Start.state.gz")
    pm = ProgressManager(progress_dir=str(tmp_path), state_file=start_path)
    pm._saved_states.append("./progress_states/pallet_exterior.state")

    rng = random.Random(2026)
    n_samples = 10_000
    genesis_targets = {start_path, "./progress_states/pallet_exterior.state"}
    genesis_hits = 0
    start_hits = 0
    for _ in range(n_samples):
        picked = pm.sample_state(rng=rng)
        if picked in genesis_targets:
            genesis_hits += 1
        if picked == start_path:
            start_hits += 1

    stats = pm.stats_dict()
    frontier_ratio = stats["progress/frontier_samples"] / n_samples
    intermediate_ratio = stats["progress/intermediate_samples"] / n_samples
    initial_ratio = stats["progress/initial_samples"] / n_samples

    assert abs(frontier_ratio - CFG.frontier_prob) < 0.03
    assert abs(intermediate_ratio - CFG.intermediate_prob) < 0.03
    assert abs(initial_ratio - CFG.initial_prob) < 0.03
    assert genesis_hits / n_samples >= 0.10
    assert start_hits / n_samples >= 0.045
    assert stats["progress/genesis_samples"] == genesis_hits
    assert stats["progress/start_state_samples"] == start_hits
    assert abs(stats["progress/genesis_ratio"] - (genesis_hits / n_samples)) < 1e-6
    assert abs(stats["progress/start_state_ratio"] - (start_hits / n_samples)) < 1e-6


def test_post_pokedex_events_124_outranks_pre_delivery_and_saves_route2(tmp_path):
    for fl in [112, 113, 114, 116, 117, 118, 120, 121, 122, 123, 124]:
        (tmp_path / f"events_{fl:03d}.state").write_bytes(b"STATE")
    for pre in [
        "events_120_m01_3_19.state",
        "events_121_m05_5_3.state",
        "events_121_m09_4_3.state",
    ]:
        (tmp_path / pre).write_bytes(b"PRE")

    start_path = str(tmp_path / "Start.state.gz")
    pallet_path = str(tmp_path / "pallet_exterior.state")
    (tmp_path / "Start.state.gz").write_bytes(b"START")
    (tmp_path / "pallet_exterior.state").write_bytes(b"PALLET")

    pm = ProgressManager(progress_dir=str(tmp_path), state_file=start_path)
    pm._saved_states.append(pallet_path)

    ordered = sorted(pm._saved_states, key=pm._progression_key)
    assert ordered[0] == start_path
    assert ordered[1] == pallet_path
    s124 = str(tmp_path / "events_124.state")
    s121_m09 = str(tmp_path / "events_121_m09_4_3.state")
    assert ordered.index(s124) > ordered.index(s121_m09)

    fake_emu = SimpleNamespace(em=SimpleNamespace(get_state=lambda: b"ROUTE2_POKEDEX"))
    pm.check_and_save_map_milestone(fake_emu, 124, (3, 20), env_id=0, has_pokedex=True)
    r2_file = str(tmp_path / "events_124_m13_3_20.state")
    assert os.path.exists(r2_file)

    ordered2 = sorted(pm._saved_states, key=pm._progression_key)
    assert ordered2[-1] == r2_file
    assert ordered2[0] == start_path and ordered2[1] == pallet_path


def test_visited_grid_stationary_novelty_and_frontier_top_share(tmp_path):
    import numpy as np

    stats = EpisodeStats()
    stats.reset({})
    for _ in range(7):
        stats.update_exploration((3, 19, 5, 5), (3, 19), include_new_map_bonus=False)
    for _ in range(2):
        stats.update_exploration((3, 19, 6, 4), (3, 19), include_new_map_bonus=False)

    vg = stats.get_visited_grid((3, 19), 5, 5, 9)
    assert vg.shape == (9, 9)
    assert vg.dtype == np.float32
    assert np.isclose(vg[4, 4], 1.0)  # clampado em 5/5
    assert np.isclose(vg[3, 5], 0.4)  # (6,4) com 2 visitas
    assert np.isclose(vg[0, 0], 0.0)

    # Piso estacionario: mesmo apos 1000 visitas historicas, tile novo no episodio mantem floor >= 50%
    stats.persistent_tile_visits[(3, 19, 9, 9)] = 1000
    stats.reset({})
    rew_floor, _ = stats.update_exploration((3, 19, 9, 9), (3, 19), include_new_map_bonus=False)
    expected_min = (CFG.explore_weight + CFG.explore_weight_outdoor_bonus) * CFG.explore_novelty_floor
    assert rew_floor >= expected_min - 1e-6

    # Frontier top state share: events_124.state recebe ~50% de frontier_prob (~25% do total)
    for fl in [116, 120, 121, 124]:
        (tmp_path / f"events_{fl:03d}.state").write_bytes(b"S")
    (tmp_path / "events_121_m08_3_0.state").write_bytes(b"S")
    (tmp_path / "events_121_m09_4_3.state").write_bytes(b"S")
    pm = ProgressManager(progress_dir=str(tmp_path), state_file="/tmp/Start.state.gz")
    rng = random.Random(101)
    top_path = str(tmp_path / "events_124.state")
    hits_top = sum(1 for _ in range(2000) if pm.sample_state(rng=rng) == top_path)
    assert 420 <= hits_top <= 580


def test_heal_reward_anti_farming_prior_low_hp_guard():
    stats = EpisodeStats()
    stats.reset({"p1_lvl": 8, "p1_hp": 20, "p1_maxhp": 26})

    # HP previo >= 65% (20/26 = 76.9%) NAO pode recompensar cura no Centro Pokemon
    assert stats.update_heal_reward({"p1_hp": 26, "p1_maxhp": 26}) == 0.0

    # HP previo < 65% (7/26 = 26.9%), ganho pequeno (<3 HP, ex levelup bump) tambem nao recompensa
    stats.last_party_hp = 7
    assert stats.update_heal_reward({"p1_hp": 9, "p1_maxhp": 26}) == 0.0

    # HP previo < 65% (7/26) e cura real para 26/26 recompensa proporcionalmente
    rew = stats.update_heal_reward({"p1_hp": 26, "p1_maxhp": 26})
    expected = ((26 - 7) / 26.0) * CFG.heal_reward_scale
    assert abs(rew - expected) < 1e-6

    # 3 Pokemon feridos a 50% HP (< 65%) curados para 100% recebem 0.50 * heal_reward_scale
    injured_3 = {"p1_lvl": 8, "p1_hp": 13, "p1_maxhp": 26, "p2_lvl": 5, "p2_hp": 9, "p2_maxhp": 18, "p3_lvl": 5, "p3_hp": 8, "p3_maxhp": 16}
    full_3 = {"p1_lvl": 8, "p1_hp": 26, "p1_maxhp": 26, "p2_lvl": 5, "p2_hp": 18, "p2_maxhp": 18, "p3_lvl": 5, "p3_hp": 16, "p3_maxhp": 16}
    stats.reset(injured_3)
    rew_3 = stats.update_heal_reward(full_3)
    assert abs(rew_3 - 0.50 * CFG.heal_reward_scale) < 1e-6

    # 3 Pokemon com HP total >= 65% (42/57 = 73.7%), mas 1 desmaiado revivido no Centro -> tambem recompensa
    fainted_one = {"p1_lvl": 8, "p1_hp": 0, "p1_maxhp": 15, "p2_lvl": 5, "p2_hp": 21, "p2_maxhp": 21, "p3_lvl": 5, "p3_hp": 21, "p3_maxhp": 21}
    revived_all = {"p1_lvl": 8, "p1_hp": 15, "p1_maxhp": 15, "p2_lvl": 5, "p2_hp": 21, "p2_maxhp": 21, "p3_lvl": 5, "p3_hp": 21, "p3_maxhp": 21}
    stats.reset(fainted_one)
    rew_rev = stats.update_heal_reward(revived_all)
    assert abs(rew_rev - ((15.0 / 57.0) * CFG.heal_reward_scale)) < 1e-6


def test_cutscene_script_lock_prevents_false_stuck_and_antimem_config():
    stats = EpisodeStats()
    stats.reset({"map_bank": 4, "map_id": 3, "player_x": 6, "player_y": 4})

    # 200 steps parado no mesmo tile durante cutscene (script_lock=True) nao acumulam stuck
    for _ in range(200):
        stats.update_exploration((4, 3, 6, 4), (4, 3), include_new_map_bonus=False, script_lock=True)
    assert stats.steps_on_cur_tile == 0

    # Assim que script_lock=False, volta a incrementar normalmente
    stats.update_exploration((4, 3, 6, 4), (4, 3), include_new_map_bonus=False, script_lock=False)
    stats.update_exploration((4, 3, 6, 4), (4, 3), include_new_map_bonus=False, script_lock=False)
    assert stats.steps_on_cur_tile == 2

    assert 0.0 < CFG.sticky_action_prob <= 0.25
    assert CFG.reset_random_noop_max > 0
    assert CFG.reset_random_walk_max > 0
    assert 0.0 < CFG.coord_dropout_prob <= 0.5
