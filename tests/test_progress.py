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
    map_key = (4, 0)  # interior cap = 50.0
    # Satura o cap de exploração no episódio 1
    for x in range(30):
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
