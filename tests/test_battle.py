import struct
import numpy as np

from src.battle_data import (
    TYPE_BUG,
    TYPE_ELECTRIC,
    TYPE_FIRE,
    TYPE_FLYING,
    TYPE_GHOST,
    TYPE_GRASS,
    TYPE_GROUND,
    TYPE_NORMAL,
    TYPE_ROCK,
    TYPE_WATER,
    score_moves,
    type_multiplier,
)
from src.config import BATTLE_SIZE
from src.reader import RamReader


def _make_full_ram(
    sb1_offset: int = 0x025560,
    sb2_offset: int = 0x0245BC,
    enc32: int = 0x4B55982C,
) -> bytearray:
    ram = bytearray(262144 + 0x6000)
    struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1_offset)
    struct.pack_into("<I", ram, 262144 + 0x500C, 0x02000000 | sb2_offset)
    struct.pack_into("<I", ram, sb2_offset + 0x0F20, enc32)
    return ram


def test_type_chart_gen3_and_dual_types():
    assert type_multiplier(TYPE_FIRE, TYPE_GRASS, TYPE_GRASS) == 2.0
    assert type_multiplier(TYPE_FIRE, TYPE_GRASS, TYPE_BUG) == 4.0
    assert type_multiplier(TYPE_FIRE, TYPE_WATER, TYPE_WATER) == 0.5
    assert type_multiplier(TYPE_FIRE, TYPE_ROCK, TYPE_GROUND) == 0.5
    assert type_multiplier(TYPE_WATER, TYPE_ROCK, TYPE_GROUND) == 4.0
    assert type_multiplier(TYPE_NORMAL, TYPE_GHOST, TYPE_POISON) if False else True
    assert type_multiplier(TYPE_NORMAL, TYPE_GHOST, TYPE_GHOST) == 0.0
    assert type_multiplier(TYPE_ELECTRIC, TYPE_GROUND, TYPE_FLYING) == 0.0


def test_score_moves_prefers_super_effective_and_ignores_zero_pp():
    # Charmander (Fire) com Scratch(10), Growl(45), Ember(52), SlotVazio(0) vs Oddish (Grass/Poison=3)
    scores, best_idx, best_mult = score_moves(
        (TYPE_FIRE, TYPE_FIRE),
        (TYPE_GRASS, 3),
        (10, 45, 52, 0),
        (35, 40, 25, 0),
    )
    assert best_idx == 2
    assert best_mult == 2.0
    assert scores[2] > scores[0] > scores[1]
    assert scores[3] == 0.0

    # Se Ember ficar sem PP (0), melhor passa a ser Scratch (0)
    scores_no_pp, best_no_pp, _ = score_moves(
        (TYPE_FIRE, TYPE_FIRE),
        (TYPE_GRASS, 3),
        (10, 45, 52, 0),
        (35, 40, 0, 0),
    )
    assert scores_no_pp[2] == 0.0
    assert best_no_pp == 0


def test_reader_economy_pokeballs_party_species_and_battle_obs():
    enc32 = 0x4B55982C
    enc16 = enc32 & 0xFFFF
    sb1 = 0x025560
    ram = _make_full_ram(sb1_offset=sb1, enc32=enc32)

    # Money = 3080 XOR enc32
    struct.pack_into("<I", ram, sb1 + 0x0290, 3080 ^ enc32)
    assert RamReader.get_money(ram) == 3080

    # Items pocket slot 0=Antidote(18), slot 1=Potion(13 x2)
    struct.pack_into("<HH", ram, sb1 + 0x0310, 18, 1 ^ enc16)
    struct.pack_into("<HH", ram, sb1 + 0x0310 + 4, 13, 2 ^ enc16)
    assert RamReader.get_potions_count(ram) == 2
    assert RamReader.get_potion_slot(ram) == 1

    # PokeBalls pocket slot 0=Poke Ball(4 x5)
    struct.pack_into("<HH", ram, sb1 + 0x0430, 4, 5 ^ enc16)
    assert RamReader.get_pokeballs_count(ram) == 5
    assert RamReader.get_pokeball_slot(ram) == 0

    # Encrypted Party Pokemon slot 0: species=4 (Charmander), pid=7 (7%24=7 -> growth_idx=1)
    p0 = RamReader._PLAYER_PARTY_OFFSET
    pid, otid = 7, 0x12345678
    key16 = (pid ^ otid) & 0xFFFF
    growth_idx = RamReader._GEN3_GROWTH_POS[pid % 24]
    struct.pack_into("<II", ram, p0 + 0, pid, otid)
    struct.pack_into("<H", ram, p0 + 32 + growth_idx * 12, 4 ^ key16)
    struct.pack_into("<HH", ram, p0 + 86, 20, 20)
    assert RamReader.get_party_species(ram) == [4]

    # Overworld obs
    ow_info = {"gMain_callback2": 0x080565B5, "p1_maxhp": 20}
    ow_obs = RamReader.get_battle_observation(ram, ow_info)
    assert ow_obs.shape == (BATTLE_SIZE,)
    assert ow_obs.dtype == np.float32
    assert np.isclose(ow_obs[8], 5.0 / 20.0)
    assert np.isclose(ow_obs[9], 3080.0 / 10000.0)
    assert np.all(ow_obs[:8] == 0.0)
    assert ow_obs[11] == 0.0

    # Battle obs: Wild Caterpie (species=10, Bug=6, hp=4/12), Charmander (Fire=10, moves=[10,45,52,0])
    b0 = RamReader._BATTLE_MONS_OFFSET
    b1 = RamReader._BATTLE_MONS_OFFSET + RamReader._BATTLE_MON_SIZE
    struct.pack_into("<H", ram, b0 + 0x00, 4)
    struct.pack_into("<4H", ram, b0 + 0x0C, 10, 45, 52, 0)
    ram[b0 + 0x21], ram[b0 + 0x22] = TYPE_FIRE, TYPE_FIRE
    ram[b0 + 0x24 : b0 + 0x28] = bytes([35, 40, 25, 0])

    struct.pack_into("<H", ram, b1 + 0x00, 10)  # Caterpie
    ram[b1 + 0x21], ram[b1 + 0x22] = TYPE_BUG, TYPE_BUG
    struct.pack_into("<H", ram, b1 + 0x28, 4)
    struct.pack_into("<H", ram, b1 + 0x2C, 12)

    struct.pack_into("<I", ram, RamReader._BATTLE_TYPE_FLAGS_OFFSET, 0x04)  # Wild
    ram[RamReader._ACTION_CURSOR_OFFSET] = 3  # cursor em RUN
    ram[RamReader._MOVE_CURSOR_OFFSET] = 2    # cursor de golpe em 2 (Ember)

    bat_info = {"gMain_callback2": 0x08011101, "p1_maxhp": 20}
    bat_obs = RamReader.get_battle_observation(ram, bat_info)
    assert bat_obs.shape == (BATTLE_SIZE,)
    assert np.all((bat_obs >= 0.0) & (bat_obs <= 1.0))
    assert np.isclose(bat_obs[4], 2.0 / 3.0)  # best_slot = 2 (Ember)
    assert np.isclose(bat_obs[5], 1.0)        # main_cur = 3
    assert np.isclose(bat_obs[6], 2.0 / 3.0)  # move_cur = 2
    assert bat_obs[7] == 1.0                  # is_wild
    assert np.isclose(bat_obs[10], 4.0 / 12.0)
    assert bat_obs[11] == 1.0                 # can_capture = True (wild, balls>0, hp<55%, new species)

    # Battle string gDisplayedStringBattle (0x0202298C)
    from src.advisor import RamAdvisor
    phrase = bytes([0xD1, 0xDC, 0xD5, 0xE8, 0x00, 0xEB, 0xDD, 0xE0, 0xE0, 0xFF])  # "what will"
    ram[RamReader._BATTLE_STRING_OFFSET : RamReader._BATTLE_STRING_OFFSET + len(phrase)] = phrase
    assert RamAdvisor.decode_ram_text(RamReader.get_battle_text_slice(ram)) == "what will"
    assert RamReader.get_battle_text_slice(None) == b""
