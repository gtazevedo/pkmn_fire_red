import struct
import numpy as np

from src.reader import RamReader


def _make_ram(
    sb1_offset: int = 0x025000,
    map_width: int = 24,
    map_height: int = 24,
    map_offset: int = 0x031000,
) -> bytearray:
    ram = bytearray(262144 + 0x6000)
    if sb1_offset:
        struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1_offset)
    if map_offset:
        struct.pack_into("<iiI", ram, 262144 + 0x5040, map_width, map_height, 0x02000000 | map_offset)
    return ram


def test_saveblock1_and_event_flags():
    ram = _make_ram(sb1_offset=0x025000)
    assert RamReader.get_saveblock1_ptr(ram) == 0x025000

    # Escreve bits em event flags (offset 0x0EE0 dentro do SaveBlock1)
    flags_start = 0x025000 + 0x0EE0
    ram[flags_start] = 0b10101100  # 4 bits
    ram[flags_start + 5] = 0b11111111  # 8 bits
    assert RamReader.event_flags_sum(ram) == 12

    # Key items (Oak's Parcel = 349, qty = 1)
    key_items_start = 0x025000 + 0x03B8
    struct.pack_into("<HH", ram, key_items_start, 349, 1)
    assert RamReader.get_key_items(ram) == [349]
    assert RamReader.has_oaks_parcel(ram) is True

    # Potions (item_id = 13) e TMs (não vazias)
    items_start = 0x025000 + 0x02B8
    struct.pack_into("<HH", ram, items_start, 13, 3)
    assert RamReader.get_potions_count(ram) == 3

    tms_start = 0x025000 + 0x040C
    struct.pack_into("<HH", ram, tms_start, 289, 2)
    assert RamReader.get_tms_count(ram) == 2

    # P1 status (offset 0x0294 + 80)
    assert RamReader.get_p1_status(ram) == 0
    struct.pack_into("<I", ram, 0x025000 + 0x0294 + 80, 0x08)  # Poison
    assert RamReader.get_p1_status(ram) == 1


def test_collision_grid_all_tile_types_and_object_events():
    w, h = 24, 24
    map_off = 0x031000
    sb1_off = 0x025000
    ram = _make_ram(sb1_offset=sb1_off, map_width=w, map_height=h, map_offset=map_off)

    # Player no SaveBlock1 em (px=5, py=5)
    px, py = 5, 5
    struct.pack_into("<hh", ram, sb1_off, px, py)

    # Preenche todo o mapa com walkable (collision=0, id=0x001)
    for gy in range(h):
        for gx in range(w):
            struct.pack_into("<H", ram, map_off + 2 * (gy * w + gx), 0x0001)

    def set_tile(tx: int, ty: int, raw_entry: int) -> None:
        gx, gy = tx + 7, ty + 7
        struct.pack_into("<H", ram, map_off + 2 * (gy * w + gx), raw_entry)

    # Parede com collision=1 ao norte (5, 4) -> TILE_BLOCKED (0)
    set_tile(5, 4, (1 << 10) | 0x001)
    # Borda do mapa 0x03FF a oeste (4, 5) -> TILE_BLOCKED (0)
    set_tile(4, 5, 0x03FF)
    # Mato/encontro a leste (6, 5) -> TILE_GRASS (2)
    set_tile(6, 5, 0x000D)
    # Warp / porta ao sul (5, 6) e warp com bit de colisão (4, 6) -> TILE_WARP (3)
    set_tile(5, 6, 0x0013)
    set_tile(4, 6, (1 << 10) | 0x01B)

    # Slot 0: jogador com bit 16 isPlayer (0x10000) — NÃO pode aparecer como NPC no centro!
    struct.pack_into("<I", ram, 0x36E38, 0x00018001)
    struct.pack_into("<hh", ram, 0x36E38 + 16, px + 7, py + 7)

    # Slot 1: NPC ativo visível em (6, 6) -> TILE_NPC (4)
    base1 = 0x36E38 + 36
    struct.pack_into("<I", ram, base1, 0x00000001)
    struct.pack_into("<hh", ram, base1 + 16, 6 + 7, 6 + 7)

    # Slot 2: NPC invisível (bit 13 = 0x2000) em (6, 4) -> deve permanecer TILE_WALKABLE (1)
    base2 = 0x36E38 + 2 * 36
    struct.pack_into("<I", ram, base2, 0x00002001)
    struct.pack_into("<hh", ram, base2 + 16, 6 + 7, 4 + 7)

    # Passa info com valor lixo de ponteiro (21884) como o stable-retro real faz
    grid = RamReader.get_collision_grid(ram, info={"player_x": 21884, "player_y": 21884})
    assert grid.shape == (9, 9)
    assert grid.dtype == np.uint8

    # Centro (4, 4) é o jogador em tile caminhável
    assert grid[4, 4] == RamReader.TILE_WALKABLE
    assert grid[3, 4] == RamReader.TILE_BLOCKED  # (5, 4)
    assert grid[4, 3] == RamReader.TILE_BLOCKED  # (4, 5)
    assert grid[4, 5] == RamReader.TILE_GRASS    # (6, 5)
    assert grid[5, 4] == RamReader.TILE_WARP     # (5, 6)
    assert grid[5, 3] == RamReader.TILE_WARP     # (4, 6)
    assert grid[5, 5] == RamReader.TILE_NPC      # (6, 6)
    assert grid[3, 5] == RamReader.TILE_WALKABLE # (6, 4) invisível ignorado

    norm = RamReader.get_collision_grid_normalized(ram, info={"player_x": 21884, "player_y": 21884})
    assert norm.dtype == np.float32
    assert np.isclose(norm[5, 5], 1.0)
    assert np.isclose(norm[5, 4], 0.75)
    assert np.isclose(norm[4, 5], 0.50)
    assert np.isclose(norm[4, 4], 0.25)
    assert np.isclose(norm[3, 4], 0.0)


def test_collision_grid_out_of_bounds_and_null_safety():
    assert np.all(RamReader.get_collision_grid(None) == 0)

    ram = _make_ram(sb1_offset=0, map_width=10, map_height=10, map_offset=0x031000)
    # Sem SaveBlock1, fallback usa info válido; posição (0, 0) em mapa 10x10 deixa parte sul/leste fora do mapa
    for i in range(10 * 10):
        struct.pack_into("<H", ram, 0x031000 + 2 * i, 0x0001)

    grid = RamReader.get_collision_grid(ram, info={"player_x": 0, "player_y": 0})
    # Para px=0, py=0: gx = 0 + 7 + dx >= 10 quando dx >= 3 (colunas 7,8 e linhas 7,8 ficam fora do mapa)
    assert np.all(grid[7:, :] == RamReader.TILE_BLOCKED)
    assert np.all(grid[:, 7:] == RamReader.TILE_BLOCKED)
    assert grid[4, 4] == RamReader.TILE_WALKABLE


def test_enrich_info_from_ram_party_enemy_coords_and_badges():
    sb1_off = 0x026000
    ram = _make_ram(sb1_offset=sb1_off)

    # Coordenadas reais no SaveBlock1: (12, 9)
    struct.pack_into("<hh", ram, sb1_off, 12, 9)

    # 3 insígnias ligadas no byte 0x0FE4 do SaveBlock1
    ram[sb1_off + RamReader._BADGES_FLAG_OFFSET] = 0b00000111

    # gPlayerParty slot 1 (0x24284): lvl=16, hp=35, maxhp=44, status=0x08 (Poison)
    p1 = RamReader._PLAYER_PARTY_OFFSET
    ram[p1 + 84] = 16
    struct.pack_into("<I", ram, p1 + 80, 0x08)
    struct.pack_into("<HH", ram, p1 + 86, 35, 44)

    # gEnemyParty slot 1 (0x2402C): lvl=14, hp=18, maxhp=36
    e1 = RamReader._ENEMY_PARTY_OFFSET
    ram[e1 + 84] = 14
    struct.pack_into("<HH", ram, e1 + 86, 18, 36)

    # Simula info corrompido pelo stable-retro data.json
    info = {
        "player_x": 21884,
        "player_y": 21884,
        "p1_lvl": 16,
        "p1_hp": 14,
        "p1_maxhp": 0,
        "e1_lvl": 4,
        "e1_hp": 27920,
    }

    enriched = RamReader.enrich_info_from_ram(info, ram)
    assert enriched["player_x"] == 12
    assert enriched["player_y"] == 9
    assert RamReader.party_level(enriched) == 16
    assert RamReader.party_hp(enriched) == 35
    assert RamReader.party_max_hp(enriched) == 44
    assert RamReader.party_size(enriched) == 1
    assert RamReader.enemy_hp(enriched) == 18
    assert RamReader.max_enemy_level(enriched) == 14
    assert RamReader.badges(enriched) == 3
    assert RamReader.get_p1_status(ram) == 1
