import numpy as np
from src.config import CFG, GRID_SIZE, log


class RamReader:
    TILE_BLOCKED  = 0
    TILE_WALKABLE = 1
    TILE_GRASS    = 2
    TILE_WARP     = 3
    TILE_NPC      = 4

    GRASS_METATILES = frozenset({0x008, 0x009, 0x00A, 0x00B, 0x00D, 0x015, 0x016})
    WARP_METATILES = frozenset({
        0x012, 0x013, 0x014, 0x01A, 0x01B, 0x01C, 0x01D, 0x01E, 0x01F,
        0x025, 0x026, 0x027, 0x038, 0x039, 0x03A,
        0x060, 0x061, 0x062, 0x063, 0x064, 0x065, 0x066, 0x067, 0x068, 0x069, 0x06A,
    })

    _IRAM_START = 262144
    _BACKUP_MAP_LAYOUT_OFFSET = _IRAM_START + 0x5040
    _OBJECT_EVENTS_OFFSET = 0x36E38
    _OBJECT_EVENTS_COUNT = 16
    _OBJECT_EVENT_SIZE = 36
    _MAP_BORDER_OFFSET = 7
    _PLAYER_PARTY_OFFSET = 0x24284
    _ENEMY_PARTY_OFFSET = 0x2402C
    _MON_STRUCT_SIZE = 100
    _BADGES_FLAG_OFFSET = 0x0EE0 + (0x820 >> 3)  # 0x0FE4 em SaveBlock1 (bits 0..7 = 8 insígnias)
    _POKEDEX_FLAG_OFFSET = 0x0EE0 + (0x829 >> 3)  # 0x0FE5 (FLAG_SYS_POKEDEX_GET = 0x829)
    _POKEDEX_FLAG_MASK = 1 << (0x829 & 7)         # 0x02

    @staticmethod
    def _clamp(val, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(val)))

    @staticmethod
    def _read_u16(ram_array, idx: int) -> int:
        return int(ram_array[idx]) | (int(ram_array[idx + 1]) << 8)

    @classmethod
    def _read_s16(cls, ram_array, idx: int) -> int:
        val = cls._read_u16(ram_array, idx)
        return val - 0x10000 if val >= 0x8000 else val

    @staticmethod
    def _read_u32(ram_array, idx: int) -> int:
        return (
            int(ram_array[idx])
            | (int(ram_array[idx + 1]) << 8)
            | (int(ram_array[idx + 2]) << 16)
            | (int(ram_array[idx + 3]) << 24)
        )

    @classmethod
    def _read_s32(cls, ram_array, idx: int) -> int:
        val = cls._read_u32(ram_array, idx)
        return val - 0x100000000 if val >= 0x80000000 else val

    @classmethod
    def coord(cls, info: dict, key: str) -> int:
        return cls._clamp(info.get(key, 0), 0, CFG.max_coord)

    @classmethod
    def get_saveblock1_ptr(cls, ram_array) -> int:
        """Retorna o offset do SaveBlock1 na WRAM"""
        if ram_array is None:
            return 0
        
        iram_start = 262144
        ptr_idx = iram_start + 0x5008
        if len(ram_array) <= ptr_idx + 3:
            return 0
            
        ptr_val = cls._read_u32(ram_array, ptr_idx)
        
        # WRAM address check (e.g. 0x02025eb4)
        if (ptr_val >> 24) != 0x02:
            return 0
            
        return ptr_val & 0x00FFFFFF

    @classmethod
    def get_saveblock2_ptr(cls, ram_array) -> int:
        """Retorna o offset do SaveBlock2 na WRAM (IRAM 0x0300500C)."""
        if ram_array is None:
            return 0
        ptr_idx = cls._IRAM_START + 0x500C
        if len(ram_array) <= ptr_idx + 3:
            return 0
        ptr_val = cls._read_u32(ram_array, ptr_idx)
        if (ptr_val >> 24) != 0x02:
            return 0
        return ptr_val & 0x00FFFFFF

    @classmethod
    def get_encryption_key(cls, ram_array) -> int:
        """Retorna a chave XOR de 16 bits em SaveBlock2 + 0x0F20."""
        sb2 = cls.get_saveblock2_ptr(ram_array)
        if sb2 == 0 or len(ram_array) <= sb2 + 0x0F21:
            return 0
        return cls._read_u16(ram_array, sb2 + 0x0F20)

    @classmethod
    def is_in_battle(cls, info: dict) -> bool:
        """Detecta batalha ativa via ponteiro gMain_callback2 (CB2_InitBattle/CB2_BattleMain)."""
        cb2 = int(info.get("gMain_callback2", 0))
        return 0x08010000 <= cb2 < 0x08018000

    @classmethod
    def event_flags_sum(cls, ram_array) -> int:
        """
        Lê o bloco de Event Flags (offset 0x0EE0) e soma todos os bits ligados.
        """
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return 0
            
        flags_start = wram_offset + 0x0EE0
        flags_end = wram_offset + 0x1000
        
        if len(ram_array) <= flags_end:
            return 0
            
        flags_block = ram_array[flags_start:flags_end]
        
        # Conta a quantidade de bits '1' no bloco inteiro
        total_flags = 0
        for byte in flags_block:
            total_flags += byte.bit_count()
            
        return total_flags

    @classmethod
    def get_key_items(cls, ram_array) -> list[int]:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0: return []
        key_items_start = wram_offset + 0x03B8
        if len(ram_array) <= key_items_start + (30 * 4): return []
        found_items = []
        for i in range(30):
            idx = key_items_start + i * 4
            item_id = int(ram_array[idx]) | (int(ram_array[idx+1]) << 8)
            item_qty = int(ram_array[idx+2]) | (int(ram_array[idx+3]) << 8)
            if item_id != 0 and item_qty > 0:
                found_items.append(item_id)
        return found_items

    @classmethod
    def has_oaks_parcel(cls, ram_array) -> bool:
        return 349 in cls.get_key_items(ram_array)

    @classmethod
    def has_pokedex(cls, ram_array) -> bool:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return False
        idx = wram_offset + cls._POKEDEX_FLAG_OFFSET
        if len(ram_array) <= idx:
            return False
        return bool(int(ram_array[idx]) & cls._POKEDEX_FLAG_MASK)

    @classmethod
    def get_potions_count(cls, ram_array) -> int:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return 0

        enc_key = cls.get_encryption_key(ram_array)
        potion_ids = (13, 14, 15, 16, 17)

        def _scan_potions(base_offset: int, decrypt: bool) -> int:
            if len(ram_array) <= base_offset + (42 * 4):
                return 0
            total = 0
            for i in range(42):
                idx = base_offset + i * 4
                item_id = cls._read_u16(ram_array, idx)
                raw_qty = cls._read_u16(ram_array, idx + 2)
                item_qty = (raw_qty ^ enc_key) if (decrypt and enc_key != 0) else raw_qty
                if item_id in potion_ids and 0 < item_qty <= 999:
                    total += item_qty
            return total

        total_potions = _scan_potions(wram_offset + 0x0310, decrypt=True)
        if total_potions == 0 and enc_key == 0:
            total_potions = _scan_potions(wram_offset + 0x02B8, decrypt=False)
        return total_potions

    @classmethod
    def get_tms_count(cls, ram_array) -> int:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return 0

        enc_key = cls.get_encryption_key(ram_array)

        def _scan_tms(base_offset: int, decrypt: bool) -> int:
            if len(ram_array) <= base_offset + (58 * 4):
                return 0
            total = 0
            for i in range(58):
                idx = base_offset + i * 4
                item_id = cls._read_u16(ram_array, idx)
                raw_qty = cls._read_u16(ram_array, idx + 2)
                item_qty = (raw_qty ^ enc_key) if (decrypt and enc_key != 0) else raw_qty
                if item_id != 0 and 0 < item_qty <= 99:
                    total += item_qty
            return total

        total_tms = _scan_tms(wram_offset + 0x0464, decrypt=True)
        if total_tms == 0 and enc_key == 0:
            total_tms = _scan_tms(wram_offset + 0x040C, decrypt=False)
        return total_tms

    @classmethod
    def get_p1_status(cls, ram_array) -> int:
        if ram_array is None:
            return 0
        if len(ram_array) >= cls._PLAYER_PARTY_OFFSET + cls._MON_STRUCT_SIZE:
            maxhp = cls._read_u16(ram_array, cls._PLAYER_PARTY_OFFSET + 88)
            if maxhp > 0:
                status_val = cls._read_u32(ram_array, cls._PLAYER_PARTY_OFFSET + 80)
                return 1 if status_val > 0 else 0

        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return 0
        # Fallback: Pokemon 1 começa no offset 0x0294. Status está no byte 80 da struct (0x50).
        status_start = wram_offset + 0x0294 + 80
        if len(ram_array) <= status_start + 3:
            return 0

        status_val = cls._read_u32(ram_array, status_start)
        # Se for > 0, tem alguma condição negativa (Poison, Paralyze, Sleep, etc)
        return 1 if status_val > 0 else 0

    @classmethod
    def enrich_info_from_ram(cls, info: dict, ram_array) -> dict:
        """
        Corrige e completa `info` do stable-retro lendo diretamente a RAM do FireRed:
        - Coordenadas reais (player_x, player_y) do SaveBlock1 (evita ponteiro bruto 21884)
        - Party do jogador em EWRAM gPlayerParty (0x02024284)
        - Party inimiga em EWRAM gEnemyParty (0x0202402C)
        - Insígnias nas flags de SaveBlock1 (offset 0x0FE4)
        """
        if ram_array is None:
            return info

        px, py = cls.get_player_coords(ram_array, info)
        info["player_x"] = px
        info["player_y"] = py

        ram_len = len(ram_array)
        if ram_len >= cls._PLAYER_PARTY_OFFSET + 6 * cls._MON_STRUCT_SIZE:
            if cls._read_u16(ram_array, cls._PLAYER_PARTY_OFFSET + 88) > 0:
                for i in range(1, 7):
                    base = cls._PLAYER_PARTY_OFFSET + (i - 1) * cls._MON_STRUCT_SIZE
                    maxhp = cls._read_u16(ram_array, base + 88)
                    info[f"p{i}_lvl"] = int(ram_array[base + 84]) if maxhp > 0 else 0
                    info[f"p{i}_hp"] = cls._read_u16(ram_array, base + 86) if maxhp > 0 else 0
                    info[f"p{i}_maxhp"] = maxhp

        if ram_len >= cls._ENEMY_PARTY_OFFSET + 6 * cls._MON_STRUCT_SIZE:
            e1_maxhp = cls._read_u16(ram_array, cls._ENEMY_PARTY_OFFSET + 88)
            if 0 < e1_maxhp <= CFG.max_enemy_hp:
                for i in range(1, 7):
                    base = cls._ENEMY_PARTY_OFFSET + (i - 1) * cls._MON_STRUCT_SIZE
                    maxhp = cls._read_u16(ram_array, base + 88)
                    valid = 0 < maxhp <= CFG.max_enemy_hp
                    info[f"e{i}_lvl"] = int(ram_array[base + 84]) if valid else 0
                    info[f"e{i}_hp"] = cls._read_u16(ram_array, base + 86) if valid else 0
                    info[f"e{i}_maxhp"] = maxhp if valid else 0

        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset > 0 and ram_len > wram_offset + cls._BADGES_FLAG_OFFSET:
            badge_byte = int(ram_array[wram_offset + cls._BADGES_FLAG_OFFSET])
            if "badges" not in info or info["badges"] == 0:
                info["badges"] = badge_byte

        return info

    @classmethod
    def map_id(cls, info: dict) -> int:
        return cls._clamp(info.get("map_id", 0), 0, CFG.max_map_id)

    @classmethod
    def map_bank(cls, info: dict) -> int:
        return cls._clamp(info.get("map_bank", 0), 0, CFG.max_map_id)

    @classmethod
    def player_moving(cls, info: dict) -> int:
        return cls._clamp(info.get("player_moving", 0), 0, 1)

    @classmethod
    def party_level(cls, info: dict) -> int:
        total = sum(info.get(f"p{i}_lvl", 0) for i in range(1, 7))
        return cls._clamp(total, 0, CFG.max_party_level_sum)

    @classmethod
    def party_hp(cls, info: dict) -> int:
        total = sum(info.get(f"p{i}_hp", 0) for i in range(1, 7))
        return cls._clamp(total, 0, CFG.max_party_hp)

    @classmethod
    def party_size(cls, info: dict) -> int:
        return sum(1 for i in range(1, 7) if info.get(f"p{i}_maxhp", 0) > 0)

    @classmethod
    def party_max_hp(cls, info: dict) -> int:
        return sum(info.get(f"p{i}_maxhp", 0) for i in range(1, 7))

    @classmethod
    def enemy_hp(cls, info: dict) -> int:
        total = sum(info.get(f"e{i}_hp", 0) for i in range(1, 7))
        return cls._clamp(total, 0, CFG.max_enemy_hp)

    @classmethod
    def max_enemy_level(cls, info: dict) -> int:
        return max((info.get(f"e{i}_lvl", 0) for i in range(1, 7)), default=0)

    @classmethod
    def badges(cls, info: dict) -> int:
        raw = info.get("badges", 0)
        return cls._clamp(bin(int(raw)).count("1"), 0, 8)

    @classmethod
    def get_active_npcs(cls, ram_array) -> set[tuple[int, int]]:
        """Lê gObjectEvents (EWRAM 0x02036E38) e retorna coords (x, y) de NPCs ativos."""
        npcs: set[tuple[int, int]] = set()
        if ram_array is None:
            return npcs

        ram_len = len(ram_array)
        for i in range(cls._OBJECT_EVENTS_COUNT):
            base = cls._OBJECT_EVENTS_OFFSET + i * cls._OBJECT_EVENT_SIZE
            if ram_len <= base + 19:
                break
            flags = cls._read_u32(ram_array, base)
            is_active = (flags & 0x01) != 0
            is_player = (i == 0) or ((flags & 0x10000) != 0)
            is_invisible = (flags & 0x2000) != 0
            if not is_active or is_player or is_invisible:
                continue

            curr_x = cls._read_s16(ram_array, base + 16) - cls._MAP_BORDER_OFFSET
            curr_y = cls._read_s16(ram_array, base + 18) - cls._MAP_BORDER_OFFSET
            npcs.add((curr_x, curr_y))
        return npcs

    @classmethod
    def get_player_coords(cls, ram_array, info: dict | None = None) -> tuple[int, int]:
        """Obtém (x, y) reais do jogador via SaveBlock1, com fallback seguro para info."""
        sb_x, sb_y = 0, 0
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        has_sb = wram_offset > 0 and ram_array is not None and len(ram_array) >= wram_offset + 4
        if has_sb:
            sb_x = cls._read_s16(ram_array, wram_offset)
            sb_y = cls._read_s16(ram_array, wram_offset + 2)

        if info is not None and "player_x" in info and "player_y" in info:
            ix = int(info["player_x"])
            iy = int(info["player_y"])
            info_valid = 0 <= ix <= CFG.max_coord and 0 <= iy <= CFG.max_coord
            if info_valid and (not has_sb or (sb_x == 0 and sb_y == 0)):
                return ix, iy

        if has_sb:
            return sb_x, sb_y
        return 0, 0

    @classmethod
    def get_collision_grid(
        cls,
        ram_array,
        info: dict | None = None,
        grid_size: int = GRID_SIZE,
    ) -> np.ndarray:
        """
        Extrai a matriz grid_size x grid_size de colisão/warps/NPCs ao redor do jogador.
        Valores: 0=bloqueado, 1=caminhável, 2=mato, 3=warp/porta, 4=NPC.
        """
        grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        if ram_array is None or len(ram_array) < cls._BACKUP_MAP_LAYOUT_OFFSET + 12:
            return grid

        width = cls._read_s32(ram_array, cls._BACKUP_MAP_LAYOUT_OFFSET)
        height = cls._read_s32(ram_array, cls._BACKUP_MAP_LAYOUT_OFFSET + 4)
        map_ptr = cls._read_u32(ram_array, cls._BACKUP_MAP_LAYOUT_OFFSET + 8)
        if width <= 0 or width > 1024 or height <= 0 or height > 1024 or (map_ptr >> 24) != 0x02:
            return grid

        map_offset = map_ptr & 0x00FFFFFF
        px, py = cls.get_player_coords(ram_array, info)
        active_npcs = cls.get_active_npcs(ram_array)
        radius = grid_size // 2
        ram_len = len(ram_array)

        for row, dy in enumerate(range(-radius, radius + 1)):
            ty = py + dy
            gy = ty + cls._MAP_BORDER_OFFSET
            if gy < 0 or gy >= height:
                continue
            for col, dx in enumerate(range(-radius, radius + 1)):
                tx = px + dx
                gx = tx + cls._MAP_BORDER_OFFSET
                if gx < 0 or gx >= width:
                    continue
                tile_idx = map_offset + 2 * (gy * width + gx)
                if tile_idx + 1 >= ram_len:
                    continue

                entry = cls._read_u16(ram_array, tile_idx)
                metatile_id = entry & 0x03FF
                collision = (entry >> 10) & 0x03

                if metatile_id == 0x03FF:
                    tile = cls.TILE_BLOCKED
                elif (tx, ty) in active_npcs:
                    tile = cls.TILE_NPC
                elif metatile_id in cls.WARP_METATILES:
                    tile = cls.TILE_WARP
                elif collision != 0:
                    tile = cls.TILE_BLOCKED
                elif metatile_id in cls.GRASS_METATILES:
                    tile = cls.TILE_GRASS
                else:
                    tile = cls.TILE_WALKABLE

                grid[row, col] = tile

        return grid

    @classmethod
    def get_collision_grid_normalized(
        cls,
        ram_array,
        info: dict | None = None,
        grid_size: int = GRID_SIZE,
    ) -> np.ndarray:
        """Retorna matriz 9x9 normalizada em [0.0, 1.0] float32 para observação neural."""
        return cls.get_collision_grid(ram_array, info=info, grid_size=grid_size).astype(np.float32) / 4.0

    @staticmethod
    def debug_dump(info: dict, env_id: int, label: str) -> None:
        log.debug(
            f"[Env {env_id}] {label} | "
            f"x={info.get('player_x','?')} y={info.get('player_y','?')} "
            f"map_bank={info.get('map_bank','?')} map_id={info.get('map_id','?')} | "
        )
