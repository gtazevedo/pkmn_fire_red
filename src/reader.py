import numpy as np
from src.battle_data import score_moves
from src.config import BATTLE_SIZE, CFG, GRID_SIZE, log


class RamReader:
    TILE_BLOCKED  = 0
    TILE_WALKABLE = 1
    TILE_GRASS    = 2
    TILE_WARP     = 3
    TILE_NPC      = 4

    GRASS_METATILES = frozenset({0x008, 0x009, 0x00A, 0x00B, 0x00D, 0x015, 0x016})
    WARP_METATILES = frozenset({
        0x007,
        0x012, 0x013, 0x014, 0x01D,
        0x025, 0x026, 0x027,
        0x038, 0x039, 0x03A,
        0x191,
        0x290, 0x291, 0x292,
    })
    SOLID_DOOR_METATILES = frozenset({0x062, 0x2A3, 0x2AC})
    HM_OBSTACLE_GFX_IDS = frozenset({95, 96, 97})  # Cuttable Tree (0x5F), Rock Smash (0x60), Strength Boulder (0x61)
    ITEM_TEACHY_TV = 366
    POTION_IDS = (13, 14, 15, 16, 17)
    POKEBALL_IDS = (1, 2, 3, 4)

    _IRAM_START = 262144
    _SCRIPT_CONTEXT1_MODE_OFFSET = _IRAM_START + 0x0EB0
    _LOCK_FIELD_CONTROLS_OFFSET = _IRAM_START + 0x0F9C
    _BACKUP_MAP_LAYOUT_OFFSET = _IRAM_START + 0x5040
    _OBJECT_EVENTS_OFFSET = 0x36E38
    _OBJECT_EVENTS_COUNT = 16
    _OBJECT_EVENT_SIZE = 36
    _MAP_BORDER_OFFSET = 7
    _PLAYER_PARTY_OFFSET = 0x24284
    _ENEMY_PARTY_OFFSET = 0x2402C
    _MON_STRUCT_SIZE = 100
    _BATTLE_MONS_OFFSET = 0x23BE4
    _BATTLE_MON_SIZE = 0x58
    _BATTLE_TYPE_FLAGS_OFFSET = 0x22B4C
    _BATTLE_TYPE_TRAINER_MASK = 0x08
    _ACTION_CURSOR_OFFSET = 0x23FF8
    _MOVE_CURSOR_OFFSET = 0x23FFC
    _BATTLE_STRING_OFFSET = 0x2298C
    _MONEY_OFFSET = 0x0290
    _POKEBALLS_POCKET_OFFSET = 0x0430
    _POKEBALLS_SLOTS = 13
    _GEN3_GROWTH_POS = (
        0, 0, 0, 0, 0, 0,
        1, 1, 2, 3, 2, 3,
        1, 1, 2, 3, 2, 3,
        1, 1, 2, 3, 2, 3,
    )
    _BADGES_FLAG_OFFSET = 0x0EE0 + (0x820 >> 3)  # 0x0FE4 em SaveBlock1 (bits 0..7 = 8 insígnias)
    _POKEDEX_FLAG_OFFSET = 0x0EE0 + (0x829 >> 3)  # 0x0FE5 (FLAG_SYS_POKEDEX_GET = 0x829)
    _POKEDEX_FLAG_MASK = 1 << (0x829 & 7)         # 0x02
    _HIDE_OAK_LAB_FLAG_OFFSET = 0x0EE0 + (0x02B >> 3)  # 0x0EE5 (FLAG_HIDE_OAK_IN_HIS_LAB = 0x02B)
    _HIDE_OAK_LAB_FLAG_MASK = 1 << (0x02B & 7)        # 0x08

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
    def get_encryption_key32(cls, ram_array) -> int:
        """Retorna a chave XOR de 32 bits em SaveBlock2 + 0x0F20."""
        sb2 = cls.get_saveblock2_ptr(ram_array)
        if sb2 == 0 or ram_array is None or len(ram_array) <= sb2 + 0x0F23:
            return 0
        return cls._read_u32(ram_array, sb2 + 0x0F20)

    @classmethod
    def get_encryption_key(cls, ram_array) -> int:
        """Retorna a chave XOR de 16 bits em SaveBlock2 + 0x0F20."""
        return cls.get_encryption_key32(ram_array) & 0xFFFF

    @classmethod
    def is_in_battle(cls, info: dict) -> bool:
        """Detecta batalha ativa via ponteiro gMain_callback2 (CB2_InitBattle/CB2_BattleMain)."""
        cb2 = int(info.get("gMain_callback2", 0))
        return 0x08010000 <= cb2 < 0x08018000

    @classmethod
    def is_scripted_tutorial(cls, info: dict) -> bool:
        """Detecta tutoriais/cutscenes scriptados como o tutorial de captura do Old Man."""
        cb2 = int(info.get("gMain_callback2", 0) or 0)
        return 0x08100000 <= cb2 < 0x08110000

    @classmethod
    def is_script_locked(cls, ram_array) -> bool:
        """Detecta bloqueio real de controles/scripts na IWRAM do FireRed (USA v1.0)."""
        if ram_array is None or len(ram_array) <= cls._LOCK_FIELD_CONTROLS_OFFSET:
            return False
        return bool(
            int(ram_array[cls._LOCK_FIELD_CONTROLS_OFFSET]) != 0
            or int(ram_array[cls._SCRIPT_CONTEXT1_MODE_OFFSET]) != 0
        )

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
    def has_teachy_tv(cls, ram_array) -> bool:
        """Verifica se o jogador já recebeu a Teachy TV (key item 366) do Old Man em Viridian."""
        return cls.ITEM_TEACHY_TV in cls.get_key_items(ram_array)

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
    def is_oak_in_lab(cls, ram_array) -> bool:
        """Retorna True se Prof. Oak está no Laboratório (FLAG_HIDE_OAK_IN_HIS_LAB = 0x02B desligada)."""
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0:
            return True
        idx = wram_offset + cls._HIDE_OAK_LAB_FLAG_OFFSET
        if len(ram_array) <= idx:
            return True
        return not bool(int(ram_array[idx]) & cls._HIDE_OAK_LAB_FLAG_MASK)

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
    def get_money(cls, ram_array) -> int:
        """Lê o dinheiro atual do jogador em SaveBlock1 + 0x0290 XOR encKey32."""
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0 or ram_array is None or len(ram_array) <= wram_offset + cls._MONEY_OFFSET + 3:
            return 0
        raw_money = cls._read_u32(ram_array, wram_offset + cls._MONEY_OFFSET)
        enc32 = cls.get_encryption_key32(ram_array)
        val = (raw_money ^ enc32) if enc32 != 0 else raw_money
        return cls._clamp(val, 0, 999_999) if val <= 999_999 else 0

    @classmethod
    def get_pokeballs_count(cls, ram_array) -> int:
        """Soma quantidade de Pokébolas no bolso bagPocket_PokeBalls (SaveBlock1 + 0x0430)."""
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0 or ram_array is None:
            return 0
        base = wram_offset + cls._POKEBALLS_POCKET_OFFSET
        if len(ram_array) <= base + cls._POKEBALLS_SLOTS * 4:
            return 0
        enc_key = cls.get_encryption_key(ram_array)
        total = 0
        for i in range(cls._POKEBALLS_SLOTS):
            idx = base + i * 4
            item_id = cls._read_u16(ram_array, idx)
            raw_qty = cls._read_u16(ram_array, idx + 2)
            qty = (raw_qty ^ enc_key) if enc_key != 0 else raw_qty
            if item_id in cls.POKEBALL_IDS and 0 < qty <= 999:
                total += qty
        return total

    @classmethod
    def get_potion_slot(cls, ram_array) -> int:
        """Retorna índice 0-based da primeira Poção no bolso Items, ou -1 se ausente."""
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0 or ram_array is None:
            return -1
        enc_key = cls.get_encryption_key(ram_array)

        def _find(base_offset: int, decrypt: bool) -> int:
            if len(ram_array) <= base_offset + 42 * 4:
                return -1
            for i in range(42):
                idx = base_offset + i * 4
                item_id = cls._read_u16(ram_array, idx)
                raw_qty = cls._read_u16(ram_array, idx + 2)
                qty = (raw_qty ^ enc_key) if (decrypt and enc_key != 0) else raw_qty
                if item_id in cls.POTION_IDS and 0 < qty <= 999:
                    return i
            return -1

        slot = _find(wram_offset + 0x0310, decrypt=True)
        if slot == -1 and enc_key == 0:
            slot = _find(wram_offset + 0x02B8, decrypt=False)
        return slot

    @classmethod
    def get_pokeball_slot(cls, ram_array) -> int:
        """Retorna índice 0-based da primeira Pokébola no bolso Poké Balls, ou -1."""
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0 or ram_array is None:
            return -1
        base = wram_offset + cls._POKEBALLS_POCKET_OFFSET
        if len(ram_array) <= base + cls._POKEBALLS_SLOTS * 4:
            return -1
        enc_key = cls.get_encryption_key(ram_array)
        for i in range(cls._POKEBALLS_SLOTS):
            idx = base + i * 4
            item_id = cls._read_u16(ram_array, idx)
            raw_qty = cls._read_u16(ram_array, idx + 2)
            qty = (raw_qty ^ enc_key) if enc_key != 0 else raw_qty
            if item_id in cls.POKEBALL_IDS and 0 < qty <= 999:
                return i
        return -1

    @classmethod
    def get_party_species(cls, ram_array) -> list[int]:
        """Descriptografa as espécies (Gen 3 Growth block) de gPlayerParty."""
        if ram_array is None or len(ram_array) < cls._PLAYER_PARTY_OFFSET + 6 * cls._MON_STRUCT_SIZE:
            return []
        species_list: list[int] = []
        for i in range(6):
            base = cls._PLAYER_PARTY_OFFSET + i * cls._MON_STRUCT_SIZE
            maxhp = cls._read_u16(ram_array, base + 88)
            if maxhp == 0 or maxhp > CFG.max_party_hp:
                continue
            pid = cls._read_u32(ram_array, base + 0)
            otid = cls._read_u32(ram_array, base + 4)
            key16 = (pid ^ otid) & 0xFFFF
            growth_idx = cls._GEN3_GROWTH_POS[pid % 24]
            raw_species = cls._read_u16(ram_array, base + 32 + growth_idx * 12)
            species = raw_species ^ key16
            if 1 <= species <= 411:
                species_list.append(species)
        return species_list

    @classmethod
    def is_trainer_battle(cls, ram_array) -> bool:
        """Retorna True quando gBattleTypeFlags (0x02022B4C) indica batalha contra treinador."""
        if ram_array is None or len(ram_array) <= cls._BATTLE_TYPE_FLAGS_OFFSET + 3:
            return False
        flags = cls._read_u32(ram_array, cls._BATTLE_TYPE_FLAGS_OFFSET)
        return bool(flags & cls._BATTLE_TYPE_TRAINER_MASK)

    @classmethod
    def get_battle_cursors(cls, ram_array) -> tuple[int, int]:
        """Retorna (main_cursor, move_cursor) em 0..3 a partir de gAction/MoveSelectionCursor."""
        if ram_array is None or len(ram_array) <= cls._MOVE_CURSOR_OFFSET:
            return 0, 0
        main_cur = int(ram_array[cls._ACTION_CURSOR_OFFSET]) & 0x03
        move_cur = int(ram_array[cls._MOVE_CURSOR_OFFSET]) & 0x03
        return main_cur, move_cur

    @classmethod
    def get_battle_text_slice(cls, ram_array):
        """Retorna a fatia de EWRAM gDisplayedStringBattle (0x0202298C, 160 bytes)."""
        if ram_array is None or len(ram_array) <= cls._BATTLE_STRING_OFFSET:
            return b""
        return ram_array[cls._BATTLE_STRING_OFFSET : cls._BATTLE_STRING_OFFSET + 160]

    @classmethod
    def get_battle_mon(cls, ram_array, slot: int = 0) -> dict:
        """Lê struct BattlePokemon em gBattleMons (0=jogador ativo, 1=inimigo ativo)."""
        empty = {
            "species": 0, "moves": (0, 0, 0, 0), "pp": (0, 0, 0, 0),
            "type1": 0, "type2": 0, "hp": 0, "max_hp": 0, "level": 0,
        }
        if ram_array is None or slot not in (0, 1):
            return empty
        base = cls._BATTLE_MONS_OFFSET + slot * cls._BATTLE_MON_SIZE
        if len(ram_array) <= base + 0x2D:
            return empty
        return {
            "species": cls._read_u16(ram_array, base + 0x00),
            "moves": tuple(cls._read_u16(ram_array, base + 0x0C + i * 2) for i in range(4)),
            "type1": int(ram_array[base + 0x21]),
            "type2": int(ram_array[base + 0x22]),
            "pp": tuple(int(ram_array[base + 0x24 + i]) for i in range(4)),
            "hp": cls._read_u16(ram_array, base + 0x28),
            "level": int(ram_array[base + 0x2A]),
            "max_hp": cls._read_u16(ram_array, base + 0x2C),
        }

    @classmethod
    def get_battle_observation(cls, ram_array, info: dict) -> np.ndarray:
        """Vetor de batalha/economia de dimensão BATTLE_SIZE (12) normalizado em [0.0, 1.0]."""
        obs = np.zeros(BATTLE_SIZE, dtype=np.float32)
        if ram_array is None:
            return obs
        balls = cls.get_pokeballs_count(ram_array)
        money = cls.get_money(ram_array)
        obs[8] = min(balls, 20) / 20.0
        obs[9] = min(money, 10000) / 10000.0
        if not cls.is_in_battle(info):
            return obs

        p_mon = cls.get_battle_mon(ram_array, 0)
        e_mon = cls.get_battle_mon(ram_array, 1)
        scores, best_slot, _ = score_moves(
            (p_mon["type1"], p_mon["type2"]),
            (e_mon["type1"], e_mon["type2"]),
            p_mon["moves"],
            p_mon["pp"],
        )
        main_cur, move_cur = cls.get_battle_cursors(ram_array)
        is_wild = not cls.is_trainer_battle(ram_array)
        e_max = max(e_mon["max_hp"], 1) if e_mon["max_hp"] > 0 else max(cls.enemy_hp(info), 1)
        e_hp = e_mon["hp"] if e_mon["max_hp"] > 0 else cls.enemy_hp(info)
        enemy_hp_pct = max(0.0, min(1.0, e_hp / float(e_max)))
        party_species = set(cls.get_party_species(ram_array))
        party_sz = cls.party_size(info)
        can_capture = (
            is_wild
            and balls > 0
            and 0 < e_hp
            and enemy_hp_pct < 0.55
            and party_sz < 6
            and e_mon["species"] > 0
            and e_mon["species"] not in party_species
        )

        obs[0:4] = scores
        obs[4] = best_slot / 3.0
        obs[5] = main_cur / 3.0
        obs[6] = move_cur / 3.0
        obs[7] = float(is_wild)
        obs[10] = enemy_hp_pct
        obs[11] = float(can_capture)
        return obs

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

        info["script_lock"] = 1 if (info.get("script_lock", 0) or cls.is_script_locked(ram_array)) else 0
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
    def get_active_objects(cls, ram_array) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Lê gObjectEvents (EWRAM 0x02036E38) separando (npcs, hm_obstacles)."""
        npcs: set[tuple[int, int]] = set()
        hm_obstacles: set[tuple[int, int]] = set()
        if ram_array is None:
            return npcs, hm_obstacles

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
            gfx_id = int(ram_array[base + 5])
            if gfx_id in cls.HM_OBSTACLE_GFX_IDS:
                hm_obstacles.add((curr_x, curr_y))
            else:
                npcs.add((curr_x, curr_y))
        return npcs, hm_obstacles

    @classmethod
    def get_active_npcs(cls, ram_array) -> set[tuple[int, int]]:
        """Lê gObjectEvents (EWRAM 0x02036E38) e retorna coords (x, y) de NPCs ativos."""
        return cls.get_active_objects(ram_array)[0]

    @classmethod
    def get_player_coords(cls, ram_array, info: dict | None = None) -> tuple[int, int]:
        """Obtém (x, y) reais do jogador via SaveBlock1, com fallback seguro para info."""
        sb_x, sb_y = 0, 0
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        has_sb = wram_offset > 0 and ram_array is not None and len(ram_array) >= wram_offset + 4
        if has_sb:
            sb_x = cls._read_s16(ram_array, wram_offset)
            sb_y = cls._read_s16(ram_array, wram_offset + 2)

        if info is not None and ("player_x" in info or "player_y" in info):
            ix = int(info.get("player_x", 0))
            iy = int(info.get("player_y", 0))
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
        active_npcs, hm_obstacles = cls.get_active_objects(ram_array)
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

                if metatile_id == 0x03FF or (tx, ty) in hm_obstacles:
                    tile = cls.TILE_BLOCKED
                elif (tx, ty) in active_npcs:
                    tile = cls.TILE_NPC
                elif metatile_id in cls.SOLID_DOOR_METATILES or (
                    collision == 0 and metatile_id in cls.WARP_METATILES
                ):
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
