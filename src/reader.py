from src.config import CFG, log

class RamReader:
    @staticmethod
    def _clamp(val, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(val)))

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
            
        b1, b2, b3, b4 = int(ram_array[ptr_idx]), int(ram_array[ptr_idx+1]), int(ram_array[ptr_idx+2]), int(ram_array[ptr_idx+3])
        ptr_val = b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
        
        # WRAM address check (e.g. 0x02025eb4)
        if (ptr_val >> 24) != 0x02:
            return 0
            
        return ptr_val & 0x00FFFFFF

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
    def get_potions_count(cls, ram_array) -> int:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0: return 0
        items_start = wram_offset + 0x02B8
        if len(ram_array) <= items_start + (42 * 4): return 0
        
        total_potions = 0
        for i in range(42):
            idx = items_start + i * 4
            item_id = int(ram_array[idx]) | (int(ram_array[idx+1]) << 8)
            item_qty = int(ram_array[idx+2]) | (int(ram_array[idx+3]) << 8)
            if item_id == 13: # 13 = Potion no Fire Red
                total_potions += item_qty
        return total_potions

    @classmethod
    def get_tms_count(cls, ram_array) -> int:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0: return 0
        tms_start = wram_offset + 0x040C
        if len(ram_array) <= tms_start + (58 * 4): return 0
        
        total_tms = 0
        for i in range(58):
            idx = tms_start + i * 4
            item_id = int(ram_array[idx]) | (int(ram_array[idx+1]) << 8)
            item_qty = int(ram_array[idx+2]) | (int(ram_array[idx+3]) << 8)
            if item_id != 0 and item_qty > 0:
                total_tms += item_qty
        return total_tms

    @classmethod
    def get_p1_status(cls, ram_array) -> int:
        wram_offset = cls.get_saveblock1_ptr(ram_array)
        if wram_offset == 0: return 0
        # Pokemon 1 começa no offset 0x0294. Status está no byte 80 da struct (0x50).
        status_start = wram_offset + 0x0294 + 80
        if len(ram_array) <= status_start + 3: return 0
        
        b1, b2, b3, b4 = int(ram_array[status_start]), int(ram_array[status_start+1]), int(ram_array[status_start+2]), int(ram_array[status_start+3])
        status_val = b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
        
        # Se for > 0, tem alguma condição negativa (Poison, Paralyze, Sleep, etc)
        return 1 if status_val > 0 else 0

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

    @staticmethod
    def debug_dump(info: dict, env_id: int, label: str) -> None:
        log.debug(
            f"[Env {env_id}] {label} | "
            f"x={info.get('player_x','?')} y={info.get('player_y','?')} "
            f"map_bank={info.get('map_bank','?')} map_id={info.get('map_id','?')} | "
        )
