from src.config import CFG, log

class RamReader:
    @staticmethod
    def _clamp(val, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(val)))

    @classmethod
    def coord(cls, info: dict, key: str) -> int:
        return cls._clamp(info.get(key, 0), 0, CFG.max_coord)

    @classmethod
    def map_id(cls, info: dict) -> int:
        return cls._clamp(info.get("map_id", 0), 0, CFG.max_map_id)

    @classmethod
    def map_bank(cls, info: dict) -> int:
        return cls._clamp(info.get("map_bank", 0), 0, CFG.max_map_id)

    @classmethod
    def script_lock(cls, info: dict) -> int:
        return cls._clamp(info.get("script_lock", 0), 0, 1)

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
    def enemy_hp(cls, info: dict) -> int:
        total = sum(info.get(f"e{i}_hp", 0) for i in range(1, 7))
        return cls._clamp(total, 0, CFG.max_enemy_hp)

    @classmethod
    def in_battle(cls, info: dict) -> bool:
        return any(info.get(f"e{i}_hp", 0) > 0 for i in range(1, 7))

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
            f"in_battle={RamReader.in_battle(info)} "
        )
