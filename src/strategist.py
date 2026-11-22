from __future__ import annotations

import json
import time
import threading
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.advisor import RamAdvisor
from src.config import CFG, GOAL_SIZE, log
from src.reader import RamReader

PRIORITIES: tuple[str, ...] = ("explore", "heal", "battle_gym", "use_hm")
NAV_HINTS: tuple[str, ...] = ("none", "north", "south", "east", "west", "interact")

# Mapa macro de progressão em Kanto (pokefirered map_bank, map_id) ->
# (sub_goal, target_map_bank, target_map_id, priority, navigation_hint)
KANTO_PROGRESSION_GRAPH: dict[tuple[int, int], tuple[str, int, int, str, str]] = {
    (3, 1):  ("Advance north from Viridian City to Route 2", 3, 20, "explore", "north"),
    (3, 20): ("Walk north on Route 2 into Viridian Forest South Gate", 15, 0, "explore", "north"),
    (15, 0): ("Walk north through Viridian Forest South Gate into Viridian Forest", 1, 0, "explore", "north"),
    (1, 0):  ("Navigate Viridian Forest north to Viridian Forest North Gate", 15, 2, "explore", "north"),
    (15, 2): ("Walk north through Viridian Forest North Gate onto Route 2 north", 3, 20, "explore", "north"),
    (3, 2):  ("Challenge Pewter Gym or continue east on Route 3", 3, 21, "battle_gym", "east"),
    (3, 21): ("Cross Route 3 east toward Mt. Moon", 1, 1, "explore", "east"),
    (1, 1):  ("Traverse Mt. Moon exit east to Route 4", 3, 22, "explore", "east"),
    (3, 22): ("Head east on Route 4 into Cerulean City", 3, 3, "explore", "east"),
    (3, 3):  ("Explore Cerulean City, Nugget Bridge, and Misty Gym", 3, 43, "battle_gym", "north"),
    (3, 43): ("Cross Nugget Bridge north toward Route 25", 3, 44, "explore", "east"),
    (3, 44): ("Visit Bill's Sea Cottage and return south to Cerulean", 3, 3, "explore", "south"),
    (3, 23): ("Head south on Route 5 toward Vermilion City", 3, 24, "explore", "south"),
    (3, 24): ("Head south on Route 6 into Vermilion City", 3, 5, "explore", "south"),
    (3, 5):  ("Obtain HM01 Cut on S.S. Anne and challenge Lt. Surge", 3, 27, "use_hm", "north"),
    (3, 27): ("Cross Route 9 east toward Rock Tunnel", 3, 28, "explore", "east"),
    (3, 28): ("Enter Rock Tunnel south toward Lavender Town", 3, 4, "explore", "south"),
    (3, 4):  ("Head west from Lavender Town onto Route 8", 3, 26, "explore", "west"),
    (3, 26): ("Take Underground Path west toward Celadon City", 3, 6, "explore", "west"),
    (3, 25): ("Enter Celadon City west from Route 7", 3, 6, "explore", "west"),
    (3, 6):  ("Challenge Erika and clear Silph/Rocket path to Saffron", 3, 10, "battle_gym", "east"),
    (3, 10): ("Challenge Sabrina at Saffron Gym and head south", 3, 7, "battle_gym", "south"),
    (3, 7):  ("Obtain HM03 Surf in Safari Zone and challenge Koga", 3, 8, "use_hm", "south"),
    (3, 8):  ("Challenge Blaine on Cinnabar Island and head north", 3, 1, "battle_gym", "north"),
    (3, 39): ("Surf south on Route 21 toward Cinnabar Island", 3, 40, "explore", "south"),
    (3, 40): ("Surf south on Route 21 to Cinnabar Island", 3, 8, "explore", "south"),
    (3, 41): ("Return east from Route 22 to Viridian City", 3, 1, "explore", "east"),
    (3, 42): ("Pass badge checks north on Route 23 to Indigo Plateau", 3, 9, "explore", "north"),
    (3, 9):  ("Challenge the Elite Four at Indigo Plateau", 3, 9, "battle_gym", "north"),
}

VALID_KANTO_MAPS: frozenset[tuple[int, int]] = frozenset(
    set(KANTO_PROGRESSION_GRAPH.keys())
    | {(t_b, t_id) for _, t_b, t_id, _, _ in KANTO_PROGRESSION_GRAPH.values()}
    | {
        (3, 0), (4, 0), (4, 1), (4, 2), (4, 3),
        (5, 0), (5, 1), (5, 2), (5, 3), (5, 4),
        (6, 0), (6, 1), (6, 2),
    }
)


@dataclass
class StrategicGoal:
    sub_goal: str = "Explore current area and advance north"
    target_map_bank: int = 3
    target_map_id: int = 0
    priority: str = "explore"
    navigation_hint: str = "north"
    issued_on_map: tuple[int, int] = (-1, -1)

    def to_embedding(self) -> np.ndarray:
        """Converte o sub-objetivo macro em vetor float32 de dimensão GOAL_SIZE (6) em [0, 1]."""
        vec = np.zeros(GOAL_SIZE, dtype=np.float32)
        prio_idx = PRIORITIES.index(self.priority) if self.priority in PRIORITIES else 0
        vec[prio_idx] = 1.0
        clamped_map_id = max(0, min(255, int(self.target_map_id)))
        vec[4] = clamped_map_id / 255.0
        hint_idx = NAV_HINTS.index(self.navigation_hint) if self.navigation_hint in NAV_HINTS else 0
        vec[5] = hint_idx / float(len(NAV_HINTS) - 1)
        return vec


class Strategist:
    """
    Planejador estratégico assíncrono (Event-Driven) usando Ollama local com LLM.
    Integrado ao RamAdvisor para ler diálogos da EWRAM e definir sub-objetivos macro
    injetados na observação do PPO+LSTM sem nunca bloquear o loop env.step().
    """

    OLLAMA_RESPONSE_SCHEMA: dict = {
        "type": "object",
        "properties": {
            "sub_goal": {"type": "string"},
            "target_map_bank": {"type": "integer"},
            "target_map_id": {"type": "integer"},
            "priority": {"type": "string", "enum": list(PRIORITIES)},
            "navigation_hint": {"type": "string", "enum": list(NAV_HINTS)},
        },
        "required": [
            "sub_goal",
            "target_map_bank",
            "target_map_id",
            "priority",
            "navigation_hint",
        ],
    }

    def __init__(
        self,
        advisor: Optional[RamAdvisor] = None,
        ollama_url: str = CFG.ollama_url,
        model_name: str = CFG.ollama_model,
        async_mode: bool = True,
        timeout_sec: float = CFG.ollama_timeout_sec,
    ):
        self._advisor = advisor
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.async_mode = async_mode
        self.timeout_sec = timeout_sec

        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="strategist-llm")
            if async_mode
            else None
        )
        self._pending_future: Optional[Future] = None

        self._current_goal = StrategicGoal()
        self._last_map_key: Optional[tuple[int, int]] = None
        self._last_flags: int = 0
        self._last_in_battle: Optional[bool] = None
        self._last_has_parcel: Optional[bool] = None
        self._steps_on_current_map: int = 0
        self._rewarded_targets: set[tuple[int, int]] = set()
        self._visited_maps_ep: set[tuple[int, int]] = set()

        self.queries_count: int = 0
        self.llm_success_count: int = 0
        self.goal_alignments: int = 0
        self.total_alignment_bonus: float = 0.0
        self._ollama_cooldown_until: float = 0.0
        self._tactical_signal: float = 0.0
        self._last_battle_phase: str = ""
        self._last_needs_heal: Optional[bool] = None
        self._heal_alignment_available: bool = True

    @property
    def current_goal(self) -> StrategicGoal:
        with self._lock:
            return self._current_goal

    def get_tactical_signal(self) -> float:
        with self._lock:
            return float(self._tactical_signal)

    @staticmethod
    def _extract_party_metrics(info: dict, ram_array) -> dict:
        slots = RamReader.get_party_slots(ram_array) if ram_array is not None else []
        if not slots:
            for i in range(1, 7):
                maxhp = int(info.get(f"p{i}_maxhp", 0) or 0)
                if maxhp > 0:
                    slots.append({
                        "slot": i - 1,
                        "level": int(info.get(f"p{i}_lvl", 0) or 0),
                        "hp": int(info.get(f"p{i}_hp", 0) or 0),
                        "max_hp": maxhp,
                    })
        party_hp = RamReader.party_hp(info)
        party_max_hp = RamReader.party_max_hp(info)
        party_hp_pct = (party_hp / party_max_hp) if party_max_hp > 0 else 1.0
        party_level = RamReader.party_level(info)
        if not slots:
            return {
                "party_size": RamReader.party_size(info),
                "max_mon_level": party_level,
                "lead_level": party_level,
                "lead_hp_pct": round(party_hp_pct, 3),
                "fainted_count": 0,
                "injured_count": 0,
            }
        lead = slots[0]
        lead_hp_pct = round(lead["hp"] / float(max(lead["max_hp"], 1)), 3)
        return {
            "party_size": len(slots),
            "max_mon_level": max(int(s["level"]) for s in slots),
            "lead_level": int(lead["level"]),
            "lead_hp_pct": lead_hp_pct,
            "fainted_count": sum(1 for s in slots if int(s["hp"]) <= 0),
            "injured_count": sum(1 for s in slots if int(s["hp"]) < int(s["max_hp"]) * 0.60),
        }

    @staticmethod
    def _compute_needs_center_heal(
        map_bank: int,
        map_id: int,
        party_hp: int,
        party_hp_pct: float,
        party_size: int,
        lead_hp_pct: float,
        fainted_count: int,
        injured_count: int,
        has_parcel: bool,
    ) -> bool:
        if party_hp <= 0 or party_size <= 0:
            return False
        if (map_bank, map_id) in ((5, 4), (6, 1)):
            return party_hp_pct < 0.90 or fainted_count >= 1
        if (map_bank, map_id) == (4, 0):
            return party_hp_pct < 0.60 or fainted_count >= 1
        if has_parcel:
            return party_hp_pct < 0.35 or fainted_count >= 1
        return (
            party_hp_pct < 0.55
            or fainted_count >= 1
            or lead_hp_pct < 0.45
            or (party_size >= 2 and injured_count >= 2)
        )

    @staticmethod
    def _classify_tactical_phase(context: dict) -> tuple[str, float]:
        if context.get("in_battle", False):
            potions = int(context.get("potions", 0))
            active_hp_pct = float(context.get("active_hp_pct", 1.0))
            party_hp_pct = float(context.get("party_hp_pct", 1.0))
            if context.get("active_fainted", False) and context.get("has_alive_backup", False):
                return "switch", 1.0
            if potions > 0 and 0.0 < active_hp_pct < 0.50:
                return "potion", 1.0
            if context.get("can_capture", False):
                return "capture", 1.0
            if (
                context.get("is_wild", False)
                and potions == 0
                and 0.0 < active_hp_pct < 0.35
                and party_hp_pct < 0.35
            ):
                return "flee", 0.5
            return "fight", 0.0

        bank = int(context.get("map_bank", 0))
        mid = int(context.get("map_id", 0))
        potions = int(context.get("potions", 0))
        balls = int(context.get("pokeballs", 0))
        money = int(context.get("money", 0))
        if (
            (bank, mid) in ((5, 3), (6, 0))
            and not context.get("has_oaks_parcel", False)
            and context.get("has_pokedex", False)
            and ((potions < 2 and money >= 300) or (balls < 3 and money >= 200))
        ):
            return "mart_buy", 1.0
        return "overworld", 0.0

    def should_trigger(
        self,
        map_key: tuple[int, int],
        total_flags: int,
        in_battle: bool,
    ) -> bool:
        if self._last_map_key is None or map_key != self._last_map_key:
            return True
        if total_flags > self._last_flags:
            return True
        if self._last_in_battle is None or in_battle != self._last_in_battle:
            return True
        if self._steps_on_current_map >= CFG.strategist_stagnation_steps:
            return True
        return False

    def build_context(
        self,
        info: dict,
        ram_array,
        in_battle: bool,
        total_flags: int,
    ) -> dict:
        map_bank = RamReader.map_bank(info)
        map_id = RamReader.map_id(info)
        px, py = RamReader.get_player_coords(ram_array, info)
        party_hp = RamReader.party_hp(info)
        party_max_hp = RamReader.party_max_hp(info)
        party_hp_pct = (party_hp / party_max_hp) if party_max_hp > 0 else 1.0
        party_level = RamReader.party_level(info)
        badges = RamReader.badges(info)

        has_parcel = RamReader.has_oaks_parcel(ram_array) if ram_array is not None else False
        has_pokedex = RamReader.has_pokedex(ram_array) if ram_array is not None else False
        oak_in_lab = RamReader.is_oak_in_lab(ram_array) if ram_array is not None else True
        potions = RamReader.get_potions_count(ram_array) if ram_array is not None else 0
        pokeballs = RamReader.get_pokeballs_count(ram_array) if ram_array is not None else 0
        money = RamReader.get_money(ram_array) if ram_array is not None else 0
        tms = RamReader.get_tms_count(ram_array) if ram_array is not None else 0

        if in_battle and ram_array is not None:
            p_mon = RamReader.get_battle_mon(ram_array, 0)
            active_hp = p_mon["hp"] if p_mon["max_hp"] > 0 else party_hp
            active_max_hp = p_mon["max_hp"] if p_mon["max_hp"] > 0 else max(party_max_hp, 1)
            active_hp_pct = round(active_hp / float(max(active_max_hp, 1)), 3)
            active_fainted = bool(p_mon["max_hp"] > 0 and p_mon["hp"] == 0)
            has_alive_backup = bool(RamReader.get_best_alive_switch_slot(ram_array) > 0)
            is_wild = not RamReader.is_trainer_battle(ram_array)
            battle_obs = RamReader.get_battle_observation(ram_array, info)
            can_capture = bool(battle_obs[11] > 0.5)
            enemy_hp_pct = round(float(battle_obs[10]), 3)
        else:
            active_hp_pct = round(party_hp_pct, 3)
            active_fainted = False
            has_alive_backup = bool(RamReader.get_best_alive_switch_slot(ram_array) > 0) if ram_array is not None else False
            is_wild = False
            can_capture = False
            enemy_hp_pct = 0.0

        last_dialogue = self._advisor.last_decoded_text.strip() if self._advisor else ""
        recent_dialogues = list(self._advisor.recent_dialogues) if self._advisor else []
        pm = self._extract_party_metrics(info, ram_array)
        needs_center_heal = self._compute_needs_center_heal(
            map_bank=map_bank,
            map_id=map_id,
            party_hp=party_hp,
            party_hp_pct=party_hp_pct,
            party_size=int(pm["party_size"]),
            lead_hp_pct=float(pm["lead_hp_pct"]),
            fainted_count=int(pm["fainted_count"]),
            injured_count=int(pm["injured_count"]),
            has_parcel=has_parcel,
        )

        return {
            "map_bank": map_bank,
            "map_id": map_id,
            "player_x": px,
            "player_y": py,
            "party_level": party_level,
            "party_size": int(pm["party_size"]),
            "max_mon_level": int(pm["max_mon_level"]),
            "lead_level": int(pm["lead_level"]),
            "lead_hp_pct": float(pm["lead_hp_pct"]),
            "fainted_count": int(pm["fainted_count"]),
            "injured_count": int(pm["injured_count"]),
            "needs_center_heal": needs_center_heal,
            "party_hp": party_hp,
            "party_max_hp": party_max_hp,
            "party_hp_pct": round(party_hp_pct, 3),
            "active_hp_pct": active_hp_pct,
            "active_fainted": active_fainted,
            "has_alive_backup": has_alive_backup,
            "is_wild": is_wild,
            "can_capture": can_capture,
            "enemy_hp_pct": enemy_hp_pct,
            "badges": badges,
            "event_flags": total_flags,
            "has_oaks_parcel": has_parcel,
            "has_pokedex": has_pokedex,
            "oak_in_lab": oak_in_lab,
            "potions": potions,
            "pokeballs": pokeballs,
            "money": money,
            "tms": tms,
            "in_battle": in_battle,
            "steps_on_map": self._steps_on_current_map,
            "last_dialogue": last_dialogue,
            "recent_dialogues": recent_dialogues,
        }

    def build_prompt(
        self,
        context: dict,
        suggested_goal: Optional[StrategicGoal] = None,
    ) -> str:
        dialog_snippet = context.get("last_dialogue") or "none"
        recent = " | ".join(context.get("recent_dialogues", [])[-3:]) or "none"
        suggestion_line = (
            f"Baseline: sub_goal=\"{suggested_goal.sub_goal}\", "
            f"target_map_bank={suggested_goal.target_map_bank}, "
            f"target_map_id={suggested_goal.target_map_id}, "
            f"priority=\"{suggested_goal.priority}\", "
            f"navigation_hint=\"{suggested_goal.navigation_hint}\".\n"
            if suggested_goal is not None
            else ""
        )
        return (
            "You are a Pokemon FireRed strategic & tactical planner. Output one JSON object.\n"
            "Rules:\n"
            "- sub_goal must be a concise English action phrase (never \"none\").\n"
            "- priority: \"heal\" when needs_center_heal=True, active_fainted/low HP, needing Potion, or fleeing wild battle at critical HP; "
            "\"battle_gym\" only for Gym/Elite4; \"use_hm\" for HM obstacles; else \"explore\".\n"
            "- Center & Preparation: when needs_center_heal=True (e.g. multiple injured Pokemon, fainted party member, or low HP), route to ViridianCenter (5,4) or PewterCenter (6,1) and stay inside until HP >= 90% and fainted == 0 before exiting south.\n"
            "- Rival Gary (Route 22 has Pidgey Lv 9 + starter): do NOT fight Gary until max_mon_lvl >= 11, lead_lvl >= 10, fainted == 0, and hp_pct >= 0.75; otherwise train on Route 1/Route 2 or heal in Viridian first.\n"
            "- Tactics: switch to alive backup on faint; use Potion when active_hp_pct < 0.50; throw Poke Ball if can_capture=True; "
            "flee wild battle if active_hp_pct < 0.35 and potions == 0; buy Potions/Balls in Mart when low and money >= 300.\n"
            "- Keep target_map_bank and target_map_id aligned with Kanto maps or the baseline unless low HP or dialogue requires overriding.\n"
            "- Kanto maps (bank,id): Pallet=(3,0), RedHouse=(4,0), OakLab=(4,3), "
            "Route1=(3,19), Viridian=(3,1), ViridianMart=(5,3), ViridianCenter=(5,4), "
            "Route2=(3,20), ViridianGateS=(15,0), ViridianForest=(1,0), ViridianGateN=(15,2), Pewter=(3,2), PewterCenter=(6,1), PewterGym=(6,2), Route22=(3,41).\n"
            f"State: map_bank={context['map_bank']}, map_id={context['map_id']}, "
            f"pos=({context['player_x']},{context['player_y']}), "
            f"party_lvl={context['party_level']}, party_size={context.get('party_size', 1)}, "
            f"max_mon_lvl={context.get('max_mon_level', context['party_level'])}, lead_lvl={context.get('lead_level', context['party_level'])}, "
            f"hp_pct={context['party_hp_pct']}, lead_hp_pct={context.get('lead_hp_pct', context['party_hp_pct'])}, "
            f"active_hp_pct={context.get('active_hp_pct', context['party_hp_pct'])}, "
            f"fainted={context.get('fainted_count', 0)}, injured={context.get('injured_count', 0)}, "
            f"needs_center_heal={context.get('needs_center_heal', False)}, "
            f"badges={context['badges']}, flags={context['event_flags']}, "
            f"has_oaks_parcel={context['has_oaks_parcel']}, has_pokedex={context.get('has_pokedex', False)}, "
            f"potions={context['potions']}, pokeballs={context['pokeballs']}, money={context['money']}, "
            f"in_battle={context['in_battle']}, is_wild={context.get('is_wild', False)}, "
            f"can_capture={context.get('can_capture', False)}, steps_on_map={context['steps_on_map']}.\n"
            f"{suggestion_line}"
            f"RamAdvisor last_dialogue: \"{dialog_snippet}\" | recent: \"{recent}\"\n"
            "Return strict JSON matching the schema."
        )

    @classmethod
    def parse_llm_response(cls, raw_text: str) -> Optional[StrategicGoal]:
        if not raw_text or not isinstance(raw_text, str):
            return None

        text = raw_text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None

        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        sub_goal = str(payload.get("sub_goal", "Explore area")).strip() or "Explore area"
        try:
            target_bank = max(0, min(255, int(payload.get("target_map_bank", 3))))
        except (TypeError, ValueError):
            target_bank = 3
        try:
            target_id = max(0, min(255, int(payload.get("target_map_id", 0))))
        except (TypeError, ValueError):
            target_id = 0

        priority = str(payload.get("priority", "explore")).strip().lower()
        if priority not in PRIORITIES:
            priority = "explore"

        nav_hint = str(payload.get("navigation_hint", "none")).strip().lower()
        if nav_hint not in NAV_HINTS:
            nav_hint = "none"

        return StrategicGoal(
            sub_goal=sub_goal,
            target_map_bank=target_bank,
            target_map_id=target_id,
            priority=priority,
            navigation_hint=nav_hint,
        )

    @staticmethod
    def _heal_route_goal(context: dict) -> StrategicGoal:
        bank = int(context.get("map_bank", 0))
        mid = int(context.get("map_id", 0))
        px = int(context.get("player_x", 0))
        py = int(context.get("player_y", 0))

        if (bank, mid) in ((5, 4), (6, 1)):
            if py > 4:
                return StrategicGoal(
                    sub_goal="Walk north to Nurse Joy counter in Pokemon Center to heal party",
                    target_map_bank=bank,
                    target_map_id=mid,
                    priority="heal",
                    navigation_hint="north",
                )
            return StrategicGoal(
                sub_goal="Talk to Nurse Joy to heal injured party",
                target_map_bank=bank,
                target_map_id=mid,
                priority="heal",
                navigation_hint="interact",
            )
        if (bank, mid) == (4, 0):
            return StrategicGoal(
                sub_goal="Talk to Mom in Red's House to heal injured party",
                target_map_bank=4,
                target_map_id=0,
                priority="heal",
                navigation_hint="interact",
            )
        if (bank, mid) == (3, 1):
            hint = "south" if py < 25 else ("west" if px > 28 else ("east" if px < 24 else "north"))
            return StrategicGoal(
                sub_goal="Enter Viridian Pokemon Center to heal injured party",
                target_map_bank=5,
                target_map_id=4,
                priority="heal",
                navigation_hint=hint,
            )
        if (bank, mid) == (3, 2):
            hint = "south" if py < 24 else ("west" if px > 20 else ("east" if px < 14 else "north"))
            return StrategicGoal(
                sub_goal="Enter Pewter Pokemon Center to heal injured party",
                target_map_bank=6,
                target_map_id=1,
                priority="heal",
                navigation_hint=hint,
            )
        if (bank, mid) == (3, 20):
            if py >= 45:
                return StrategicGoal(
                    sub_goal="Return south from Route 2 to Viridian Pokemon Center to heal party",
                    target_map_bank=3,
                    target_map_id=1,
                    priority="heal",
                    navigation_hint="south",
                )
            return StrategicGoal(
                sub_goal="Continue north on Route 2 to Pewter Pokemon Center to heal party",
                target_map_bank=3,
                target_map_id=2,
                priority="heal",
                navigation_hint="north",
            )
        if (bank, mid) == (3, 41):
            return StrategicGoal(
                sub_goal="Retreat east from Route 22 to Viridian Pokemon Center to heal party",
                target_map_bank=3,
                target_map_id=1,
                priority="heal",
                navigation_hint="east",
            )
        if (bank, mid) == (3, 19):
            if py <= 22:
                return StrategicGoal(
                    sub_goal="Head north on Route 1 to Viridian Pokemon Center to heal party",
                    target_map_bank=3,
                    target_map_id=1,
                    priority="heal",
                    navigation_hint="north",
                )
            return StrategicGoal(
                sub_goal="Return south on Route 1 to Pallet Town to heal injured party",
                target_map_bank=3,
                target_map_id=0,
                priority="heal",
                navigation_hint="south",
            )
        if (bank, mid) == (3, 0):
            return StrategicGoal(
                sub_goal="Enter Red's House to heal injured party",
                target_map_bank=4,
                target_map_id=0,
                priority="heal",
                navigation_hint="north",
            )
        if (bank, mid) == (15, 0):
            return StrategicGoal(
                sub_goal="Exit Viridian Forest South Gate toward Viridian Pokemon Center to heal",
                target_map_bank=3,
                target_map_id=20,
                priority="heal",
                navigation_hint="south",
            )
        if (bank, mid) == (1, 0):
            if py < 28:
                return StrategicGoal(
                    sub_goal="Exit Viridian Forest north toward Pewter Pokemon Center to heal",
                    target_map_bank=15,
                    target_map_id=2,
                    priority="heal",
                    navigation_hint="north",
                )
            return StrategicGoal(
                sub_goal="Retreat south from Viridian Forest toward Viridian Pokemon Center to heal",
                target_map_bank=15,
                target_map_id=0,
                priority="heal",
                navigation_hint="south",
            )
        if (bank, mid) == (15, 2):
            return StrategicGoal(
                sub_goal="Exit Viridian Forest North Gate toward Pewter Pokemon Center to heal",
                target_map_bank=3,
                target_map_id=20,
                priority="heal",
                navigation_hint="north",
            )
        if bank == 5:
            return StrategicGoal(
                sub_goal="Exit Viridian building south to reach Pokemon Center",
                target_map_bank=3,
                target_map_id=1,
                priority="heal",
                navigation_hint="south",
            )
        if bank == 6:
            return StrategicGoal(
                sub_goal="Exit Pewter building south to reach Pokemon Center",
                target_map_bank=3,
                target_map_id=2,
                priority="heal",
                navigation_hint="south",
            )
        if bank == 4:
            return StrategicGoal(
                sub_goal="Exit building south to heal injured party",
                target_map_bank=3,
                target_map_id=0,
                priority="heal",
                navigation_hint="south",
            )
        return StrategicGoal(
            sub_goal="Return to nearest Pokemon Center to heal injured party",
            target_map_bank=5,
            target_map_id=4,
            priority="heal",
            navigation_hint="south",
        )

    def _heuristic_goal(self, context: dict) -> StrategicGoal:
        bank = context["map_bank"]
        mid = context["map_id"]
        hp_pct = context["party_hp_pct"]
        party_hp = context["party_hp"]
        party_lvl = context["party_level"]
        has_parcel = context["has_oaks_parcel"]
        in_battle = context["in_battle"]
        last_dialog = context.get("last_dialogue", "")

        if in_battle:
            phase, _ = self._classify_tactical_phase(context)
            if phase == "switch":
                return StrategicGoal(
                    sub_goal="Switch to next alive Pokemon",
                    target_map_bank=bank,
                    target_map_id=mid,
                    priority="heal",
                    navigation_hint="interact",
                )
            if phase == "potion":
                return StrategicGoal(
                    sub_goal="Use Potion to heal active Pokemon",
                    target_map_bank=bank,
                    target_map_id=mid,
                    priority="heal",
                    navigation_hint="interact",
                )
            if phase == "capture":
                return StrategicGoal(
                    sub_goal="Throw Poke Ball to capture new wild Pokemon",
                    target_map_bank=bank,
                    target_map_id=mid,
                    priority="explore",
                    navigation_hint="interact",
                )
            if phase == "flee":
                return StrategicGoal(
                    sub_goal="Flee wild battle at critical HP and heal",
                    target_map_bank=bank,
                    target_map_id=mid,
                    priority="heal",
                    navigation_hint="south",
                )
            return StrategicGoal(
                sub_goal="Win active Pokemon battle",
                target_map_bank=bank,
                target_map_id=mid,
                priority="battle_gym" if context["badges"] == 0 and bank in (5, 6) else "explore",
                navigation_hint="interact",
            )

        if context.get("needs_center_heal", False):
            return self._heal_route_goal(context)

        if "yes" in last_dialog and "no" in last_dialog:
            return StrategicGoal(
                sub_goal="Resolve dialogue prompt and proceed",
                target_map_bank=bank,
                target_map_id=mid,
                priority="explore",
                navigation_hint="interact",
            )

        if has_parcel:
            if bank == 5:
                return StrategicGoal(
                    sub_goal="Exit Viridian interior south to return Oak's Parcel",
                    target_map_bank=3,
                    target_map_id=1,
                    priority="explore",
                    navigation_hint="south",
                )
            if (bank, mid) == (3, 1):
                return StrategicGoal(
                    sub_goal="Leave Viridian City south onto Route 1",
                    target_map_bank=3,
                    target_map_id=19,
                    priority="explore",
                    navigation_hint="south",
                )
            if (bank, mid) == (3, 19):
                return StrategicGoal(
                    sub_goal="Head south down Route 1 to Pallet Town",
                    target_map_bank=3,
                    target_map_id=0,
                    priority="explore",
                    navigation_hint="south",
                )
            if (bank, mid) == (3, 0):
                return StrategicGoal(
                    sub_goal="Enter Prof. Oak's Lab to deliver Oak's Parcel",
                    target_map_bank=4,
                    target_map_id=3,
                    priority="explore",
                    navigation_hint="south",
                )
            if (bank, mid) == (4, 3):
                return StrategicGoal(
                    sub_goal="Deliver Oak's Parcel to Prof. Oak",
                    target_map_bank=4,
                    target_map_id=3,
                    priority="explore",
                    navigation_hint="interact",
                )
            return StrategicGoal(
                sub_goal="Return south to deliver Oak's Parcel",
                target_map_bank=4,
                target_map_id=3,
                priority="explore",
                navigation_hint="south",
            )

        if (bank, mid) == (4, 1):
            return StrategicGoal(
                sub_goal="Go downstairs from Red's bedroom to Red's House 1F",
                target_map_bank=4,
                target_map_id=0,
                priority="explore",
                navigation_hint="east",
            )
        if (bank, mid) == (4, 0):
            return StrategicGoal(
                sub_goal="Leave Red's house south into Pallet Town",
                target_map_bank=3,
                target_map_id=0,
                priority="explore",
                navigation_hint="south",
            )
        if (bank, mid) == (4, 3) and party_lvl == 0:
            if not context.get("oak_in_lab", True):
                return StrategicGoal(
                    sub_goal="Exit empty Oak's Lab south and walk north to Pallet grass for Prof. Oak",
                    target_map_bank=3,
                    target_map_id=0,
                    priority="explore",
                    navigation_hint="south",
                )
            return StrategicGoal(
                sub_goal="Choose your starter Pokemon from the table in Prof. Oak's Lab",
                target_map_bank=4,
                target_map_id=3,
                priority="explore",
                navigation_hint="interact",
            )

        if bank == 4:
            return StrategicGoal(
                sub_goal="Exit interior building to overworld",
                target_map_bank=3,
                target_map_id=0 if mid <= 3 else 1,
                priority="explore",
                navigation_hint="south",
            )

        if bank == 3 and mid == 0:
            if party_lvl == 0:
                return StrategicGoal(
                    sub_goal="Walk north toward Route 1 grass so Prof. Oak leads you to his Lab",
                    target_map_bank=4,
                    target_map_id=3,
                    priority="explore",
                    navigation_hint="north",
                )
            return StrategicGoal(
                sub_goal="Leave Pallet Town north onto Route 1",
                target_map_bank=3,
                target_map_id=19,
                priority="explore",
                navigation_hint="north",
            )

        if bank == 3 and mid == 19:
            return StrategicGoal(
                sub_goal="Cross Route 1 north to Viridian City",
                target_map_bank=3,
                target_map_id=1,
                priority="explore",
                navigation_hint="north",
            )

        if (
            bank == 3
            and mid == 1
            and not has_parcel
            and not context.get("has_pokedex", False)
            and context.get("event_flags", 0) < 124
            and context.get("badges", 0) == 0
        ):
            return StrategicGoal(
                sub_goal="Enter Viridian Poke Mart to pick up Oak's Parcel",
                target_map_bank=5,
                target_map_id=3,
                priority="explore",
                navigation_hint="east",
            )

        if (
            (bank, mid) == (3, 1)
            and not has_parcel
            and context.get("has_pokedex", False)
            and context.get("badges", 0) == 0
            and context.get("money", 0) >= 600
            and context.get("potions", 0) == 0
        ):
            return StrategicGoal(
                sub_goal="Enter Viridian Poke Mart to buy Potions before Route 2",
                target_map_bank=5,
                target_map_id=3,
                priority="explore",
                navigation_hint="east",
            )

        if (
            (bank, mid) in ((5, 3), (6, 0))
            and not has_parcel
            and context.get("has_pokedex", False)
            and context.get("money", 0) >= 500
            and (context.get("potions", 0) < 3 or context.get("pokeballs", 0) < 5)
        ):
            return StrategicGoal(
                sub_goal="Buy Potions and Poke Balls from the Mart clerk",
                target_map_bank=bank,
                target_map_id=mid,
                priority="explore",
                navigation_hint="interact",
            )

        if bank == 5:
            return StrategicGoal(
                sub_goal="Exit Viridian building south to Viridian City",
                target_map_bank=3,
                target_map_id=1,
                priority="explore",
                navigation_hint="south",
            )

        if bank == 6 and mid != 2:
            return StrategicGoal(
                sub_goal="Exit Pewter building south to Pewter City",
                target_map_bank=3,
                target_map_id=2,
                priority="explore",
                navigation_hint="south",
            )

        if (bank, mid) == (3, 20) and context.get("player_y", 79) < 45:
            return StrategicGoal(
                sub_goal="Continue north on Route 2 into Pewter City",
                target_map_bank=3,
                target_map_id=2,
                priority="explore",
                navigation_hint="north",
            )

        max_mon_lvl = int(context.get("max_mon_level", party_lvl))
        lead_lvl = int(context.get("lead_level", max_mon_lvl))

        if (bank, mid) == (3, 41) and context.get("badges", 0) >= 8:
            return StrategicGoal(
                sub_goal="Take Route 22 west toward Victory Road gate",
                target_map_bank=3,
                target_map_id=42,
                priority="explore",
                navigation_hint="west",
            )
        if (bank, mid) == (3, 41) and context.get("badges", 0) < 8 and (max_mon_lvl < 11 or lead_lvl < 10 or hp_pct < 0.75):
            return StrategicGoal(
                sub_goal="Train lead Pokemon to Lv 11+ on Route 1/Route 2 before fighting Rival Gary (Pidgey Lv 9); return east to Viridian City",
                target_map_bank=3,
                target_map_id=1,
                priority="explore",
                navigation_hint="east",
            )

        if (
            (bank, mid) == (3, 1)
            and context.get("has_pokedex", False)
            and context.get("badges", 0) == 0
            and not has_parcel
            and max_mon_lvl < 11
        ):
            return StrategicGoal(
                sub_goal="Advance north to Route 2 to train lead Pokemon to Lv 11+ before Rival Gary / Pewter Gym",
                target_map_bank=3,
                target_map_id=20,
                priority="explore",
                navigation_hint="north",
            )

        if (bank, mid) == (3, 2) and context.get("badges", 0) == 0:
            if max_mon_lvl < 13:
                return StrategicGoal(
                    sub_goal="Train lead Pokemon to Lv 13+ on Route 2 before challenging Brock at Pewter Gym",
                    target_map_bank=3,
                    target_map_id=20,
                    priority="explore",
                    navigation_hint="south",
                )
            return StrategicGoal(
                sub_goal="Challenge Brock at Pewter Gym",
                target_map_bank=6,
                target_map_id=2,
                priority="battle_gym",
                navigation_hint="north",
            )

        if (bank, mid) in KANTO_PROGRESSION_GRAPH:
            sub_goal, t_bank, t_id, prio, hint = KANTO_PROGRESSION_GRAPH[(bank, mid)]
            return StrategicGoal(
                sub_goal=sub_goal,
                target_map_bank=t_bank,
                target_map_id=t_id,
                priority=prio,
                navigation_hint=hint,
            )

        return StrategicGoal(
            sub_goal="Explore current region and advance",
            target_map_bank=3,
            target_map_id=mid,
            priority="explore",
            navigation_hint="north",
        )

    def _call_ollama(self, prompt: str) -> Optional[StrategicGoal]:
        now = time.monotonic()
        if now < self._ollama_cooldown_until:
            return None

        payload = json.dumps({
            "model": self.model_name,
            "prompt": prompt,
            "format": self.OLLAMA_RESPONSE_SCHEMA,
            "stream": False,
            "keep_alive": "24h",
            "options": {
                "temperature": 0.1,
                "num_predict": 96,
                "num_ctx": 1024,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            self.ollama_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            self._ollama_cooldown_until = time.monotonic() + 30.0
            log.debug(f"[Strategist] Ollama indisponível ({exc}); usando heurística local.")
            return None

        raw_response = body.get("response", "") if isinstance(body, dict) else ""
        return self.parse_llm_response(raw_response)

    def _query_and_apply(
        self,
        prompt: str,
        issued_on_map: tuple[int, int],
        fallback_goal: Optional[StrategicGoal] = None,
        in_battle: bool = False,
        hp_pct: float = 1.0,
        needs_center_heal: bool = False,
    ) -> Optional[StrategicGoal]:
        goal = self._call_ollama(prompt)
        if goal is not None:
            if fallback_goal is not None:
                if goal.priority == "heal" and not needs_center_heal and hp_pct >= 0.65 and fallback_goal.priority != "heal":
                    goal.priority = fallback_goal.priority

                goal_target = (goal.target_map_bank, goal.target_map_id)
                fallback_target = (fallback_goal.target_map_bank, fallback_goal.target_map_id)
                valid_heal_override = (
                    goal.priority == "heal"
                    and (needs_center_heal or hp_pct < 0.55)
                    and goal_target in ((4, 0), (5, 4), (6, 1), (3, 1), (3, 2), (3, 0))
                )

                if goal.target_map_bank <= 0 and fallback_goal.target_map_bank > 0:
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    goal.navigation_hint = fallback_goal.navigation_hint
                elif goal_target not in VALID_KANTO_MAPS:
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    goal.navigation_hint = fallback_goal.navigation_hint
                elif (
                    fallback_goal.priority == "heal"
                    and not in_battle
                    and issued_on_map in ((5, 4), (6, 1), (4, 0))
                    and goal_target != issued_on_map
                ):
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    goal.navigation_hint = fallback_goal.navigation_hint
                    goal.sub_goal = fallback_goal.sub_goal
                elif (
                    not in_battle
                    and fallback_target != issued_on_map
                    and goal_target != fallback_target
                    and not valid_heal_override
                ):
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    goal.navigation_hint = fallback_goal.navigation_hint
                elif (
                    not in_battle
                    and goal_target == issued_on_map
                    and fallback_target != issued_on_map
                    and "deliver" not in goal.sub_goal.lower()
                ):
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    if goal.navigation_hint == "none":
                        goal.navigation_hint = fallback_goal.navigation_hint

                if (
                    (goal.target_map_bank, goal.target_map_id) == fallback_target
                    and fallback_goal.navigation_hint != "none"
                    and goal.navigation_hint != fallback_goal.navigation_hint
                ):
                    goal.navigation_hint = fallback_goal.navigation_hint

                if (
                    goal.sub_goal.strip().lower() in ("", "none", "explore", "heal", "battle_gym", "use_hm")
                    or (in_battle and "choose pokemon" in goal.sub_goal.lower())
                ):
                    goal.sub_goal = fallback_goal.sub_goal
            goal.issued_on_map = issued_on_map
            with self._lock:
                self._current_goal = goal
                self.llm_success_count += 1
                cnt = self.llm_success_count
            if cnt == 1 or cnt % 25 == 0:
                log.info(
                    f"[Strategist LLM] #{cnt} on {issued_on_map} -> "
                    f"({goal.target_map_bank},{goal.target_map_id}) {goal.navigation_hint}: {goal.sub_goal}"
                )
        return goal

    def update(
        self,
        info: dict,
        ram_array,
        in_battle: bool,
        total_flags: int,
    ) -> float:
        map_key = (RamReader.map_bank(info), RamReader.map_id(info))

        empty_oak_lab_entry = (
            map_key == (4, 3)
            and RamReader.party_level(info) == 0
            and ram_array is not None
            and not RamReader.is_oak_in_lab(ram_array)
        )

        party_hp_now = RamReader.party_hp(info)
        party_max_now = RamReader.party_max_hp(info)
        party_hp_pct_now = (party_hp_now / party_max_now) if party_max_now > 0 else 1.0
        has_parcel_now = RamReader.has_oaks_parcel(ram_array) if ram_array is not None else False
        pm_now = self._extract_party_metrics(info, ram_array)
        needs_heal_now = (
            not in_battle
            and self._compute_needs_center_heal(
                map_bank=map_key[0],
                map_id=map_key[1],
                party_hp=party_hp_now,
                party_hp_pct=party_hp_pct_now,
                party_size=int(pm_now["party_size"]),
                lead_hp_pct=float(pm_now["lead_hp_pct"]),
                fainted_count=int(pm_now["fainted_count"]),
                injured_count=int(pm_now["injured_count"]),
                has_parcel=has_parcel_now,
            )
        )
        if party_hp_now > 0 and party_hp_pct_now >= 0.90 and int(pm_now["fainted_count"]) == 0:
            self._heal_alignment_available = True
        health_state_changed = (
            self._last_needs_heal is not None and needs_heal_now != self._last_needs_heal
        )

        alignment_bonus = 0.0
        with self._lock:
            cur_goal = self._current_goal
        is_heal_center_arrival = (
            cur_goal.priority == "heal"
            and map_key in ((5, 4), (6, 1), (4, 0))
            and self._heal_alignment_available
        )
        if (
            cur_goal.issued_on_map != (-1, -1)
            and map_key != cur_goal.issued_on_map
            and (cur_goal.target_map_bank, cur_goal.target_map_id) == map_key
            and (
                (map_key not in self._rewarded_targets and map_key not in self._visited_maps_ep)
                or is_heal_center_arrival
            )
            and not empty_oak_lab_entry
        ):
            if is_heal_center_arrival:
                self._heal_alignment_available = False
            else:
                self._rewarded_targets.add(map_key)
            self.goal_alignments += 1
            alignment_bonus = float(CFG.strategist_goal_bonus)
            self.total_alignment_bonus += alignment_bonus
            log.info(
                f"[Strategist] Goal aligned on map {map_key}! "
                f"bonus=+{alignment_bonus:.1f} ({cur_goal.sub_goal})"
            )
        if not empty_oak_lab_entry:
            self._visited_maps_ep.add(map_key)

        if self._last_map_key is None or map_key != self._last_map_key:
            self._steps_on_current_map = 0
        else:
            self._steps_on_current_map += 1

        parcel_changed = (
            self._last_has_parcel is not None and has_parcel_now != self._last_has_parcel
        )
        in_pokecenter_heal = needs_heal_now and map_key in ((5, 4), (6, 1))
        need_context = (
            self.should_trigger(map_key, total_flags, in_battle)
            or parcel_changed
            or health_state_changed
            or in_battle
            or map_key in ((5, 3), (6, 0))
            or in_pokecenter_heal
            or self._last_battle_phase not in ("", "overworld")
        )
        context = self.build_context(info, ram_array, in_battle, total_flags) if need_context else None
        if context is not None:
            phase_now, sig_now = self._classify_tactical_phase(context)
        else:
            phase_now, sig_now = "overworld", 0.0
        phase_changed = (phase_now != self._last_battle_phase)

        with self._lock:
            self._tactical_signal = sig_now

        if self.should_trigger(map_key, total_flags, in_battle) or parcel_changed or phase_changed or health_state_changed:
            if context is None:
                context = self.build_context(info, ram_array, in_battle, total_flags)
            fallback = self._heuristic_goal(context)
            fallback.issued_on_map = map_key
            with self._lock:
                self._current_goal = fallback

            self.queries_count += 1
            stagnation_triggered = self._steps_on_current_map >= CFG.strategist_stagnation_steps
            if stagnation_triggered:
                self._steps_on_current_map = 0

            battle_only_trigger = (
                (
                    (self._last_in_battle is not None and in_battle != self._last_in_battle)
                    or (in_battle and phase_changed)
                )
                and (self._last_map_key == map_key)
                and (total_flags <= self._last_flags)
                and not parcel_changed
                and not stagnation_triggered
            )
            if not battle_only_trigger:
                prompt = self.build_prompt(context, suggested_goal=fallback)
                hp_pct = float(context.get("party_hp_pct", 1.0))
                needs_center = bool(context.get("needs_center_heal", False))
                if self.async_mode and self._executor is not None:
                    if self._pending_future is None or self._pending_future.done():
                        self._pending_future = self._executor.submit(
                            self._query_and_apply, prompt, map_key, fallback, in_battle, hp_pct, needs_center
                        )
                else:
                    self._query_and_apply(
                        prompt,
                        map_key,
                        fallback_goal=fallback,
                        in_battle=in_battle,
                        hp_pct=hp_pct,
                        needs_center_heal=needs_center,
                    )
        elif in_pokecenter_heal and context is not None:
            pc_goal = self._heal_route_goal(context)
            pc_goal.issued_on_map = map_key
            with self._lock:
                if (
                    pc_goal.target_map_bank,
                    pc_goal.target_map_id,
                    pc_goal.navigation_hint,
                ) != (
                    self._current_goal.target_map_bank,
                    self._current_goal.target_map_id,
                    self._current_goal.navigation_hint,
                ):
                    self._current_goal = pc_goal

        self._last_map_key = map_key
        self._last_flags = total_flags
        self._last_in_battle = in_battle
        self._last_has_parcel = has_parcel_now
        self._last_battle_phase = phase_now
        self._last_needs_heal = needs_heal_now
        return alignment_bonus

    def get_goal_embedding(self) -> np.ndarray:
        with self._lock:
            return self._current_goal.to_embedding()

    def reset(self, info: Optional[dict] = None, ram_array=None) -> None:
        self._last_map_key = None
        self._last_flags = RamReader.event_flags_sum(ram_array) if ram_array is not None else 0
        self._last_in_battle = None
        self._last_has_parcel = RamReader.has_oaks_parcel(ram_array) if ram_array is not None else None
        self._steps_on_current_map = 0
        self._rewarded_targets.clear()
        self._visited_maps_ep.clear()
        self._heal_alignment_available = True
        self.queries_count = 0
        self.llm_success_count = 0
        self.goal_alignments = 0
        self.total_alignment_bonus = 0.0

        with self._lock:
            if info is not None:
                map_key = (RamReader.map_bank(info), RamReader.map_id(info))
                self._visited_maps_ep.add(map_key)
                context = self.build_context(info, ram_array, False, self._last_flags)
                goal = self._heuristic_goal(context)
                goal.issued_on_map = map_key
                self._current_goal = goal
                phase0, sig0 = self._classify_tactical_phase(context)
                self._last_battle_phase = phase0
                self._tactical_signal = sig0
                self._last_needs_heal = bool(context.get("needs_center_heal", False))
            else:
                self._current_goal = StrategicGoal()
                self._last_battle_phase = ""
                self._tactical_signal = 0.0
                self._last_needs_heal = None

    def stats_dict(self) -> dict:
        return {
            "strategist/queries_count": self.queries_count,
            "strategist/llm_success_count": self.llm_success_count,
            "strategist/goal_alignments": self.goal_alignments,
            "strategist/alignment_bonus": self.total_alignment_bonus,
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
