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
    (3, 20): ("Enter Viridian Forest north via Route 2", 1, 0, "explore", "north"),
    (1, 0):  ("Navigate Viridian Forest north toward Pewter City", 3, 2, "explore", "north"),
    (3, 2):  ("Challenge Pewter Gym or continue east on Route 3", 3, 21, "battle_gym", "east"),
    (3, 21): ("Cross Route 3 east toward Mt. Moon", 1, 1, "explore", "east"),
    (1, 1):  ("Traverse Mt. Moon exit east to Route 4", 3, 22, "explore", "east"),
    (3, 22): ("Head east on Route 4 into Cerulean City", 3, 3, "explore", "east"),
    (3, 3):  ("Explore Cerulean City, Nugget Bridge, and Misty Gym", 3, 42, "battle_gym", "north"),
    (3, 42): ("Cross Nugget Bridge north toward Route 25", 3, 43, "explore", "east"),
    (3, 43): ("Visit Bill's Sea Cottage and return south to Cerulean", 3, 3, "explore", "south"),
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
    (3, 40): ("Take Route 22 west toward Victory Road gate", 3, 41, "explore", "west"),
    (3, 41): ("Pass badge checks north on Route 23 to Indigo Plateau", 3, 9, "explore", "north"),
    (3, 9):  ("Challenge the Elite Four at Indigo Plateau", 3, 9, "battle_gym", "north"),
}


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

        self.queries_count: int = 0
        self.llm_success_count: int = 0
        self.goal_alignments: int = 0
        self.total_alignment_bonus: float = 0.0
        self._ollama_cooldown_until: float = 0.0

    @property
    def current_goal(self) -> StrategicGoal:
        with self._lock:
            return self._current_goal

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
        potions = RamReader.get_potions_count(ram_array) if ram_array is not None else 0
        tms = RamReader.get_tms_count(ram_array) if ram_array is not None else 0

        last_dialogue = self._advisor.last_decoded_text.strip() if self._advisor else ""
        recent_dialogues = list(self._advisor.recent_dialogues) if self._advisor else []

        return {
            "map_bank": map_bank,
            "map_id": map_id,
            "player_x": px,
            "player_y": py,
            "party_level": party_level,
            "party_hp": party_hp,
            "party_max_hp": party_max_hp,
            "party_hp_pct": round(party_hp_pct, 3),
            "badges": badges,
            "event_flags": total_flags,
            "has_oaks_parcel": has_parcel,
            "potions": potions,
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
            "You are a Pokemon FireRed strategic planner. Output one JSON object.\n"
            "Rules:\n"
            "- sub_goal must be a concise English action phrase (never \"none\").\n"
            "- priority: \"heal\" ONLY if hp_pct < 0.35; \"battle_gym\" only for Gym/Elite4; \"use_hm\" for HM obstacles; else \"explore\".\n"
            "- Keep target_map_bank and target_map_id aligned with Kanto maps or the baseline unless low HP or dialogue requires overriding.\n"
            "- Kanto maps (bank,id): Pallet=(3,0), RedHouse=(4,0), OakLab=(4,3), "
            "Route1=(3,19), Viridian=(3,1), ViridianMart=(5,3), ViridianCenter=(5,4), "
            "Route2=(3,20), ViridianForest=(1,0), Pewter=(3,2), PewterGym=(6,2).\n"
            f"State: map_bank={context['map_bank']}, map_id={context['map_id']}, "
            f"pos=({context['player_x']},{context['player_y']}), "
            f"party_lvl={context['party_level']}, hp_pct={context['party_hp_pct']}, "
            f"badges={context['badges']}, flags={context['event_flags']}, "
            f"has_oaks_parcel={context['has_oaks_parcel']}, potions={context['potions']}, "
            f"in_battle={context['in_battle']}, steps_on_map={context['steps_on_map']}.\n"
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
            return StrategicGoal(
                sub_goal="Win active Pokemon battle",
                target_map_bank=bank,
                target_map_id=mid,
                priority="battle_gym" if context["badges"] == 0 and bank in (5, 6) else "explore",
                navigation_hint="interact",
            )

        if party_hp > 0 and hp_pct < 0.30:
            heal_bank, heal_id = (
                (4, 0) if (bank == 4 or (bank == 3 and mid in (0, 19))) else (5, 4)
            )
            return StrategicGoal(
                sub_goal="Heal injured party at Pokemon Center",
                target_map_bank=heal_bank,
                target_map_id=heal_id,
                priority="heal",
                navigation_hint="interact",
            )

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
                    sub_goal="Enter Prof. Oak's Lab to receive starter",
                    target_map_bank=4,
                    target_map_id=3,
                    priority="explore",
                    navigation_hint="east",
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
            and context.get("event_flags", 0) < 126
            and context.get("badges", 0) == 0
        ):
            return StrategicGoal(
                sub_goal="Enter Viridian Poke Mart to pick up Oak's Parcel",
                target_map_bank=5,
                target_map_id=3,
                priority="explore",
                navigation_hint="east",
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
    ) -> Optional[StrategicGoal]:
        goal = self._call_ollama(prompt)
        if goal is not None:
            if fallback_goal is not None:
                if goal.target_map_bank <= 0 and fallback_goal.target_map_bank > 0:
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                if (
                    not in_battle
                    and (goal.target_map_bank, goal.target_map_id) == issued_on_map
                    and (fallback_goal.target_map_bank, fallback_goal.target_map_id) != issued_on_map
                    and "deliver" not in goal.sub_goal.lower()
                ):
                    goal.target_map_bank = fallback_goal.target_map_bank
                    goal.target_map_id = fallback_goal.target_map_id
                    if goal.navigation_hint == "none":
                        goal.navigation_hint = fallback_goal.navigation_hint
                if goal.priority == "heal" and hp_pct >= 0.45 and fallback_goal.priority != "heal":
                    goal.priority = fallback_goal.priority
                if goal.sub_goal.strip().lower() in ("", "none", "explore", "heal", "battle_gym", "use_hm"):
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

        alignment_bonus = 0.0
        with self._lock:
            cur_goal = self._current_goal
        if (
            cur_goal.issued_on_map != (-1, -1)
            and map_key != cur_goal.issued_on_map
            and (cur_goal.target_map_bank, cur_goal.target_map_id) == map_key
            and map_key not in self._rewarded_targets
        ):
            self._rewarded_targets.add(map_key)
            self.goal_alignments += 1
            alignment_bonus = float(CFG.strategist_goal_bonus)
            self.total_alignment_bonus += alignment_bonus
            log.info(
                f"[Strategist] Goal aligned on map {map_key}! "
                f"bonus=+{alignment_bonus:.1f} ({cur_goal.sub_goal})"
            )

        if self._last_map_key is None or map_key != self._last_map_key:
            self._steps_on_current_map = 0
        else:
            self._steps_on_current_map += 1

        has_parcel_now = RamReader.has_oaks_parcel(ram_array) if ram_array is not None else False
        parcel_changed = (
            self._last_has_parcel is not None and has_parcel_now != self._last_has_parcel
        )
        if self.should_trigger(map_key, total_flags, in_battle) or parcel_changed:
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
                (self._last_in_battle is not None and in_battle != self._last_in_battle)
                and (self._last_map_key == map_key)
                and (total_flags <= self._last_flags)
                and not parcel_changed
                and not stagnation_triggered
            )
            if not battle_only_trigger:
                prompt = self.build_prompt(context, suggested_goal=fallback)
                hp_pct = float(context.get("party_hp_pct", 1.0))
                if self.async_mode and self._executor is not None:
                    if self._pending_future is None or self._pending_future.done():
                        self._pending_future = self._executor.submit(
                            self._query_and_apply, prompt, map_key, fallback, in_battle, hp_pct
                        )
                else:
                    self._query_and_apply(
                        prompt,
                        map_key,
                        fallback_goal=fallback,
                        in_battle=in_battle,
                        hp_pct=hp_pct,
                    )

        self._last_map_key = map_key
        self._last_flags = total_flags
        self._last_in_battle = in_battle
        self._last_has_parcel = has_parcel_now
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
        self.queries_count = 0
        self.llm_success_count = 0
        self.goal_alignments = 0
        self.total_alignment_bonus = 0.0

        with self._lock:
            if info is not None:
                map_key = (RamReader.map_bank(info), RamReader.map_id(info))
                context = self.build_context(info, ram_array, False, self._last_flags)
                goal = self._heuristic_goal(context)
                goal.issued_on_map = map_key
                self._current_goal = goal
            else:
                self._current_goal = StrategicGoal()

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
