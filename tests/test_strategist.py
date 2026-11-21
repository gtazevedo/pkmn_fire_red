import io
import json
import time
import urllib.error
import numpy as np

from src.advisor import RamAdvisor
from src.config import CFG, GOAL_SIZE
from src.strategist import StrategicGoal, Strategist
from src.vision import GameState, VisionEngine


def _encode_gen3(text: str) -> bytes:
    rev = {v: k for k, v in RamAdvisor.CHAR_MAP.items() if len(v) == 1}
    out = []
    for ch in text:
        out.append(rev.get(ch, 0x00))
    out.append(0xFF)
    return bytes(out)


def _ram_with_dialogue(text: str) -> np.ndarray:
    ram = np.zeros(262144 + 0x6000, dtype=np.uint8)
    encoded = _encode_gen3(text)
    ram[138520 : 138520 + len(encoded)] = list(encoded)
    return ram


def test_advisor_decode_and_repeat_micro_penalty():
    advisor = RamAdvisor()
    ram = _ram_with_dialogue("PALLET TOWN")

    # Primeira leitura do diálogo -> sem penalidade de repetição
    r1 = advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram)
    assert advisor.last_decoded_text == "pallet town"
    assert advisor.repeat_penalties == 0
    assert r1 == 0.0

    # Mesmo frame enquanto ainda está no diálogo -> não penaliza de novo
    r_same = advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram)
    assert advisor.repeat_penalties == 0
    assert r_same == 0.0

    # Sai do diálogo e interage novamente com a mesma placa -> -0.001
    advisor.update(in_battle=False, script_lock=False, action_idx=0, ram_array=ram)
    r_repeat = advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram)
    assert advisor.repeat_penalties == 1
    assert np.isclose(r_repeat, CFG.dialog_repeat_penalty)
    assert advisor.stats_dict()["advisor/repeat_penalties"] == 1

    advisor.reset()
    assert advisor.repeat_penalties == 0
    assert advisor.seen_dialogues == set()


def test_strategic_goal_embedding_and_parse():
    raw = 'Here is the plan:\n```json\n{"sub_goal": "Challenge Brock", "target_map_bank": 4, "target_map_id": 14, "priority": "battle_gym", "navigation_hint": "north"}\n```'
    goal = Strategist.parse_llm_response(raw)
    assert goal is not None
    assert goal.sub_goal == "Challenge Brock"
    assert goal.target_map_bank == 4
    assert goal.target_map_id == 14
    assert goal.priority == "battle_gym"
    assert goal.navigation_hint == "north"

    emb = goal.to_embedding()
    assert emb.shape == (GOAL_SIZE,)
    assert emb.dtype == np.float32
    assert np.allclose(emb[:4], [0.0, 0.0, 1.0, 0.0])  # battle_gym
    assert np.isclose(emb[4], 14.0 / 255.0)
    assert np.isclose(emb[5], 1.0 / 5.0)  # north


def test_strategist_ollama_integration_with_advisor(monkeypatch):
    advisor = RamAdvisor()
    ram = _ram_with_dialogue("OAK! Choose a POKeMON!")
    advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram)

    captured_requests = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode("utf-8"))
        captured_requests.append(body)
        resp_obj = {
            "response": json.dumps({
                "sub_goal": "Exit Oak Lab to Pallet Town",
                "target_map_bank": 3,
                "target_map_id": 0,
                "priority": "explore",
                "navigation_hint": "south",
            })
        }
        return FakeResp(json.dumps(resp_obj).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    strat = Strategist(advisor=advisor, model_name="qwen2.5:7b", async_mode=False)
    info_lab = {"map_bank": 4, "map_id": 2, "p1_lvl": 5, "p1_hp": 20, "p1_maxhp": 20}

    # Gatilho inicial (primeiro mapa)
    bonus = strat.update(info_lab, ram, in_battle=False, total_flags=10)
    assert bonus == 0.0
    assert strat.queries_count == 1
    assert strat.llm_success_count == 1
    assert len(captured_requests) == 1
    assert captured_requests[0]["model"] == "qwen2.5:7b"
    assert "oak! choose a pokemon!" in captured_requests[0]["prompt"]
    assert strat.current_goal.target_map_bank == 3
    assert strat.current_goal.target_map_id == 0

    # Segundo step no mesmo mapa sem evento -> não consulta LLM
    strat.update(info_lab, ram, in_battle=False, total_flags=10)
    assert strat.queries_count == 1

    # Alcançou o target_map (3, 0) -> recebe bônus de alinhamento (+3.0) e dispara novo gatilho
    info_pallet = {"map_bank": 3, "map_id": 0, "p1_lvl": 5, "p1_hp": 20, "p1_maxhp": 20}
    bonus_align = strat.update(info_pallet, ram, in_battle=False, total_flags=10)
    assert np.isclose(bonus_align, CFG.strategist_goal_bonus)
    assert strat.goal_alignments == 1
    assert strat.queries_count == 2

    # Gatilho por estagnação (500 steps no mesmo mapa)
    for _ in range(CFG.strategist_stagnation_steps):
        strat.update(info_pallet, ram, in_battle=False, total_flags=10)
    assert strat.queries_count == 3

    # Gatilho por nova Event Flag e mudança de batalha
    strat.update(info_pallet, ram, in_battle=False, total_flags=11)
    assert strat.queries_count == 4
    strat.update(info_pallet, ram, in_battle=True, total_flags=11)
    assert strat.queries_count == 5


def test_strategist_async_non_blocking_and_offline_fallback(monkeypatch):
    def slow_urlopen(req, timeout=None):
        time.sleep(0.15)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", slow_urlopen)

    strat = Strategist(advisor=RamAdvisor(), async_mode=True)
    info = {"map_bank": 3, "map_id": 0, "p1_lvl": 5, "p1_hp": 4, "p1_maxhp": 20}

    t0 = time.monotonic()
    strat.update(info, ram_array=None, in_battle=False, total_flags=0)
    elapsed = time.monotonic() - t0

    # Loop principal nunca deve bloquear aguardando Ollama
    assert elapsed < 0.05
    # Heurística imediata detecta HP < 30% -> prioridade heal
    assert strat.current_goal.priority == "heal"

    # Aguarda future terminar e limpa executor
    if strat._pending_future is not None:
        strat._pending_future.result(timeout=2.0)
    assert strat._ollama_cooldown_until > 0.0
    strat.close()


def test_strategist_oaks_parcel_chain_and_viridian_mart():
    import struct

    strat = Strategist(advisor=RamAdvisor(), async_mode=False)

    # Sem inicial em Pallet Town (3, 0) -> deve mirar Lab do Prof. Oak (4, 3)
    ctx_no_starter = strat.build_context(
        {"map_bank": 3, "map_id": 0, "p1_lvl": 0}, None, in_battle=False, total_flags=113
    )
    g_lab = strat._heuristic_goal(ctx_no_starter)
    assert (g_lab.target_map_bank, g_lab.target_map_id) == (4, 3)

    # Em Viridian City (3, 1) antes de pegar Oak's Parcel -> deve apontar Viridian Poke Mart (5, 3)
    ctx_viridian = strat.build_context(
        {"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24},
        None,
        in_battle=False,
        total_flags=122,
    )
    g_mart = strat._heuristic_goal(ctx_viridian)
    assert (g_mart.target_map_bank, g_mart.target_map_id) == (5, 3)

    # RAM simulada com Oak's Parcel (item_id=349, qty=1 em SaveBlock1 + 0x03B8)
    ram = np.zeros(262144 + 0x6000, dtype=np.uint8)
    sb1 = 0x025000
    struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1)
    struct.pack_into("<HH", ram, sb1 + 0x03B8, 349, 1)

    expected_chain = [
        ((5, 3), (3, 1), "south"),
        ((3, 1), (3, 19), "south"),
        ((3, 19), (3, 0), "south"),
        ((3, 0), (4, 3), "south"),
        ((4, 3), (4, 3), "interact"),
    ]
    for (cur_b, cur_id), (tgt_b, tgt_id), hint in expected_chain:
        ctx = strat.build_context(
            {"map_bank": cur_b, "map_id": cur_id, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24},
            ram,
            in_battle=False,
            total_flags=122,
        )
        assert ctx["has_oaks_parcel"] is True
        g = strat._heuristic_goal(ctx)
        assert (g.target_map_bank, g.target_map_id) == (tgt_b, tgt_id)
        assert g.navigation_hint == hint

    strat.close()


def test_vision_two_line_dialogue_detection():
    vision = VisionEngine()
    frame = np.full((160, 240, 3), 100, dtype=np.uint8)
    # Simula caixa de dialogo branca com ~20% de pixels de texto escuro (mean ~198)
    box = np.full((40, 224, 3), 245, dtype=np.uint8)
    box[8:16, 16:180] = 40
    box[22:30, 16:150] = 40
    frame[112:152, 8:232] = box
    assert vision.detect_state(frame) == GameState.DIALOGUE


def test_strategist_anti_echo_and_kanto_prompt_and_battle_skip(monkeypatch):
    class FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(json.loads(req.data.decode("utf-8"))["prompt"])
        return FakeResp(
            json.dumps(
                {
                    "response": json.dumps(
                        {
                            "sub_goal": "Explore Viridian",
                            "target_map_bank": 3,
                            "target_map_id": 1,
                            "priority": "explore",
                            "navigation_hint": "none",
                        }
                    )
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    strat = Strategist(advisor=RamAdvisor(), async_mode=False)
    info_vir = {"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24}

    strat.update(info_vir, ram_array=None, in_battle=False, total_flags=122)
    assert len(calls) == 1
    assert "Pallet=(3,0)" in calls[0]
    assert "ViridianMart=(5,3)" in calls[0]
    assert "target_map_bank=5, target_map_id=3" in calls[0]
    # Eco preguiçoso (3,1) deve preservar alvo (5,3) e hint 'east' da heuristica
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (5, 3)
    assert strat.current_goal.navigation_hint == "east"

    # Transicao pura de batalha no mesmo mapa usa heuristica sem chamar HTTP Ollama
    strat.update(info_vir, ram_array=None, in_battle=True, total_flags=122)
    assert strat.queries_count == 2
    assert len(calls) == 1
    assert strat.current_goal.navigation_hint == "interact"
    strat.close()


def test_advisor_anti_dialogue_freeze_and_vision_blue_text():
    advisor = RamAdvisor()
    ram1 = _ram_with_dialogue("OAK")
    ram2 = _ram_with_dialogue("You need your own POKeMON!")
    ram3 = _ram_with_dialogue("Here, choose a POKeMON!")

    # Abertura inicial da fala: sem bonus de avanco, dialogue_stuck_steps == 0
    r_open = advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram1)
    assert r_open == 0.0
    assert advisor.dialogue_stuck_steps == 0

    # Avancar para segunda fala inedita com A recompensa +dialogue_advance_bonus
    r_adv = advisor.update(in_battle=False, script_lock=True, action_idx=0, ram_array=ram2)
    assert np.isclose(r_adv, CFG.dialogue_advance_bonus)
    assert advisor.dialogue_stuck_steps == 0

    # Ficar travado 24 steps apertando UP (2) nao penaliza ainda; no step 25 aplica penalidade leve
    for _ in range(CFG.dialogue_stuck_grace - 1):
        r_wait = advisor.update(in_battle=False, script_lock=True, action_idx=2, ram_array=ram2)
        assert r_wait == 0.0
    r_stuck = advisor.update(in_battle=False, script_lock=True, action_idx=2, ram_array=ram2)
    assert np.isclose(r_stuck, CFG.dialogue_stuck_penalty)
    assert advisor.dialogue_stuck_steps == CFG.dialogue_stuck_grace

    # Mudar para nova fala reseta dialogue_stuck_steps
    r_next = advisor.update(in_battle=False, script_lock=True, action_idx=1, ram_array=ram3)
    assert np.isclose(r_next, CFG.dialogue_advance_bonus)
    assert advisor.dialogue_stuck_steps == 0

    # VisionEngine detecta dialogo com texto azul masculino (luminancia ~114 < 140)
    vision = VisionEngine()
    frame = np.full((160, 240, 3), 100, dtype=np.uint8)
    box = np.full((40, 224, 3), 245, dtype=np.uint8)
    box[8:20, 20:190] = (60, 110, 210)
    frame[112:152, 8:232] = box
    assert vision.detect_state(frame) == GameState.DIALOGUE


def test_strategist_schema_and_3b_guardrails(monkeypatch):
    captured = []

    class FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=None):
        captured.append(json.loads(req.data.decode("utf-8")))
        return FakeResp(
            json.dumps(
                {
                    "response": json.dumps(
                        {
                            "sub_goal": "none",
                            "target_map_bank": 0,
                            "target_map_id": 5,
                            "priority": "heal",
                            "navigation_hint": "east",
                        }
                    )
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    strat = Strategist(advisor=RamAdvisor(), async_mode=False)
    info_vir = {"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24}
    strat.update(info_vir, ram_array=None, in_battle=False, total_flags=122)

    assert len(captured) == 1
    assert captured[0]["format"] == Strategist.OLLAMA_RESPONSE_SCHEMA
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (5, 3)
    assert strat.current_goal.priority == "explore"
    assert strat.current_goal.sub_goal != "none"
    assert "Viridian Poke Mart" in strat.current_goal.sub_goal
    strat.close()


def test_strategist_early_game_bedroom_house_and_starter_goals():
    strat = Strategist(advisor=RamAdvisor(), async_mode=False)

    g_bed = strat._heuristic_goal(
        strat.build_context({"map_bank": 4, "map_id": 1, "p1_lvl": 0}, None, False, 0)
    )
    assert (g_bed.target_map_bank, g_bed.target_map_id) == (4, 0)
    assert g_bed.navigation_hint == "east"

    g_house = strat._heuristic_goal(
        strat.build_context({"map_bank": 4, "map_id": 0, "p1_lvl": 0}, None, False, 0)
    )
    assert (g_house.target_map_bank, g_house.target_map_id) == (3, 0)
    assert g_house.navigation_hint == "south"

    g_pallet_no_starter = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 0, "p1_lvl": 0}, None, False, 113)
    )
    assert (g_pallet_no_starter.target_map_bank, g_pallet_no_starter.target_map_id) == (4, 3)
    assert g_pallet_no_starter.navigation_hint == "north"

    g_lab_no_starter = strat._heuristic_goal(
        strat.build_context({"map_bank": 4, "map_id": 3, "p1_lvl": 0}, None, False, 114)
    )
    assert (g_lab_no_starter.target_map_bank, g_lab_no_starter.target_map_id) == (4, 3)
    assert g_lab_no_starter.navigation_hint == "interact"

    g_lab_with_starter = strat._heuristic_goal(
        strat.build_context(
            {"map_bank": 4, "map_id": 3, "p1_lvl": 5, "p1_hp": 20, "p1_maxhp": 20},
            None,
            False,
            116,
        )
    )
    assert (g_lab_with_starter.target_map_bank, g_lab_with_starter.target_map_id) == (3, 0)
    assert g_lab_with_starter.navigation_hint == "south"
    strat.close()


def test_strategist_viridian_routes_north_after_pokedex():
    import struct

    ram = np.zeros(262144 + 0x6000, dtype=np.uint8)
    sb1 = 0x025000
    struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1)
    # Marca FLAG_SYS_POKEDEX_GET (0x829 -> byte 0x0FE5 bit 0x02)
    ram[sb1 + 0x0FE5] = 0x02

    strat = Strategist(advisor=RamAdvisor(), async_mode=False)
    ctx = strat.build_context(
        {"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24},
        ram,
        in_battle=False,
        total_flags=124,
    )
    assert ctx["has_pokedex"] is True
    g = strat._heuristic_goal(ctx)
    assert (g.target_map_bank, g.target_map_id) == (3, 20)
    assert g.navigation_hint == "north"
    strat.close()


def test_vision_ignores_oaks_lab_floor_false_dialogue():
    vision = VisionEngine()
    frame = np.full((160, 240, 3), 100, dtype=np.uint8)
    # Simula piso claro do Lab com sombras/mobiliario (bright_ratio=0.580 < 0.65, dark_ratio=0.100, mean=197.8)
    box = np.full((40, 224, 3), 155, dtype=np.uint8)
    box[:, :130] = 240
    box[25:, 130:190] = 90
    frame[112:152, 8:232] = box
    assert vision.detect_state(frame) == GameState.OVERWORLD


def test_strategist_3b_hallucination_guardrails_viridian_exits_and_anti_bounce(monkeypatch):
    class FakeResp:
        def __init__(self, payload: bytes):
            self._buf = io.BytesIO(payload)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    canned_responses = iter([
        {"sub_goal": "Rest at Oak Lab", "target_map_bank": 4, "target_map_id": 4, "priority": "explore", "navigation_hint": "north"},
        {"sub_goal": "Rest at Oak Lab", "target_map_bank": 3, "target_map_id": 4, "priority": "explore", "navigation_hint": "none"},
        {"sub_goal": "Explore", "target_map_bank": 3, "target_map_id": 3, "priority": "explore", "navigation_hint": "north"},
    ])

    def fake_urlopen(req, timeout=None):
        return FakeResp(json.dumps({"response": json.dumps(next(canned_responses))}).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    strat = Strategist(advisor=RamAdvisor(), async_mode=False)

    # 1. Alucinacao (4,1) -> (4,4) north deve manter fallback (4,0) east
    strat.update({"map_bank": 4, "map_id": 1, "p1_lvl": 0}, None, in_battle=False, total_flags=111)
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (4, 0)
    assert strat.current_goal.navigation_hint == "east"

    # 2. Alucinacao (3,1) -> (3,4) none deve manter fallback (5,3) east
    strat.update({"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24}, None, in_battle=False, total_flags=122)
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (5, 3)
    assert strat.current_goal.navigation_hint == "east"

    # 3. Alucinacao (5,3) -> (3,3) north sem Parcel deve manter fallback (3,1) south
    strat.update({"map_bank": 5, "map_id": 3, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24}, None, in_battle=False, total_flags=122)
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (3, 1)
    assert strat.current_goal.navigation_hint == "south"

    # Heuristica de saida para Viridian Center (5,4) e interior de Pewter (6,1)
    g_vc = strat._heuristic_goal(strat.build_context({"map_bank": 5, "map_id": 4, "p1_lvl": 7, "p1_hp": 22, "p1_maxhp": 24}, None, False, 124))
    assert (g_vc.target_map_bank, g_vc.target_map_id) == (3, 1) and g_vc.navigation_hint == "south"
    g_pew = strat._heuristic_goal(strat.build_context({"map_bank": 6, "map_id": 1, "p1_lvl": 9, "p1_hp": 26, "p1_maxhp": 26}, None, False, 124))
    assert (g_pew.target_map_bank, g_pew.target_map_id) == (3, 2) and g_pew.navigation_hint == "south"

    # Anti-bounce de border: (3,19) -> (3,1) paga 1x; voltar a (3,19) e depois (3,1) paga 0
    strat.reset({"map_bank": 3, "map_id": 19, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24}, None)
    strat._ollama_cooldown_until = 1e18
    strat.update({"map_bank": 3, "map_id": 19, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24}, None, False, 122)
    b1 = strat.update({"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24}, None, False, 122)
    assert b1 == CFG.strategist_goal_bonus
    b_back = strat.update({"map_bank": 3, "map_id": 19, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24}, None, False, 122)
    b_revisit = strat.update({"map_bank": 3, "map_id": 1, "p1_lvl": 7, "p1_hp": 24, "p1_maxhp": 24}, None, False, 122)
    assert b_back == 0.0 and b_revisit == 0.0
    strat.close()


def test_route22_to_25_graph_and_context_aware_yes_no():
    strat = Strategist(advisor=RamAdvisor(), async_mode=False)

    # Route 22 (3, 41) sem 8 insignias deve retornar a leste para Viridian City (3, 1)
    g_r22_early = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 41, "p1_lvl": 8, "p1_hp": 20, "p1_maxhp": 26, "badges": 0}, None, False, 124)
    )
    assert (g_r22_early.target_map_bank, g_r22_early.target_map_id) == (3, 1)
    assert g_r22_early.navigation_hint == "east"

    # Route 22 (3, 41) com 8 insignias avanca a oeste para Route 23 (3, 42)
    g_r22_e4 = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 41, "p1_lvl": 55, "p1_hp": 180, "p1_maxhp": 180, "badges": 0xFF}, None, False, 300)
    )
    assert (g_r22_e4.target_map_bank, g_r22_e4.target_map_id) == (3, 42)
    assert g_r22_e4.navigation_hint == "west"

    # Cerulean (3,3) -> Route 24 (3,43) -> Route 25 (3,44) -> Cerulean (3,3)
    g_cer = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 3, "p1_lvl": 20, "p1_hp": 50, "p1_maxhp": 50}, None, False, 200)
    )
    assert (g_cer.target_map_bank, g_cer.target_map_id) == (3, 43)
    g_r24 = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 43, "p1_lvl": 20, "p1_hp": 50, "p1_maxhp": 50}, None, False, 200)
    )
    assert (g_r24.target_map_bank, g_r24.target_map_id) == (3, 44)
    g_r25 = strat._heuristic_goal(
        strat.build_context({"map_bank": 3, "map_id": 44, "p1_lvl": 20, "p1_hp": 50, "p1_maxhp": 50}, None, False, 200)
    )
    assert (g_r25.target_map_bank, g_r25.target_map_id) == (3, 3)
    strat.close()

    # RamAdvisor YES/NO contextual
    adv = RamAdvisor()
    ram_starter = _ram_with_dialogue("So! You want the fire POKeMON, CHARMANDER? YES NO")
    bonus_a_starter = adv.update(
        in_battle=False, script_lock=True, action_idx=adv.A_IDX,
        ram_array=ram_starter, party_level=0, map_key=(4, 3), party_hp_pct=1.0,
    )
    assert np.isclose(bonus_a_starter, adv.YES_NO_BONUS)
    assert np.isclose(adv.get_embedding()[1], adv.A_IDX / 5.0)

    # Nickname prompt com party_level=5 volta a recomendar B
    adv.reset()
    ram_nick = _ram_with_dialogue("Do you want to give a nickname? YES NO")
    bonus_b_nick = adv.update(
        in_battle=False, script_lock=True, action_idx=adv.B_IDX,
        ram_array=ram_nick, party_level=5, map_key=(4, 3), party_hp_pct=1.0,
    )
    assert np.isclose(bonus_b_nick, adv.YES_NO_BONUS)
    assert np.isclose(adv.get_embedding()[1], adv.B_IDX / 5.0)

    # Nurse Joy com HP baixo (<80%) recomenda A; com HP 100% volta a recomendar B
    adv.reset()
    ram_heal = _ram_with_dialogue("Would you like to rest your tired POKeMON? YES NO")
    bonus_a_heal = adv.update(
        in_battle=False, script_lock=True, action_idx=adv.A_IDX,
        ram_array=ram_heal, party_level=8, map_key=(5, 4), party_hp_pct=0.30,
    )
    assert np.isclose(bonus_a_heal, adv.YES_NO_BONUS)
    assert np.isclose(adv.get_embedding()[1], adv.A_IDX / 5.0)

    bonus_a_full = adv.update(
        in_battle=False, script_lock=True, action_idx=adv.A_IDX,
        ram_array=ram_heal, party_level=8, map_key=(5, 4), party_hp_pct=1.0,
    )
    assert bonus_a_full == 0.0
    assert np.isclose(adv.get_embedding()[1], adv.B_IDX / 5.0)


def test_strategist_empty_oaks_lab_vs_oak_present():
    import struct

    ram = np.zeros(262144 + 0x6000, dtype=np.uint8)
    sb1 = 0x025000
    struct.pack_into("<I", ram, 262144 + 0x5008, 0x02000000 | sb1)
    # 0x02B ON -> Oak nao esta no Lab ainda
    ram[sb1 + 0x0EE0 + (0x02B >> 3)] = 1 << (0x02B & 7)

    strat = Strategist(advisor=RamAdvisor(), async_mode=False)
    strat._ollama_cooldown_until = 1e18

    # Em Pallet (3,0) alvo inicial e (4,3)
    strat.reset({"map_bank": 3, "map_id": 0, "p1_lvl": 0}, ram)
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (4, 3)

    # Entrar no Lab vazio (0x02B ON) nao paga alinhamento e manda sair ao sul
    b_empty = strat.update({"map_bank": 4, "map_id": 3, "p1_lvl": 0}, ram, in_battle=False, total_flags=112)
    assert b_empty == 0.0
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (3, 0)
    assert strat.current_goal.navigation_hint == "south"

    # Sair para Pallet (3,0) e entrar no Lab apos Oak aparecer (0x02B OFF) paga alinhamento e manda interagir
    strat.update({"map_bank": 3, "map_id": 0, "p1_lvl": 0}, ram, in_battle=False, total_flags=113)
    ram[sb1 + 0x0EE0 + (0x02B >> 3)] = 0
    b_real = strat.update({"map_bank": 4, "map_id": 3, "p1_lvl": 0}, ram, in_battle=False, total_flags=114)
    assert np.isclose(b_real, CFG.strategist_goal_bonus)
    assert (strat.current_goal.target_map_bank, strat.current_goal.target_map_id) == (4, 3)
    assert strat.current_goal.navigation_hint == "interact"
    strat.close()
