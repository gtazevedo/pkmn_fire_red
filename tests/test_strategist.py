import io
import json
import time
import urllib.error
import numpy as np

from src.advisor import RamAdvisor
from src.config import CFG, GOAL_SIZE
from src.strategist import StrategicGoal, Strategist


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
