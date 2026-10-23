"""WAVE 2 B4/B5: compose priced CESMM qty + amount when synthesis hangs.

Live Master Corpus on 2ceef76 (#545) asked B4:

    Answer only from the client project documents. What is the quantity
    and amount for breaking out existing carriageway including road
    markings (D599.5)?

#547 composed D599.5 (340,904 m2 @ 31.00 = 10,568,024) from the
isolated window. Live REAL SHIP #547 tip d94b740 then FAILed B5:

    Answer only from the client project documents. What is the amount
    for removal of existing chain link fence (D549.2)?

Ship-pack OCR is ``3,504 m @ SAR 80.00 = SAR 280,320.00`` — a currency
token the B4 triple regex did not accept. Expected: 3,504 @ 80.00 =
280,320. G4 (D529.3) stays Rate Only. Kill-switch:
COMPOSE_PRICED_BOQ_ROW=0 restores the empty hang.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import (
    Agent,
    _CG_REFUSAL,
    _EMPTY_RESPONSE_FALLBACK,
    _compose_priced_boq_instead_of_retry,
    _final_text_needs_forced_retry,
    _graft_priced_boq_item,
    _graft_rate_only_item,
    _postprocess_answer,
    _should_short_circuit_priced_boq,
)
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.retriever import (
    answer_states_priced_boq,
    chunk_states_excluded_item,
    chunk_states_priced_item,
    chunk_states_rate_only_item,
    compose_priced_boq_row,
    format_priced_boq_line,
    priced_boq_compose_enabled,
)
from app.core.rag.vector_store import Chunk
from tests._fixture_values import STUB_TOKEN_02


LIVE_PREFIX = "Answer only from the client project documents. "
B4_ASK = (
    "What is the quantity and amount for breaking out existing "
    "carriageway including road markings (D599.5)?"
)
B5_ASK = (
    "What is the amount for removal of existing chain link fence (D549.2)?"
)
G4_ASK = (
    "What is the total amount for removal of storm water culverts (D529.3)?"
)
LIVE_B4 = LIVE_PREFIX + B4_ASK
LIVE_B5 = LIVE_PREFIX + B5_ASK
LIVE_G4 = LIVE_PREFIX + G4_ASK

B4_QTY = "340904"
B4_RATE = "31.00"
B4_AMT = "10568024"
B5_QTY = "3504"
B5_RATE = "80.00"
B5_AMT = "280320"

SOUP = (
    "D529.3 Removal of storm water culverts — m 1,370.00 Rate Only "
    f"D549.2 Removal of existing chain link fence {B5_QTY} m {B5_RATE} "
    f"{B5_AMT}.00 "
    "D599.5 Breaking out existing carriageway including road markings "
    f"{B4_QTY} m2 {B4_RATE} {B4_AMT}"
)
OCR_SPACED = (
    "D 529.3 Removal of storm water culverts — m 1,370.00 Rate Only "
    f"D 549.2 Removal of existing chain link fence 3,504 m {B5_RATE} "
    "280,320.00 "
    "D 599.5 Breaking out existing carriageway including road markings "
    "340,904 m2 31.00 10,568,024"
)
PIPE_SOUP = (
    "| D529.3 | Removal of storm water culverts | — | m | 1,370.00 "
    "| Rate Only | D549.2 | Removal of existing chain link fence | "
    f"{B5_QTY} | m | {B5_RATE} | {B5_AMT} | D599.5 | Breaking out "
    f"existing carriageway including road markings | {B4_QTY} | m2 | "
    f"{B4_RATE} | {B4_AMT} |"
)
EXCLUDED_CULVERT = (
    "J |Breakout and remove existing storm water culverts D599.5 "
    "| sum 1 Excluded"
)
SEARCH_PROMISE = (
    "Let me search the demolition BOQ more specifically for D599.5 "
    "quantity and amount."
)
B5_SEARCH_PROMISE = (
    "Let me search the demolition BOQ more specifically for D549.2 "
    "amount."
)
QUESTION_ECHO = LIVE_B4
B5_QUESTION_ECHO = LIVE_B5
# Live ship-pack OCR (rag_inject identifier-retrieval fixture).
LIVE_B5_SAR = (
    "D 549.2 Removal of existing chain link fence 3,504 m @ SAR 80.00 "
    "= SAR 280,320.00"
)
LIVE_B4_SAR = (
    "D599.5 Breaking out existing carriageway including road markings "
    "340,904 m2 @ SAR 31.00 = SAR 10,568,024"
)
MIXED_SAR_SOUP = (
    "D529.3 Removal of storm water culverts — m 1,370.00 Rate Only "
    + LIVE_B5_SAR
    + " "
    + LIVE_B4_SAR
)
CHROME_FLUFF = (
    "Answer only from the client project documents. I will look through "
    "the retrieved excerpts for this item."
)
STORM_WATER = "removal of storm water culverts"


def _sys(*texts: str) -> dict:
    blocks = [
        f"[doc_id=soup chunk={i} score=0.90] {t}"
        for i, t in enumerate(texts)
    ]
    return {"role": "system", "content": "\n\n".join(blocks)}


def _msgs(ask: str) -> list[dict]:
    return [{"role": "user", "content": ask}]


def _has_b4_figures(text: str) -> bool:
    blob = (text or "").replace(",", "")
    return B4_QTY in blob and B4_AMT in blob


def _has_b5_figures(text: str) -> bool:
    blob = (text or "").replace(",", "")
    return B5_AMT in blob


def test_compose_parses_priced_d599_5_from_soup():
    for blob in (SOUP, OCR_SPACED, PIPE_SOUP):
        parsed = compose_priced_boq_row(LIVE_B4, blob)
        assert parsed, blob
        assert _plain(parsed["qty"]) == B4_QTY
        assert _plain(parsed["amount"]) == B4_AMT
        assert "carriageway" in (parsed.get("description") or "").lower()


def test_compose_parses_priced_d549_2_from_soup():
    parsed = compose_priced_boq_row(LIVE_B5, SOUP)
    assert parsed
    assert _plain(parsed["amount"]) == B5_AMT
    assert _plain(parsed["qty"]) == B5_QTY


def test_compose_parses_live_b5_sar_ocr():
    """Ship-pack OCR: 3,504 m @ SAR 80.00 = SAR 280,320.00."""
    parsed = compose_priced_boq_row(LIVE_B5, LIVE_B5_SAR)
    assert parsed, LIVE_B5_SAR
    assert _plain(parsed["qty"]) == B5_QTY
    assert abs(parsed["rate"] - 80.0) < 1e-9
    assert _plain(parsed["amount"]) == B5_AMT
    assert "chain link" in (parsed.get("description") or "").lower()


def test_compose_still_parses_b4_with_currency_prefix():
    """Do not weaken #547 — B4 still composes when OCR prints SAR."""
    parsed = compose_priced_boq_row(LIVE_B4, LIVE_B4_SAR)
    assert parsed
    assert _plain(parsed["qty"]) == B4_QTY
    assert _plain(parsed["amount"]) == B4_AMT


def test_compose_does_not_invent_from_rate_only_or_excluded():
    assert compose_priced_boq_row(LIVE_G4, SOUP) is None
    assert compose_priced_boq_row(LIVE_B4, EXCLUDED_CULVERT) is None
    # Currency-enabled lookalike: do not treat Excluded sum 1 as 1 × 80.
    priced_excluded = (
        "J |Breakout and remove existing storm water culverts D599.5 "
        "| sum 1 Excluded @ SAR 80.00 = SAR 80.00"
    )
    assert compose_priced_boq_row(LIVE_B4, priced_excluded) is None


def test_kill_switch_restores_the_empty_hang(monkeypatch):
    monkeypatch.setenv("COMPOSE_PRICED_BOQ_ROW", "0")
    assert priced_boq_compose_enabled() is False
    assert compose_priced_boq_row(LIVE_B4, SOUP) is None
    assert compose_priced_boq_row(LIVE_B5, LIVE_B5_SAR) is None
    rag = _sys(SOUP)
    out = _graft_priced_boq_item("", rag, _msgs(LIVE_B4))
    assert out == ""
    assert not _has_b4_figures(out)
    b5 = _graft_priced_boq_item("", _sys(LIVE_B5_SAR), _msgs(LIVE_B5))
    assert b5 == ""
    assert not _has_b5_figures(b5)


def _plain(value: float) -> str:
    return str(int(round(value)))


def test_graft_replaces_empty_question_echo_and_search_promise():
    rag = _sys(SOUP)
    for hung in ("", QUESTION_ECHO, SEARCH_PROMISE, CHROME_FLUFF,
                 _EMPTY_RESPONSE_FALLBACK, _CG_REFUSAL,
                 "I'm ready to help. Please let me know what you need."):
        out = _graft_priced_boq_item(hung, rag, _msgs(LIVE_B4))
        assert _has_b4_figures(out), hung
        assert STORM_WATER not in out.lower()
        assert "Rate Only" not in out


def test_graft_keeps_an_already_priced_answer():
    rag = _sys(SOUP)
    already = format_priced_boq_line(compose_priced_boq_row(LIVE_B4, SOUP))
    assert _graft_priced_boq_item(already, rag, _msgs(LIVE_B4)) == already


def test_priced_b5_drops_rate_only_label_when_amount_is_present():
    """B5 D549.2 has an amount. Rate Only on that line is the wrong half."""
    rag = _sys(LIVE_B5_SAR)
    parsed = compose_priced_boq_row(LIVE_B5, LIVE_B5_SAR)
    contradictory = (
        "D549.2 Removal of existing chain link fence is Rate Only. "
        "Quantity 3,504 m @ 80.00 = 280,320."
    )
    assert parsed
    assert answer_states_priced_boq(contradictory, parsed) is False
    out = _graft_priced_boq_item(contradictory, rag, _msgs(LIVE_B5))
    assert _has_b5_figures(out)
    assert "Rate Only" not in out


def test_graft_replaces_storm_water_rate_only_misroute():
    rag = _sys(SOUP)
    wrong = "D599.5 (removal of storm water culverts) is Rate Only."
    out = _graft_priced_boq_item(wrong, rag, _msgs(LIVE_B4))
    assert _has_b4_figures(out)
    assert STORM_WATER not in out.lower()


def test_postprocess_empty_hang_states_b4_figures():
    rag = _sys(SOUP)
    out = _postprocess_answer("", rag, _msgs(LIVE_B4))
    assert _has_b4_figures(out)
    assert out != _CG_REFUSAL
    assert STORM_WATER not in out.lower()


def test_postprocess_question_echo_states_b4_figures():
    rag = _sys(SOUP)
    out = _postprocess_answer(QUESTION_ECHO, rag, _msgs(LIVE_B4))
    assert _has_b4_figures(out)
    assert STORM_WATER not in out.lower()


def test_postprocess_b5_empty_still_states_fence_amount():
    rag = _sys(SOUP)
    out = _postprocess_answer("", rag, _msgs(LIVE_B5))
    assert _has_b5_figures(out)
    assert STORM_WATER not in out.lower()


def test_postprocess_g4_still_states_rate_only_not_priced():
    rag = _sys(SOUP)
    out = _postprocess_answer(
        "I'm ready to help. Please let me know what you need.",
        rag, _msgs(LIVE_G4),
    )
    assert "Rate Only" in out
    assert "D529.3" in out
    assert B4_AMT not in out.replace(",", "")
    assert B5_AMT not in out.replace(",", "")


def test_g4_fence_on_mixed_sar_soup():
    """G4 stays Rate Only when the same page has priced SAR D549.2 / D599.5."""
    assert compose_priced_boq_row(LIVE_G4, MIXED_SAR_SOUP) is None
    parsed_b5 = compose_priced_boq_row(LIVE_B5, MIXED_SAR_SOUP)
    assert parsed_b5 and _plain(parsed_b5["amount"]) == B5_AMT
    parsed_b4 = compose_priced_boq_row(LIVE_B4, MIXED_SAR_SOUP)
    assert parsed_b4 and _plain(parsed_b4["amount"]) == B4_AMT
    rag = _sys(MIXED_SAR_SOUP)
    g4 = _postprocess_answer(
        "I'm ready to help. Please let me know what you need.",
        rag, _msgs(LIVE_G4),
    )
    assert "Rate Only" in g4
    assert "D529.3" in g4
    assert B5_AMT not in g4.replace(",", "")
    assert B4_AMT not in g4.replace(",", "")
    b5 = _postprocess_answer("", rag, _msgs(LIVE_B5))
    assert _has_b5_figures(b5)
    assert STORM_WATER not in b5.lower()
    assert "Rate Only" not in b5


def test_retry_skip_composes_instead_of_another_llm_hop():
    rag = _sys(SOUP)
    assert _final_text_needs_forced_retry(SEARCH_PROMISE, user_message=LIVE_B4)
    filled = _compose_priced_boq_instead_of_retry(
        SEARCH_PROMISE, rag, _msgs(LIVE_B4),
    )
    assert _has_b4_figures(filled)
    parsed = compose_priced_boq_row(LIVE_B4, SOUP)
    assert answer_states_priced_boq(filled, parsed)


def test_retry_skip_is_off_when_kill_switch(monkeypatch):
    monkeypatch.setenv("COMPOSE_PRICED_BOQ_ROW", "0")
    rag = _sys(SOUP)
    assert _compose_priced_boq_instead_of_retry(
        SEARCH_PROMISE, rag, _msgs(LIVE_B4),
    ) == ""


def test_inject_fires_priced_header_on_b4_not_g4():
    chunks = [
        Chunk(
            chunk_id="soup", project_id="p_master", doc_id="soup",
            chunk_index=0, text=SOUP, score=0.9,
        )
    ]
    b4 = format_chunks_as_system_message(chunks, 4, query=LIVE_B4)["content"]
    b5 = format_chunks_as_system_message(chunks, 4, query=LIVE_B5)["content"]
    g4 = format_chunks_as_system_message(chunks, 4, query=LIVE_G4)["content"]
    assert "PRICED BOQ ROW" in b4
    assert "PRICED BOQ ROW" in b5
    assert "RATE ONLY" not in b4
    assert "RATE ONLY" not in b5
    assert "RATE ONLY" in g4
    assert "PRICED BOQ ROW" not in g4
    assert B4_QTY in b4
    assert B5_AMT in b5
    sar = format_chunks_as_system_message(
        [
            Chunk(
                chunk_id="sar", project_id="p_master", doc_id="sar",
                chunk_index=0, text=LIVE_B5_SAR, score=0.9,
            )
        ],
        4, query=LIVE_B5,
    )["content"]
    assert "PRICED BOQ ROW" in sar
    assert B5_AMT in sar.replace(",", "")


def test_rate_only_graft_still_wins_for_g4_on_soup():
    rag = _sys(SOUP)
    out = _graft_rate_only_item(
        "I'm ready to help. Please let me know what you need.",
        rag, _msgs(LIVE_G4),
    )
    assert "Rate Only" in out
    priced = _graft_priced_boq_item(out, rag, _msgs(LIVE_G4))
    assert priced == out


def test_short_circuit_skips_provider_when_priced_row_is_in_rag():
    rag = _sys(SOUP)
    out = _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B4), has_predispatch=False,
    )
    assert _has_b4_figures(out)
    assert STORM_WATER not in out.lower()
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B4), has_predispatch=True,
    ) == ""
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_G4), has_predispatch=False,
    ) == ""


def test_short_circuit_kill_switch_restores_provider_hop(monkeypatch):
    monkeypatch.setenv("COMPOSE_PRICED_BOQ_ROW", "0")
    rag = _sys(SOUP)
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B4), has_predispatch=False,
    ) == ""


@pytest.mark.asyncio
async def test_chat_short_circuits_b4_without_calling_llm(monkeypatch):
    """Live 2ceef76 retest: provider died with the unavailable banner.

    When RAG already has priced D599.5, do not call the LLM. Health can
    stay 49/49 while OpenRouter is owner-gated — compose from excerpts.
    """
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True,
    )
    calls = {"n": 0}

    async def fake_call_llm(*_a, **_k):
        calls["n"] += 1
        return {
            "status": "error",
            "error": "connection refused",
        }

    rag = _sys(SOUP)
    audit = {
        "project_id": "p_master",
        "chunks": [
            {
                "doc_id": "soup",
                "chunk_index": 0,
                "chunk_id": "p_master:soup:0",
                "score": 0.90,
            }
        ],
    }

    def fake_rag_inject(**_k):
        return rag, audit

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant.",
        allowed_blocks=[],
    )
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)

    result = await agent.chat(LIVE_B4, project_id="p_master")
    assert calls["n"] == 0
    assert result["status"] == "success"
    assert _has_b4_figures(result["answer"])
    assert STORM_WATER not in result["answer"].lower()
    assert result["iterations"] == 0


@pytest.mark.asyncio
async def test_chat_stream_short_circuits_b4_without_calling_llm(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True,
    )
    calls = {"n": 0}

    async def fake_call_llm(*_a, **_k):
        calls["n"] += 1
        raise ConnectionError("connection refused")

    rag = _sys(SOUP)
    audit = {
        "project_id": "p_master",
        "chunks": [
            {
                "doc_id": "soup",
                "chunk_index": 0,
                "chunk_id": "p_master:soup:0",
                "score": 0.90,
            }
        ],
    }
    monkeypatch.setattr(
        "app.agents.runtime.rag_inject",
        lambda **_k: (rag, audit),
    )
    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant.",
        allowed_blocks=[],
    )
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    events = []
    async for evt in agent.chat_stream(LIVE_B4, project_id="p_master"):
        events.append(evt)
    assert calls["n"] == 0
    tokens = "".join(
        e.get("content") or "" for e in events if e.get("type") == "token"
    )
    end = next(e for e in events if e.get("type") == "end")
    assert _has_b4_figures(tokens) or _has_b4_figures(end.get("content") or "")
    assert not any(e.get("type") == "error" for e in events)


def test_graft_replaces_b5_empty_echo_and_search_promise():
    rag = _sys(LIVE_B5_SAR)
    for hung in ("", B5_QUESTION_ECHO, B5_SEARCH_PROMISE,
                 _EMPTY_RESPONSE_FALLBACK, _CG_REFUSAL,
                 "I'm ready to help. Please let me know what you need."):
        out = _graft_priced_boq_item(hung, rag, _msgs(LIVE_B5))
        assert _has_b5_figures(out), hung
        assert STORM_WATER not in out.lower()
        assert "Rate Only" not in out


def test_short_circuit_skips_provider_when_b5_priced_row_is_in_rag():
    rag = _sys(LIVE_B5_SAR)
    out = _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B5), has_predispatch=False,
    )
    assert _has_b5_figures(out)
    assert STORM_WATER not in out.lower()
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B5), has_predispatch=True,
    ) == ""
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_G4), has_predispatch=False,
    ) == ""


def test_short_circuit_b5_kill_switch_restores_provider_hop(monkeypatch):
    monkeypatch.setenv("COMPOSE_PRICED_BOQ_ROW", "0")
    rag = _sys(LIVE_B5_SAR)
    assert _should_short_circuit_priced_boq(
        rag, _msgs(LIVE_B5), has_predispatch=False,
    ) == ""
    assert _compose_priced_boq_instead_of_retry(
        B5_SEARCH_PROMISE, rag, _msgs(LIVE_B5),
    ) == ""


@pytest.mark.asyncio
async def test_chat_short_circuits_b5_without_calling_llm(monkeypatch):
    """Same provider-unavailable fix as #547 B4, for D549.2."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True,
    )
    calls = {"n": 0}

    async def fake_call_llm(*_a, **_k):
        calls["n"] += 1
        return {
            "status": "error",
            "error": "connection refused",
        }

    rag = _sys(LIVE_B5_SAR)
    audit = {
        "project_id": "p_master",
        "chunks": [
            {
                "doc_id": "soup",
                "chunk_index": 0,
                "chunk_id": "p_master:soup:0",
                "score": 0.90,
            }
        ],
    }

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant.",
        allowed_blocks=[],
    )
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "app.agents.runtime.rag_inject",
        lambda **_k: (rag, audit),
    )

    result = await agent.chat(LIVE_B5, project_id="p_master")
    assert calls["n"] == 0
    assert result["status"] == "success"
    assert _has_b5_figures(result["answer"])
    assert STORM_WATER not in result["answer"].lower()
    assert result["iterations"] == 0


# Live WAVE 2 B5 on be93dee: Rate Only (PART NR. 3) vs Excluded
# (priced BOQ …53-0000-EXC-BOQ-CA-000007-B) suppressed 280,320.
PART_NR_3_PRICED = (
    "PART NR. 3 DEMOLITION\n"
    "D529.3 Removal of storm water culverts — m 1,370.00 Rate Only\n"
    "D 549.2 Removal of existing chain link fence 3,504 m @ SAR 80.00 "
    "= SAR 280,320.00\n"
)
RATE_ONLY_D549_SIBLING = (
    "PART NR. 3\n"
    "D549.2 Removal of existing chain link fence — m Rate Only\n"
)
EXCLUDED_D549_SIBLING = (
    "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B Bill of Quantities (Priced)\n"
    "D549.2 Removal of existing chain link fence | sum 1 Excluded\n"
)
STAINED_PART_NR_3 = (
    "PART NR. 3 DEMOLITION Rate Only items listed separately. "
    "D 549.2 Removal of existing chain link fence 3,504 m @ SAR 80.00 "
    "= SAR 280,320.00 Rate Only notes continue.\n"
)
CONFLICT_REFUSE = (
    "The retrieved documents conflict: PART NR. 3 states Rate Only "
    "while the priced BOQ QZ-ENG-310-0000-EXC-BOQ-CA-000007-B states "
    "Excluded. I cannot confirm the amount."
)
B5_SIBLING_RAG = _sys(
    PART_NR_3_PRICED, RATE_ONLY_D549_SIBLING, EXCLUDED_D549_SIBLING,
)


def test_compose_prefers_priced_part_nr_3_over_rate_only_and_excluded():
    blob = "\n".join(
        (PART_NR_3_PRICED, RATE_ONLY_D549_SIBLING, EXCLUDED_D549_SIBLING),
    )
    parsed = compose_priced_boq_row(LIVE_B5, blob)
    assert parsed, blob
    assert _plain(parsed["qty"]) == B5_QTY
    assert abs(parsed["rate"] - 80.0) < 1e-9
    assert _plain(parsed["amount"]) == B5_AMT
    assert "chain link" in (parsed.get("description") or "").lower()
    assert chunk_states_priced_item(PART_NR_3_PRICED, ["d549.2"])
    assert chunk_states_rate_only_item(RATE_ONLY_D549_SIBLING, ["d549.2"])
    assert chunk_states_excluded_item(EXCLUDED_D549_SIBLING, ["d549.2"])
    assert not chunk_states_rate_only_item(PART_NR_3_PRICED, ["d549.2"])
    assert not chunk_states_excluded_item(PART_NR_3_PRICED, ["d549.2"])


def test_stained_part_nr_3_window_still_composes_280320():
    """Rate Only page notes must not hide the priced D549.2 triple."""
    parsed = compose_priced_boq_row(LIVE_B5, STAINED_PART_NR_3)
    assert parsed
    assert _plain(parsed["amount"]) == B5_AMT
    assert not chunk_states_rate_only_item(STAINED_PART_NR_3, ["d549.2"])
    assert chunk_states_priced_item(STAINED_PART_NR_3, ["d549.2"])


def test_rate_only_plus_excluded_conflict_does_not_suppress_priced_amount():
    """Live be93dee refuse: Rate Only vs Excluded must not hide 280,320."""
    for hung in ("", CONFLICT_REFUSE, _CG_REFUSAL, B5_QUESTION_ECHO):
        out = _graft_priced_boq_item(hung, B5_SIBLING_RAG, _msgs(LIVE_B5))
        assert _has_b5_figures(out), hung
        assert "Rate Only" not in out
        assert "Excluded" not in out
        assert "cannot confirm" not in out.lower()
    assert _graft_rate_only_item(
        CONFLICT_REFUSE, B5_SIBLING_RAG, _msgs(LIVE_B5),
    ) == CONFLICT_REFUSE
    post = _postprocess_answer(CONFLICT_REFUSE, B5_SIBLING_RAG, _msgs(LIVE_B5))
    assert _has_b5_figures(post)
    assert "cannot confirm" not in post.lower()
    g4 = _postprocess_answer(
        "I'm ready to help. Please let me know what you need.",
        B5_SIBLING_RAG, _msgs(LIVE_G4),
    )
    assert "Rate Only" in g4
    assert "D529.3" in g4
    assert B5_AMT not in g4.replace(",", "")


def test_inject_prefers_priced_header_when_rate_only_and_excluded_siblings():
    chunks = [
        Chunk(
            chunk_id="part3", project_id="p_master", doc_id="part3",
            chunk_index=0, text=PART_NR_3_PRICED, score=0.4,
        ),
        Chunk(
            chunk_id="ro", project_id="p_master", doc_id="ro",
            chunk_index=0, text=RATE_ONLY_D549_SIBLING, score=0.9,
        ),
        Chunk(
            chunk_id="ex", project_id="p_master", doc_id="ex",
            chunk_index=0, text=EXCLUDED_D549_SIBLING, score=0.85,
        ),
    ]
    b5 = format_chunks_as_system_message(chunks, 4, query=LIVE_B5)["content"]
    assert "PRICED BOQ ROW" in b5
    assert "RATE ONLY" not in b5
    assert B5_AMT in b5.replace(",", "")
    g4 = format_chunks_as_system_message(chunks, 4, query=LIVE_G4)["content"]
    assert "RATE ONLY" in g4
    assert "PRICED BOQ ROW" not in g4
