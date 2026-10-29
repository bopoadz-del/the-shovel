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
280,320. G4 (D529.3) stays Rate Only.
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
