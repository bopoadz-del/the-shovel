"""Cost-grounding gate — a chat cost/rate figure must trace to a rate-semantic
retrieved chunk or a computed tool result, else the answer is refused.

Anchored on the 2026-07-14 live incident: a cost query answered "450 SAR/m³"
by lifting "450" out of a the client project drawing dimension-table chunk. The gate MUST fail
that answer even though "450" is present in retrieval (it is NOT present in rate
semantics), and MUST pass a real rate answer grounded in the rates chunk.
"""
from __future__ import annotations

from app.agents.runtime import _cost_grounding_gate, _CG_REFUSAL


def _sys(*chunk_texts: str) -> dict:
    """Build a RAG system message the way format_chunks_as_system_message does:
    each chunk prefixed with a [doc_id=... chunk=N ...] marker."""
    body = "\n\n".join(
        f"[doc_id=doc{i} chunk={i} score=0.80] {t}" for i, t in enumerate(chunk_texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


# ── The preserved live incident (must FAIL the gate) ────────────────────────
# Verbatim shape of the retrieved the client project drawing dimension-table chunk: pure
# numbers + bend/pipe labels. "450" IS in this text — but not as a rate.
_DRAWING_SOUP = (
    "700 100 100 100 100 620 450 350 300 1960 11950 5350 1250 1850 9650 4300 "
    "1050 1820 8450 3750 950 11.25 BEND 22.5 BEND 45 BEND TEE H mm A mm B C "
    "200 500 500 900 1250 3400 2000 450 900 2100 1700 250 600 1400 1350"
)
_FABRICATED_ANSWER = (
    "The unit rate for ready-mix concrete (250 kg/cm2) in Saudi Arabia is "
    "450 SAR per m3.\n\nSource: construction_rates_gulf_ksa_2025.md, chunk 52."
)


def test_incident_fabrication_is_refused_even_though_number_is_in_retrieval():
    out = _cost_grounding_gate(_FABRICATED_ANSWER, _sys(_DRAWING_SOUP), messages=[])
    # 450 is present in the drawing soup, but not in rate semantics -> refuse.
    assert out == _CG_REFUSAL
    assert "450" not in out


# ── The real rate answer (must PASS) ────────────────────────────────────────
_RATE_CHUNK = (
    "## Detailed material prices — Saudi Arabia (SAR) | Material | Unit | Price "
    "range (SAR) | Ready-mix concrete (250 kg/cm2) | /m3 | 220 — 280 | "
    "Source: Turner & Townsend KSAMI 2025."
)


def test_real_rate_answer_grounded_in_rate_chunk_passes():
    answer = (
        "Ready-mix concrete (250 kg/cm2) in Saudi Arabia is about 220–280 SAR/m3 "
        "(indicative; Turner & Townsend KSAMI 2025)."
    )
    out = _cost_grounding_gate(answer, _sys(_RATE_CHUNK), messages=[])
    assert out == answer  # untouched — figures trace to the rate-semantic chunk


def test_rate_chunk_present_but_wrong_number_still_refused():
    # The rate chunk grounds 220–280; an answer claiming 450 is still ungrounded.
    answer = "The rate is 450 SAR/m3."
    out = _cost_grounding_gate(answer, _sys(_RATE_CHUNK, _DRAWING_SOUP), messages=[])
    assert out == _CG_REFUSAL


def test_non_cost_answer_is_never_touched():
    answer = "The project has 12 work packages and 3 open RFIs across 2 zones."
    out = _cost_grounding_gate(answer, _sys(_DRAWING_SOUP), messages=[])
    assert out == answer  # no money/rate figure -> passes through


def test_tool_computed_cost_is_grounded():
    # A real cost_estimate tool result is authoritative: its numbers ground the
    # prose even without a RAG rate chunk (prevents false refusal of deliverables).
    answer = "Estimated total cost is SAR 1,250,000 for the concrete package."
    tool_msgs = [{"role": "tool", "content": '{"total_cost": 1250000, "currency": "SAR"}'}]
    out = _cost_grounding_gate(answer, None, messages=tool_msgs)
    assert out == answer


def test_dimension_not_mistaken_for_rate():
    # "450 mm" in the answer is a dimension, not a rate -> not a money figure ->
    # never triggers the gate.
    answer = "Provide a 450 mm pipe with 200 mm bedding."
    out = _cost_grounding_gate(answer, _sys(_DRAWING_SOUP), messages=[])
    assert out == answer


def test_magnitude_suffix_matches_raw_tool_number():
    # Tool returns 1250000; model writes "SAR 1.25 million" — must PASS, not be
    # false-refused (Codex #223 magnitude-suffix finding).
    answer = "The estimated total cost is SAR 1.25 million for the package."
    tool_msgs = [{"role": "tool", "content": '{"total_cost": 1250000}'}]
    out = _cost_grounding_gate(answer, None, messages=tool_msgs)
    assert out == answer


def test_magnitude_suffix_still_refuses_ungrounded():
    # A magnitude-suffixed figure with nothing to ground it is still refused.
    answer = "The cost is USD 2 million."
    out = _cost_grounding_gate(answer, None, messages=[])
    assert out == _CG_REFUSAL


def test_magnitude_in_rate_chunk_grounds_repeated_answer():
    # The retrieved rate chunk itself says "SAR 1.25 million"; an answer that
    # repeats it must PASS even with no tool emitting the raw number (Codex #226
    # — grounded side must fold magnitude symmetrically with the answer side).
    chunk = "Contract sum for the package: SAR 1.25 million (priced BOQ)."
    answer = "The package cost is SAR 1.25 million per the priced BOQ."
    out = _cost_grounding_gate(answer, _sys(chunk), messages=[])
    assert out == answer


def test_gate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("COST_GROUNDING_GATE", "0")
    out = _cost_grounding_gate(_FABRICATED_ANSWER, _sys(_DRAWING_SOUP), messages=[])
    assert out == _FABRICATED_ANSWER  # flag off -> pass through unchanged


# Live Wave-1 A5 (29c4bdd): the model quoted the Contract Data Delay Damages
# particular (and sometimes a SAR-per-day expansion). The gate treated that
# as an ungrounded BOQ unit rate and replaced the whole answer with
# "upload your priced BOQ". A Contract Data lookup is not a unit-rate quote.
_A5_ASK = "What are the Delay Damages for the whole of the Works?"
_A5_GC_CLAUSE = (
    "Sub-Clause 8.8 Delay Damages. The Contractor shall pay delay damages "
    "for the whole of the Works at the rate stated in the Contract Data "
    "for every calendar day."
)
_A5_ANSWER_WITH_SAR = (
    "Delay Damages are 0.1% of the Contract Price per calendar day "
    "(SAR 2,017,680.12 per day)."
)


def test_delay_damages_particular_is_not_wiped_as_a_boq_rate():
    """A5 live: Sources cited AB-2023-101 Conditions of Contract; the
    answer was the BOQ-rate refusal. A Contract Data fact lookup must
    keep the particular even when a SAR expansion is ungrounded."""
    msgs = [{"role": "user", "content": _A5_ASK}]
    out = _cost_grounding_gate(
        _A5_ANSWER_WITH_SAR, _sys(_A5_GC_CLAUSE), messages=msgs,
    )
    assert out == _A5_ANSWER_WITH_SAR
    assert out != _CG_REFUSAL


def test_unit_rate_ask_still_refuses_an_ungrounded_price():
    """The A5 exception is the question class, not a hole in the gate."""
    msgs = [{"role": "user", "content": "What is the unit rate for ready-mix?"}]
    out = _cost_grounding_gate(
        _FABRICATED_ANSWER, _sys(_DRAWING_SOUP), messages=msgs,
    )
    assert out == _CG_REFUSAL


def test_e1_arithmetic_delay_damages_in_sar_is_still_gated():
    """E1 asks for a SAR figure. A percentage-only excerpt cannot ground
    an invented daily amount — that reservation is retrieval's job, and
    the gate still refuses a fabricated SAR total."""
    ask = (
        "Calculate the delay damages per calendar day in SAR for the "
        "whole of the Works."
    )
    msgs = [{"role": "user", "content": ask}]
    answer = "The daily delay damages are SAR 9,936.00."
    rate_only = (
        "CONTRACT DATA particulars.\n"
        "8.8 Delay Damages: 0.1% of the Contract Price per calendar day"
    )
    out = _cost_grounding_gate(answer, _sys(rate_only), messages=msgs)
    assert out == _CG_REFUSAL
