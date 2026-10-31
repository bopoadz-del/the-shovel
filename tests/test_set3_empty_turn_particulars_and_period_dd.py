"""Set3 empty-turn / unbound delay-damages (A7, A9, E1, E3).

Live a8498b3 standing-order:

    A7  "What is the Advance Payment amount under the contract?"
        empty answer; expect ~10%.
    A9  "What is the Limitation of Liability under the contract?"
        empty answer; expect ~100%.
    E1  "Calculate the Advance Payment in SAR."
        empty answer; expect 175,450,445.6
        (10% × excl-VAT ACA 1,754,504,456.25).
    E3  "If Milestones 3 and 4 are each 20 days late, what are the
        combined milestone delay damages in SAR?"
        got "0% of SAR 0.00"; expect ~10,527,026/027
        (0.015% × ACA × 20 × 2).

Class, not one-row special cases: a named Contract Data percentage
row is recovered when synthesis hangs empty, a percentage-of-ACA
money ask is composed when both operands are in the excerpts, and
``delay_damages_daily`` with empty rates is not a deliverable.
Do not call this a #634 regression. Parked D1 / C3 / #643/#645/#649
are out of scope.
"""
from __future__ import annotations

import json

from app.agents.runtime import (
    _EMPTY_RESPONSE_FALLBACK,
    _format_any_calc_result,
    _graft_asked_contract_particular,
    _postprocess_answer,
    _recover_answer_from_tool_messages,
    _should_force_synthesis,
)
from app.lib.construction_formulas import run_calculation
from app.lib.construction_formulas_commercial import (
    compose_delay_damages_over_period_from_excerpts,
    compose_percentage_of_aca_from_excerpts,
    delay_damages_daily,
    extract_named_percentage_particular,
    query_asks_delay_damages_over_a_period,
    query_asks_named_percentage_particular,
    query_asks_percentage_particular_in_money,
)

# Live asks (no client names). Figures already used elsewhere in this repo.
A7 = "What is the Advance Payment amount under the contract?"
A9 = "What is the Limitation of Liability under the contract?"
E1 = "Calculate the Advance Payment in SAR."
E3 = (
    "If Milestones 3 and 4 are each 20 days late, what are the "
    "combined milestone delay damages in SAR?"
)
A5 = "What are the Delay Damages for the whole of the Works?"
E2 = "If Milestone 1 is 30 days late, what are the milestone delay damages?"

ACA_EXCL = "SAR 1,754,504,456.25"
LABEL = (
    "CONTRACT DATA particulars — filled-in amount / duration / percentage "
    "[XX-2099-001_Contract Data.pdf].\nCONTRACT DATA\n"
)
CD_ADVANCE = LABEL + (
    "14.2.1: | | Advance Payment: 10% of the Accepted Contract Amount | |\n"
)
CD_LIABILITY = LABEL + (
    "17.6: | | Limitation of Liability: 100% of the Accepted Contract Amount | |\n"
)
CD_ACA = LABEL + (
    "1.1.1: | | Accepted Contract Amount: "
    f"{ACA_EXCL} excluding VAT |\n"
)
CD_MILESTONE_RATES = LABEL + (
    "8.8.1: | | Delay Damages (if applicable per Milestone): "
    "Milestone | Delay Damages\n"
    "|: | | Milestone 1 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 2 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 3 | 0.015% of the Contract Price per calendar day\n"
    "|: | | Milestone 4 | 0.015% of the Contract Price per calendar day\n"
)
CD_WHOLE_WORKS_RATE = LABEL + (
    "8.8.1: | | Delay Damages (for the whole of the Works): "
    "0.1% of the Contract Price per calendar day |\n"
)
UNBOUND_NOTE = "0% of SAR 0.00 = SAR 0.00 per calendar day."


def _sys(*texts: str) -> dict:
    body = "\n\n".join(
        f"[doc_id=cd{i} chunk={i} score=0.80] {t}"
        for i, t in enumerate(texts)
    )
    return {"role": "system", "content": "Reference context:\n" + body}


def _msgs(ask: str, *extra: dict) -> list[dict]:
    return [{"role": "user", "content": ask}, *extra]


def _calc_tool(payload: dict) -> dict:
    return {
        "role": "tool",
        "name": "construction_calc",
        "content": json.dumps(payload),
    }


# ── Ask class ──────────────────────────────────────────────────────────────


def test_a7_a9_are_named_percentage_particulars_not_money_or_delay():
    assert query_asks_named_percentage_particular(A7)
    assert query_asks_named_percentage_particular(A9)
    assert not query_asks_percentage_particular_in_money(A7)
    assert not query_asks_percentage_particular_in_money(A9)
    assert not query_asks_delay_damages_over_a_period(A7)
    assert not query_asks_delay_damages_over_a_period(A5)


def test_e1_advance_payment_in_sar_is_percentage_of_aca_money():
    assert query_asks_named_percentage_particular(E1)
    assert query_asks_percentage_particular_in_money(E1)
    assert not query_asks_delay_damages_over_a_period(E1)


def test_e3_combined_milestones_is_a_period_delay_ask():
    assert query_asks_delay_damages_over_a_period(E3)
    assert query_asks_delay_damages_over_a_period(E2)
    assert not query_asks_delay_damages_over_a_period(A5)
    assert not query_asks_percentage_particular_in_money(E3)


# ── A7 / A9: empty turn recovers the named percentage ──────────────────────


def test_a7_empty_turn_states_the_10_percent_advance_payment():
    rag = _sys(CD_ADVANCE, CD_ACA)
    parsed = extract_named_percentage_particular(A7, rag["content"])
    assert parsed is not None
    assert parsed["percent"] == 10.0
    out = _postprocess_answer("", rag, _msgs(A7))
    assert out.strip()
    assert out != _EMPTY_RESPONSE_FALLBACK
    assert __import__("re").search(r"\b10\s*%", out)


def test_a9_empty_turn_states_the_100_percent_limitation():
    rag = _sys(CD_LIABILITY)
    parsed = extract_named_percentage_particular(A9, rag["content"])
    assert parsed is not None
    assert parsed["percent"] == 100.0
    out = _postprocess_answer("", rag, _msgs(A9))
    assert out.strip()
    assert __import__("re").search(r"\b100\s*%", out)


def test_a7_does_not_invent_when_the_row_is_absent():
    rag = _sys(CD_ACA, CD_WHOLE_WORKS_RATE)
    assert extract_named_percentage_particular(A7, rag["content"]) is None
    missing = "I could not confirm the Advance Payment in the retrieved excerpts."
    assert _graft_asked_contract_particular(missing, rag, _msgs(A7)) == missing


# ── E1: empty turn composes 10% × ACA ──────────────────────────────────────


def test_e1_empty_turn_composes_advance_payment_in_sar():
    rag = _sys(CD_ADVANCE, CD_ACA)
    composed = compose_percentage_of_aca_from_excerpts(E1, rag["content"])
    assert composed is not None
    assert "175,450,445.6" in f"{composed['amount']:,.2f}"
    out = _postprocess_answer("", rag, _msgs(E1))
    assert "175,450,445.6" in out


def test_e1_does_not_compose_without_the_aca():
    rag = _sys(CD_ADVANCE)
    assert compose_percentage_of_aca_from_excerpts(E1, rag["content"]) is None
    out = _postprocess_answer("", rag, _msgs(E1))
    # Rate-only recovery is honest; it must not invent the SAR product.
    assert "175,450,445" not in out


# ── E3: unbound 0% of 0 is not a deliverable; compose the period sum ───────


def test_delay_damages_daily_empty_args_are_unbound_not_zero():
    """Empty-args class: defaults must not emit '0% of SAR 0.00'."""
    bare = delay_damages_daily()
    assert isinstance(bare, dict) and bare.get("error")
    assert "0% of SAR 0.00" not in str(bare.get("note") or "")

    via_registry = run_calculation("delay_damages_daily", {})
    assert via_registry["status"] == "error"
    assert not _should_force_synthesis({
        "name": "construction_calc",
        "ok": True,
        "result": via_registry,
    })

    # A genuine zero *rate* against a real base stays valid (audit pin).
    zero_rate = delay_damages_daily(rate_percent=0, contract_amount=1_000_000)
    assert not zero_rate.get("error")
    assert zero_rate["daily_amount"] == 0.0


def test_unbound_zero_note_is_not_recovered_as_the_answer():
    payload = {
        "status": "success",
        "calculation": "delay_damages_daily",
        "result": {
            "daily_amount": 0.0,
            "rate_percent": 0.0,
            "contract_amount": 0.0,
            "currency": "SAR",
            "note": UNBOUND_NOTE,
        },
    }
    assert _format_any_calc_result(payload) == ""
    recovered = _recover_answer_from_tool_messages(
        "", _msgs(E3, _calc_tool(payload)),
    )
    assert UNBOUND_NOTE not in recovered


def test_e3_empty_turn_composes_combined_milestone_damages():
    rag = _sys(CD_MILESTONE_RATES, CD_ACA, CD_WHOLE_WORKS_RATE)
    composed = compose_delay_damages_over_period_from_excerpts(E3, rag["content"])
    assert composed is not None
    # 0.015% × 1,754,504,456.25 × 20 × 2 = 10,527,026.7375
    assert abs(composed["amount"] - 10_527_026.74) < 0.5
    out = _postprocess_answer("", rag, _msgs(E3))
    assert __import__("re").search(r"10,527,02[67]", out)
    # Must not elect the whole-of-Works 0.1% (that product is ~70M).
    assert "70,180,178" not in out


def test_e3_replaces_the_unbound_zero_percent_answer():
    rag = _sys(CD_MILESTONE_RATES, CD_ACA)
    out = _postprocess_answer(UNBOUND_NOTE, rag, _msgs(E3))
    assert UNBOUND_NOTE not in out
    assert __import__("re").search(r"10,527,02[67]", out)


def test_e3_does_not_invent_when_the_milestone_rate_is_absent():
    rag = _sys(CD_ACA, CD_WHOLE_WORKS_RATE)
    assert compose_delay_damages_over_period_from_excerpts(E3, rag["content"]) is None
    out = _postprocess_answer(UNBOUND_NOTE, rag, _msgs(E3))
    assert "10,527,02" not in out


def test_a5_rate_lookup_is_not_stolen_onto_period_compose():
    rag = _sys(CD_WHOLE_WORKS_RATE, CD_ACA)
    assert compose_delay_damages_over_period_from_excerpts(A5, rag["content"]) is None
    out = _postprocess_answer("", rag, _msgs(A5))
    assert "10,527,02" not in out
    assert "1,754,504.46" not in out
