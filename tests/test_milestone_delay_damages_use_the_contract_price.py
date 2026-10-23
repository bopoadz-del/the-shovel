"""The 8.8.1 delay-damages rate is per Milestone, on the Contract Price.

Live SET4.1 M3, "What are the delay damages in SAR per calendar day for
a section?":

  * 4/6 runs multiplied 0.015% by SAR 144,042,486.50 (SAR 21,606.37).
    That figure arrived in a higher-ranked excerpt — the executed cover
    and a pre-kickoff note — ahead of clause 1.1.1.
  * 2/6 runs multiplied by the Contract Price SAR 1,754,504,456.25
    (SAR 263,175.67) and then called it the whole-of-the-Works rate.

Contract Data 8.8.1 states the rate per calendar day per Milestone.
A question that says Section selects that same rate, and the sentence
states it as per Milestone. The base is the Contract Price, excluding VAT.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas_commercial as cc

SECTION_ASK = "What are the delay damages in SAR per calendar day for a section?"
MILESTONE_ASK = (
    "What are the delay damages in SAR per calendar day for a Milestone?"
)
WHOLE_ASK = (
    "What are the delay damages in SAR per calendar day for the whole of the Works?"
)
CONTRACT_PRICE = 1_754_504_456.25
OTHER_AMOUNT = 144_042_486.50
DAILY = 263_175.67
MONEY_LINE = (
    "Delay damages per Milestone are SAR 263,175.67 per calendar day "
    "(0.015% of the Contract Price SAR 1,754,504,456.25)."
)

# Contract Data 8.8.1. Quoted back; not paraphrased as the whole of the Works.
MILESTONE_CLAUSE = (
    "8.8.1: Delay Damages: 0.015% of the Contract Price per calendar day "
    "per Milestone"
)
WHOLE_CLAUSE = (
    "8.8.1 Delay Damages (for the whole of the Works): 0.1% of the "
    "Contract Price per calendar day"
)
PRICE_ROW = (
    "1.1.1 Accepted Contract Amount: SAR 1,754,504,456.25 excluding VAT"
)

# Live shape: the wrong amount is in other documents and is concatenated
# first. The Milestone rate and the Contract Price share the conditions.
LIVE_BUNDLE = (
    "[doc_id=ocr9demo chunk=0] Form of Agreement | package value "
    f"SAR {OTHER_AMOUNT:,.2f} | "
    "[doc_id=fixture-doc-b chunk=11] FIXTURE-c-20260927 kickoff note | "
    f"Accepted Contract Amount excluding VAT SAR {OTHER_AMOUNT:,.2f} | "
    "[doc_id=coc chunk=20] CONTRACT DATA | "
    f"{MILESTONE_CLAUSE} | {WHOLE_CLAUSE} | {PRICE_ROW}"
)

# The wrong amount leads, inside one window, with only "Contract Price"
# nearby — the bleed that used to elect it as the Accepted Contract Amount.
UNMARKED_BLEED = (
    f"{MILESTONE_CLAUSE}. contract value SAR {OTHER_AMOUNT:,.2f}. "
    f"{PRICE_ROW}. {WHOLE_CLAUSE}."
)


@pytest.mark.parametrize("ask", [SECTION_ASK, MILESTONE_ASK])
def test_either_ask_uses_the_contract_price_and_says_per_milestone(ask):
    out = cc.compose_delay_damages_daily_from_excerpts(ask, LIVE_BUNDLE)
    assert out is not None
    assert out["rate_percent"] == pytest.approx(0.015)
    assert out["contract_amount"] == pytest.approx(CONTRACT_PRICE)
    assert out["daily_amount"] == pytest.approx(DAILY)
    assert out["basis"] == "milestone"
    line = cc.format_delay_damages_daily_line(out)
    assert line.startswith(MONEY_LINE)
    assert "per Section" not in line
    assert "whole of the Works" not in line
    assert "144,042,486.50" not in line
    assert "The contract states:" in line
    assert "8.8.1" in line
    assert "per calendar day per Milestone" in line


def test_a_leading_other_sum_does_not_beat_clause_1_1_1():
    out = cc.compose_delay_damages_daily_from_excerpts(SECTION_ASK, UNMARKED_BLEED)
    assert out is not None
    assert out["contract_amount"] == pytest.approx(CONTRACT_PRICE)
    assert out["daily_amount"] == pytest.approx(DAILY)
    assert out["basis"] == "milestone"


def test_another_documents_amount_is_not_the_milestone_base():
    """The rate's document has no Contract Price. Do not borrow 144M."""
    bundle = (
        "[doc_id=coc chunk=20] CONTRACT DATA | "
        f"{MILESTONE_CLAUSE} | "
        "[doc_id=fixture-doc-b chunk=11] FIXTURE-c-20260927 kickoff note | "
        f"Accepted Contract Amount excluding VAT SAR {OTHER_AMOUNT:,.2f}"
    )
    assert cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, bundle) is None
    assert cc.compose_delay_damages_daily_from_excerpts(SECTION_ASK, bundle) is None


def test_the_whole_of_the_works_line_is_unchanged():
    bundle = (
        "[doc_id=coc chunk=20] CONTRACT DATA | "
        f"{WHOLE_CLAUSE} | {MILESTONE_CLAUSE} | {PRICE_ROW}"
    )
    out = cc.compose_delay_damages_daily_from_excerpts(WHOLE_ASK, bundle)
    assert out is not None
    assert out["rate_percent"] == pytest.approx(0.1)
    assert out["contract_amount"] == pytest.approx(CONTRACT_PRICE)
    assert out.get("basis") == "whole"
    line = cc.format_delay_damages_daily_line(out)
    assert line.startswith("Whole-of-Works delay damages: ")
    assert "per Milestone" not in line
    assert "per Section" not in line
    assert "The contract states:" not in line
