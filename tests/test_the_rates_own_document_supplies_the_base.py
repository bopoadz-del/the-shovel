"""A rate and its base must belong to the same contract.

#701 made the delay-damages composer fire for a Milestone ask and was reverted
within the hour: on live it took the base from whatever excerpt was in the
bundle and answered 0.015% x SAR 144,042,486.50 = SAR 21,606.37, on an amount
belonging to another contract entirely. 4/10 correct became 0/10 with a
confident wrong figure, which is worse than the refusal it replaced.

Two changes, together:
  * the ask selects the rate (a Milestone ask must not be scored by
    whole-of-Works preferences) -- the sound half of #701;
  * the base may only be paired while ONE contract is in view. Two contract
    identifiers in the bundle means no pairing is safe, and nothing composes.

A per-DOCUMENT pairing was tried first and removed: parsing one segment at a
time loses the preference ordering inside parse_accepted_contract_amount (a
filled particular beats a scanned window) and picked a partial figure over the
real Contract Data row on the live A2 fixture.

Synthetic contract text and synthetic amounts throughout.
"""
import pytest

from app.lib import construction_formulas_commercial as cc

MILESTONE_ASK = ("What are the delay damages in SAR per calendar day for a single "
                 "Milestone, calculated on the Accepted Contract Amount?")
WHOLE_ASK = "What are the delay damages in SAR per calendar day for the whole of the Works?"

ONE_DOC = (
    "[doc_id=cd1 chunk=0] Contract Data 8.8.1 | Delay Damages for the whole of the "
    "Works: 0.1% of the Contract Price per calendar day. | Delay Damages per "
    "Milestone: 0.015% of the Contract Price per calendar day. | Maximum Amount of "
    "Delay Damages: 10% of the Contract Price. | Accepted Contract Amount "
    "(excluding VAT): SAR 2,000,000,000.00"
)
SPLIT_DOCS = (
    "[doc_id=cd1 chunk=0] Contract Data 8.8.1 | Delay Damages per Milestone: 0.015% "
    "of the Contract Price per calendar day. "
    "[doc_id=ccf9 chunk=3] Change Control Form Nr. 21 | contract value SAR 144,042,486.50"
)


# ── the revert's lesson: no cross-document arithmetic ──────────────────────

def test_a_base_from_another_document_composes_nothing():
    assert cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, SPLIT_DOCS) is None


def test_the_same_document_composes():
    out = cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, ONE_DOC)
    assert out is not None
    assert out["daily_amount"] == pytest.approx(300_000.0)     # 0.015% x 2,000,000,000
    assert out["rate_percent"] == pytest.approx(0.015)


def test_the_pairing_reports_which_document_it_used():
    found = cc.rate_and_base_from_one_document(ONE_DOC, MILESTONE_ASK)
    assert found is not None
    _rate, _base, doc_id = found
    assert doc_id == "cd1"


def test_unmarked_text_is_treated_as_one_document():
    # Callers that pass raw text (tests, tools) still work: one segment.
    plain = ONE_DOC.split("] ", 1)[1]
    out = cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, plain)
    assert out is not None and out["daily_amount"] == pytest.approx(300_000.0)


def test_a_rate_alone_composes_nothing():
    rate_only = ("[doc_id=cd1 chunk=0] Delay Damages per Milestone: 0.015% of the "
                 "Contract Price per calendar day.")
    assert cc.rate_and_base_from_one_document(rate_only, MILESTONE_ASK) is None


# ── the ask selects the rate ───────────────────────────────────────────────

@pytest.mark.parametrize("ask,expected", [
    (MILESTONE_ASK, 0.015),
    (WHOLE_ASK, 0.1),
    ("", 0.1),                      # no ask: the whole-of-Works default stands
])
def test_the_ask_selects_the_rate(ask, expected):
    assert cc.parse_delay_damages_rate_percent(ONE_DOC, ask) == pytest.approx(expected)


def test_the_cap_row_is_never_a_rate():
    for ask in (MILESTONE_ASK, WHOLE_ASK):
        assert cc.parse_delay_damages_rate_percent(ONE_DOC, ask) != pytest.approx(10.0)


def test_a_milestone_ask_is_recognised_and_a_general_one_is_not():
    assert cc.ask_is_about_a_milestone_or_section(MILESTONE_ASK)
    assert cc.ask_is_about_a_milestone_or_section("delay damages for Section 2")
    assert not cc.ask_is_about_a_milestone_or_section(WHOLE_ASK)


def test_the_composed_amount_follows_the_selected_rate():
    whole = cc.compose_delay_damages_daily_from_excerpts(WHOLE_ASK, ONE_DOC)
    milestone = cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, ONE_DOC)
    assert whole["daily_amount"] == pytest.approx(2_000_000.0)
    assert milestone["daily_amount"] == pytest.approx(300_000.0)


def test_nothing_composes_without_both_halves():
    rate_only = "[doc_id=cd1 chunk=0] Delay Damages per Milestone: 0.015% per calendar day."
    base_only = "[doc_id=cd1 chunk=0] Accepted Contract Amount: SAR 2,000,000,000.00"
    assert cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, rate_only) is None
    assert cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, base_only) is None


def test_the_question_form_reaches_the_composer():
    # "What are ... in SAR per calendar day" must count as an arithmetic ask,
    # or the composer never runs at all -- which is why M3 sat at 0/10.
    from app.core.rag import retriever as r
    assert r.query_needs_a_monetary_base(MILESTONE_ASK)
    assert not r.query_needs_a_monetary_base("What is the Defects Notification Period?")


TWO_CONTRACTS = (
    "[doc_id=cdB chunk=0] SYN-2019-777 Contract Data | Accepted Contract Amount "
    "(excluding VAT): SAR 500,000,000.00 "
    "[doc_id=cdA chunk=0] SYN-2024-001 Contract Data | Delay Damages per Milestone: "
    "0.015% of the Contract Price per calendar day. | Accepted Contract Amount "
    "(excluding VAT): SAR 2,000,000,000.00"
)


def test_two_contracts_in_view_compose_nothing():
    # Even though the rate's own contract states an amount, a second contract's
    # Accepted Contract Amount is in the bundle. Picking one is exactly how
    # #701 answered against another project's figure, so the honest result is
    # no figure at all.
    assert cc.compose_delay_damages_daily_from_excerpts(MILESTONE_ASK, TWO_CONTRACTS) is None
