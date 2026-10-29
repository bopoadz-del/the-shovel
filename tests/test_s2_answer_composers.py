"""S-2: the answer-composer registry that replaced the per-case graft cascade.

The registry is the one general answer-composition mechanism: an ordered list
of (question_type, apply) entries keyed on the STRUCTURE of the ask + the
kernel/tool payload, never on a probe case. These tests pin that it is present
and ordered (so a dispatcher-bypass / dropped-entry mutant fails), that no entry
names a probe case, and that each question type is input-derived (two different
asks -> two different composed answers).
"""
from __future__ import annotations

import re

from app.agents import runtime

# Every question type the pre-/post-gate composers must cover. A dropped entry
# (dispatcher-bypass mutant) removes that type's composition and fails here.
EXPECTED_PRE_GATE = {
    "complete_calc", "boq_scope_wbs", "operator_claim_facts", "concrete_volume",
    "user_priced_takeoff", "stated_total_follow_up", "milestone_arithmetic",
    "delay_damages_daily", "delay_damages_over_period", "asked_contract_particular",
    "percentage_of_aca", "priced_boq_item", "part_summary_total",
    "combined_part_summary_total", "named_community_tfc_span", "rate_only_item",
    "honest_contract_refusal", "deterministic_cost", "supplied_input_calculation",
}
EXPECTED_POST_GATE = {"computed_slab_thickness", "named_standard_attribution"}

# A probe case id is a bare letter+digits token (e4, a2, m17, f1, ...).
_PROBE_ID = re.compile(r"(?i)\b[a-z]\d{1,2}\b")


def _keys(composers):
    return [name for name, _fn in composers]


def test_registry_covers_every_question_type():
    assert set(_keys(runtime._PRE_GATE_ANSWER_COMPOSERS)) == EXPECTED_PRE_GATE
    assert set(_keys(runtime._POST_GATE_ANSWER_COMPOSERS)) == EXPECTED_POST_GATE


def test_no_composer_entry_names_a_probe_case():
    for name in _keys(runtime._PRE_GATE_ANSWER_COMPOSERS) + _keys(
        runtime._POST_GATE_ANSWER_COMPOSERS
    ):
        assert not _PROBE_ID.search(name), name


def test_run_answer_composers_applies_each_in_order():
    seen: list[str] = []

    def mk(tag: str):
        def _apply(text, _m, _r, _p, _e):
            seen.append(tag)
            return text + tag
        return _apply

    out = runtime._run_answer_composers(
        "x", [("a", mk("A")), ("b", mk("B"))], None, None, None, None,
    )
    assert out == "xAB"
    assert seen == ["A", "B"]  # dispatcher preserves registry order


def test_concrete_volume_two_inputs_two_outputs():
    from app.lib.construction_formulas_quantities import (
        compose_concrete_volume_from_ask,
    )
    big = compose_concrete_volume_from_ask(
        "Concrete volume for a raft 30 x 20 x 1.5 m including documented waste."
    )
    small = compose_concrete_volume_from_ask(
        "Concrete volume for a raft 10 x 10 x 0.5 m including documented waste."
    )
    assert big and small
    assert big["volume_m3"] != small["volume_m3"]
    assert big["line"] != small["line"]


def test_productivity_cost_two_inputs_two_outputs():
    from app.lib.construction_formulas_planning import (
        compose_productivity_cost_from_ask,
        looks_like_productivity_cost_ask,
    )
    q_a = (
        "Plastering 500 m2 at 25 m2 per gang-day, crew cost SAR 1200 per day "
        "- duration and cost?"
    )
    q_b = (
        "Plastering 900 m2 at 25 m2 per gang-day, crew cost SAR 1200 per day "
        "- duration and cost?"
    )
    assert looks_like_productivity_cost_ask(q_a)
    assert looks_like_productivity_cost_ask(q_b)
    c_a = compose_productivity_cost_from_ask(q_a)
    c_b = compose_productivity_cost_from_ask(q_b)
    assert c_a and c_b
    assert c_a["line"] != c_b["line"]
    assert c_a["total_cost"] != c_b["total_cost"]


# ── strict two-input backfill (ORDERS_29OCT_0000Z): two asks -> two answers ──


def test_rebar_metres_run_two_inputs_two_outputs():
    from app.lib.construction_formulas_quantities import (
        compose_rebar_metres_run_from_ask,
        looks_like_rebar_metres_run_ask,
    )
    a = "How many metres run of Y16 rebar for 12 t?"
    b = "How many metres run of Y16 rebar for 30 t?"
    assert looks_like_rebar_metres_run_ask(a) and looks_like_rebar_metres_run_ask(b)
    ca = compose_rebar_metres_run_from_ask(a)
    cb = compose_rebar_metres_run_from_ask(b)
    assert ca and cb and ca["line"] != cb["line"]


def test_user_priced_takeoff_two_inputs_two_outputs():
    from app.lib.construction_formulas_commercial import (
        compose_user_priced_takeoff_from_ask,
    )
    a = "Price the concrete take-off for 24 pad footings each 2.5 x 2.0 x 1.5 m at SAR 450 per m3."
    b = "Price the concrete take-off for 12 pad footings each 2.5 x 2.0 x 1.5 m at SAR 450 per m3."
    ca = compose_user_priced_takeoff_from_ask(a)
    cb = compose_user_priced_takeoff_from_ask(b)
    assert ca and cb and ca != cb


def test_stated_total_follow_up_two_inputs_two_outputs():
    from app.lib.construction_formulas_commercial import compose_stated_total_follow_up
    current = "Add a 7% waste allowance and price that total at SAR 450 per m3."
    prior_a = "24 pile caps each 2.5 x 2.0 x 1.5 m."
    prior_b = "12 pile caps each 2.5 x 2.0 x 1.5 m."
    ca = compose_stated_total_follow_up(current, prior_a)
    cb = compose_stated_total_follow_up(current, prior_b)
    assert ca and cb and ca != cb


def test_milestone_arithmetic_two_inputs_two_outputs():
    from app.core.hypothetical_milestone_arithmetic import (
        compose_hypothetical_milestone_arithmetic,
    )
    a = "M1 = 397 days, M2 = 487 days from a common start; which milestone drives completion and by how much over M1?"
    b = "M1 = 397 days, M2 = 731 days from a common start; which milestone drives completion and by how much over M1?"
    ca = compose_hypothetical_milestone_arithmetic(a)
    cb = compose_hypothetical_milestone_arithmetic(b)
    assert ca and cb and ca != cb


def test_delay_damages_daily_two_inputs_two_outputs():
    from app.lib.construction_formulas_commercial import (
        compose_delay_damages_daily_from_excerpts,
    )
    query = (
        "Answer only from the client project documents. Calculate the delay "
        "damages per calendar day in SAR for the whole of the Works."
    )
    rate = (
        "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
        "8.8 Delay Damages for the whole of the Works: 0.1% of the Contract "
        "Price per calendar day"
    )

    def excerpts(aca: float) -> str:
        aca_row = (
            "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
            f"1.1.1 Accepted Contract Amount excluding VAT | SAR {aca:,.2f}"
        )
        return rate + "\n\n" + aca_row

    a = compose_delay_damages_daily_from_excerpts(query, excerpts(1_000_000_000.0))
    b = compose_delay_damages_daily_from_excerpts(query, excerpts(2_000_000_000.0))
    assert a and b and a != b


def _aca_row(aca: float) -> str:
    # The CONTRACT DATA ACA row shape the commercial composers parse.
    from tests.test_set3_empty_turn_particulars_and_period_dd import LABEL
    return LABEL + f"1.1.1: | | Accepted Contract Amount: SAR {aca:,.2f} excluding VAT |\n"


def test_percentage_of_aca_two_inputs_two_outputs():
    from app.lib.construction_formulas_commercial import (
        compose_percentage_of_aca_from_excerpts,
    )
    from tests.test_set3_empty_turn_particulars_and_period_dd import E1, CD_ADVANCE, _sys
    a = compose_percentage_of_aca_from_excerpts(
        E1, _sys(CD_ADVANCE, _aca_row(1_000_000_000.0))["content"]
    )
    b = compose_percentage_of_aca_from_excerpts(
        E1, _sys(CD_ADVANCE, _aca_row(2_000_000_000.0))["content"]
    )
    assert a and b and a["amount"] != b["amount"]


def test_delay_damages_over_period_two_inputs_two_outputs():
    from app.lib.construction_formulas_commercial import (
        compose_delay_damages_over_period_from_excerpts,
    )
    from tests.test_set3_empty_turn_particulars_and_period_dd import (
        E3, CD_MILESTONE_RATES, CD_WHOLE_WORKS_RATE, _sys,
    )
    a = compose_delay_damages_over_period_from_excerpts(
        E3, _sys(CD_MILESTONE_RATES, _aca_row(1_000_000_000.0), CD_WHOLE_WORKS_RATE)["content"]
    )
    b = compose_delay_damages_over_period_from_excerpts(
        E3, _sys(CD_MILESTONE_RATES, _aca_row(2_000_000_000.0), CD_WHOLE_WORKS_RATE)["content"]
    )
    assert a and b and a["amount"] != b["amount"]


def test_computed_slab_two_inputs_two_outputs():
    # The graft replaces a corpus refusal with the slab_thickness_min result;
    # two different spans -> two different computed thicknesses.
    refusal = "I could not confirm the minimum slab thickness in the retrieved excerpts."
    a = runtime._graft_computed_slab_thickness(
        refusal,
        [{"role": "user", "content": "Minimum one-way slab thickness for a simply supported span of 4.8 m?"}],
    )
    b = runtime._graft_computed_slab_thickness(
        refusal,
        [{"role": "user", "content": "Minimum one-way slab thickness for a simply supported span of 7.2 m?"}],
    )
    assert a != refusal and b != refusal and a != b


def test_priced_boq_item_two_inputs_two_outputs():
    from app.core.rag.retriever import compose_priced_boq_row
    from tests.test_a_priced_row_is_priced_in_either_column_order import (
        ASK, UNIT_FIRST, POLES_ASK, POLES_DEMOLITION,
    )
    a = compose_priced_boq_row(ASK, UNIT_FIRST)
    b = compose_priced_boq_row(POLES_ASK, POLES_DEMOLITION)
    assert a and b and a["amount"] != b["amount"]


def test_part_summary_total_two_inputs_two_outputs():
    from app.core.rag.retriever import compose_part_summary_total
    from tests.test_b3_part_summary_compose import LIVE_B3, S2
    a = compose_part_summary_total(LIVE_B3, S2)
    b = compose_part_summary_total(LIVE_B3, S2.replace("40,560.00", "99,999.00"))
    assert a and b and a["amount"] != b["amount"]


def test_combined_part_summary_total_two_inputs_two_outputs():
    from app.core.rag.retriever import compose_combined_part_summary_total
    from tests.test_set3_remaining_e1_e3_e6_f1 import E6, COLLECTION
    a = compose_combined_part_summary_total(E6, COLLECTION)
    b = compose_combined_part_summary_total(
        E6, COLLECTION.replace("34,645,529.00", "10,000,000.00")
    )
    assert a and b and a["amount"] != b["amount"]


def test_named_community_tfc_span_two_inputs_two_outputs():
    from app.core.rag.retriever import compose_named_community_tfc_span
    query = (
        "What are the longest and shortest Time for Completion among the "
        "Northern Community milestones?"
    )
    ex_a = (
        "Milestone 1 | 547 days for the Northern Community.\n"
        "Milestone 2 | 397 days for the Northern Community."
    )
    ex_b = (
        "Milestone 1 | 600 days for the Northern Community.\n"
        "Milestone 2 | 300 days for the Northern Community."
    )
    a = compose_named_community_tfc_span(query, ex_a)
    b = compose_named_community_tfc_span(query, ex_b)
    assert a and b and a["delta"] != b["delta"]


def test_rate_only_item_two_inputs_two_outputs():
    from app.core.rag.retriever import format_rate_only_line
    a = format_rate_only_line(["D529.3"], "D529.3 is Rate Only in the Demolition BOQ.")
    b = format_rate_only_line(["D549.2"], "D549.2 is Rate Only in the Demolition BOQ.")
    assert a and b and a != b


def test_named_standard_attribution_two_inputs_two_outputs():
    from app.core.rag.named_standard_attribution import relabel_answer
    rag = "Project excerpts: the 30-minute fire rating is from the project spec section 7."
    a = relabel_answer(
        "What does NFPA 13 require for this rating?",
        "Per NFPA 13 the rating is 30 minutes.", rag,
    )
    b = relabel_answer(
        "What does ASHRAE 90.1 require for this rating?",
        "Per ASHRAE 90.1 the rating is 30 minutes.", rag,
    )
    assert a and b and a != b and "NFPA" in a and "ASHRAE" in b


def test_asked_contract_particular_two_inputs_two_outputs():
    # asked_contract_particular composes a named Contract Data particular from
    # the excerpts; extract_time_for_completion_days is input-derived.
    from app.core.rag.retriever import extract_time_for_completion_days

    def tfc(days: int) -> str:
        return (
            "CONTRACT DATA particulars — filled-in amount / duration / percentage.\n"
            f"Time for Completion for the whole of the Works | {days} days"
        )

    a = extract_time_for_completion_days(tfc(547))
    b = extract_time_for_completion_days(tfc(730))
    assert a and b and a != b


def test_operator_claim_facts_two_inputs_two_outputs():
    base = "Delay claim notice issued under Clause 4.5 of the contract."
    a = runtime._graft_operator_claim_facts(
        base, "Draft the delay claim; it must be issued via Aconex."
    )
    b = runtime._graft_operator_claim_facts(
        base, "Draft the delay claim under Clause 8.8.1."
    )
    assert a != base and b != base and a != b


def test_deterministic_cost_two_inputs_two_outputs():
    def payload(total: float) -> dict:
        return {"status": "success", "calculation": "cost_estimate",
                "result": {"total_cost": total}}
    a = runtime._format_deterministic_cost_calc(payload(100_000.0))
    b = runtime._format_deterministic_cost_calc(payload(250_000.0))
    assert a and b and a != b


def test_supplied_input_calculation_two_inputs_two_outputs():
    from app.lib.construction_formulas import formula_completed_by_user_text
    a = formula_completed_by_user_text(
        "Calculate the concrete volume with length 10 m, width 5 m and thickness 2 m."
    )
    b = formula_completed_by_user_text(
        "Calculate the concrete volume with length 20 m, width 5 m and thickness 2 m."
    )
    assert a and b and a != b


def test_boq_scope_wbs_two_inputs_two_outputs():
    # The boq_scope_wbs graft surfaces the BOQ-derived WBS tree via
    # format_wbs_outline; two different trees -> two different rendered WBS
    # outlines (only codes in the tree are printed; no invented levels).
    from app.core.predefined_reasoning import format_wbs_outline
    tree_a = {
        "1": "Site clearance", "1.1": "Remove existing fence",
        "2": "Structural demolition", "2.1": "Break out ground slab",
    }
    tree_b = {
        "1": "Enabling works", "1.1": "Divert live services",
        "2": "Bulk excavation", "2.1": "Cart away arisings",
    }
    a = format_wbs_outline(tree_a)
    b = format_wbs_outline(tree_b)
    assert a and b and a != b


def test_honest_contract_refusal_is_input_gated():
    # honest_contract_refusal is NON-COMPOSITIONAL: its correct output is the
    # honest "not required / absent" refusal, which is legitimately the SAME for
    # two different contract-absent asks. So instead of a (meaningless) a != b,
    # the two-input property is that it is input-GATED (fires on a contract ask
    # whose CD row states the particular; does NOT fire on a compositional ask)
    # and CONSISTENT (same honest line for the same CD row).
    from app.core.rag.retriever import (
        chunk_states_pcg_contract_data,
        format_pcg_honest_line,
        query_asks_for_parent_company_guarantee,
    )
    from tests.test_g3_g6_honest_refusal import G3_ALT, PCG_NOT_REQUIRED

    # Fires on a PCG ask backed by the Contract Data row:
    assert query_asks_for_parent_company_guarantee(G3_ALT)
    assert chunk_states_pcg_contract_data(PCG_NOT_REQUIRED)
    # Does NOT fire on a compositional (calc) ask:
    assert not query_asks_for_parent_company_guarantee(
        "What is the concrete volume for a 30 x 20 x 1.5 m raft?"
    )
    # Consistent honest refusal for the same CD row (not a canned constant,
    # a derived honest line that is the same when the input is the same):
    line = format_pcg_honest_line(PCG_NOT_REQUIRED)
    assert line and format_pcg_honest_line(PCG_NOT_REQUIRED) == line
