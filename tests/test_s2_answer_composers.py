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
