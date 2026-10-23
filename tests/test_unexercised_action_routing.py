"""Lock routing for the 14 orchestrator actions the leftover matrix never sent.

The 2026-08-21 prod sweep BLOCKED these because fixture projects were missing.
This test does not need those fixtures: it asserts the keyword router still
selects each action from the canonical matrix prompt.
"""
from __future__ import annotations

import pytest

from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.core.clash_intent import message_wants_clash
from tests.conftest import requires_construction_kit

# Matrix BLOCKED rows that leftover/prod chat also never hit.
# document_metadata is a matrix feature name; the router action is process_document.
_UNSENT = [
    ("estimate_costs", "generate a cost estimate for a 2km 400mm sewer line in Riyadh",
     ["estimate_costs"]),
    ("spec_analyze", "analyze the concrete specification requirements - what grades and standards apply?",
     ["spec_analyze", "construction_advisor"]),
    ("document_metadata", "list the documents in this project and what type each one is",
     ["process_document"]),
    ("procurement_list_generator", "generate a procurement list for the MEP package",
     ["procurement_list_generator"]),
    ("rfi_management", "how many RFIs are open and which ones are overdue?",
     ["rfi_management", "rfi_generator"]),
    ("change_order_impact", "assess the cost and time impact of variation order VO-12 adding 300m of storm drain",
     ["change_order_impact", "variation_order_manager"]),
    ("extract_quantities", "take off the concrete quantities for the ground floor slabs",
     ["extract_quantities", "drawing_qto"]),
    ("procurement_optimizer", "optimize procurement - which supplier mix gives the cheapest viable steel package?",
     ["procurement_optimizer", "procurement_list_generator"]),
    ("process_specification_full", "walk through the full specification section for cast-in-place concrete, CSI division 03",
     ["process_specification_full", "spec_analyze"]),
    ("progress_tracker", "how is actual progress tracking against planned - where are we slipping?",
     ["progress_tracker"]),
    ("forensic_delay_analysis", "run a forensic delay analysis for the 6-week steel delivery delay - what EOT is supportable?",
     ["forensic_delay_analysis", "claims_builder"]),
    ("bim_clash_detection", "run clash detection between the structural and MEP models",
     ["bim_clash_detection"]),
    ("variation_order_manager", "update the variation log - what's the status and value of the open VOs?",
     ["variation_order_manager", "change_order_impact"]),
    ("value_engineering", "value engineer the basement - options to cut cost without losing parking spaces",
     ["value_engineering"]),
]


def _matched_actions(message: str) -> list[str]:
    block = SmartOrchestratorBlock()
    return [m["action"] for m in block._match_actions(message, None)]


@requires_construction_kit
@pytest.mark.parametrize("action,prompt,expect_any", _UNSENT)
def test_previously_unsent_matrix_action_still_routes(action, prompt, expect_any):
    matched = _matched_actions(prompt)
    assert matched, f"{action}: router returned no actions for {prompt!r}"
    assert any(a in expect_any for a in matched), (
        f"{action}: expected one of {expect_any} in {matched} for {prompt!r}"
    )


@requires_construction_kit
def test_clash_keyword_does_not_route_unless_present():
    block = SmartOrchestratorBlock()
    without = [m["action"] for m in block._match_actions(
        "count IfcWall in sample_office.ifc", ".ifc"
    )]
    with_clash = [m["action"] for m in block._match_actions(
        "run clash detection on sample_office.ifc", ".ifc"
    )]
    assert "bim_clash_detection" not in without
    assert "bim_clash_detection" in with_clash


@requires_construction_kit
def test_mep_conflict_still_routes_to_clash():
    """Fable W3: synonym keywords must survive the post-match clash filter."""
    assert message_wants_clash("find MEP conflict issues in the coordinated model")
    matched = _matched_actions("find MEP conflict issues in the coordinated model")
    assert "bim_clash_detection" in matched


@requires_construction_kit
def test_do_not_run_clash_does_not_route_to_clash():
    """Leftover project-assistant: negated clash must not steal the turn."""
    # Fable W1 — real asks must stay on.
    assert message_wants_clash("no rush, run clash detection")
    assert message_wants_clash("Without further delay, run clash detection")
    assert message_wants_clash(
        "Run clash detection but do not include clashes below 5mm"
    )
    # Fable W2 — common opt-outs.
    assert not message_wants_clash("skip clash detection")
    assert not message_wants_clash("avoid clash detection")
    assert not message_wants_clash("disable clash detection")
    assert not message_wants_clash("clash detection is not needed")
    assert message_wants_clash("run clash detection on sample_office.ifc")
    assert not message_wants_clash("Do not run clash detection.")
    assert not message_wants_clash("count IfcWall")

    prompt = (
        "Process the bill of quantities sample_mini_boq.xlsx — billed "
        "excavation 81.2 m3 vs site remeasure 92 m3. What is the variance? "
        "Do not run clash detection."
    )
    matched = _matched_actions(prompt)
    assert "bim_clash_detection" not in matched
    # Lock the desired leftover path: BOQ work, not clash.
    assert "boq_process" in matched, matched