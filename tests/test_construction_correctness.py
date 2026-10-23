"""Construction-correctness: delay split, no alias collapse, site vocab.

Pins the chat-path rules that must hold after Aconex P0:
- a reported slip is not a claim
- NCR / stop-work / PC cert do not collapse onto analysis templates
- site language (WIR, hold point, IFC drawings, SI vs VO, nominated sub)
  routes without waiting on embeddings
"""
from __future__ import annotations

import asyncio

import pytest

from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.core.cm_step_aliases import (
    NO_AUTO_DISPATCH,
    is_auto_dispatch,
    is_plan_only,
    resolve_action,
    resolve_step,
)
from app.core.cross_domain_reasoner import CrossDomainReasoner
from app.core.delay_advice import DelayKind, claims_builder_permitted, classify_delay
from app.core.dependency_graph import SuggestedAction
from app.core.procedure_actions import procedure_metadata
from app.core.site_vocab import (
    message_is_site_instruction_not_vo,
    message_issues_ncr,
    message_issues_pc_cert,
    message_issues_stop_work,
    message_wants_ifc_drawings,
    message_wants_nominated_sub,
    message_wants_wir,
)
from tests.conftest import requires_construction_kit


def _matched(message: str) -> list[str]:
    block = SmartOrchestratorBlock()
    result = asyncio.run(block.process({"user_message": message}))
    return [m["action"] for m in (result.get("matched_actions") or [])] or list(
        result.get("action_queue") or []
    )


def _queue(message: str) -> list[str]:
    block = SmartOrchestratorBlock()
    result = asyncio.run(block.process({"user_message": message}))
    return list(result.get("action_queue") or [])


# ── 1. Delay advice split ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message,kind",
    [
        (
            "Assess EOT entitlement for the late foundation release — time only, no cost.",
            DelayKind.EOT_ONLY,
        ),
        (
            "What prolongation cost is supportable for the 4-week standing time?",
            DelayKind.COST_ONLY,
        ),
        (
            "There is concurrent delay between the steel late delivery and our rebar fix.",
            DelayKind.CONCURRENT,
        ),
        (
            "This is a culpable delay — we missed our own hold point.",
            DelayKind.CULPABLE,
        ),
        (
            "The chiller delivery has slipped four weeks.",
            DelayKind.ASK,
        ),
        (
            "Help me build a delay claim for the electrical package.",
            DelayKind.CLAIM,
        ),
    ],
)
def test_delay_kind_split(message, kind):
    assert classify_delay(message) == kind


def test_slipped_delivery_is_not_a_claim():
    msg = "The chiller delivery has slipped four weeks."
    assert classify_delay(msg) == DelayKind.ASK
    assert not claims_builder_permitted(msg)
    assert "claims_builder" not in _matched(msg)
    assert "claims_builder" not in _queue(msg)
    analysis = CrossDomainReasoner().analyze_turn(msg)
    assert analysis["matched_template"] != "delay_to_claim"
    assert "claims_builder" not in (analysis.get("suggested_tools") or [])
    ctx = analysis.get("cross_domain_context") or ""
    assert "claim eligibility" not in ctx.lower()
    assert "create_claim" not in ctx.lower()


def test_explicit_delay_claim_still_reaches_claims_builder():
    msg = "Help me build a delay claim for the electrical package."
    assert claims_builder_permitted(msg)
    analysis = CrossDomainReasoner().analyze_turn(msg)
    assert analysis["matched_template"] == "delay_to_claim"
    assert "claims_builder" in _matched(msg) or analysis["matched_template"] == "delay_to_claim"


def test_eot_only_does_not_recommend_a_claim():
    msg = "Assess EOT entitlement for the late foundation release — time only."
    assert classify_delay(msg) == DelayKind.EOT_ONLY
    assert not claims_builder_permitted(msg)
    assert "claims_builder" not in _matched(msg)
    analysis = CrossDomainReasoner().analyze_turn(msg)
    assert analysis["matched_template"] != "delay_to_claim"
    assert "claims_builder" not in (analysis.get("suggested_tools") or [])


def test_r004_default_actions_are_not_a_claim():
    from app.core.dependency_graph import DEFAULT_RULES

    r004 = next(r for r in DEFAULT_RULES if r.rule_id == "R004")
    assert SuggestedAction.CREATE_CLAIM not in r004.suggested_actions
    assert SuggestedAction.CHECK_EOT_ENTITLEMENT not in r004.suggested_actions
    assert SuggestedAction.CALCULATE_PROLONGATION_COST not in r004.suggested_actions


# ── 2. Alias collapse ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "step,collapsed_onto",
    [
        ("ncr_issue", "qa_qc_inspection"),
        ("stop_work_order", "safety_compliance_audit"),
        ("pc_cert", "commissioning_checklist"),
    ],
)
def test_formal_instruments_do_not_collapse_to_templates(step, collapsed_onto):
    assert resolve_step(step) == NO_AUTO_DISPATCH
    assert not is_auto_dispatch(resolve_step(step))
    assert is_plan_only(step)
    assert resolve_action(step) != collapsed_onto
    assert resolve_action(step) == step


def test_issue_ncr_does_not_run_qa_qc_inspection():
    msg = "Issue an NCR for out-of-tolerance column verticality on level 3."
    assert message_issues_ncr(msg)
    matched = _matched(msg)
    assert "ncr_management" in matched
    assert "qa_qc_inspection" not in matched
    meta = procedure_metadata("ncr_management")
    assert meta["execution_mode"] == "metadata_only"
    assert meta.get("delegate_action") in (None, "")


def test_issue_stop_work_does_not_run_safety_audit():
    msg = "Issue a stop-work order for the east elevation scaffold."
    assert message_issues_stop_work(msg)
    assert "safety_compliance_audit" not in _matched(msg)


def test_issue_pc_cert_does_not_run_commissioning_checklist():
    msg = "Issue the practical completion certificate for the energy centre."
    assert message_issues_pc_cert(msg)
    assert "commissioning_checklist" not in _matched(msg)
    meta = procedure_metadata("handover_management")
    assert meta["execution_mode"] == "metadata_only"
    assert meta.get("delegate_action") in (None, "")


@pytest.mark.asyncio
@requires_construction_kit
async def test_ncr_and_pc_routes_are_metadata_not_templates():
    from app.containers.construction import ConstructionContainer

    container = ConstructionContainer()
    ncr = await container.route("ncr_management", {}, {"action": "ncr_management"})
    assert ncr.get("status") == "success"
    assert ncr.get("execution_mode") == "metadata_only"
    assert ncr.get("action") != "qa_qc_inspection"
    assert ncr.get("procedure_context") is None

    pc = await container.route("handover_management", {}, {"action": "handover_management"})
    assert pc.get("status") == "success"
    assert pc.get("execution_mode") == "metadata_only"
    assert pc.get("action") != "commissioning_checklist"


def test_qa_defect_plan_marks_ncr_as_operator_step():
    from app.core.cross_domain_reasoner import MultiDomainPlanBuilder

    plan = MultiDomainPlanBuilder().build_from_template("qa_defect_closeout")
    ncr = next(s for s in plan["steps"] if s.get("step_type") == "ncr_issue")
    assert ncr["dispatch"] is False
    assert ncr.get("needs_operator") is True
    assert ncr["params"].get("action") is None

    safety = MultiDomainPlanBuilder().build_from_template("safety_incident_response")
    stop = next(s for s in safety["steps"] if s.get("step_type") == "stop_work_order")
    assert stop["dispatch"] is False
    assert stop.get("needs_operator") is True

    handover = MultiDomainPlanBuilder().build_from_template("commissioning_to_handover")
    pc = next(s for s in handover["steps"] if s.get("step_type") == "pc_cert")
    assert pc["dispatch"] is False
    assert pc.get("needs_operator") is True


# ── 3. Site vocabulary ─────────────────────────────────────────────────────


def test_wir_routes_to_inspection_request():
    msg = "Prepare a WIR for the Week 53 manhole collar pour."
    assert message_wants_wir(msg)
    assert "inspection_request" in _matched(msg)


def test_hold_point_routes_to_inspection_not_commissioning():
    msg = "List hold points before pour and witness points for slump."
    matched = _matched(msg)
    assert "inspection_request" in matched
    assert "commissioning_checklist" not in matched


def test_ifc_drawings_are_drawings_not_bim_model():
    msg = "Pull quantities from the IFC drawings for the podium slab."
    assert message_wants_ifc_drawings(msg)
    matched = _matched(msg)
    assert "drawing_qto" in matched
    assert "bim_analysis" not in matched


def test_site_instruction_is_not_a_variation():
    msg = "Issue a site instruction for the grid C setting-out."
    assert message_is_site_instruction_not_vo(msg)
    matched = _matched(msg)
    assert "change_order_impact" not in matched
    assert "variation_order_manager" not in matched
    assert "design_directive" in matched


def test_si_vs_vo_reaches_site_instruction_route():
    matched = _matched("What is the difference between SI vs VO on this contract?")
    assert "design_directive" in matched


def test_nominated_sub_routes_to_contract_or_procurement():
    msg = "What are the obligations of the nominated subcontractor for the lifts?"
    assert message_wants_nominated_sub(msg)
    matched = _matched(msg)
    assert "process_contract" in matched or "procurement_list_generator" in matched


def test_delivery_has_slipped_does_not_auto_route_claims_builder():
    msg = "the chiller delivery has slipped four weeks"
    assert "claims_builder" not in _matched(msg)
    assert "claims_builder" not in _queue(msg)


# ── 4. Optional clash → CDE RFI ────────────────────────────────────────────


def test_clash_plus_rfi_posts_to_cde_not_local_register():
    msg = "Run clash detection and post an RFI to Aconex for the duct/beam clash."
    matched = _matched(msg)
    assert "bim_clash_detection" in matched
    assert "cde_post_rfi" in matched
    assert "rfi_generator" not in matched
