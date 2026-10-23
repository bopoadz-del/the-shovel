"""Workflow template step type → canonical ConstructionContainer / block actions.

WorkflowTemplateLibrary uses a rich step vocabulary (procurement_plan,
float_analysis, payment_cert, …). ConstructionContainer.route and specialist
blocks expose a smaller set of canonical action names. This module is the
single alias table used when plans are built or executed so every template
step resolves to a known handler.

Do not invent parallel routers here — smart_orchestrator remains the keyword
owner; this only translates step types → executable actions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# (block_name, canonical_action_or_None)
# - block_name: registry key passed to /v1/execute; empty string = no auto-dispatch
# - action: params["action"] for containers; None means the block's default process
#   (or, with empty block_name, that the caller must materialize / deliver)
StepTarget = Tuple[str, Optional[str]]

# Sentinel: step is in the alias table but must NOT be routed to a container /
# block action. Plan builder emits dispatch=False so callers
# (PlanExecutor._step_render_artifact, agent synthesis, etc.) handle it.
# Never map these to health_check — that falsely reports delivery success.
NO_AUTO_DISPATCH: StepTarget = ("", None)

# Formal instruments (NCR, stop-work, PC cert) are CDE / operator writes.
# They must not collapse onto analysis templates (qa_qc_inspection,
# safety_compliance_audit, commissioning_checklist). Plan or ask.
PLAN_ONLY_STEPS = frozenset({
    "ncr_issue",
    "stop_work_order",
    "pc_cert",
})

# Every STEP_* constant in workflow_templates.py must appear here.
STEP_TO_TARGET: Dict[str, StepTarget] = {
    # Document
    "extract_document": ("document_engine", None),
    "extract_risks": ("document_engine", None),
    "extract_equipment_list": ("document_engine", None),
    "extract_clauses": ("document_engine", None),
    # Schedule / WBS
    "build_wbs": ("construction", "generate_wbs"),
    "float_analysis": ("construction", "parse_primavera_schedule"),
    "recovery_options": ("construction", "schedule_risk"),
    "time_impact_analysis": ("construction", "forensic_delay_analysis"),
    "delay_analysis": ("construction", "forensic_delay_analysis"),
    "critical_path_impact": ("construction", "forensic_delay_analysis"),
    "link_procurement": ("construction", "procurement_list_generator"),
    "progress_tracker": ("construction", "progress_tracker"),
    "look_ahead": ("construction", "look_ahead"),
    "evm_calculate": ("construction", "evm_calculate"),
    "hold_point": ("construction", "qa_qc_inspection"),
    # Cost / BOQ
    "extract_boq": ("boq_processor", None),
    "cost_load": ("construction", "estimate_costs"),
    # Classic EVM (PV/EV/AC) — not the progress_tracker % proxy.
    "earned_value": ("construction", "evm_calculate"),
    "cost_variance": ("construction", "evm_calculate"),
    "cost_impact": ("construction", "change_order_impact"),
    "prolongation_cost": ("construction", "claims_builder"),
    "rework_cost": ("construction", "estimate_costs"),
    "carbon_cost_premium": ("construction", "carbon_footprint_calculator"),
    # Quality
    "inspection_summary": ("construction", "qa_qc_inspection"),
    "defect_trends": ("construction", "qa_qc_inspection"),
    "defect_register": ("construction", "qa_qc_inspection"),
    "severity_scoring": ("construction", "qa_qc_inspection"),
    "incoming_inspection": ("construction", "qa_qc_inspection"),
    "test_results": ("construction", "commissioning_checklist"),
    "deficiency_tracking": ("construction", "commissioning_checklist"),
    "ncr_issue": NO_AUTO_DISPATCH,
    "rectification_track": ("construction", "qa_qc_inspection"),
    "close_out": ("construction", "commissioning_checklist"),
    # Safety
    "incident_report": ("construction", "safety_compliance_audit"),
    "root_cause": ("construction", "safety_compliance_audit"),
    "stop_work_order": NO_AUTO_DISPATCH,
    "safety_summary": ("construction", "safety_compliance_audit"),
    "incident_summary": ("construction", "safety_compliance_audit"),
    # Procurement
    "procurement_plan": ("construction", "procurement_list_generator"),
    "approval_tracking": ("construction", "submittal_log_generator"),
    "po_trigger": ("construction", "procurement_optimizer"),
    # Contract / claims
    "entitlement_check": ("construction", "process_contract"),
    "clause_reference": ("construction", "process_contract"),
    "eot_entitlement": ("construction", "process_contract"),
    "obligation_register": ("construction", "process_contract"),
    "claim_strength": ("construction", "claims_builder"),
    "counter_arguments": ("construction", "claims_builder"),
    # Risk
    "populate_risk_register": ("construction", "risk_register_auto_populate"),
    "update_risk_register": ("construction", "risk_register_auto_populate"),
    "escalate_risk": ("construction", "risk_register_auto_populate"),
    "mitigation_plan": ("construction", "risk_register_auto_populate"),
    "coordination_risks": ("construction", "bim_clash_detection"),
    "regulatory_compliance": ("construction", "risk_register_auto_populate"),
    # Payment / cash
    "payment_cert": ("construction", "payment_certificate"),
    "cash_flow_update": ("construction", "cash_flow_forecast"),
    "generate_pay_app": ("construction", "payment_certificate"),
    # Commissioning / handover
    "pc_cert": NO_AUTO_DISPATCH,
    "om_manual": ("construction", "om_manual_generator"),
    "training_plan": ("construction", "om_manual_generator"),
    "warranty_register": ("construction", "warranty_maintenance_schedule"),
    "update_commissioning_plan": ("construction", "commissioning_checklist"),
    # BIM
    "bim_analysis": ("construction", "bim_analysis"),
    "clash_detection": ("construction", "bim_clash_detection"),
    "qto_extraction": ("construction", "bim_extract"),
    "wbs_from_bim": ("construction", "generate_wbs"),
    "boq_reconcile": ("boq_processor", None),
    # Specs / docs / RFI
    "submittal_log": ("construction", "submittal_log_generator"),
    "spec_analyze": ("spec_analyzer", None),
    "rfi_generation": ("construction", "rfi_generator"),
    "generate_rfi": ("construction", "rfi_generator"),
    "update_drawings": ("construction", "as_built_deviation_report"),
    "create_submittal": ("construction", "submittal_log_generator"),
    "lessons_learned": ("construction", "daily_site_report"),
    "release_hold_point": ("construction", "qa_qc_inspection"),
    # Sustainability
    "carbon_calc": ("construction", "carbon_footprint_calculator"),
    "sustainable_alternatives": ("construction", "procurement_optimizer"),
    "esg_report": ("construction", "esg_sustainability_report"),
    # Delivery / render — caller-handled; no container action produces the
    # promised report/package. Keep step_type visible; do not alias to
    # health_check (that masks missing delivery with a false success).
    "render_artifact": NO_AUTO_DISPATCH,
    "deliver": NO_AUTO_DISPATCH,
}

# Flat action alias: step type OR short name → ConstructionContainer action
# (for route() when block is already "construction").
ACTION_ALIASES: Dict[str, str] = {
    step: action
    for step, (block, action) in STEP_TO_TARGET.items()
    if block == "construction" and action
}


def is_auto_dispatch(target: Optional[StepTarget]) -> bool:
    """True when resolve_step yielded a block that /v1/execute can run."""
    return bool(target and target[0])


def is_plan_only(step_type: str) -> bool:
    """True when the step is a formal instrument — plan or ask, do not run."""
    return step_type in PLAN_ONLY_STEPS


def resolve_step(step_type: str) -> Optional[StepTarget]:
    """Return (block, action) for a workflow step type, or None if unknown.

    Delivery steps (render_artifact, deliver) resolve to NO_AUTO_DISPATCH
    (``("", None)``) — known, but not auto-executable. Callers must check
    ``is_auto_dispatch`` / plan ``dispatch`` before routing.
    """
    return STEP_TO_TARGET.get(step_type)


def resolve_action(action: str) -> str:
    """Map a template step type / alias to a ConstructionContainer action name.

    Unknown names pass through unchanged so existing callers keep working.
    """
    if not action:
        return action
    if action in ACTION_ALIASES:
        return ACTION_ALIASES[action]
    target = STEP_TO_TARGET.get(action)
    if target and target[0] == "construction" and target[1]:
        return target[1]
    return action


def all_template_step_types() -> List[str]:
    """Import STEP_* constants from workflow_templates (single source of truth)."""
    from app.core import workflow_templates as wt

    return sorted(
        {
            getattr(wt, name)
            for name in dir(wt)
            if name.startswith("STEP_") and isinstance(getattr(wt, name), str)
        }
    )


def unmapped_step_types() -> List[str]:
    """Step types defined in workflow_templates but missing from STEP_TO_TARGET."""
    return [s for s in all_template_step_types() if s not in STEP_TO_TARGET]
