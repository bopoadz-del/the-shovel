"""Procedure orchestrator actions → catalogue metadata and optional delegate handlers.

SmartOrchestratorBlock routes user language to procedure-specific action names
(design_review_workflow, rfi_management, …). ConstructionContainer.route must
never return Unknown action for these — either delegate to a real handler or
return honest metadata-only guidance from the procedures knowledge base.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# orchestrator action → (procedure kind, optional ConstructionContainer delegate action)
PROCEDURE_ACTION_MAP: Dict[str, Tuple[str, Optional[str]]] = {
    "design_review_workflow": ("design_review", None),
    "design_directive": ("design_directive", None),
    "rfi_management": ("request_for_information", "rfi_generator"),
    "work_package_control": ("work_package_control", None),
    "qa_audit": ("quality_audit", "qa_qc_inspection"),
    # NCR and PC cert are CDE / operator writes — metadata only. Do not
    # collapse onto qa_qc_inspection / commissioning_checklist templates.
    "ncr_management": ("non_conformance", None),
    "handover_management": ("handover", None),
    "inspection_request": ("inspection_request", "wir_form"),
    "job_requisition": ("job_requisition", "job_requisition"),
    "rfp_management": ("request_for_proposals", "rfp_draft"),
    "contract_award": ("contract_award", None),
}

# Keys commonly present on procedure records but not in the generic statuses/rules shape.
_SCHEMA_SPECIFIC_KEYS = (
    "review_statuses",
    "forbidden_term",
    "timeline",
    "workflow",
    "raci",
    "prerequisites",
    "handover_documents",
    "templates",
    "acceptance_forms",
    "design_phases",
    "document_prefix",
    "document_format",
    "notice_periods",
    "result_options",
    "result_rules",
    "key_rule",
    "critical_rule",
    "scoring",
    "impact_categories",
    "risk_format",
    "cadence",
    "stop_work",
    "category",
)


def is_procedure_action(action: str) -> bool:
    return action in PROCEDURE_ACTION_MAP


def procedure_metadata(
    action: str,
    project_id: Optional[str] = None,
    document_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return honest metadata-only payload for a procedure action.

    Preserves the raw catalogue record (and schema-specific fields) so callers
    see the design-review review_statuses / forbidden_term / timeline /
    workflow / raci and the handover prerequisites / handover_documents — not
    empty guidance.

    ``procedure_id`` is resolved at run time: the code of the active project's
    own procedure document of this kind when one exists (``document_names``,
    else the project's documents), otherwise the kind id.
    """
    from app.core.construction_knowledge import ConstructionKnowledge
    from app.core.procedure_catalogue import live_document_names, resolve_procedure

    kind, delegate = PROCEDURE_ACTION_MAP.get(action, (None, None))
    if not kind:
        return {
            "status": "error",
            "error": f"Unknown procedure action: {action}",
        }
    ck = ConstructionKnowledge()
    proc = ck.get_procedure(kind) or {}
    names = document_names if document_names is not None else live_document_names(project_id)
    resolved = resolve_procedure(kind, names)
    payload: Dict[str, Any] = {
        "status": "success",
        "action": action,
        "execution_mode": "metadata_only",
        "procedure_id": resolved["procedure_id"],
        "procedure_kind": kind,
        "procedure_document": resolved["procedure_document"],
        "procedure_title": proc.get("title", ""),
        "purpose": proc.get("purpose", ""),
        "roles": proc.get("roles") or {},
        "statuses": proc.get("statuses") or [],
        "rules": proc.get("rules") or [],
        "required_fields": proc.get("required_fields") or [],
        # Full DB record — honesty: return what is stored, invent nothing.
        "procedure": proc,
        "delegate_action": delegate,
        "note": (
            "Procedure guidance from the knowledge base — not a fabricated "
            "execution result. Use delegate_action when a runnable handler exists."
        ),
    }
    # Surface schema-specific fields at top level when present (no invention).
    for key in _SCHEMA_SPECIFIC_KEYS:
        if key in proc and proc[key] not in (None, "", [], {}):
            payload[key] = proc[key]
    return payload


def normalize_rfi_issues(data: Dict[str, Any], params: Optional[Dict] = None) -> List[Any]:
    """Build a runnable issues list for rfi_generator from data/params.

    Accepts ``issues``, ``auto_risks``, or a singular ``issue`` (str or dict).
    Returns an empty list when nothing runnable is present.
    """
    p = params or {}
    issues = p.get("issues") or data.get("issues") or data.get("auto_risks")
    if isinstance(issues, list) and issues:
        return list(issues)

    singular = p.get("issue") if "issue" in p else data.get("issue")
    if singular is None or singular == "":
        return []
    if isinstance(singular, dict):
        return [singular]
    return [{"description": str(singular)}]


def resolve_procedure_route(action: str) -> Tuple[str, Optional[str]]:
    """Return (procedure kind, delegate_action) for an orchestrator procedure action."""
    return PROCEDURE_ACTION_MAP.get(action, (None, None))  # type: ignore[return-value]
