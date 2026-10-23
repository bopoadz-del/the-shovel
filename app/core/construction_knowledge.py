"""
construction_knowledge.py
==========================
Live knowledge module for the construction platform.
Any block imports this to get procedure rules, validate inputs,
enforce critical business rules, and generate correct document numbers.

Place at: app/core/construction_knowledge.py

Usage:
    from app.core.construction_knowledge import (
        ConstructionKnowledge,
        validate_design_status,
        generate_doc_number,
        get_procedure,
        enforce_critical_rules,
    )
"""
from __future__ import annotations

from app.lib.formula_registry import formula

import json

import logging

import re

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.hats.commercial.formulas.construction_knowledge import (  # noqa: F401 -- moved
    calculate_evm,
    calculate_payment,
)
from app.agents.hats.contracts.formulas.construction_knowledge import (  # noqa: F401 -- moved
    score_risk,
)
from app.agents.hats.procurement.formulas.construction_knowledge import (  # noqa: F401 -- moved
    evaluate_tender,
)

logger = logging.getLogger(__name__)

_DB_PATH = Path(__file__).parent.parent / "data" / "procedures" / "procedures_db.json"

_SYSTEM_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "construction_expert.txt"

_procedures_db: Optional[Dict] = None

def _load_db() -> Dict:
    global _procedures_db
    if _procedures_db is None:
        if _DB_PATH.exists():
            _procedures_db = json.loads(_DB_PATH.read_text(encoding="utf-8"))
        else:
            _procedures_db = {}
    return _procedures_db

def get_procedure(procedure_id: str) -> Optional[Dict]:
    """Return the catalogue record for a procedure kind, e.g. ``"non_conformance"``."""
    db = _load_db()
    return db.get("procedures", {}).get(procedure_id)

def get_system_prompt() -> str:
    """Return the construction expert system prompt text for injection into chat."""
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return ""

CRITICAL_RULES = {

    "no_approved_on_design": {
        "rule": "Never use 'APPROVED' on design documents.",
        "correct": ["accepted", "for comment", "buy-off"],
        "procedure": "design_review",
        "violation_message": (
            "The word 'APPROVED' is contractually prohibited on design documents. "
            "Use 'accepted', 'for comment', or 'buy-off' instead."
        ),
    },

    "no_work_before_dd_approval": {
        "rule": "No work may proceed on any instruction until the Design Directive has Employer approval.",
        "procedure": "design_directive",
        "violation_message": (
            "Work cannot proceed on this instruction. "
            "A Design Directive (DD) must be approved by the Employer first (design directive procedure)."
        ),
    },

    "rfm_not_vo": {
        "rule": "An RFM is an instruction only - NOT a Variation Order. Only a signed VO changes the contract.",
        "procedure": "change_management",
        "violation_message": (
            "An RFM (Request for Modification) is an instruction, not a contract amendment. "
            "A signed Variation Order (VO) is required to change the contract (change management procedure)."
        ),
    },

    "stop_work_resumption": {
        "rule": "No work resumption after STOP WORK without PMC Project Director written sign-off.",
        "procedure": "hse_audit",
        "violation_message": (
            "Work cannot resume after a STOP WORK order without the PMC Project Director's "
            "written sign-off (HSE audit and inspection procedure)."
        ),
    },

    "payment_form_controlled": {
        "rule": "Payment Request Form is a controlled document. Cannot be modified without programme-management approval.",
        "procedure": "interim_payment",
        "violation_message": (
            "The Payment Request Form is a controlled document and cannot be modified "
            "without programme-management approval (interim payment procedure)."
        ),
    },

    "ncr_not_ir": {
        "rule": "An NCR (non-conformance) and an Inspection Rejection (IR) are different.",
        "procedure": "non_conformance/inspection_request",
        "violation_message": (
            "An Inspection Rejection (inspection request procedure) is a routine hold that may or may not "
            "escalate to an NCR. An NCR (non-conformance procedure) is a formal non-conformance record."
        ),
    },

    "design_review_min_distribution": {
        "rule": "Design review package must be distributed at least 7 calendar days before the workshop.",
        "procedure": "design_review",
        "violation_message": (
            "The design review package must be distributed a minimum of 7 calendar days "
            "before the workshop date (design review and acceptance procedure)."
        ),
    },
}

def enforce_critical_rules(text: str, document_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Scan text for critical rule violations.
    Returns a list of violation dicts: {rule_id, message, procedure, procedure_id}.

    ``procedure`` is the catalogue kind. ``procedure_id`` is the live
    procedure document's own code when ``document_names`` (the active
    project's documents, read at run time) contains one of that kind, else
    the kind id.
    """
    violations = []
    text_lower = text.lower()

    # Check for forbidden "APPROVED" on design docs
    if re.search(r"\bapprove[ds]?\b|\bapproval\b", text_lower):
        design_context_keywords = [
            "design", "drawing", "package", "review", "consultant",
            "architect", "engineer", "document"
        ]
        if any(kw in text_lower for kw in design_context_keywords):
            violations.append({
                "rule_id": "no_approved_on_design",
                **CRITICAL_RULES["no_approved_on_design"]
            })

    if violations:
        from app.core.procedure_catalogue import resolve_procedure

        for v in violations:
            kind = str(v.get("procedure") or "").split("/")[0]
            v["procedure_id"] = resolve_procedure(kind, document_names)["procedure_id"]
    return violations

def generate_doc_number(doc_type: str, sequence: int, year: Optional[int] = None) -> str:
    """
    Generate a correctly formatted document number.

    Examples:
        generate_doc_number("RFI", 42)       -> "RFI-0042"
        generate_doc_number("NCR", 1, 2024)  -> "NCR-2024-001"
        generate_doc_number("VO", 15)        -> "VO-015"
        generate_doc_number("DD", 23)        -> "DD-023"
    """
    templates = {
        "RFI": f"RFI-{sequence:04d}",
        "NCR": f"NCR-{year}-{sequence:03d}" if year else f"NCR-{sequence:03d}",
        "IR":  f"IR-{sequence:04d}",
        "VO":  f"VO-{sequence:03d}",
        "RFM": f"RFM-{sequence:03d}",
        "JR":  f"JR-{sequence:04d}",
        "PDN": f"PDN-{year}-{sequence:03d}" if year else f"PDN-{sequence:03d}",
        "DD":  f"DD-{sequence:03d}",
        "PR":  f"PR-{sequence:04d}",
        "WP":  f"WP-{sequence:04d}",
    }
    return templates.get(doc_type.upper(), f"{doc_type.upper()}-{sequence:04d}")

VALID_DESIGN_STATUSES = {"FOR_COMMENT", "ACCEPTANCE", "BUY_OFF", "PENDING_DEBRIEF", "SUPERSEDED"}

FORBIDDEN_DESIGN_STATUSES = {"APPROVED", "APPROVAL", "SIGN_OFF"}

def validate_design_status(status: str) -> Tuple[bool, str]:
    """
    Validate a design review status.
    Returns (is_valid, message).
    """
    s = status.upper().replace(" ", "_").replace("-", "_")
    if s in FORBIDDEN_DESIGN_STATUSES:
        return (
            False,
            f"'{status}' is forbidden on design documents (design review and acceptance procedure). "
            f"Use one of: {', '.join(VALID_DESIGN_STATUSES)}"
        )
    if s not in VALID_DESIGN_STATUSES:
        return (
            False,
            f"'{status}' is not a valid design review status. "
            f"Valid statuses: {', '.join(VALID_DESIGN_STATUSES)}"
        )
    return True, f"Status '{status}' is valid."

def check_review_timeline(distribution_date: str, workshop_date: str) -> Tuple[bool, str]:
    """
    Check that the review distribution period meets the design review procedure.
    distribution_date and workshop_date: 'YYYY-MM-DD' strings.
    Returns (compliant, message).
    """
    from datetime import date, timedelta
    try:
        dist = date.fromisoformat(distribution_date)
        workshop = date.fromisoformat(workshop_date)
        delta = (workshop - dist).days
        if delta < 7:
            return (
                False,
                f"Only {delta} calendar days between distribution and workshop. "
                f"The design review procedure requires minimum 7 days. Workshop must be no earlier than "
                f"{(dist + timedelta(days=7)).isoformat()}."
            )
        if delta > 14:
            return (
                True,
                f"Distribution period is {delta} days. The design review procedure maximum is 14 days - "
                f"consider whether the package can be issued closer to the workshop."
            )
        return True, f"Timeline compliant: {delta} calendar days distribution period."
    except Exception as e:
        return False, f"Could not parse dates: {e}"

VALID_NCR_DISPOSITIONS = {"USE_AS_IS", "REPAIR", "REJECT", "CONCESSION"}

NCR_WORKFLOW_SEQUENCE = [
    "RAISED", "ACKNOWLEDGED", "DISPOSITION_PROPOSED",
    "DISPOSITION_REVIEWED", "APPROVED", "IMPLEMENTING", "VERIFYING", "CLOSED"
]

def validate_ncr_disposition(disposition: str) -> Tuple[bool, str]:
    d = disposition.upper().replace(" ", "_").replace("-", "_")
    if d not in VALID_NCR_DISPOSITIONS:
        return (
            False,
            f"'{disposition}' is not a valid NCR disposition (non-conformance procedure). "
            f"Valid options: {', '.join(VALID_NCR_DISPOSITIONS)}"
        )
    descriptions = {
        "USE_AS_IS": "Non-conformance acceptable without repair",
        "REPAIR": "Bring into conformance by rework",
        "REJECT": "Remove and replace",
        "CONCESSION": "Waiver from Employer required",
    }
    return True, f"Valid disposition: {d} - {descriptions[d]}"

def next_ncr_status(current_status: str) -> Optional[str]:
    """Return the next status in the NCR workflow sequence."""
    current = current_status.upper()
    try:
        idx = NCR_WORKFLOW_SEQUENCE.index(current)
    except ValueError:
        logger.warning("unknown NCR status %r; no next step", current, exc_info=True)
        return None
    return NCR_WORKFLOW_SEQUENCE[idx + 1] if idx + 1 < len(NCR_WORKFLOW_SEQUENCE) else None

class ConstructionKnowledge:
    """
    Single import point for all construction domain knowledge.
    Blocks instantiate this once and call methods as needed.

    Example:
        from app.core.construction_knowledge import ConstructionKnowledge
        ck = ConstructionKnowledge()

        # Validate design status
        ok, msg = ck.validate_design_status("APPROVED")  # -> False, violation message

        # Generate a doc number
        num = ck.generate_doc_number("NCR", 1, year=2024)  # -> "NCR-2024-001"

        # Get full procedure rules
        proc = ck.get_procedure("non_conformance")  # -> dict with all NCR rules

        # Score a risk
        risk = ck.score_risk(4, 3)  # -> {score: 12, band: "RED", ...}

        # Get system prompt for chat injection
        prompt = ck.get_system_prompt()
    """

    def validate_design_status(self, status: str) -> Tuple[bool, str]:
        return validate_design_status(status)

    def check_review_timeline(self, distribution_date: str, workshop_date: str) -> Tuple[bool, str]:
        return check_review_timeline(distribution_date, workshop_date)

    def validate_ncr_disposition(self, disposition: str) -> Tuple[bool, str]:
        return validate_ncr_disposition(disposition)

    def next_ncr_status(self, current_status: str) -> Optional[str]:
        return next_ncr_status(current_status)

    def score_risk(self, probability: int, impact: int) -> Dict:
        return score_risk(probability, impact)

    def calculate_payment(self, claimed: float, certified: float, **kwargs) -> Dict:
        return calculate_payment(claimed, certified, **kwargs)

    def calculate_evm(
        self,
        bac=None,
        bcwp=None,
        bcws=None,
        acwp=None,
        *,
        pv=None,
        ev=None,
        ac=None,
    ) -> Dict:
        return calculate_evm(
            bac=bac, bcwp=bcwp, bcws=bcws, acwp=acwp, pv=pv, ev=ev, ac=ac,
        )

    def evaluate_tender(self, tenderers: List[Dict], weights: Optional[Dict] = None) -> Dict:
        return evaluate_tender(tenderers, weights)

    def generate_doc_number(self, doc_type: str, sequence: int, year: Optional[int] = None) -> str:
        return generate_doc_number(doc_type, sequence, year)

    def enforce_critical_rules(self, text: str) -> List[Dict]:
        return enforce_critical_rules(text)

    def get_procedure(self, procedure_id: str) -> Optional[Dict]:
        return get_procedure(procedure_id)

    def get_system_prompt(self) -> str:
        return get_system_prompt()

    def get_workflow(self, procedure_id: str) -> Optional[List[str]]:
        proc = get_procedure(procedure_id)
        if not proc:
            return None
        return proc.get("workflow") or proc.get("workflow_type_a")

    def get_roles(self, procedure_id: str) -> Optional[Dict]:
        proc = get_procedure(procedure_id)
        return proc.get("roles") if proc else None

    def get_critical_rule(self, rule_id: str) -> Optional[Dict]:
        return CRITICAL_RULES.get(rule_id)

    def list_procedures(self) -> List[str]:
        db = _load_db()
        return list(db.get("procedures", {}).keys())
