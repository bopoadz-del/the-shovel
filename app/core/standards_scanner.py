"""
Standards-Advisory Scanner (W8)

Broadens the single `no_approved_on_design` rule to a multi-rule scanner that
flags common standards deviations in construction deliverables.

Rules are ADVISORY ONLY -- deviations are highlighted, the task proceeds.
This is a post-synthesis flag gate (like GROUNDING_GATE), not a blocker.

Flag-gated: STANDARDS_SCANNER_BROADENED env var. OFF = only the original
`no_approved_on_design` rule runs. ON = all rules active.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

_ENV_FLAG = os.environ.get("STANDARDS_SCANNER_BROADENED", "").strip()
BROADENED = _ENV_FLAG in ("1", "true", "True", "TRUE", "yes")


@dataclass
class StandardsFlag:
    rule_id: str
    severity: str  # "advisory" | "warning" | "critical"
    message: str
    standard_reference: str = ""
    triggered: bool = False


# -- Rule definitions -----------------------------------------------------

# Each rule is a (name, pattern, message, standard_ref) tuple.
# Patterns are checked against the deliverable text with re.IGNORECASE.

_RULES_CORE = [
    (
        "no_approved_on_design",
        None,  # Special: checked via structured data, not regex
        "No 'APPROVED' stamp found on design document -- fabrication risk",
        "ISO 9001 / project QAP",
    ),
]

_RULES_BROADENED = [
    (
        "missing_itp_reference",
        r"\bITP\b|\binspection\s+test\s+plan\b|\binspection\s+and\s+test\s+plan\b",
        "Deliverable references inspection but cites no ITP number -- traceability gap",
        "ISO 9001 / project ITP register",
    ),
    (
        "unverified_material_substitution",
        r"\bsubstitut|\bequivalent\b|\bor\s+equal\b|\bapproved\s+equal\b",
        "Material substitution claimed without documented approval -- compliance risk",
        "ASTM / BS material specs + project VAR register",
    ),
    (
        "deviated_test_frequency",
        r"\bsample\b|\bfrequency\b|\btest\s+rate\b|\b1\s*(?:in|per)\s*\d+",
        "Test frequency stated without reference to project specification -- may deviate from contract",
        "Contract spec section on QC / testing frequencies",
    ),
    (
        "undocumented_welding_procedure",
        r"\bweld|\bWPS\b|\bPQR\b|\bwelder\s+qualification\b|\bWPQR\b",
        "Welding referenced without WPS/PQR citation -- code compliance gap",
        "AWS D1.1 / BS EN ISO 15614-1 / project welding spec",
    ),
    (
        "missing_commissioning_signoff",
        r"\bcommission|\bhandover\b|\bTOC\b|\btaking\s+over\b|\bpractical\s+completion\b",
        "Commissioning/handover mentioned without sign-off reference -- completion process gap",
        "Project Cx plan / FIDIC Clause 10",
    ),
    (
        "uncited_regulatory_requirement",
        r"\bauthority\b|\bcivil\s+defense\b|\bmunicipalit|\bNWC\b|\bSEC\b|\bMOMRA\b|\bDEWA\b|\bRTA\b|\bDM\b",
        "Regulatory authority requirement stated without specific clause/regulation citation",
        "Jurisdiction-specific building codes (KSA MOMRA / UAE DM etc.)",
    ),
    (
        "missing_method_statement",
        r"\bmethod\s+statement\b|\bwork\s+procedure\b|\bsafe\s+work\s+method\b",
        "Method statement referenced but not provided -- high-risk activity gap",
        "Project HSE plan / OSHA / SABIC work-permit requirements",
    ),
    (
        "unverified_as_built_status",
        r"\bas[-\s]?built\b|\bas\s+constructed\b|\brecord\s+drawing\b",
        "As-built claim without survey/verification reference -- may not reflect actual construction",
        "Project as-built procedure / survey verification",
    ),
]


class StandardsScanner:
    """
    Post-synthesis scanner that flags standards deviations in construction deliverables.

    Usage:
        scanner = StandardsScanner()
        flags = scanner.scan(text, context={"document_type": "design_drawing"})
        # flags is a list of StandardsFlag -- advisory only, not blocking
    """

    def __init__(self, broadened: Optional[bool] = None) -> None:
        self.broadened = BROADENED if broadened is None else broadened
        self.rules = self._build_rules()

    def _build_rules(self) -> List[tuple]:
        rules = list(_RULES_CORE)
        if self.broadened:
            rules.extend(_RULES_BROADENED)
        return rules

    def scan(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[StandardsFlag]:
        """
        Scan deliverable text and return flags for any detected deviations.

        Args:
            text: The deliverable text to scan.
            context: Optional context dict with keys like:
                - document_type: "design_drawing" | "test_report" | "method_statement" etc.
                - has_approved_stamp: bool
                - itp_reference: str
                - wps_reference: str

        Returns:
            List of StandardsFlag, one per triggered rule.
        """
        context = context or {}
        flags: List[StandardsFlag] = []

        for rule_id, pattern, message, standard_ref in self.rules:
            triggered = False

            if rule_id == "no_approved_on_design":
                # Special: check structured context
                has_approved = context.get("has_approved_stamp", None)
                doc_type = context.get("document_type", "")
                if has_approved is False and doc_type in ("design_drawing", "design_doc", "specification"):
                    triggered = True
                elif has_approved is None and "design" in doc_type:
                    # Ambiguous: flag advisory
                    triggered = True
                    message = "Cannot verify APPROVED stamp on design document -- check manually"
            elif pattern is not None:
                # Regex-based rule (case-insensitive)
                if re.search(pattern, text, re.IGNORECASE):
                    triggered = True

            if triggered:
                flags.append(StandardsFlag(
                    rule_id=rule_id,
                    severity="advisory",
                    message=message,
                    standard_reference=standard_ref,
                    triggered=True,
                ))

        return flags

    def scan_to_markdown(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Scan and return a markdown advisory block for appending to deliverables.

        Returns empty string if no flags triggered.
        """
        flags = self.scan(text, context)
        if not flags:
            return ""

        lines = ["\n---", "**Standards Advisory:**"]
        for f in flags:
            ref = f" ({f.standard_reference})" if f.standard_reference else ""
            lines.append(f"- [{f.severity.upper()}] {f.message}{ref}")
        lines.append("\n*These are advisory flags only -- the deliverable proceeds. Review and address as appropriate.*")
        return "\n".join(lines)

    def rule_count(self) -> int:
        """Return total number of active rules."""
        return len(self.rules)

    def rule_ids(self) -> List[str]:
        """Return IDs of all active rules."""
        return [r[0] for r in self.rules]


# -- Module-level convenience ---------------------------------------------

_default_scanner: Optional[StandardsScanner] = None


def get_scanner() -> StandardsScanner:
    """Return singleton scanner instance."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = StandardsScanner()
    return _default_scanner


def scan_text(text: str, context: Optional[Dict[str, Any]] = None) -> List[StandardsFlag]:
    """One-shot scan."""
    return get_scanner().scan(text, context)


def scan_to_md(text: str, context: Optional[Dict[str, Any]] = None) -> str:
    """One-shot scan returning markdown advisory block."""
    return get_scanner().scan_to_markdown(text, context)
