"""Contracts hat: the procedure rule checks, declared as calculators.

The manifest offers these three as ``construction_calc:<name>``. They were
plain functions in ``app.core.construction_knowledge`` with no declaration,
so the calculator path answered 'unknown calculation'. Each is declared
here, through the same ``@formula`` mechanism every calculator uses, and
answers from the rule function it always meant -- no rule is restated here.

Input names avoid ``text`` and ``status``: those are envelope keys the
calculator path strips before binding, so an input of that name never binds.
"""
from __future__ import annotations

from app.core import construction_knowledge as _ck
from app.lib.formula_registry import Param, formula


@formula(
    owner='contracts',
    display_name='Document number',
    description='The register number of a project document from its type and sequence number.',
    inputs={
        'doc_type': Param('-', label='document type'),
        'sequence': Param('count', 1, 99999, label='sequence number'),
        'year': Param('count', 1900, 2200, label='year'),
    },
    outputs={'document_number': '-', 'doc_type': '-'},
    main_output='document_number',
)
def generate_doc_number(doc_type: str, sequence: int, year: int | None = None) -> dict:
    kind = str(doc_type or "").strip()
    if not kind:
        return {"error": "A document type is needed (for example RFI, NCR or VO)."}
    return {"document_number": _ck.generate_doc_number(kind, int(sequence), year),
            "doc_type": kind.upper()}


@formula(
    owner='contracts',
    display_name='Critical-rules gate',
    description='Checks a piece of document text against the critical procedure rules and lists every rule it breaks.',
    inputs={'document_text': Param('-', label='document text')},
    outputs={'compliant': '-', 'violation_count': '-', 'violations': '-'},
    main_output='compliant',
)
def enforce_critical_rules(document_text: str) -> dict:
    body = str(document_text or "")
    if not body.strip():
        return {"error": "The document text to check is needed."}
    violations = _ck.enforce_critical_rules(body)
    return {"compliant": not violations, "violation_count": len(violations),
            "violations": violations}


@formula(
    owner='contracts',
    display_name='NCR next status',
    description='The next step of the non-conformance (NCR) workflow after a given status.',
    inputs={'ncr_status': Param('-', label='current NCR status')},
    outputs={'current_status': '-', 'next_status': '-', 'workflow_complete': '-'},
    main_output='next_status',
)
def next_ncr_status(ncr_status: str) -> dict:
    current = str(ncr_status or "").strip().upper().replace(" ", "_").replace("-", "_")
    if current not in _ck.NCR_WORKFLOW_SEQUENCE:
        return {"error": (f"'{ncr_status}' is not an NCR workflow status. The workflow is: "
                          f"{' -> '.join(_ck.NCR_WORKFLOW_SEQUENCE)}.")}
    following = _ck.next_ncr_status(current)
    return {"current_status": current, "next_status": following,
            "workflow_complete": following is None}
