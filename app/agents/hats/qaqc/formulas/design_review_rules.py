"""QA/QC hat: the design-review procedure checks, declared as calculators.

The manifest offers these two as formulas. They were plain functions in
``app.core.construction_knowledge`` with no declaration, so the calculator
path answered 'unknown calculation'. Each is declared here, through the same
``@formula`` mechanism every calculator uses, and answers from the rule
function it always meant -- no rule is restated here.

The input is ``design_status``, not ``status``: ``status`` is an envelope
key the calculator path strips before binding.
"""
from __future__ import annotations

from datetime import date

from app.core import construction_knowledge as _ck
from app.lib.formula_registry import Param, formula


@formula(
    owner='qaqc',
    display_name='Design status check',
    description='Whether a status may be written on a design document under the design review procedure.',
    inputs={'design_status': Param('-', label='design review status')},
    outputs={'design_status': '-', 'valid': '-', 'message': '-'},
    main_output='valid',
)
def validate_design_status(design_status: str) -> dict:
    given = str(design_status or "").strip()
    if not given:
        return {"error": "The design review status to check is needed."}
    valid, message = _ck.validate_design_status(given)
    return {"design_status": given, "valid": valid, "message": message}


def _iso_date(value: str) -> date:
    return date.fromisoformat(str(value or "").strip())


@formula(
    owner='qaqc',
    display_name='Design review timeline check',
    description=('Whether the period between a design review package distribution and the '
                 'review workshop meets the design review procedure.'),
    inputs={
        'distribution_date': Param('-', label='distribution date (YYYY-MM-DD)'),
        'workshop_date': Param('-', label='workshop date (YYYY-MM-DD)'),
    },
    outputs={'compliant': '-', 'distribution_period_days': 'd', 'message': '-'},
    main_output='compliant',
)
def check_review_timeline(distribution_date: str, workshop_date: str) -> dict:
    try:
        days = (_iso_date(workshop_date) - _iso_date(distribution_date)).days
    except ValueError:
        return {"error": (f"The dates must be written YYYY-MM-DD; got distribution "
                          f"'{distribution_date}' and workshop '{workshop_date}'.")}
    compliant, message = _ck.check_review_timeline(str(distribution_date).strip(),
                                                   str(workshop_date).strip())
    return {"compliant": compliant, "distribution_period_days": days, "message": message}
