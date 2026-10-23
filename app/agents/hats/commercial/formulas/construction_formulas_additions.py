"""Formulas from app.lib.construction_formulas_additions owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='commercial',
    display_name='Interim payment',
    description='Net interim payment from a gross valuation and a retention percentage.',
    inputs={
        'gross_valuation': Param('currency', 0, 1e13, label='gross valuation'),
        'retention_percent': Param('%', 0, 100, label='retention'),
    },
    outputs={'gross_valuation': 'currency', 'retention_percent': '%', 'retention_amount': 'currency', 'net_payment': 'currency'},
)
def calculate_interim_payment(
    gross_valuation: float | int,
    retention_percent: float | int = 10.0,
) -> dict:
    """
    Calculate interim payment after retention deduction.

    FIDIC Red Book Clause 14.3: the Engineer issues an Interim Payment
    Certificate within 28 days of receiving the Statement. Retention is
    deducted per Clause 14.3 unless the limit (typically 5% of Accepted
    Contract Amount) has been reached.

    Args:
        gross_valuation: Certified gross amount for this period.
        retention_percent: Retention percentage (default 10%).

    Returns:
        dict with gross, retention, net payment, and breakdown.
    """
    gross = float(gross_valuation)
    if gross < 0:
        return {"error": "gross_valuation must be >= 0."}
    if not (0.0 <= float(retention_percent) <= 100.0):
        return {"error": "retention_percent must be between 0 and 100."}
    retention_rate = float(retention_percent) / 100.0
    retention_amount = gross * retention_rate
    net_payment = gross - retention_amount

    return {
        "gross_valuation": round(gross, 2),
        "retention_percent": float(retention_percent),
        "retention_amount": round(retention_amount, 2),
        "net_payment": round(net_payment, 2),
        "calculation": f"{gross} - ({gross} x {retention_percent}%) = {round(net_payment, 2)}",
        "standard": "FIDIC Red Book Clause 14.3",
        "note": (
            "Retention default 10%; FIDIC cap is 5% of Accepted Contract Amount. "
            "Verify if retention cap has been reached on this project."
        ),
    }
