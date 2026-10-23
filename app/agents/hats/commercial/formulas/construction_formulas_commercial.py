"""Formulas from app.lib.construction_formulas_commercial owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='commercial',
    display_name='Return on investment',
    description='Return on investment as a percentage of cost.',
    inputs={
        'gain': Param('currency', -1e13, 1e13, label='amount gained'),
        'cost': Param('currency', 0, 1e13, label='investment cost'),
    },
    outputs={'net_profit': '-', 'roi_percent': '%'},
)
def roi_calculator(gain: float, cost: float) -> dict:
    """Return on investment: ROI% = (gain - cost) / cost * 100."""
    g, c = float(gain), float(cost)
    if c == 0:
        return {"error": "cost must be non-zero to compute ROI"}
    net = g - c
    roi = net / c * 100.0
    return {
        "net_profit": round(net, 2),
        "roi_percent": round(roi, 2),
        "standard": "arithmetic",
        "note": f"ROI = (gain - cost)/cost*100 = ({g} - {c})/{c}*100 = {roi:.2f}%.",
    }
