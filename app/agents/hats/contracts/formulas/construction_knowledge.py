"""Formulas from app.core.construction_knowledge owned by the contracts hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Dict


@formula(
    owner='contracts',
    display_name='Risk score',
    description='Risk score and band from probability and impact ratings.',
    inputs={
        'probability': Param('count', 1, 5, label='probability'),
        'impact': Param('count', 1, 5, label='impact rating'),
    },
    outputs={'probability': '-', 'impact': '-', 'score': '-', 'band': '-', 'requires_action': '-', 'description': '-'},
)
def score_risk(probability: int, impact: int) -> Dict:
    """
    Score a risk per the risk management procedure methodology.
    probability: 1-5, impact: 1-5
    Returns dict with score, band, and formatted statement.
    """
    if not (1 <= probability <= 5 and 1 <= impact <= 5):
        return {"error": "Probability and impact must each be 1-5"}
    score = probability * impact
    if score <= 4:
        band = "GREEN"
    elif score <= 9:
        band = "AMBER"
    else:
        band = "RED"
    return {
        "probability": probability,
        "impact": impact,
        "score": score,
        "band": band,
        "requires_action": band in ("AMBER", "RED"),
        "description": f"Risk Score {score}/25 - {band}",
    }
