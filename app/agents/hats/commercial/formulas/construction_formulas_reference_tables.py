"""Formulas from app.lib.construction_formulas_reference_tables owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


# LEED v4 BD+C certification thresholds (points out of 110).
_LEED_LEVELS = [(80, "Platinum"), (60, "Gold"), (50, "Silver"), (40, "Certified")]

@formula(
    owner='commercial',
    display_name='LEED certification level',
    description='Maps a LEED point total to its certification level.',
    inputs={
        'points': Param('count', 0, 110, label='LEED points'),
    },
    outputs={'points': '-', 'certification_level': '-'},
)
def leed_points_estimate(points: float) -> dict:
    """Map a LEED v4 point total to a certification level. Certified 40-49,
    Silver 50-59, Gold 60-79, Platinum 80+."""
    p = float(points)
    level = "Not certified"
    for threshold, name in _LEED_LEVELS:
        if p >= threshold:
            level = name
            break
    return {
        "kind": "reference_table",
        "points": p,
        "certification_level": level,
        "standard": "LEED v4 BD+C thresholds",
        "note": f"{p} points -> {level} (Certified>=40, Silver>=50, Gold>=60, Platinum>=80).",
    }
