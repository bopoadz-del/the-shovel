"""Formulas from app.lib.construction_formulas_additions owned by the safety hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import formula


@formula(
    owner='safety',
    display_name='Guardrail top-rail height',
    description='Required guardrail top-rail height for fall protection.',
    inputs={},
    outputs={'top_rail_height_in': '-', 'top_rail_height_mm': 'mm', 'top_rail_height_m': 'm', 'mid_rail_height_in': '-', 'mid_rail_height_mm': 'mm', 'mid_rail_height_m': 'm', 'tolerance_in': '-', 'tolerance_mm': 'mm', 'scope': '-'},
)
def guardrail_top_rail_height() -> dict:
    """
    OSHA guardrail top-rail height for fall protection.

    OSHA 29 CFR 1926.502(b): top edge height of top rails shall be
    42 inches +/- 3 inches (1.07 m +/- 76 mm) above the walking/working
    level. Mid-rails at halfway point.

    Returns dict with heights in imperial and metric.
    """
    return {
        "top_rail_height_in": 42,
        "top_rail_height_mm": 1067,
        "top_rail_height_m": 1.07,
        "mid_rail_height_in": 21,
        "mid_rail_height_mm": 533,
        "mid_rail_height_m": 0.53,
        "tolerance_in": 3,
        "tolerance_mm": 76,
        "standard": "OSHA 29 CFR 1926.502(b)",
        "scope": "construction",
        "note": (
            "Top rail 42 in +/- 3 in above walking/working level. "
            "General industry (1910.29) same requirement."
        ),
    }
