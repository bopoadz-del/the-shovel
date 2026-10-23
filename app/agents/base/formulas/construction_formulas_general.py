"""Formulas from app.lib.construction_formulas_general owned by the base package (every hat).

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner="base",
    display_name='Tolerance check',
    description=("Checks a measured value against a specified value and tolerance: whether it is "
                 "within tolerance, its deviation, and the margin left. The specified value and "
                 "the tolerance come from the user or the project documents."),
    inputs={
        'measured': Param('ratio', -1e9, 1e9, label='measured value'),
        'specified': Param('ratio', -1e9, 1e9, label='specified value'),
        'tolerance_plus': Param('ratio', 0, 1e9, label='plus tolerance'),
        'tolerance_minus': Param('ratio', 0, 1e9, label='minus tolerance'),
    },
    outputs={"within_tolerance": "-", "deviation": "same unit as specified",
             "lower_limit": "same unit as specified", "upper_limit": "same unit as specified",
             "margin": "same unit as specified"},
)
def tolerance_check(measured: float, specified: float, tolerance_plus: float,
                    tolerance_minus: float | None = None) -> dict:
    """Measured vs specified with a +/- tolerance (minus defaults to the plus value)."""
    plus = abs(float(tolerance_plus))
    minus = abs(float(tolerance_minus)) if tolerance_minus is not None else plus
    m, s = float(measured), float(specified)
    low, high = s - minus, s + plus
    within = low <= m <= high
    return {
        "within_tolerance": within,
        "deviation": round(m - s, 6),
        "lower_limit": round(low, 6),
        "upper_limit": round(high, 6),
        "margin": round(min(m - low, high - m), 6),
        "note": (f"measured {m} vs specified {s} (+{plus} / -{minus}): "
                 + ("within tolerance" if within else "outside tolerance") + "."),
    }
