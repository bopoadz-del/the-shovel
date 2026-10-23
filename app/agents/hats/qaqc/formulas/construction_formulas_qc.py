"""Formulas from app.lib.construction_formulas_qc owned by the qaqc hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.agents.hats.qaqc.results import qaqc_calculator
from app.agents.hats.qaqc.validation import STANDARD_SPECIMEN_MM
from app.lib.formula_registry import Param, formula
import math


@formula(
    owner='qaqc',
    display_name='Concrete cylinder strength',
    description='Compressive strength from a cylinder test load and diameter.',
    inputs={
        'failure_load_kn': Param('kN', 0, 100000, label='failure load'),
        'cylinder_diameter_mm': Param('mm', STANDARD_SPECIMEN_MM[0], STANDARD_SPECIMEN_MM[1], label='cylinder diameter'),
    },
    outputs={'compressive_strength_mpa': 'MPa', 'cylinder_area_mm2': 'mm2'},
)
@qaqc_calculator
def concrete_cylinders(failure_load_kn: float, cylinder_diameter_mm: float = 150.0) -> dict:
    """Compressive strength from a cylinder test: f = P / A, A = pi/4 * d^2.
    Standard cylinder is 150 mm dia x 300 mm."""
    if float(cylinder_diameter_mm) <= 0:
        return {"error": "cylinder_diameter_mm must be > 0."}
    if float(failure_load_kn) < 0:
        return {"error": "failure_load_kn must be >= 0."}
    P_n = float(failure_load_kn) * 1000.0
    area = math.pi / 4.0 * float(cylinder_diameter_mm) ** 2
    f = P_n / area
    return {
        "compressive_strength_mpa": round(f, 2),
        "cylinder_area_mm2": round(area, 1),
        "standard": "ASTM C39 / BS EN 12390-3",
        "note": (f"A = pi/4*{cylinder_diameter_mm}^2 = {area:.0f} mm2; "
                 f"f = {P_n:.0f}/{area:.0f} = {f:.2f} MPa."),
    }

@formula(
    owner='qaqc',
    display_name='Concrete drying shrinkage',
    description='Drying shrinkage strain of concrete at a given age.',
    inputs={
        'time_days': Param('days', 0, 36500, label='age'),
        'ultimate_shrinkage_microstrain': Param('-', 0, 3000, label='ultimate shrinkage'),
        'time_constant_days': Param('days', 1, 3650, label='shrinkage time constant'),
    },
    outputs={'shrinkage_microstrain': '-', 'shrinkage_strain': '-', 'fraction_of_ultimate': '-'},
)
@qaqc_calculator
def concrete_shrinkage(
    time_days: float,
    ultimate_shrinkage_microstrain: float = 780.0,
    time_constant_days: float = 35.0,
) -> dict:
    """Drying shrinkage strain at time t (ACI 209): esh(t) = t/(f+t)*esh_ult,
    f=35 (moist-cured), esh_ult ~780 microstrain (adjust for RH/size)."""
    t = float(time_days)
    if t < 0:
        return {"error": "time_days must be >= 0."}
    if float(time_constant_days) < 0:
        return {"error": "time_constant_days must be >= 0."}
    f = float(time_constant_days)
    frac = t / (f + t)
    esh = frac * float(ultimate_shrinkage_microstrain)
    return {
        "shrinkage_microstrain": round(esh, 2),
        "shrinkage_strain": round(esh * 1e-6, 8),
        "fraction_of_ultimate": round(frac, 4),
        "standard": "ACI 209R (hyperbolic)",
        "note": (f"esh = t/(f+t)*esh_ult = {t}/({f}+{t})*{ultimate_shrinkage_microstrain} "
                 f"= {esh:.1f} microstrain."),
    }

@formula(
    owner='qaqc',
    display_name='Concrete curing time',
    description='Days of curing for concrete to reach a fraction of its 28-day strength.',
    inputs={
        'target_strength_fraction': Param('fraction', 0, 1, label='target fraction of 28-day strength'),
        'gain_a': Param('-', 0, 50, label='strength-gain constant a'),
        'gain_b': Param('-', 0, 1, label='strength-gain constant b'),
    },
    outputs={'days_to_target': 'days', 'target_fraction': '-'},
)
@qaqc_calculator
def concrete_curing_time(
    target_strength_fraction: float = 0.70,
    gain_a: float = 4.0,
    gain_b: float = 0.85,
) -> dict:
    """Days of moist curing to reach a fraction of f28 (ACI 209 strength gain
    f(t)/f28 = t/(a+b*t); Type I moist a=4, b=0.85). Inverse:
    t = a*frac/(1 - b*frac)."""
    frac = float(target_strength_fraction)
    if frac <= 0:
        return {"error": "target_strength_fraction must be > 0."}
    denom = 1.0 - gain_b * frac
    if denom <= 0:
        return {"error": "target_strength_fraction is unreachable for the given ACI 209 b coefficient."}
    days = gain_a * frac / denom
    return {
        "days_to_target": round(days, 2),
        "target_fraction": frac,
        "standard": "ACI 209R (strength gain) / ACI 308 curing",
        "note": (f"t = a*frac/(1 - b*frac) = {gain_a}*{frac}/(1 - {gain_b}*{frac}) "
                 f"= {days:.2f} days to reach {frac*100:.0f}% f28."),
    }

ADDITIONAL_CALCULATORS = {
    "concrete_cylinders": concrete_cylinders,
    "concrete_shrinkage": concrete_shrinkage,
    "concrete_curing_time": concrete_curing_time,
}
