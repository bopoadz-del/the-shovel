"""Formulas from app.lib.construction_formulas_earthwork owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math


@formula(
    owner='design',
    display_name='Slope factor of safety',
    description='Factor of safety of an infinite slope, with and without cohesion.',
    inputs={
        'friction_angle_deg': Param('deg', 0, 60, label='friction angle'),
        'slope_angle_deg': Param('deg', 0, 90, label='slope angle'),
        'cohesion_kpa': Param('kPa', 0, 1000, label='cohesion'),
        'unit_weight_kn_m3': Param('kN/m3', 5, 30, label='soil unit weight'),
        'depth_m': Param('m', 0, 500, label='depth'),
    },
    outputs={'factor_of_safety': '-', 'frictional_term': '-', 'cohesive_term': '-'},
)
def slope_fos_simple(
    friction_angle_deg: float,
    slope_angle_deg: float,
    cohesion_kpa: float = 0.0,
    unit_weight_kn_m3: float = 18.0,
    depth_m: float = 0.0,
) -> dict:
    """Infinite-slope factor of safety (dry). Cohesionless: FoS = tan(phi)/tan(beta).
    With cohesion: FoS = c'/(gamma*z*sin(beta)*cos(beta)) + tan(phi)/tan(beta)."""
    phi = math.radians(float(friction_angle_deg))
    beta = math.radians(float(slope_angle_deg))
    frictional = math.tan(phi) / math.tan(beta)
    cohesive = 0.0
    if cohesion_kpa > 0 and depth_m > 0:
        cohesive = cohesion_kpa / (unit_weight_kn_m3 * depth_m
                                   * math.sin(beta) * math.cos(beta))
    fos = cohesive + frictional
    return {
        "factor_of_safety": round(fos, 3),
        "frictional_term": round(frictional, 3),
        "cohesive_term": round(cohesive, 3),
        "standard": "infinite-slope stability",
        "note": (f"tan(phi)/tan(beta) = tan{friction_angle_deg}/tan{slope_angle_deg} "
                 f"= {frictional:.3f}; cohesive term = {cohesive:.3f}; FoS = {fos:.3f}."),
    }
