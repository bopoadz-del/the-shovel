"""Formulas from app.lib.construction_formulas owned by the safety hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Dict, Optional


@formula(
    owner='safety',
    display_name='Wind load on formwork',
    description='Wind force and overturning moment on climbing formwork from wind speed and exposed area.',
    inputs={
        'wind_velocity_m_s': Param('m/s', 0, 120, label='wind speed'),
        'formwork_area_m2': Param('m2', 0, 1e6, label='formwork area'),
        'formwork_height_m': Param('m', 0, 200, label='formwork height'),
        'formwork_width_m': Param('m', 0, 1000, label='formwork width'),
        'shape_factor': Param('ratio', 0, 3, label='shape factor'),
    },
    outputs={'wind_pressure_kpa': 'kPa', 'wind_force_kn': 'kN', 'overturning_moment_kn_m': 'kN.m', 'bending_stress_n_m2': 'N/m2'},
)
def wind_load_on_formwork(
    wind_velocity_m_s: float,
    formwork_area_m2: Optional[float] = None,
    formwork_height_m: float = 0.0,
    formwork_width_m: float = 0.0,
    shape_factor: float = 1.0,
) -> Dict[str, float]:
    """Wind on climbing formwork: q = 0.613 x V^2, F = q x A, M = F x h/2.

    When the area is not given, it is the product of the width and the height.
    """
    width = float(formwork_width_m or 0)
    height = float(formwork_height_m or 0)
    area = formwork_area_m2
    if area is None or float(area) <= 0:
        if width > 0 and height > 0:
            area = width * height
        else:
            return {
                "error": (
                    "The wind load check needs the formwork area, or both "
                    "the width and the height."
                ),
            }
    q = 0.613 * wind_velocity_m_s**2
    force_n = q * float(area) * shape_factor
    moment_n_m = force_n * (height / 2)
    z = width * height**2 / 6
    return {
        "wind_pressure_kpa": round(q / 1000, 3),
        "wind_force_kn": round(force_n / 1000, 3),
        "overturning_moment_kn_m": round(moment_n_m / 1000, 3),
        "bending_stress_n_m2": round(moment_n_m / z, 2) if z > 0 else 0,
    }
