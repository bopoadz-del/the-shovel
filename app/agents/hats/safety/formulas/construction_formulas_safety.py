"""Formulas from app.lib.construction_formulas_safety owned by the safety hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from typing import Optional

from app.lib.formula_registry import Param, formula


# Scaffold duty rated loads (OSHA 1926.451): light 25 psf, medium 50, heavy 75.
_SCAFFOLD_DUTY_KPA = {"light": 1.2, "medium": 2.4, "heavy": 3.6}

_OSHA_MAF_KN = 8.0  # OSHA 1926.502 maximum arresting force (body harness)

@formula(
    owner='safety',
    display_name='Scaffold load capacity',
    description='Intended scaffold load and the capacity it must be built to support.',
    inputs={
        'platform_area_m2': Param('m2', 0, 10000, label='platform area'),
        'duty': Param('-', label='scaffold duty'),
        'safety_factor': Param('ratio', 1, 10, label='safety factor'),
    },
    outputs={'duty': '-', 'duty_load_kpa': 'kPa', 'intended_load_kn': 'kN', 'required_capacity_kn': 'kN', 'safety_factor': '-'},
)
def scaffold_load_capacity(
    platform_area_m2: float,
    duty: str = "medium",
    safety_factor: float = 4.0,
) -> dict:
    """Scaffold intended load and the capacity it must be built to support.
    OSHA 1926.451: scaffolds support their own weight + 4x the intended load."""
    d = str(duty or "").strip().lower()
    if d not in _SCAFFOLD_DUTY_KPA:
        return {
            "error": (
                "The scaffold duty must be light, medium or heavy, "
                f"not {duty}. What duty should I use?"
            ),
        }
    duty_kpa = _SCAFFOLD_DUTY_KPA[d]
    area = float(platform_area_m2)
    intended_kn = duty_kpa * area
    required_kn = intended_kn * safety_factor
    return {
        "duty": d,
        "duty_load_kpa": duty_kpa,
        "intended_load_kn": round(intended_kn, 2),
        "required_capacity_kn": round(required_kn, 2),
        "safety_factor": safety_factor,
        "standard": "OSHA 1926.451 Subpart L (4:1)",
        "note": (f"Intended = {duty_kpa} kPa * {area} m2 = {intended_kn:.1f} kN; "
                 f"must support {safety_factor}x = {required_kn:.1f} kN."),
    }

@formula(
    owner='safety',
    display_name='Fall arrest force',
    description='Peak arrest force on a falling worker from mass, free-fall distance and deceleration distance.',
    inputs={
        'worker_mass_kg': Param('kg', 20, 300, label='worker mass'),
        'free_fall_m': Param('m', 0, 10, label='free-fall distance'),
        'deceleration_distance_m': Param('m', 0, 10, label='deceleration distance'),
        'g': Param('-', 9.7, 9.9, label='gravitational acceleration'),
    },
    outputs={'max_arrest_force_kn': 'kN', 'within_osha_limit': '-', 'osha_limit_kn': 'kN'},
)
def fall_arrest_force(
    worker_mass_kg: float = 100.0,
    free_fall_m: float = 1.8,
    deceleration_distance_m: float = 1.0,
    g: float = 9.81,
) -> dict:
    """Peak arresting force from an energy balance: F = W*(1 + H/d), where W =
    m*g, H = free-fall distance, d = deceleration (energy-absorber) distance.
    OSHA 1926.502 caps the maximum arresting force at 8 kN with a body harness."""
    w_n = float(worker_mass_kg) * g
    H = float(free_fall_m)
    d = float(deceleration_distance_m)
    if worker_mass_kg <= 0:
        return {"error": "worker_mass_kg must be > 0."}
    if H < 0:
        return {"error": "free_fall_m must be >= 0."}
    if d <= 0:
        return {"error": "deceleration_distance_m must be > 0."}
    f_n = w_n * (1.0 + H / d)
    f_kn = f_n / 1000.0
    within = f_kn <= _OSHA_MAF_KN
    return {
        "max_arrest_force_kn": round(f_kn, 2),
        "within_osha_limit": within,
        "osha_limit_kn": _OSHA_MAF_KN,
        "standard": "OSHA 1926.502 (MAF <= 8 kN)",
        "note": (
            f"F = W×(1 + H/d) = {w_n:.0f}×(1 + {H}/{d}) = {f_n:.0f} N = "
            f"{f_kn:.2f} kN, "
            f"{'within' if within else 'above'} the 8 kN limit."
        ),
    }

_DEFAULT_MAX_UTILIZATION = 0.85


@formula(
    owner='safety',
    display_name='Crane lift capacity and utilisation',
    description="Net crane capacity after rigging deductions, and the lift's utilisation.",
    inputs={
        'chart_capacity_t': Param('t', 0, 10000, label='chart capacity'),
        'deductions_t': Param('t', 0, 10000, label='deductions'),
        'load_t': Param('t', 0, 10000, label='load'),
        'max_utilization': Param('fraction', 0, 1, label='maximum utilisation'),
    },
    outputs={'net_capacity_t': 't', 'utilization_percent': '%', 'passed': '-', 'max_utilization_percent': '%'},
)
def crane_lift_capacity(
    chart_capacity_t: float,
    deductions_t: float,
    load_t: float,
    max_utilization: Optional[float] = None,
) -> dict:
    """Net crane capacity after rigging/block/jib deductions, and the lift
    utilisation. Net = chart - deductions; util = load/net.

    A load above net capacity fails with a reason and stops. A utilisation
    limit the caller did not give is not treated as a project figure; the
    built-in guide is labelled as a default when it is used.
    """
    if float(chart_capacity_t) <= 0:
        return {"error": "chart_capacity_t must be > 0."}
    if float(deductions_t) < 0 or float(load_t) < 0:
        return {"error": "deductions_t and load_t must be >= 0."}
    net = float(chart_capacity_t) - float(deductions_t)
    if net <= 0:
        return {"error": "deductions_t must be less than chart_capacity_t (net capacity would be <= 0)."}
    load = float(load_t)
    util = load / net
    limit_given = max_utilization is not None
    limit = float(max_utilization) if limit_given else _DEFAULT_MAX_UTILIZATION
    if load > net:
        out = {
            "net_capacity_t": round(net, 2),
            "utilization_percent": round(util * 100.0, 2),
            "passed": False,
            "reason": (
                f"The load {load:g} t exceeds the net capacity {net:g} t."
            ),
            "max_utilization_percent": round(limit * 100.0, 1),
            "standard": "crane lift plan (de-rated chart)",
            "note": (
                f"The load {load:g} t exceeds the net capacity {net:g} t."
            ),
        }
        if not limit_given:
            out["max_utilization_is_default"] = True
        return out
    passed = util <= limit
    limit_words = (
        f"the default {limit*100:.0f}% limit"
        if not limit_given
        else f"the {limit*100:.0f}% limit"
    )
    out = {
        "net_capacity_t": round(net, 2),
        "utilization_percent": round(util * 100.0, 2),
        "passed": passed,
        "max_utilization_percent": round(limit * 100.0, 1),
        "standard": "crane lift plan (de-rated chart)",
        "note": (
            f"Net = {chart_capacity_t} − {deductions_t} = {net:.1f} t; "
            f"utilisation = {load_t}/{net:.1f} = {util*100:.1f}%, "
            f"which {'is within' if passed else 'exceeds'} "
            f"{limit_words}."
        ),
    }
    if not limit_given:
        out["max_utilization_is_default"] = True
    return out

ADDITIONAL_CALCULATORS = {
    "scaffold_load_capacity": scaffold_load_capacity,
    "fall_arrest_force": fall_arrest_force,
    "crane_lift_capacity": crane_lift_capacity,
}

from app.agents.hats.safety.defaults import install_default_disclosure

install_default_disclosure()
