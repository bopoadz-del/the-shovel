"""Formulas from app.lib.construction_formulas owned by the qaqc hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.agents.hats.qaqc.results import qaqc_calculator
from app.lib.formula_registry import Param, formula, grade_label
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List
from app.agents.base.formulas.construction_formulas_shared import (  # noqa: F401
    _is_aci_code,
)

_SERIES_FIGURE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _series_figures(value: Any) -> List[float]:
    """Every figure written in a series: a list, a JSON array, or numbers
    separated by commas or spaces. One figure is one reading."""
    if value is None:
        return []
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(_series_figures(item))
        return out
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            obj = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            obj = None
        if isinstance(obj, (list, tuple)):
            return _series_figures(obj)
    return [float(match.group(0)) for match in re.finditer(_SERIES_FIGURE, text)]


def _rupture_form(code: Any) -> str:
    """The ACI form when the code names it; otherwise the default form."""
    if _is_aci_code(code):
        return "aci"
    return "metric_technical"


@dataclass
class FormworkStrikingResult:
    recommended_hours: float
    bs8110_minimum_hours: float
    ciria_minimum_strength_n_mm2: float
    design_required_strength_n_mm2: float
    actual_early_strength_n_mm2: float
    based_on: str
    notes: List[str] = field(default_factory=list)
    verdict: str = ""

@formula(
    owner='qaqc',
    display_name='Formwork striking time',
    description='Earliest formwork striking time from element type, temperature and test strength.',
    inputs={
        'concrete_strength_7h': Param('N/mm2', 0, 150, label='concrete strength at striking'),
        'design_required_strength': Param('N/mm2', 0, 150, label='required striking strength'),
        'ciria_surface_strength': Param('N/mm2', 0, 150, label='CIRIA surface strength'),
        'bs8110_minimum_hours': Param('h', 0, 1000, label='minimum striking time'),
        'proposed_hours': Param('h', 0, 1000, label='proposed striking time'),
    },
    outputs={'recommended_hours': 'h', 'bs8110_minimum_hours': 'h', 'ciria_minimum_strength_n_mm2': 'N/mm2', 'design_required_strength_n_mm2': 'N/mm2', 'actual_early_strength_n_mm2': 'N/mm2'},
)
@qaqc_calculator
def formwork_striking_time(
    concrete_strength_7h: float = 7.0,
    design_required_strength: float = 0.332,
    ciria_surface_strength: float = 3.0,
    bs8110_minimum_hours: float = 12.0,
    proposed_hours: float = 8.0,
) -> FormworkStrikingResult:
    """Safe formwork striking per BS 8110 + CIRIA + cube test data."""
    notes = [
        f"BS 8110 asks for at least {bs8110_minimum_hours} hours for wall formwork.",
        f"CIRIA asks for a surface strength of at least {ciria_surface_strength} N/mm².",
        f"The design required strength is {design_required_strength} N/mm².",
        f"The concrete strength at striking is {concrete_strength_7h} N/mm².",
    ]
    if concrete_strength_7h >= ciria_surface_strength:
        notes.append(
            f"The concrete strength at striking {concrete_strength_7h} N/mm² meets the CIRIA "
            f"surface minimum of {ciria_surface_strength} N/mm²."
        )
    else:
        notes.append(
            f"The concrete strength at striking {concrete_strength_7h} N/mm² is below the CIRIA "
            f"surface minimum of {ciria_surface_strength} N/mm²."
        )
    strength_ok = (
        concrete_strength_7h >= ciria_surface_strength
        and concrete_strength_7h >= design_required_strength
    )
    based_on = "test_data" if strength_ok else "insufficient"
    recommended_hours = (
        proposed_hours if strength_ok else max(proposed_hours, bs8110_minimum_hours)
    )
    if strength_ok:
        verdict = (
            f"Formwork may be struck after {recommended_hours} hours; "
            "the concrete strength at striking meets both minima."
        )
    else:
        verdict = (
            f"Formwork should stay in place for {recommended_hours} hours; "
            "the concrete strength at striking does not yet meet both minima."
        )
    notes.append(verdict)
    return FormworkStrikingResult(
        recommended_hours=recommended_hours,
        bs8110_minimum_hours=bs8110_minimum_hours,
        ciria_minimum_strength_n_mm2=ciria_surface_strength,
        design_required_strength_n_mm2=design_required_strength,
        actual_early_strength_n_mm2=concrete_strength_7h,
        based_on=based_on,
        notes=notes,
        verdict=verdict,
    )

@formula(
    owner='qaqc',
    display_name='Aggregate fineness modulus',
    description='Fineness modulus of an aggregate from its cumulative sieve retentions.',
    inputs={
        'sieve_retained_percentages': Param('%', 0, 100, label='cumulative percentages retained'),
    },
    outputs={'fineness_modulus': '-'},
)
@qaqc_calculator
def fineness_modulus(sieve_retained_percentages: List[float]) -> float:
    """FM = (cumulative % retained on all sieves) / 100."""
    figures = _series_figures(sieve_retained_percentages)
    if not figures:
        return 0.0
    outside = next((value for value in figures if value < 0.0 or value > 100.0), None)
    if outside is not None:
        return {
            "error": (
                "Each cumulative percentage retained must be between 0 and 100 %, "
                f"not {outside}."
            ),
        }
    return round(sum(figures) / 100.0, 2)

@formula(
    owner='qaqc',
    display_name='Concrete modulus of rupture',
    description='Flexural tensile strength (modulus of rupture) of concrete, to the selected code.',
    inputs={
        'fck_n_mm2': Param('N/mm2', 5, 150, label='characteristic concrete strength', grade='concrete'),
        'code': Param('-', label='design code'),
    },
    outputs={'fck_n_mm2': 'N/mm2', 'value': 'currency', 'unit': '-', 'modulus_of_rupture_n_mm2': 'N/mm2', 'split_cylinder_aci_n_mm2': 'N/mm2', 'tensile_pct_of_compressive': '-'},
)
@qaqc_calculator
def modulus_of_rupture(fck_n_mm2: float, code: str = "metric_technical") -> Dict[str, float]:
    """Tensile strength. Two forms, and the caller's code decides which.

    * ``metric_technical`` (default, unchanged): fr = 2.4 sqrt(f'c) with f'c
      in kg/cm2, plus the split-cylinder figure -- for C30 that is 4.157 MPa.
    * ``aci``: ACI 318-19 Eq. 19.2.3.1, fr = 0.62 sqrt(f'c) in MPa (normal
      weight, lambda = 1.0) -- for C30, 3.40 MPa.

    Live, asked as a follow-up naming ACI 318-19 for f'c = 30: this
    returned 4.157 and the operator was shown 4.16 MPa. The sibling
    ``modulus_of_elasticity_concrete`` had already been given this parameter;
    this one was left behind, so an ACI question got the metric-technical
    answer with no sign that the code had been ignored.
    """
    # Not _norm_code: it defaults every unknown string to ACI, which would
    # silently change this function's default form. An unknown code keeps
    # the metric-technical form (the signature default).
    form = _rupture_form(code)
    fck = float(fck_n_mm2)
    if form == "aci":
        fr_mpa = round(0.62 * math.sqrt(fck), 3)
        return {
            "fck_n_mm2": fck_n_mm2,
            "value": fr_mpa,
            "unit": "MPa",
            "modulus_of_rupture_n_mm2": fr_mpa,
            "standard": "ACI 318-19 Eq. 19.2.3.1: 0.62*sqrt(f'c) MPa, normal weight",
        }
    fck_kg = fck * 10
    fr = 2.4 * math.sqrt(fck_kg) / 10
    ft_aci = 1.78 * math.sqrt(fck_kg) / 10
    return {
        "fck_n_mm2": fck_n_mm2,
        "value": round(fr, 3),
        "unit": "MPa",
        "modulus_of_rupture_n_mm2": round(fr, 3),
        "split_cylinder_aci_n_mm2": round(ft_aci, 3),
        "tensile_pct_of_compressive": round(ft_aci / fck_n_mm2 * 100, 1),
        "standard": "2.4*sqrt(f'c) with f'c in kg/cm2 (metric-technical form)",
    }

@formula(
    owner='qaqc',
    display_name='Mass concrete thermal check',
    description='Mass-concrete thermal check: core temperature and core-to-surface difference against limits.',
    inputs={
        'core_temp_c': Param('degC', -30, 120, label='core temperature'),
        'surface_temp_c': Param('degC', -50, 100, label='surface temperature'),
    },
    outputs={'core_temp_c': 'degC', 'surface_temp_c': 'degC', 'delta_t_c': 'degC', 'core_ok': '-', 'delta_ok': '-', 'thermal_cracking_risk': '-'},
)
@qaqc_calculator
def concrete_thermal_cracking_check(core_temp_c: float, surface_temp_c: float) -> Dict[str, Any]:
    """Mass concrete limits: core <= 70C, delta_T <= 20C."""
    delta_t = core_temp_c - surface_temp_c
    core_ok = core_temp_c <= 70
    delta_ok = delta_t <= 20
    notes = []
    if not core_ok:
        notes.append(f"The core temperature {core_temp_c} °C is above the 70 °C limit.")
    if not delta_ok:
        notes.append(f"The core-to-surface difference {delta_t} °C is above the 20 °C limit.")
    if core_ok and delta_ok:
        notes.append(
            f"The core temperature {core_temp_c} °C and the difference {delta_t} °C "
            "are within the limits."
        )
    return {
        "core_temp_c": core_temp_c, "surface_temp_c": surface_temp_c,
        "delta_t_c": round(delta_t, 1), "core_ok": core_ok, "delta_ok": delta_ok,
        "thermal_cracking_risk": not (core_ok and delta_ok), "notes": notes,
    }


def _pressure_in(n_mm2: float, unit: str) -> float:
    """A pressure in N/mm2 in another unit, by the platform's one conversion table."""
    from app.lib.construction_formulas import _convert_units

    return float(_convert_units(float(n_mm2), "N/mm2", unit)["value_out"])

@formula(
    owner='qaqc',
    display_name='Post-tensioning grout checks',
    description='Grouting pressure and strength checks for post-tensioning ducts.',
    inputs={
        'tendon_duct_diameter_mm': Param('mm', 20, 150, label='duct diameter'),
        'required_pressure_n_mm2': Param('N/mm2', 0, 10, label='grout pressure'),
        'strength_28d_n_mm2': Param('N/mm2', 1, 150, label='28-day strength'),
        'strength_7d_n_mm2': Param('N/mm2', 1, 150, label='7-day strength'),
        'mixing_time_minutes': Param('min', 0, 120, label='mixing time'),
    },
    outputs={'duct_area_mm2': 'mm2', 'grout_volume_l_m': 'm', 'pressure_n_mm2': 'N/mm2', 'pressure_kg_cm2': '-', 'pressure_psi': '-', 'strength_28d_n_mm2': 'N/mm2', 'strength_7d_n_mm2': 'N/mm2', 'mixing_time_min': '-'},
)
@qaqc_calculator
def grout_pressure_calc(
    tendon_duct_diameter_mm: float,
    required_pressure_n_mm2: float = 0.5,
    strength_28d_n_mm2: float = 35.0,
    strength_7d_n_mm2: float = 20.0,
    mixing_time_minutes: float = 2.0,
) -> Dict[str, float]:
    """Grouting for PT: 0.5 N/mm2 (5 kg/cm2 = 75 PSI). 35 N/mm2 @ 28d, >=20 @ 7d."""
    if tendon_duct_diameter_mm <= 0:
        return {"error": "tendon_duct_diameter_mm must be > 0"}
    area_mm2 = math.pi * (tendon_duct_diameter_mm / 2)**2
    return {
        "duct_area_mm2": round(area_mm2, 2),
        "grout_volume_l_m": round(area_mm2 * 1000 / 1e6, 3),
        "pressure_n_mm2": required_pressure_n_mm2,
        # One conversion table for every unit (the platform's converter), not
        # factors written here: 0.5 N/mm2 = 72.5 psi = 5.1 kgf/cm2.
        "pressure_kg_cm2": round(_pressure_in(required_pressure_n_mm2, "kgf/cm2"), 1),
        "pressure_psi": round(_pressure_in(required_pressure_n_mm2, "psi"), 1),
        "strength_28d_n_mm2": strength_28d_n_mm2,
        "strength_7d_n_mm2": strength_7d_n_mm2,
        "mixing_time_min": mixing_time_minutes,
    }

@formula(
    owner='qaqc',
    display_name='Concrete strength by maturity',
    description='Concrete strength from its temperature history by the maturity method.',
    inputs={
        'temperature_history_c': Param('degC', -30, 100, label='recorded temperatures'),
        'time_intervals_hours': Param('h', 0, 100000, label='time intervals'),
        'datum_temperature': Param('degC', -30, 10, label='datum temperature'),
        'strength_28d_n_mm2': Param('N/mm2', 1, 150, label='28-day strength', grade='concrete'),
        'reference_temperature_c': Param('degC', -30, 80, label='reference temperature'),
        'gain_a': Param('-', 0, 50, label='strength-gain constant a'),
        'gain_b': Param('-', 0, 1, label='strength-gain constant b'),
    },
    outputs={'predicted_strength_n_mm2': 'N/mm2', 'percent_of_28d': '%', 'maturity_index_c_hrs': 'degC.h', 'equivalent_age_days': 'days'},
)
@qaqc_calculator
def concrete_maturity_strength(
    temperature_history_c: List[float],
    time_intervals_hours: List[float],
    datum_temperature: float = -10.0,
    strength_28d_n_mm2: float = 40.0,
    reference_temperature_c: float = 20.0,
    gain_a: float = 4.0,
    gain_b: float = 0.85,
) -> Dict[str, Any]:
    """Nurse-Saul maturity + ACI 209 Type I moist strength gain.

    MI = sum(max(0, T - T0) * dt)  (T0 = -10 °C by default).
    Equivalent age te (hours) = MI / (Tref - T0); te_days = te / 24.
    f(t)/f28 = te_days / (a + b*te_days)  with a=4, b=0.85 (ACI 209R).
    """
    temps = _series_figures(temperature_history_c)
    hours = _series_figures(time_intervals_hours)
    if len(hours) == 1 and not isinstance(time_intervals_hours, (list, tuple)):
        hours = hours * len(temps) if temps else hours
    if not temps or not hours:
        return {
            "error": "The recorded temperatures and the time intervals are required and must not be empty.",
        }
    if len(temps) != len(hours):
        return {
            "error": "The recorded temperatures and the time intervals must have the same number of values.",
        }
    if any(float(dt) <= 0 for dt in hours):
        return {"error": "Each time interval must be greater than zero."}
    grade = grade_label(strength_28d_n_mm2)
    if grade is not None:
        strength_28d_n_mm2 = grade[1]
    if float(strength_28d_n_mm2) <= 0:
        return {"error": "strength_28d_n_mm2 must be > 0."}
    te_denom = float(reference_temperature_c) - float(datum_temperature)
    if te_denom <= 0:
        return {"error": "reference_temperature_c must be greater than datum_temperature."}

    mi = sum(
        max(0.0, float(t) - float(datum_temperature)) * float(dt)
        for t, dt in zip(temps, hours)
    )
    te_days = (mi / te_denom) / 24.0
    denom = gain_a + gain_b * te_days
    ratio = (te_days / denom) if denom > 0 else 0.0
    ratio = min(max(ratio, 0.0), 1.0)
    return {
        "maturity_index_c_hrs": round(mi, 0),
        "equivalent_age_days": round(te_days, 3),
        "predicted_strength_n_mm2": round(strength_28d_n_mm2 * ratio, 1),
        "percent_of_28d": round(ratio * 100, 1),
    }
