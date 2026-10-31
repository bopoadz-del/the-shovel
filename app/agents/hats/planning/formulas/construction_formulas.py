"""Formulas from app.lib.construction_formulas owned by the planning hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math
from typing import Any, Dict


@formula(
    owner='planning',
    display_name='Supervision manpower',
    description='Supervision manpower for given quantities, at the project\'s own supervision ratios.',
    inputs={
        'concrete_m3': Param('m3', 0, 1e8, label='concrete volume'),
        'structural_steel_t': Param('t', 0, 1e6, label='structural steel'),
        'piping_dia_inch': Param('in', 0, 1e9, label='piping'),
        'electrical_cable_km': Param('km', 0, 100000, label='electrical cable'),
        'area_m2': Param('m2', 0, 1e8, label='gross area'),
        'concrete_m3_per_supervisor': Param('m3', 1, 1e8, label='concrete per civil supervisor'),
        'steel_t_per_supervisor': Param('t', 1, 1e6, label='steel per structural supervisor'),
        'dia_inch_per_supervisor': Param('-', 1, 1e9, label='dia-inch per piping supervisor'),
        'cable_km_per_supervisor': Param('km', 1, 100000, label='cable per electrical supervisor'),
        'area_m2_per_supervisor': Param('m2', 1, 1e8, label='area per general supervisor'),
        'supervisors_per_hse_officer': Param('count', 1, 1000, label='supervisors per HSE officer'),
        'staff_per_document_controller': Param('count', 1, 1000, label='supervisors per document controller'),
    },
    outputs={'civil_supervisors': '-', 'structural_supervisors': '-', 'piping_supervisors': '-', 'electrical_supervisors': '-', 'general_supervisors': '-', 'total_supervisors': '-', 'hse_officers': '-', 'document_controllers': '-', 'total_supervision_staff': '-'},
)
def supervision_ratio(
    concrete_m3: float = 0,
    structural_steel_t: float = 0,
    piping_dia_inch: float = 0,
    electrical_cable_km: float = 0,
    area_m2: float = 0,
    concrete_m3_per_supervisor: Any = None,
    steel_t_per_supervisor: Any = None,
    dia_inch_per_supervisor: Any = None,
    cable_km_per_supervisor: Any = None,
    area_m2_per_supervisor: Any = None,
    supervisors_per_hse_officer: Any = None,
    staff_per_document_controller: Any = None,
) -> Dict[str, Any]:
    """
    Supervision manpower from quantities at the PROJECT's own ratios
    (staffing plan / tender). No ratio is built in: each quantity given
    needs its ratio, and the HSE / document-control ratios are needed
    whenever any supervisor results.
    """
    pairs = [
        ("civil_supervisors", concrete_m3, "concrete_m3_per_supervisor", concrete_m3_per_supervisor),
        ("structural_supervisors", structural_steel_t, "steel_t_per_supervisor", steel_t_per_supervisor),
        ("piping_supervisors", piping_dia_inch, "dia_inch_per_supervisor", dia_inch_per_supervisor),
        ("electrical_supervisors", electrical_cable_km, "cable_km_per_supervisor", cable_km_per_supervisor),
        ("general_supervisors", area_m2, "area_m2_per_supervisor", area_m2_per_supervisor),
    ]
    missing = [rn for _o, q, rn, r in pairs if q and q > 0 and r in (None, "")]
    if any(q and q > 0 for _o, q, _rn, _r in pairs):
        for rn, r in (("supervisors_per_hse_officer", supervisors_per_hse_officer),
                      ("staff_per_document_controller", staff_per_document_controller)):
            if r in (None, ""):
                missing.append(rn)
    if missing:
        return {"error": "project supervision ratios not given", "needs_input": missing}

    out: Dict[str, Any] = {}
    for o, q, _rn, r in pairs:
        out[o] = math.ceil(q / float(r)) if q and q > 0 else 0
    total_sup = sum(out.values())
    hse = math.ceil(total_sup / float(supervisors_per_hse_officer)) if total_sup else 0
    docs = math.ceil(total_sup / float(staff_per_document_controller)) if total_sup else 0
    out.update({
        "total_supervisors": total_sup,
        "hse_officers": hse,
        "document_controllers": docs,
        "total_supervision_staff": total_sup + hse + docs,
    })
    return out

@formula(
    owner='planning',
    display_name='Electrical installation programme',
    description='Electrical installation programme: stage durations from first fix to handover, at the project\'s own stage rates.',
    inputs={
        'floor_area_m2': Param('m2', 0, 1e8, label='floor area'),
        'num_floors': Param('count', 1, 300, label='number of floors'),
        'stage_output_rates': Param('m2/day', label='stage output rates, m2 per day by stage'),
    },
    outputs={'total_area_m2': 'm2', 'stages': '-', 'total_days': 'days'},
)
def electrical_installation_sequence(
    floor_area_m2: float, num_floors: int = 1, *, stage_output_rates: Dict[str, float],
) -> Dict[str, Any]:
    """Electrical programme: each stage's days = area / that stage's m2/day.

    The stages and their output rates are the PROJECT's (programme /
    subcontractor productivity) -- none is built in."""
    if floor_area_m2 <= 0 or num_floors <= 0:
        raise ValueError("floor_area_m2 and num_floors must be > 0")
    if not isinstance(stage_output_rates, dict) or not stage_output_rates:
        return {"error": "project stage rates not given", "needs_input": ["stage_output_rates"]}
    total = floor_area_m2 * num_floors
    stages = {k: math.ceil(total / float(v)) for k, v in stage_output_rates.items() if float(v) > 0}
    return {"total_area_m2": total, "stages": stages, "total_days": sum(stages.values())}

@formula(
    owner='planning',
    display_name='Plumbing installation programme',
    description='Plumbing installation programme: stage durations from submittal to handover.',
    inputs={
        'floor_area_m2': Param('m2', 0, 1e8, label='floor area'),
        'num_floors': Param('count', 1, 300, label='number of floors'),
    },
    outputs={'stages': '-', 'cumulative_days': 'days', 'total_days': 'days'},
)
def plumbing_flow_programme(floor_area_m2: float, num_floors: int = 1) -> Dict[str, Any]:
    """Plumbing flow: submittal -> procure -> 1st/2nd fix -> connect -> handover."""
    total = floor_area_m2 * num_floors
    stages = [
        ("material_submittal", 7), ("procurement", 21),
        ("1st_fix_marking_piping", max(1, math.ceil(total / 40))),
        ("2nd_fix_sanitary_fittings", max(1, math.ceil(total / 60))),
        ("final_fix_pumps_tanks", max(1, math.ceil(total / 80))),
        ("water_connection", 7),
        ("testing_commissioning", max(1, math.ceil(total / 50))),
        ("handover", 3),
    ]
    result = {"stages": {}, "total_days": 0}
    cum = 0
    for name, days in stages:
        cum += days
        result["stages"][name] = {"days": days, "cumulative_days": cum}
    result["total_days"] = cum
    return result
