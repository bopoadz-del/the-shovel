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
    description='Supervision manpower for given quantities, from supervision ratios flagged as indicative defaults.',
    inputs={
        'concrete_m3': Param('m3', 0, 1e8, label='concrete volume'),
        'structural_steel_t': Param('t', 0, 1e6, label='structural steel'),
        'piping_dia_inch': Param('-', 0, 1e9, label='piping'),
        'electrical_cable_km': Param('km', 0, 100000, label='electrical cable'),
        'area_m2': Param('m2', 0, 1e8, label='gross area'),
    },
    outputs={'civil_supervisors': '-', 'structural_supervisors': '-', 'piping_supervisors': '-', 'electrical_supervisors': '-', 'general_supervisors': '-', 'total_supervisors': '-', 'hse_officers': '-', 'document_controllers': '-', 'total_supervision_staff': '-'},
)
def supervision_ratio(
    concrete_m3: float = 0,
    structural_steel_t: float = 0,
    piping_dia_inch: float = 0,
    electrical_cable_km: float = 0,
    area_m2: float = 0,
) -> Dict[str, Any]:
    """
    Supervision manpower based on quantities.
    Rules of thumb:
      - 1 civil supervisor per 30,000-40,000 m3 concrete
      - 1 structural supervisor per 2,000-3,000 T steel
      - 1 piping supervisor per 15,000-20,000 dia-inch
      - 1 electrical supervisor per 50-80 km cable
    """
    civil_sup = math.ceil(concrete_m3 / 35000) if concrete_m3 > 0 else 0
    steel_sup = math.ceil(structural_steel_t / 2500) if structural_steel_t > 0 else 0
    piping_sup = math.ceil(piping_dia_inch / 17500) if piping_dia_inch > 0 else 0
    elec_sup = math.ceil(electrical_cable_km / 65) if electrical_cable_km > 0 else 0
    general_sup = math.ceil(area_m2 / 20000) if area_m2 > 0 else 0

    total_sup = civil_sup + steel_sup + piping_sup + elec_sup + general_sup
    hse_officers = max(2, math.ceil(total_sup / 8))  # 1 HSE per 8 supervisors
    document_controllers = max(1, math.ceil(total_sup / 15))

    return {
        "civil_supervisors": civil_sup,
        "structural_supervisors": steel_sup,
        "piping_supervisors": piping_sup,
        "electrical_supervisors": elec_sup,
        "general_supervisors": general_sup,
        "total_supervisors": total_sup,
        "hse_officers": hse_officers,
        "document_controllers": document_controllers,
        "total_supervision_staff": total_sup + hse_officers + document_controllers,
        "notes": [
            f"1 civil supervisor per ~35,000 m3 concrete",
            f"1 structural supervisor per ~2,500 T steel",
            f"1 piping supervisor per ~17,500 dia-inch",
            f"1 electrical supervisor per ~65 km cable",
            f"1 HSE officer per 8 supervisors",
        ],
    }

@formula(
    owner='planning',
    display_name='Electrical installation programme',
    description='Electrical installation programme: stage durations from first fix to handover.',
    inputs={
        'floor_area_m2': Param('m2', 0, 1e8, label='floor area'),
        'num_floors': Param('-', 1, 300, label='number of floors'),
    },
    outputs={'total_area_m2': 'm2', 'stages': '-', 'total_days': 'days'},
)
def electrical_installation_sequence(
    floor_area_m2: float, num_floors: int = 1,
) -> Dict[str, Any]:
    """Electrical programme: 1st fix -> 2nd fix -> test -> commission -> handover."""
    if floor_area_m2 <= 0 or num_floors <= 0:
        raise ValueError("floor_area_m2 and num_floors must be > 0")
    total = floor_area_m2 * num_floors
    rates = {"1st_fix_conduit_boxes": 50, "2nd_fix_equipment_panels": 75,
             "cabling_testing": 100, "final_test_temp_power": 100,
             "authority_inspection": 50, "final_fix_fittings": 75,
             "commissioning": 30}
    stages = {k: math.ceil(total / v) for k, v in rates.items()}
    return {"total_area_m2": total, "stages": stages, "total_days": sum(stages.values())}

@formula(
    owner='planning',
    display_name='Plumbing installation programme',
    description='Plumbing installation programme: stage durations from submittal to handover.',
    inputs={
        'floor_area_m2': Param('m2', 0, 1e8, label='floor area'),
        'num_floors': Param('-', 1, 300, label='number of floors'),
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
