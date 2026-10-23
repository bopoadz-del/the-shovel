"""Formulas from app.lib.construction_formulas owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from app.agents.base.formulas.construction_formulas_shared import (  # noqa: F401
    _is_aci_code,
)


@dataclass
class DewateringResult:
    uplift_force_t_m2: float
    counter_weight_t_m2: float
    fos: float
    can_stop: bool
    needs_tension_piles: bool
    min_floors_for_stop: int
    notes: List[str] = field(default_factory=list)

@formula(
    owner='design',
    display_name='Uplift check for stopping dewatering',
    description="Whether dewatering can stop: factor of safety of the structure's weight against groundwater uplift.",
    inputs={
        'water_depth': Param('m', 0, 200, label='groundwater head'),
        'raft_thickness': Param('m', 0, 20, label='raft thickness'),
        'floor_count': Param('count', 0, 300, label='number of floors'),
        'floor_thickness': Param('m', 0, 5, label='floor thickness'),
        'concrete_unit_weight': Param('t/m3', 1, 5, label='concrete unit weight'),
        'water_unit_weight': Param('t/m3', 0.9, 1.1, label='water unit weight'),
        'required_fos': Param('ratio', 1, 5, label='required factor of safety'),
    },
    outputs={'uplift_force_t_m2': 't/m2', 'counter_weight_t_m2': 't/m2', 'fos': '-', 'can_stop': '-', 'needs_tension_piles': '-', 'min_floors_for_stop': '-'},
)
def dewatering_uplift_check(
    water_depth: float,
    raft_thickness: float,
    floor_count: int,
    floor_thickness: float = 0.3,
    concrete_unit_weight: float = 2.5,
    water_unit_weight: float = 1.0,
    required_fos: float = 1.25,
) -> DewateringResult:
    """Check if dewatering can stop — FOS = counter-weight / uplift >= 1.25."""
    if water_depth < 0:
        raise ValueError("water_depth must be >= 0")
    if raft_thickness < 0 or floor_count < 0 or floor_thickness < 0:
        raise ValueError("raft_thickness, floor_count and floor_thickness must be >= 0")
    if concrete_unit_weight <= 0 or water_unit_weight <= 0 or required_fos <= 0:
        raise ValueError("unit weights and required_fos must be > 0")
    uplift = water_depth * water_unit_weight
    raft_weight = raft_thickness * concrete_unit_weight
    floor_weight = floor_count * floor_thickness * concrete_unit_weight
    counter_weight = raft_weight + floor_weight
    fos = counter_weight / uplift if uplift > 0 else float("inf")

    notes = [
        f"Uplift = {water_depth} x {water_unit_weight} = {uplift:.3f} T/m2",
        f"Counter = {raft_weight:.3f} + {floor_weight:.3f} = {counter_weight:.3f} T/m2",
        f"FOS = {fos:.3f} (required: {required_fos})",
    ]
    can_stop = fos >= required_fos
    if can_stop:
        notes.append(f"OK - dewatering CAN stop at {floor_count} floors")
    else:
        notes.append(f"NG - dewatering CANNOT stop, FOS {fos:.3f} < {required_fos}")
        notes.append("Tension piles required to anchor against uplift")

    min_counter = uplift * required_fos
    add_floor = floor_thickness * concrete_unit_weight
    min_floors = max(0, math.ceil((min_counter - raft_weight) / add_floor))
    notes.append(f"Minimum floors to stop dewatering: {min_floors}")

    return DewateringResult(
        uplift_force_t_m2=round(uplift, 3),
        counter_weight_t_m2=round(counter_weight, 3),
        fos=round(fos, 3),
        can_stop=can_stop,
        needs_tension_piles=not can_stop,
        min_floors_for_stop=min_floors,
        notes=notes,
    )

@formula(
    owner='design',
    display_name='Well-point spacing for dewatering',
    description='Well-point spacing and number of stages for a dewatering depth and soil type.',
    inputs={
        'soil_permeability_m_s': Param('m/s', 1e-12, 1, label='soil permeability'),
        'required_drawdown_m': Param('m', 0, 100, label='required drawdown'),
        'well_point_diameter_m': Param('m', 0, 1, label='well point diameter'),
    },
    outputs={'soil_type': '-', 'permeability_m_s': 'm/s', 'required_drawdown_m': 'm', 'stages_needed': '-', 'max_depth_capacity_m': 'm', 'well_point_spacing_m': 'm', 'well_point_diameter_m': 'm'},
)
def dewatering_well_point_spacing(
    soil_permeability_m_s: float,
    required_drawdown_m: float,
    well_point_diameter_m: float = 0.05,
) -> Dict[str, Any]:
    """Well point spacing by soil type. Multi-stage: 5m/stage, max 3 stages (15m)."""
    if soil_permeability_m_s <= 0:
        raise ValueError("soil_permeability_m_s must be > 0")
    if required_drawdown_m < 0:
        raise ValueError("required_drawdown_m must be >= 0")
    if soil_permeability_m_s > 1e-3:
        spacing_m, soil_type = 1.5, "coarse sand/gravel"
    elif soil_permeability_m_s > 1e-4:
        spacing_m, soil_type = 1.2, "medium sand"
    elif soil_permeability_m_s > 1e-5:
        spacing_m, soil_type = 0.9, "fine sand"
    else:
        spacing_m, soil_type = 0.6, "silty sand - consider vacuum well points"

    stages = math.ceil(required_drawdown_m / 5.0)
    notes = [f"Soil: {soil_type}", f"Spacing: {spacing_m}m", f"Stages: {stages}"]
    if stages > 3:
        notes.append("Exceeds 3-stage limit (15m) - use deep well system")

    return {
        "soil_type": soil_type,
        "permeability_m_s": soil_permeability_m_s,
        "required_drawdown_m": required_drawdown_m,
        "stages_needed": stages,
        "max_depth_capacity_m": stages * 5.0,
        "well_point_spacing_m": spacing_m,
        "well_point_diameter_m": well_point_diameter_m,
        "notes": notes,
    }

@formula(
    owner='design',
    display_name='Concrete mix design',
    description='Concrete mix design by the absolute-volume method from specific gravities and the water/cement ratio.',
    inputs={
        'w_c_ratio': Param('ratio', 0.2, 1, label='water-cement ratio'),
        'cement_sg': Param('ratio', 2, 4, label='cement specific gravity'),
        'fine_agg_sg': Param('ratio', 2, 3.5, label='fine aggregate specific gravity'),
        'coarse_agg_sg': Param('ratio', 2, 3.5, label='coarse aggregate specific gravity'),
        'fine_agg_ratio': Param('ratio', 0, 10, label='fine aggregate parts'),
        'coarse_agg_ratio': Param('ratio', 0, 10, label='coarse aggregate parts'),
        'dune_sand_pct': Param('%', 0, 100, label='dune sand share'),
    },
    outputs={'w_c_ratio': '-', 'cement_kg_m3': 'kg/m3', 'water_litres_m3': 'L/m3', 'fine_aggregate_kg_m3': 'kg/m3', 'coarse_aggregate_kg_m3': 'kg/m3', 'total_weight_kg_m3': 'kg/m3', 'proportions': '-'},
)
def concrete_mix_design_sg(
    w_c_ratio: float,
    cement_sg: float = 3.15,
    fine_agg_sg: float = 2.67,
    coarse_agg_sg: float = 2.54,
    fine_agg_ratio: float = 2.0,
    coarse_agg_ratio: float = 4.0,
    dune_sand_pct: float = 0.0,
) -> Dict[str, float]:
    """Mix design by Specific Gravity (absolute volume method).
    W/(SGw) + C/(SGc) + F/(SGf) + Cr/(SGcr) = 1.0 m3."""
    if w_c_ratio <= 0:
        raise ValueError("w_c_ratio must be > 0")
    if min(cement_sg, fine_agg_sg, coarse_agg_sg) <= 0:
        raise ValueError("specific gravities must be > 0")
    if fine_agg_ratio < 0 or coarse_agg_ratio < 0:
        raise ValueError("aggregate ratios must be >= 0")
    if float(dune_sand_pct) != 0.0:
        raise ValueError(
            "dune_sand_pct is not applied in the absolute-volume mix "
            "(no dune-sand SG is defined). Omit it or pass 0."
        )
    cement_tonnes = 1.0 / (w_c_ratio / 1.0 + 1.0 / cement_sg + fine_agg_ratio / fine_agg_sg + coarse_agg_ratio / coarse_agg_sg)
    cement_kg = round(cement_tonnes * 1000, 0)
    water_litres = round(cement_kg * w_c_ratio, 0)
    fine_kg = round(cement_kg * fine_agg_ratio, 0)
    coarse_kg = round(cement_kg * coarse_agg_ratio, 0)
    return {
        "w_c_ratio": w_c_ratio,
        "cement_kg_m3": cement_kg,
        "water_litres_m3": water_litres,
        "fine_aggregate_kg_m3": fine_kg,
        "coarse_aggregate_kg_m3": coarse_kg,
        "total_weight_kg_m3": cement_kg + water_litres + fine_kg + coarse_kg,
        "proportions": f"1:{int(fine_agg_ratio)}:{coarse_agg_ratio}",
    }

@formula(
    owner='design',
    display_name='Slip-form concrete mix',
    description='Mix proportions for slip-formed concrete, flagged as indicative defaults when not supplied.',
    inputs={},
    outputs={'cube_strength_n_mm2': 'N/mm2', 'slump_mm': 'mm', 'max_concrete_temp_c': 'degC', 'retarder_10c_lit_m3': 'L/m3', 'retarder_20c_lit_m3': 'L/m3'},
)
def concrete_mix_slip_form(**_kwargs: Any) -> Dict[str, Any]:
    """Slip-forming mix: 1:2:2.6, W/C=0.42, slump=150 +/- 30 mm.

    Extra kwargs are ignored. Documented constants stay locked — live
    probes send slump/w_c next to the name; forwarding them would
    TypeError a zero-arg signature (tool_error / cause).
    """
    mix = concrete_mix_design_sg(w_c_ratio=0.42, fine_agg_ratio=2.0, coarse_agg_ratio=2.6)
    mix["slump_mm"] = "150 +/- 30"
    mix["retarder_20c_lit_m3"] = 3.8
    mix["retarder_10c_lit_m3"] = 2.9
    mix["max_concrete_temp_c"] = 32
    mix["cube_strength_n_mm2"] = "50-55 @ 28 days"
    return mix

@formula(
    owner='design',
    display_name='Concrete modulus of elasticity',
    description='Modulus of elasticity of concrete from its compressive strength, to the selected code.',
    inputs={
        'fck_n_mm2': Param('N/mm2', 5, 150, label='characteristic concrete strength', grade='concrete'),
        'code': Param('-', label='design code'),
    },
    outputs={'value': 'currency', 'unit': '-', 'value_mpa': 'MPa'},
)
def modulus_of_elasticity_concrete(
    fck_n_mm2: float, code: str = "metric_technical",
) -> dict:
    """Concrete modulus of elasticity, with its unit stated.

    Two forms, because they are different numbers and a reader cannot tell
    them apart from a bare float:

    * ``metric_technical`` (default): Ec = 15000 sqrt(f'c) with f'c in
      kg/cm2. 1 N/mm² = 10.197 kg/cm². For C35 that is 283,375 kg/cm2.
    * ``aci``: ACI 318-19 Eq. 19.2.2.1b SI form, Ec = 4700 sqrt(f'c) in MPa
      -- for C35, 27,806 MPa.

    Live concrete-modulus ask: this returned the bare float 280624.0. The answer stated the ACI
    formula and its substitution correctly, then printed "28,062 MPa" -- the
    kg/cm2 figure divided by 10 instead of converted (x0.0980665 = 27,520),
    and neither number is the 27,806 the ACI form gives. The value now carries
    its unit, and the MPa conversion is done here rather than guessed.
    """
    # Not _norm_code: that helper defaults every unknown string to ACI,
    # which would silently change this function's default form.
    fck = float(fck_n_mm2)
    if _is_aci_code(code):
        ec_mpa = round(4700 * math.sqrt(fck), 0)
        return {
            "value": ec_mpa,
            "unit": "MPa",
            "standard": "ACI 318-19 Eq. 19.2.2.1b (SI): 4700*sqrt(f'c)",
        }
    ec_kg_cm2 = round(15000 * math.sqrt(fck * 10.197), 0)
    return {
        "value": ec_kg_cm2,
        "unit": "kg/cm2",
        "value_mpa": round(ec_kg_cm2 * 0.0980665, 0),
        "standard": "15000*sqrt(f'c) with f'c in kg/cm2 (metric-technical form)",
    }

# A beam's second moment of area is quoted in m4 in a question and consumed in
# mm4 by every formula below -- 1e12 apart. Taking the stated number at face
# value returns a deflection in kilometres and prints it as millimetres, so an
# implausibly small value is refused rather than used. The smallest real
# section is orders above 1 mm4 (a 10 mm square bar is 833).
_I_MM4_FLOOR = 1.0

def _second_moment_mm4(i_mm4: float) -> float:
    """Check the second moment of area is in mm4, refusing a unit mix-up."""
    if i_mm4 <= 0:
        raise ValueError("i_mm4 must be > 0")
    if i_mm4 < _I_MM4_FLOOR:
        raise ValueError(
            f"i_mm4={i_mm4:g} is too small to be mm4 -- a value this size is "
            "in m4; multiply it by 1e12 (1 m4 = 1e12 mm4) and call again")
    return float(i_mm4)

@formula(
    owner='design',
    display_name='Beam deflection under a uniform load',
    description='Midspan deflection of a simply supported beam under a uniformly distributed load.',
    inputs={
        'w_kn_m': Param('kN/m', 0, 100000, label='uniform load'),
        'span_m': Param('m', 0, 200, label='span'),
        'ec_mpa': Param('MPa', 100, 300000, label='modulus of elasticity'),
        'i_mm4': Param('mm4', 1, 1e15, label='second moment of area'),
    },
    outputs={'deflection': 'mm'},
)
def beam_deflection_ss_udl(w_kn_m: float, span_m: float, ec_mpa: float, i_mm4: float) -> float:
    """Simply supported, UDL: delta = 5wL^4 / (384EI). Returns mm."""
    if span_m < 0 or w_kn_m < 0:
        raise ValueError("span_m and w_kn_m must be >= 0")
    if ec_mpa <= 0:
        raise ValueError("ec_mpa must be > 0")
    i_mm4 = _second_moment_mm4(i_mm4)
    w_n_mm = w_kn_m  # kN/m = N/mm
    l_mm = span_m * 1e3
    return round((5 * w_n_mm * l_mm**4) / (384 * ec_mpa * i_mm4), 2)

@formula(
    owner='design',
    display_name='Cantilever deflection under a uniform load',
    description='Tip deflection of a cantilever under a uniformly distributed load.',
    inputs={
        'w_kn_m': Param('kN/m', 0, 100000, label='uniform load'),
        'span_m': Param('m', 0, 200, label='span'),
        'ec_mpa': Param('MPa', 100, 300000, label='modulus of elasticity'),
        'i_mm4': Param('mm4', 1, 1e15, label='second moment of area'),
    },
    outputs={'deflection': 'mm'},
)
def beam_deflection_cantilever_udl(w_kn_m: float, span_m: float, ec_mpa: float, i_mm4: float) -> float:
    """Cantilever, UDL: delta = wL^4 / (8EI). Returns mm."""
    if span_m < 0 or w_kn_m < 0:
        raise ValueError("span_m and w_kn_m must be >= 0")
    if ec_mpa <= 0:
        raise ValueError("ec_mpa must be > 0")
    i_mm4 = _second_moment_mm4(i_mm4)
    w_n_mm = w_kn_m
    l_mm = span_m * 1e3
    return round((w_n_mm * l_mm**4) / (8 * ec_mpa * i_mm4), 2)

@formula(
    owner='design',
    display_name='Cantilever deflection under a point load',
    description='Tip deflection of a cantilever under a point load at its free end.',
    inputs={
        'p_kn': Param('kN', 0, 1e6, label='point load'),
        'span_m': Param('m', 0, 200, label='span'),
        'ec_mpa': Param('MPa', 100, 300000, label='modulus of elasticity'),
        'i_mm4': Param('mm4', 1, 1e15, label='second moment of area'),
    },
    outputs={'deflection': 'mm'},
)
def beam_deflection_cantilever_point_load(p_kn: float, span_m: float, ec_mpa: float, i_mm4: float) -> float:
    """Cantilever, point load at the tip: delta = PL^3 / (3EI). Returns mm.

    Distinct from ``beam_deflection_cantilever_udl``: substituting a point
    load into wL^4/8EI overstates a 4 m / 30 kN tip deflection by half.
    """
    if span_m < 0 or p_kn < 0:
        raise ValueError("span_m and p_kn must be >= 0")
    if ec_mpa <= 0:
        raise ValueError("ec_mpa must be > 0")
    i_mm4 = _second_moment_mm4(i_mm4)
    p_n = p_kn * 1e3
    l_mm = span_m * 1e3
    return round((p_n * l_mm**3) / (3 * ec_mpa * i_mm4), 2)

@formula(
    owner='design',
    display_name='Beam deflection under a central point load',
    description='Midspan deflection of a simply supported beam under a central point load.',
    inputs={
        'p_kn': Param('kN', 0, 1e6, label='point load'),
        'span_m': Param('m', 0, 200, label='span'),
        'ec_mpa': Param('MPa', 100, 300000, label='modulus of elasticity'),
        'i_mm4': Param('mm4', 1, 1e15, label='second moment of area'),
    },
    outputs={'deflection': 'mm'},
)
def beam_deflection_ss_point_load_midspan(p_kn: float, span_m: float, ec_mpa: float, i_mm4: float) -> float:
    """Simply supported, point load at midspan: delta = PL^3 / (48EI). Returns mm."""
    if span_m < 0 or p_kn < 0:
        raise ValueError("span_m and p_kn must be >= 0")
    if ec_mpa <= 0:
        raise ValueError("ec_mpa must be > 0")
    i_mm4 = _second_moment_mm4(i_mm4)
    p_n = p_kn * 1e3
    l_mm = span_m * 1e3
    return round((p_n * l_mm**3) / (48 * ec_mpa * i_mm4), 2)

@formula(
    owner='design',
    display_name='Shrinkage as an equivalent temperature drop',
    description='Converts a shrinkage strain to the equivalent temperature drop.',
    inputs={
        'shrinkage_strain': Param('ratio', 0, 0.01, label='shrinkage strain'),
        'alpha_c': Param('1/degC', 0, 0.001, label='coefficient of thermal expansion'),
    },
    outputs={'shrinkage_strain': '-', 'equivalent_temp_drop_c': 'degC'},
)
def thermal_shrinkage_equivalence(shrinkage_strain: float = 0.0002, alpha_c: float = 10e-6) -> Dict[str, float]:
    """Convert shrinkage strain to equivalent temperature drop: delta_t = epsilon_sh / alpha_c."""
    return {
        "shrinkage_strain": shrinkage_strain,
        "equivalent_temp_drop_c": round(shrinkage_strain / alpha_c, 1),
    }

@formula(
    owner='design',
    display_name='Concrete unit weight',
    description='Unit weight of plain or reinforced concrete, flagged as an indicative default when not supplied.',
    inputs={
        'reinforced': Param('-', label='reinforced concrete'),
    },
    outputs={'unit_weight_kg_m3': 'kg/m3', 'type': '-', 'range_kg_m3': 'kg/m3'},
)
def unit_weight_concrete(reinforced: bool = True) -> Dict[str, float]:
    """Unit weight: RC = 2500 kg/m3, plain = 2400 kg/m3 (range 2330-2470)."""
    if reinforced:
        return {"unit_weight_kg_m3": 2500, "type": "Reinforced Concrete"}
    return {"unit_weight_kg_m3": 2400, "type": "Plain Concrete", "range_kg_m3": (2330, 2470)}

@formula(
    owner='design',
    display_name='Shear stress check',
    description='Nominal shear stress on a section from shear force, width and effective depth.',
    inputs={
        'v_kn': Param('kN', 0, 1e6, label='shear force'),
        'b_mm': Param('mm', 20, 10000, label='section width'),
        'd_mm': Param('mm', 20, 10000, label='effective depth'),
    },
    outputs={'shear_force_kn': 'kN', 'width_mm': 'mm', 'effective_depth_mm': 'mm', 'shear_stress_n_mm2': 'N/mm2'},
)
def shear_stress_check(v_kn: float, b_mm: float, d_mm: float) -> Dict[str, float]:
    """Shear stress: v = V/(b x d) in N/mm2."""
    return {
        "shear_force_kn": v_kn, "width_mm": b_mm, "effective_depth_mm": d_mm,
        "shear_stress_n_mm2": round((v_kn * 1000) / (b_mm * d_mm), 3),
    }

@formula(
    owner='design',
    display_name='Post-tensioning force',
    description='Post-tensioning force needed to balance a share of the distributed load, from span and tendon drape.',
    inputs={
        'span_m': Param('m', 0, 200, label='span'),
        'slab_thickness_m': Param('m', 0, 5, label='slab thickness'),
        'live_load_kn_m2': Param('kN/m2', 0, 100, label='live load'),
        'dead_load_kn_m2': Param('kN/m2', 0, 1000, label='dead load'),
        'tendon_stress_n_mm2': Param('N/mm2', 100, 2500, label='tendon stress'),
        'tendon_diameter_mm': Param('mm', 3, 50, label='tendon diameter'),
        'eccentricity_ratio': Param('ratio', 0, 1, label='eccentricity ratio'),
    },
    outputs={'tendon_force_kn': 'kN', 'tendon_area_mm2': 'mm2', 'num_strands': '-', 'balanced_load_kn_m': 'kN/m', 'eccentricity_mm': 'mm', 'cable_profile': '-'},
)
def post_tensioning_force(
    span_m: float, slab_thickness_m: float, live_load_kn_m2: float,
    dead_load_kn_m2: Optional[float] = None,
    tendon_stress_n_mm2: float = 1300.0,
    tendon_diameter_mm: float = 12.7,
    eccentricity_ratio: float = 0.15,
) -> Dict[str, Any]:
    """PT force: P = w x L^2 / (8 x e), balancing 75% of DL."""
    if dead_load_kn_m2 is None:
        dead_load_kn_m2 = slab_thickness_m * 25
    e = eccentricity_ratio * slab_thickness_m
    balanced = dead_load_kn_m2 * 0.75
    force_kn = balanced * span_m**2 / (8 * e) if e > 0 else 0
    area_mm2 = (force_kn * 1000) / tendon_stress_n_mm2
    # 7-wire strand fill ≈ 0.78 (ASTM A416 / EN 10138: 12.7 mm → 98.7 mm²).
    strand_area = 0.78 * math.pi * (tendon_diameter_mm / 2)**2
    strands = math.ceil(area_mm2 / strand_area) if strand_area > 0 else 0
    return {
        "tendon_force_kn": round(force_kn, 1),
        "tendon_area_mm2": round(area_mm2, 1),
        "num_strands": strands,
        "balanced_load_kn_m": round(balanced, 2),
        "eccentricity_mm": round(e * 1000, 1),
        "cable_profile": f"parabolic, e={e*1000:.0f}mm midspan",
    }

@formula(
    owner='design',
    display_name='Composite column comparison',
    description='Compares a composite column with an embedded steel section against a reinforced concrete column for a given axial load.',
    inputs={
        'axial_load_kn': Param('kN', 0, 1e7, label='axial load'),
        'column_diameter_mm': Param('mm', 100, 10000, label='column diameter'),
        'concrete_grade_n_mm2': Param('N/mm2', 5, 150, label='concrete strength', grade='concrete'),
        'use_i_beam': Param('-', label='use an I-beam'),
        'i_beam_weight_t': Param('t', 0, 1000, label='steel section weight'),
        'steel_area_mm2': Param('mm2', 0, 1e8, label='structural steel area'),
        'steel_fy_mpa': Param('MPa', 100, 2000, label='structural steel yield strength', grade='steel'),
        'rebar_area_mm2': Param('mm2', 0, 1e8, label='reinforcement area'),
        'rebar_fy_mpa': Param('MPa', 100, 2000, label='reinforcement yield strength', grade='rebar'),
    },
    outputs={'option_a': '-', 'option_b': '-', 'recommended': '-'},
)
def composite_column_design(
    axial_load_kn: float, column_diameter_mm: float,
    concrete_grade_n_mm2: float = 60.0,
    use_i_beam: bool = True, i_beam_weight_t: float = 0.0,
    steel_area_mm2: float = 0.0, steel_fy_mpa: float = 355.0,
    rebar_area_mm2: float = 0.0, rebar_fy_mpa: float = 500.0,
) -> Dict[str, Any]:
    """Squash resistance of a concrete-encased column, EN 1994-1-1 §6.7.3.2.

    Npl,Rd = Aa*fyd + 0.85*Ac*fcd + As*fsd, fcd = fck/γc (γc = 1.5),
    fyd = fy/γM0 (γM0 = 1.0), fsd = fsk/γs (γs = 1.15). Ac is the concrete
    area inside the gross section. A section mass is not an area, so
    ``i_beam_weight_t`` does not change the resistance.
    """
    if column_diameter_mm <= 0:
        raise ValueError("column_diameter_mm must be > 0")
    if axial_load_kn < 0 or concrete_grade_n_mm2 <= 0:
        raise ValueError("axial_load_kn must be >= 0 and concrete_grade_n_mm2 must be > 0")
    if steel_area_mm2 < 0 or rebar_area_mm2 < 0 or steel_fy_mpa <= 0 or rebar_fy_mpa <= 0:
        raise ValueError("steel and reinforcement areas must be >= 0 and yields must be > 0")
    gross = math.pi * (column_diameter_mm / 2) ** 2
    steel = float(steel_area_mm2) if use_i_beam else 0.0
    rebar = float(rebar_area_mm2)
    if steel + rebar > gross:
        raise ValueError("steel plus reinforcement area exceeds the gross section")
    concrete = gross - steel - rebar
    fcd = float(concrete_grade_n_mm2) / 1.5

    def resistance(steel_area: float) -> float:
        ac = gross - steel_area - rebar
        return (
            steel_area * float(steel_fy_mpa)
            + 0.85 * ac * fcd
            + rebar * (float(rebar_fy_mpa) / 1.15)
        ) / 1000.0

    cap_a = resistance(steel)
    cap_b = resistance(0.0)
    return {
        "option_a": {
            "type": f"C{concrete_grade_n_mm2:.0f}+encased steel",
            "capacity_kn": round(cap_a, 0),
            "is_safe": cap_a >= axial_load_kn,
        },
        "option_b": {
            "type": f"C{concrete_grade_n_mm2:.0f}+rebar",
            "capacity_kn": round(cap_b, 0),
            "is_safe": cap_b >= axial_load_kn,
        },
        "recommended": "B" if cap_b >= axial_load_kn and not (cap_a >= axial_load_kn) else (
            "A or B" if cap_a >= axial_load_kn and cap_b >= axial_load_kn else "A" if cap_a >= axial_load_kn else "NEITHER"),
        "note": (
            "EN 1994-1-1 §6.7.3.2. A section mass of "
            f"{i_beam_weight_t:g} t is not an area; pass steel_area_mm2."
        ),
        "concrete_area_mm2": round(concrete if use_i_beam else gross - rebar, 1),
    }

@formula(
    owner='design',
    display_name='Foundation bearing pressure',
    description='Bearing pressure under a foundation from load and area, and its factor of safety against the soil capacity.',
    inputs={
        'foundation_width_m': Param('m', 0, 100, label='foundation width'),
        'foundation_length_m': Param('m', 0, 100, label='foundation length'),
        'column_load_kn': Param('kN', 0, 1e7, label='column load'),
        'soil_bearing_capacity_kn_m2': Param('kN/m2', 0, 20000, label='allowable bearing capacity'),
        'foundation_depth_m': Param('m', 0, 50, label='foundation depth'),
    },
    outputs={'bearing_pressure_kn_m2': 'kN/m2', 'net_pressure_kn_m2': 'kN/m2', 'fos_against_bearing': '-', 'bearing_ok': '-'},
)
def foundation_bearing_pressure(
    foundation_width_m: float, foundation_length_m: float,
    column_load_kn: float, soil_bearing_capacity_kn_m2: float,
    foundation_depth_m: float = 0.0,
) -> Dict[str, float]:
    """q = P/A. The soil value is an allowable pressure, so it already
    includes the factor of safety. The foundation is adequate when the net
    pressure does not exceed that allowable (FOS >= 1). Requiring FOS >= 2
    applies a second factor."""
    if foundation_width_m <= 0 or foundation_length_m <= 0:
        return {"error": "foundation_width_m and foundation_length_m must be > 0"}
    area = foundation_width_m * foundation_length_m
    q = column_load_kn / area
    net_q = q - foundation_depth_m * 18
    fos = soil_bearing_capacity_kn_m2 / net_q if net_q > 0 else float("inf")
    return {
        "bearing_pressure_kn_m2": round(q, 2),
        "net_pressure_kn_m2": round(net_q, 2),
        "fos_against_bearing": round(fos, 2),
        "bearing_ok": fos >= 1.0,
    }

@formula(
    owner='design',
    display_name='Precast member erection demand',
    description=(
        'Erection demand of a precast member: member weight plus rigging, '
        'times the dynamic allowance, as a share of the allowable pick weight '
        'at the stated radius.'
    ),
    inputs={
        'beam_weight_t': Param('t', 0, 1000, label='member weight'),
        'beam_length_m': Param('m', 0, 200, label='member length'),
        'crane_capacity_t': Param('t', 0, 10000, label='allowable pick weight'),
        'lift_radius_m': Param('m', 0, 200, label='pick radius'),
        'rigging_weight_t': Param('t', 0, 500, label='rigging weight'),
        'dynamic_factor': Param('ratio', 1, 3, label='dynamic allowance'),
    },
    outputs={'effective_lift_weight_t': 't', 'utilization_pct': '%', 'is_safe': '-'},
)
def precast_beam_erection_check(
    beam_weight_t: float, beam_length_m: float,
    crane_capacity_t: float, lift_radius_m: float,
    rigging_weight_t: float = 0.5, dynamic_factor: float = 1.25,
) -> Dict[str, Any]:
    """Member weight plus rigging, times the dynamic allowance."""
    effective = (beam_weight_t + rigging_weight_t) * dynamic_factor
    util = effective / crane_capacity_t * 100
    return {
        "effective_lift_weight_t": round(effective, 2),
        "utilization_pct": round(util, 1),
        "is_safe": util <= 100,
    }
