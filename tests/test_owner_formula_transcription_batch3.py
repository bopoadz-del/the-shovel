"""Transcription check, batch 3: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
The stated form is the formula's own docstring; for the concrete modulus the
owner's knowledge entry ``structure.axial_shortening`` states the same
metric-technical form (Ec = 15000 sqrt(f'c), f'c in kg/cm2). Every expected
value below is a hand calculation from that stated form, written as a
literal; none is obtained by running the calculator. All inputs are invented.

The metric-technical branch of ``modulus_of_rupture`` is not pinned here: its
unit conversion differs from the stated form of its sibling, and that
difference is reported to the owner, not decided by a test.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.hats.design.formulas.construction_formulas import (
    beam_deflection_cantilever_udl,
    beam_deflection_ss_point_load_midspan,
    modulus_of_elasticity_concrete,
)
from app.agents.hats.design.formulas.construction_formulas_beam_analysis import (
    beam_moment_fixed_udl,
    beam_moment_point_load,
)
from app.agents.hats.design.formulas.construction_formulas_columns import (
    column_axial_capacity,
)
from app.agents.hats.design.formulas.construction_formulas_structural_steel import (
    bolt_shear_capacity,
    steel_tension_capacity,
    weld_capacity,
)
from app.agents.hats.qaqc.formulas.construction_formulas import modulus_of_rupture

TOL = 0.011


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


# Cantilever, UDL: delta = w*L^4/(8*E*I); w in N/mm (= kN/m), L in mm
@pytest.mark.parametrize("kw,mm", [
    (dict(w_kn_m=10, span_m=3, ec_mpa=200000, i_mm4=1e8), 5.06),
    (dict(w_kn_m=5, span_m=2, ec_mpa=30000, i_mm4=5e8), 0.67),
])
def test_cantilever_udl_deflection_is_wl4_over_8ei(kw, mm):
    _close(beam_deflection_cantilever_udl(**kw), mm)


# Simply supported, central point load: delta = P*L^3/(48*E*I); P in N, L in mm
@pytest.mark.parametrize("kw,mm", [
    (dict(p_kn=50, span_m=6, ec_mpa=200000, i_mm4=2e8), 5.63),
    (dict(p_kn=20, span_m=4, ec_mpa=25000, i_mm4=1e9), 1.07),
])
def test_ss_point_load_deflection_is_pl3_over_48ei(kw, mm):
    _close(beam_deflection_ss_point_load_midspan(**kw), mm)


# Fixed-fixed, UDL: support = w*L^2/12 (the max), midspan = w*L^2/24
@pytest.mark.parametrize("kw,sup,mid", [
    (dict(udl_w_kn_m=24, span_m=6), 72.0, 36.0),
    (dict(udl_w_kn_m=10, span_m=5), 20.833, 10.417),
])
def test_fixed_udl_moments_are_wl2_over_12_and_24(kw, sup, mid):
    r = beam_moment_fixed_udl(**kw)
    _close(r["support_moment_kn_m"], sup)
    _close(r["midspan_moment_kn_m"], mid)
    _close(r["max_moment_kn_m"], sup)


# Simply supported point load: central M = P*L/4; off-centre M = P*a*b/L
@pytest.mark.parametrize("kw,m", [
    (dict(point_load_kn=100, span_m=8), 200.0),
    (dict(point_load_kn=60, span_m=10, distance_from_left_m=3), 126.0),
])
def test_point_load_moment_is_pl_over_4_or_pab_over_l(kw, m):
    _close(beam_moment_point_load(**kw)["max_moment_kn_m"], m)


# AISC: 0.75*Fnv*Ab*n ; EN: alpha_v*fub*A*n/1.25, alpha_v from the class table
@pytest.mark.parametrize("kw,kn", [
    (dict(bolt_area_mm2=314, shear_strength_mpa=372, n_shear_planes=2), 175.21),
    (dict(bolt_area_mm2="M20", shear_strength_mpa=469), 86.18),
    (dict(bolt_area_mm2="M20", shear_strength_mpa=800, code="eurocode"), 94.08),
    (dict(bolt_area_mm2="M24", n_shear_planes=2, code="eurocode", grade="10.9"), 282.4),
])
def test_bolt_shear_is_the_stated_product(kw, kn):
    _close(bolt_shear_capacity(**kw)["capacity_kn"], kn)


# ACI: phi*alpha*(0.85*fc*(Ag-Ast) + fy*Ast); tied 0.65*0.80, spiral 0.75*0.85
# EN:  fc/1.5*(Ag-As) + fy/1.15*As
@pytest.mark.parametrize("kw,kn", [
    (dict(gross_area_mm2=160000, steel_area_mm2=2000, fc_mpa=30, fy_mpa=420), 2531.9),
    (dict(gross_area_mm2=125000, steel_area_mm2=1500, fc_mpa=35, fy_mpa=500,
          tie_type="spiral"), 2820.4),
    (dict(gross_area_mm2=160000, steel_area_mm2=2000, fc_mpa=30, fy_mpa=500,
          code="eurocode"), 4029.6),
    (dict(gross_area_mm2=90000, steel_area_mm2=1200, fc_mpa=25, fy_mpa=500,
          code="EC"), 2001.7),
])
def test_column_axial_capacity_is_the_stated_squash_load(kw, kn):
    _close(column_axial_capacity(**kw)["axial_capacity_kn"], kn, tol=0.051)


# AISC: min(0.90*Fy*Ag, 0.75*Fu*An) ; EN: min(A*fy/1.0, 0.9*Anet*fu/1.25)
@pytest.mark.parametrize("kw,cap,governs", [
    (dict(gross_area_mm2=2000, net_area_mm2=1700, fy_mpa=345, fu_mpa=450), 573.75, "rupture"),
    (dict(gross_area_mm2=1000, net_area_mm2=950, fy_mpa=250, fu_mpa=400), 225.0, "yield"),
    (dict(gross_area_mm2=2000, net_area_mm2=1700, fy_mpa=355, fu_mpa=510,
          code="eurocode"), 624.24, "rupture"),
    (dict(gross_area_mm2=1500, net_area_mm2=1400, fy_mpa=275, fu_mpa=430,
          code="eurocode"), 412.5, "yield"),
])
def test_steel_tension_is_the_lesser_limit_state(kw, cap, governs):
    r = steel_tension_capacity(**kw)
    _close(r["capacity_kn"], cap)
    assert r["governing_limit_state"] == governs


# throat = 0.707*leg ; AISC: 0.75*0.60*FEXX*throat*L ; EN: fu/(sqrt3*beta_w*1.25)*throat*L
@pytest.mark.parametrize("kw,kn", [
    (dict(leg_size_mm=8, length_mm=200, electrode_strength_mpa=490), 249.43),
    (dict(leg_size_mm=6, length_mm=150, electrode_strength_mpa=480), 137.44),
    (dict(leg_size_mm=8, length_mm=200, electrode_strength_mpa=510, beta_w=0.9,
          code="eurocode"), 296.07),
    (dict(leg_size_mm=6, length_mm=100, electrode_strength_mpa=430, beta_w=0.85,
          code="eurocode"), 99.12),
])
def test_fillet_weld_is_throat_times_design_stress_times_length(kw, kn):
    _close(weld_capacity(**kw)["capacity_kn"], kn)


# metric-technical: Ec = 15000*sqrt(f'c * 10.197) kg/cm2 ; ACI: Ec = 4700*sqrt(f'c) MPa
@pytest.mark.parametrize("kw,val,unit", [
    (dict(fck_n_mm2=35), 283375.0, "kg/cm2"),
    (dict(fck_n_mm2=25), 239496.0, "kg/cm2"),
    (dict(fck_n_mm2=35, code="aci"), 27806.0, "MPa"),
    (dict(fck_n_mm2=28, code="aci"), 24870.0, "MPa"),
])
def test_concrete_modulus_is_the_stated_root_form(kw, val, unit):
    r = modulus_of_elasticity_concrete(**kw)
    _close(r["value"], val, tol=0.51)
    assert r["unit"] == unit


def test_owner_knowledge_entry_states_the_same_modulus_form():
    kb = json.loads(Path("app/knowledge/construction_kb.json").read_text(encoding="utf-8"))
    entry = next(e for e in kb["entries"] if e.get("id") == "structure.axial_shortening")
    assert "15000 sqrt(f'c) in kg/cm2" in entry["statement"]


# ACI: fr = 0.62*sqrt(f'c) MPa
@pytest.mark.parametrize("fck,fr", [(30, 3.396), (40, 3.921)])
def test_modulus_of_rupture_aci_is_062_root_fc(fck, fr):
    r = modulus_of_rupture(fck, code="aci")
    _close(r["modulus_of_rupture_n_mm2"], fr, tol=0.0011)
    assert r["unit"] == "MPa"


# The same figures through the public route every caller uses.
@pytest.mark.parametrize("name,params,key,want", [
    ("beam_moment_fixed_udl", {"udl_w_kn_m": 24, "span_m": 6}, "support_moment_kn_m", 72.0),
    ("steel_tension_capacity", {"gross_area_mm2": 2000, "net_area_mm2": 1700,
                                "fy_mpa": 345, "fu_mpa": 450}, "capacity_kn", 573.75),
    ("weld_capacity", {"leg_size_mm": 6, "length_mm": 150,
                       "electrode_strength_mpa": 480}, "capacity_kn", 137.44),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "success"
    _close(env["result"][key], want)


# Degenerate inputs are refused, never a confident wrong figure.
@pytest.mark.parametrize("fn,kw", [
    (beam_moment_fixed_udl, dict(udl_w_kn_m=10, span_m=0)),
    (beam_moment_fixed_udl, dict(udl_w_kn_m=-1, span_m=5)),
    (beam_moment_point_load, dict(point_load_kn=10, span_m=5, distance_from_left_m=6)),
    (beam_moment_point_load, dict(point_load_kn=10, span_m=0)),
    (bolt_shear_capacity, dict(bolt_area_mm2=245, n_shear_planes=0)),
    (column_axial_capacity, dict(gross_area_mm2=1000, steel_area_mm2=1000, fc_mpa=30, fy_mpa=420)),
    (column_axial_capacity, dict(gross_area_mm2=1000, steel_area_mm2=10, fc_mpa=0, fy_mpa=420)),
])
def test_degenerate_input_is_refused(fn, kw):
    assert "error" in fn(**kw)


@pytest.mark.parametrize("fn,kw", [
    (beam_deflection_cantilever_udl, dict(w_kn_m=10, span_m=3, ec_mpa=0, i_mm4=1e8)),
    (beam_deflection_cantilever_udl, dict(w_kn_m=10, span_m=3, ec_mpa=200000, i_mm4=0.0001)),
    (beam_deflection_ss_point_load_midspan, dict(p_kn=-1, span_m=3, ec_mpa=200000, i_mm4=1e8)),
    (beam_deflection_ss_point_load_midspan, dict(p_kn=10, span_m=3, ec_mpa=200000, i_mm4=0.0001)),
])
def test_deflection_refuses_bad_modulus_load_and_m4_inertia(fn, kw):
    with pytest.raises(ValueError):
        fn(**kw)
