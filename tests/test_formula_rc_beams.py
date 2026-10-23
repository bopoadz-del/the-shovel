"""Oracle tests for beam analysis (statics) + RC design (dual-code) formulas.

Statics values are exact; RC capacities are hand-derived from ACI 318-19 / EN
1992-1-1 and shown in comments. Demand (beam_analysis) is deliberately separate
from capacity (structural_rc).
"""
import pytest

from app.lib.construction_formulas import CALCULATORS


# ── beam analysis (statics, code-agnostic) ─────────────────────────────────

def test_beam_moment_simple_udl():
    # w*L^2/8 = 20*36/8 = 90 kN.m.
    r = CALCULATORS["beam_moment_simple"](udl_w_kn_m=20, span_m=6)
    assert r["max_moment_kn_m"] == pytest.approx(90.0, abs=0.01)


def test_beam_moment_point_central():
    # P*L/4 = 50*6/4 = 75 kN.m.
    r = CALCULATORS["beam_moment_point_load"](point_load_kn=50, span_m=6)
    assert r["max_moment_kn_m"] == pytest.approx(75.0, abs=0.01)


def test_beam_moment_point_offset():
    # P*a*b/L = 50*2*4/6 = 66.667 kN.m.
    r = CALCULATORS["beam_moment_point_load"](point_load_kn=50, span_m=6,
                                              distance_from_left_m=2)
    assert r["max_moment_kn_m"] == pytest.approx(66.667, abs=0.01)


def test_beam_shear_simple():
    # w*L/2 + P/2 = 20*6/2 + 0 = 60 kN.
    r = CALCULATORS["beam_shear_simple"](udl_w_kn_m=20, span_m=6)
    assert r["max_shear_kn"] == pytest.approx(60.0, abs=0.01)


def test_beam_moment_fixed_udl():
    # support w*L^2/12 = 60; midspan w*L^2/24 = 30 kN.m.
    r = CALCULATORS["beam_moment_fixed_udl"](udl_w_kn_m=20, span_m=6)
    assert r["support_moment_kn_m"] == pytest.approx(60.0, abs=0.01)
    assert r["midspan_moment_kn_m"] == pytest.approx(30.0, abs=0.01)


# ── RC design (capacity, dual-code) ────────────────────────────────────────

def test_rc_moment_capacity_aci():
    # a = 1500*420/(0.85*30*300) = 82.35 mm; phi.Mn = 0.9*1500*420*(550-41.18)
    # = 288.50 kN.m.
    r = CALCULATORS["rc_beam_moment_capacity"](
        steel_area_mm2=1500, fy_mpa=420, width_mm=300, eff_depth_mm=550,
        fc_mpa=30, code="aci")
    assert r["moment_capacity_kn_m"] == pytest.approx(288.50, abs=0.3)


def test_rc_moment_capacity_eurocode():
    # fcd=20, fyd=365.2; x=As*fyd/(0.8*b*fcd)=114.1 mm; z=550-45.65=504.35;
    # MRd=1500*365.2*504.35 = 276.3 kN.m.
    r = CALCULATORS["rc_beam_moment_capacity"](
        steel_area_mm2=1500, fy_mpa=420, width_mm=300, eff_depth_mm=550,
        fc_mpa=30, code="eurocode")
    assert r["moment_capacity_kn_m"] == pytest.approx(276.3, abs=0.5)


def test_rc_shear_capacity_aci():
    # Table 22.5.5.1, no stirrups, ρ=0.01, d=550: φVc ≈ 76.2 kN.
    r = CALCULATORS["rc_beam_shear_capacity"](
        width_mm=300, eff_depth_mm=550, fc_mpa=30, code="aci")
    assert r["shear_capacity_kn"] == pytest.approx(76.2, abs=0.1)


def test_rc_shear_capacity_eurocode():
    # k=1+sqrt(200/550)=1.603; VRd,c=0.12*1.603*(100*0.01*30)^(1/3)*300*550
    # = 0.12*1.603*3.107*165000 = 98.6 kN.
    r = CALCULATORS["rc_beam_shear_capacity"](
        width_mm=300, eff_depth_mm=550, fc_mpa=30, code="eurocode", rho_l=0.01)
    assert r["shear_capacity_kn"] == pytest.approx(98.6, abs=0.5)


def test_slab_thickness_min_aci_simply_supported():
    # L/20 * (0.4 + 420/700) = 300 * 1.0 = 300 mm.
    r = CALCULATORS["slab_thickness_min"](span_mm=6000, support_condition="simply_supported",
                                          code="aci", fy_mpa=420)
    assert r["min_thickness_mm"] == pytest.approx(300.0, abs=0.5)


def test_slab_thickness_min_eurocode_cantilever():
    # EC2 cantilever l/d = 8: d = 750 mm; h = d + 25 mm nominal cover.
    r = CALCULATORS["slab_thickness_min"](span_mm=6000, support_condition="cantilever",
                                          code="eurocode")
    assert r["min_thickness_mm"] == pytest.approx(775.0, abs=0.5)


def test_rebar_lap_length_aci():
    # ld = (420/(1.1*sqrt(30)*1.5))*20 = 929.5 mm; lap = 1.3*ld = 1208.4 mm.
    r = CALCULATORS["rebar_lap_length"](bar_diameter_mm=20, fy_mpa=420, fc_mpa=30,
                                        code="aci", confinement_ratio=1.5)
    assert r["lap_length_mm"] == pytest.approx(1208.4, abs=2.0)


def test_rebar_lap_length_eurocode():
    # fbd=2.25*0.7*0.3*30^(2/3)/1.5=3.041; lb,rqd=(20/4)*(365.2/3.041)=600.5;
    # l0=1.5*600.5=900.8 mm.
    r = CALCULATORS["rebar_lap_length"](bar_diameter_mm=20, fy_mpa=420, fc_mpa=30,
                                        code="eurocode")
    assert r["lap_length_mm"] == pytest.approx(900.8, abs=3.0)
