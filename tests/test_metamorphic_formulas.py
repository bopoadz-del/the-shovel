"""Metamorphic relations over the deterministic formula library.

New test shape (2026-07-25): golden-value tests only re-check the numbers we
already checked. Metamorphic tests assert how outputs MUST MOVE when inputs
move — scaling, monotonicity, symmetry — which catches sign errors, swapped
parameters, and unit slips that a single golden value never exercises.

All relations run on the pure library (no LLM, no DB).
"""
import math

import pytest

from app.lib import construction_formulas as f


def test_beam_deflection_scales_linearly_with_load():
    # delta = 5wL^4/384EI — double the UDL, double the deflection.
    base = f.beam_deflection_ss_udl(w_kn_m=10, span_m=6, ec_mpa=30_000, i_mm4=5e8)
    double = f.beam_deflection_ss_udl(w_kn_m=20, span_m=6, ec_mpa=30_000, i_mm4=5e8)
    assert double == pytest.approx(2 * base, rel=1e-9)


def test_beam_deflection_scales_with_span_fourth_power():
    # Section chosen so 2-dp rounding is negligible against the values.
    base = f.beam_deflection_ss_udl(w_kn_m=10, span_m=4, ec_mpa=30_000, i_mm4=5e7)
    double_span = f.beam_deflection_ss_udl(w_kn_m=10, span_m=8, ec_mpa=30_000, i_mm4=5e7)
    assert double_span == pytest.approx((2 ** 4) * base, rel=1e-3)


def test_cantilever_deflects_more_than_simply_supported():
    # Same load, span, section: a cantilever tip deflection (wL^4/8EI) must
    # exceed the SS midspan deflection (5wL^4/384EI) by the factor 48/5.
    ss = f.beam_deflection_ss_udl(w_kn_m=10, span_m=6, ec_mpa=30_000, i_mm4=5e8)
    cant = f.beam_deflection_cantilever_udl(w_kn_m=10, span_m=6, ec_mpa=30_000, i_mm4=5e8)
    assert cant > ss
    assert cant / ss == pytest.approx(384 / (8 * 5), rel=1e-9)


def test_modulus_of_elasticity_monotonic_in_strength():
    # Stronger concrete is stiffer — Ec must be strictly increasing in fck.
    # The calculator returns {value, unit, ...} so the unit travels with the
    # number (live T12: a bare float in kg/cm2 was converted by guesswork).
    values = [f.modulus_of_elasticity_concrete(fck)["value"]
              for fck in (20, 25, 32, 40, 50)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_shear_stress_inverse_in_section_dims():
    # tau = V/(b*d): doubling the width halves the stress.
    base = f.shear_stress_check(v_kn=300, b_mm=300, d_mm=500)
    wide = f.shear_stress_check(v_kn=300, b_mm=600, d_mm=500)
    assert wide["shear_stress_n_mm2"] == pytest.approx(
        base["shear_stress_n_mm2"] / 2, rel=1e-9
    )


def test_diaphragm_wall_volume_additive_in_length():
    # Volume of a 2L panel = volume of two L panels (additivity).
    one = f.diaphragm_wall_panel_volume(
        panel_length=3, wall_thickness=0.8, excavation_depth=20)
    two = f.diaphragm_wall_panel_volume(
        panel_length=6, wall_thickness=0.8, excavation_depth=20)
    assert two["total_volume_m3"] == pytest.approx(2 * one["total_volume_m3"], rel=1e-9)
    # and N panels of length L == 1 panel of length N*L
    n_panels = f.diaphragm_wall_panel_volume(
        panel_length=3, wall_thickness=0.8, excavation_depth=20, panel_count=2)
    assert n_panels["total_volume_m3"] == pytest.approx(two["total_volume_m3"], rel=1e-9)


def test_thermal_cracking_check_monotone_in_gradient():
    # A larger core-surface gradient can never turn an unsafe check safe.
    mild = f.concrete_thermal_cracking_check(core_temp_c=45, surface_temp_c=35)
    harsh = f.concrete_thermal_cracking_check(core_temp_c=75, surface_temp_c=20)
    def risky(d):
        txt = str(d).lower()
        return ("risk" in txt and "no risk" not in txt) or d.get("cracking_risk") in (True, "high")
    assert risky(harsh) or not risky(mild)


def test_bearing_pressure_scales_inverse_with_area():
    base = f.foundation_bearing_pressure(
        foundation_width_m=2, foundation_length_m=2,
        column_load_kn=1000, soil_bearing_capacity_kn_m2=300)
    bigger = f.foundation_bearing_pressure(
        foundation_width_m=2, foundation_length_m=4,
        column_load_kn=1000, soil_bearing_capacity_kn_m2=300)
    assert bigger["bearing_pressure_kn_m2"] == pytest.approx(
        base["bearing_pressure_kn_m2"] / 2, rel=1e-9)
    # bigger footing at same load can never LOWER the factor of safety
    assert bigger["fos_against_bearing"] >= base["fos_against_bearing"]


def test_modulus_of_rupture_scales_sqrt():
    # fr ~ sqrt(fck): quadrupling fck doubles fr.
    r1 = f.modulus_of_rupture(fck_n_mm2=16)
    r2 = f.modulus_of_rupture(fck_n_mm2=64)
    assert r2["modulus_of_rupture_n_mm2"] == pytest.approx(
        2 * r1["modulus_of_rupture_n_mm2"], rel=1e-2)
