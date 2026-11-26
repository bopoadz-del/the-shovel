"""Transcription check, batch 5: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
For the mass-concrete thermal check the stated form is the owner's knowledge
base (``thermal.core_temp_limit`` and ``thermal.gradient_limit`` in
``app/knowledge/construction_kb.json``); no KB entry states the other nine,
so their stated form is each formula's own docstring. Every expected value
below is a hand calculation from that stated form, written as a literal; none
is obtained by running the calculator. All inputs are invented figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agents.hats.qaqc.formulas.construction_formulas_qc import (
    concrete_curing_time,
    concrete_cylinders,
    concrete_shrinkage,
)
from app.agents.hats.qaqc.formulas.construction_formulas import (
    concrete_maturity_strength,
    concrete_thermal_cracking_check,
)
from app.agents.hats.quantities.formulas.construction_formulas_planning import (
    concrete_mix_proportions,
)
from app.agents.hats.design.formulas.construction_formulas_masonry import masonry_wall_capacity
from app.agents.hats.design.formulas.construction_formulas import (
    post_tensioning_force,
    shear_stress_check,
)
from app.agents.hats.safety.formulas.construction_formulas_safety import scaffold_load_capacity

TOL = 0.011


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


# f = P / A, A = pi/4 * d^2.
@pytest.mark.parametrize("p_kn,d_mm,area,f", [
    (500, 150, 17671.5, 28.29),   # 500e3 / 17671.46
    (300, 100, 7854.0, 38.20),    # 300e3 / 7853.98
])
def test_concrete_cylinders_is_load_over_circle_area(p_kn, d_mm, area, f):
    out = concrete_cylinders(p_kn, d_mm)
    _close(out["compressive_strength_mpa"], f)
    _close(out["cylinder_area_mm2"], area, tol=0.11)


@pytest.mark.parametrize("d_mm", [0, -150])
def test_concrete_cylinders_refuses_a_diameter_not_above_zero(d_mm):
    assert "error" in concrete_cylinders(400, d_mm)


# esh(t) = t / (f + t) * esh_ult.
@pytest.mark.parametrize("t,ult,f,esh,frac", [
    (35, 780, 35, 390.0, 0.5),
    (100, 600, 35, 444.44, 0.7407),
])
def test_concrete_shrinkage_is_hyperbolic_in_time(t, ult, f, esh, frac):
    out = concrete_shrinkage(t, ult, f)
    _close(out["shrinkage_microstrain"], esh)
    _close(out["fraction_of_ultimate"], frac, tol=1e-4)


@pytest.mark.parametrize("t,ult", [(0, 780), (0, 500)])
def test_concrete_shrinkage_is_zero_at_age_zero(t, ult):
    assert concrete_shrinkage(t, ult, 35)["shrinkage_microstrain"] == 0


# t = a * frac / (1 - b * frac).
@pytest.mark.parametrize("frac,days", [
    (0.70, 6.91),   # 2.8 / 0.405
    (0.50, 3.48),   # 2.0 / 0.575
])
def test_concrete_curing_time_inverts_the_strength_gain_curve(frac, days):
    _close(concrete_curing_time(frac, 4.0, 0.85)["days_to_target"], days)


@pytest.mark.parametrize("frac,b", [(1.0, 1.0), (0.0, 0.85)])
def test_concrete_curing_time_refuses_an_unreachable_or_zero_target(frac, b):
    assert "error" in concrete_curing_time(frac, 4.0, b)


# MI = sum(max(0, T - T0) * dt); te = MI / (Tref - T0) / 24;
# f/f28 = te / (a + b * te).
@pytest.mark.parametrize("temps,hours,f28,te,pct,strength", [
    ([20, 20], [24, 24], 40, 2.0, 35.1, 14.0),   # MI 1440; 2 / 5.7
    ([10, 30], [12, 12], 30, 1.0, 20.6, 6.2),    # MI 720; 1 / 4.85
])
def test_concrete_maturity_strength_is_nurse_saul_then_aci209(temps, hours, f28, te, pct, strength):
    out = concrete_maturity_strength(temps, hours, -10.0, f28, 20.0, 4.0, 0.85)
    _close(out["equivalent_age_days"], te, tol=1e-3)
    _close(out["percent_of_28d"], pct)
    _close(out["predicted_strength_n_mm2"], strength)


@pytest.mark.parametrize("temps,mi", [
    ([-15, 20], 720),   # the hour below the datum adds nothing: 0 + 30 * 24
    ([-10, 0], 240),    # at the datum adds nothing: 0 + 10 * 24
])
def test_concrete_maturity_counts_nothing_at_or_below_the_datum(temps, mi):
    out = concrete_maturity_strength(temps, [24, 24], -10.0, 40, 20.0, 4.0, 0.85)
    _close(out["maturity_index_c_hrs"], mi)


# Dry = wet x 1.54 (x (1 + waste)); constituent = dry x parts / total parts.
@pytest.mark.parametrize("kw,dry,cement,sand,agg", [
    (dict(wet_volume=10, cement_parts=1, sand_parts=2, aggregate_parts=4), 15.4, 2.2, 4.4, 8.8),
    (dict(wet_volume=4, cement_parts=1, sand_parts=1.5, aggregate_parts=3, waste_factor=0.05),
     6.468, 1.176, 1.764, 3.528),
])
def test_concrete_mix_proportions_split_the_dry_volume_by_parts(kw, dry, cement, sand, agg):
    out = concrete_mix_proportions(**kw)
    _close(out["dry_volume"], dry, tol=1e-6)
    _close(out["cement_volume"], cement, tol=1e-6)
    _close(out["sand_volume"], sand, tol=1e-6)
    _close(out["aggregate_volume"], agg, tol=1e-6)


@pytest.mark.parametrize("kw,missing", [
    (dict(wet_volume=10, cement_parts=1, sand_parts=2), ["aggregate_parts"]),
    (dict(cement_parts=1, sand_parts=2, aggregate_parts=4), ["wet_volume"]),
])
def test_concrete_mix_proportions_never_invents_a_mix(kw, missing):
    out = concrete_mix_proportions(**kw)
    assert "error" in out and out["missing"] == missing


def _kb_limit(entry_id):
    kb = Path(__file__).resolve().parents[1] / "app" / "knowledge" / "construction_kb.json"
    entries = json.loads(kb.read_text(encoding="utf-8"))["entries"]
    return next(e for e in entries if e["id"] == entry_id)["statement"]


def test_the_thermal_limits_are_the_owners_kb_limits():
    assert "70 degC" in _kb_limit("thermal.core_temp_limit")
    assert "20 degC" in _kb_limit("thermal.gradient_limit")


# KB: core <= 70 degC and core - surface <= 20 degC; at the limit is within.
@pytest.mark.parametrize("core,surface,core_ok,delta_ok", [
    (65, 50, True, True),
    (70, 50, True, True),     # both exactly at the limit
    (75, 50, False, False),
    (60, 39, True, False),    # delta 21
    (71, 60, False, True),    # core 71
])
def test_concrete_thermal_check_reads_the_kb_limits(core, surface, core_ok, delta_ok):
    out = concrete_thermal_cracking_check(core, surface)
    assert out["core_ok"] is core_ok
    assert out["delta_ok"] is delta_ok
    assert out["thermal_cracking_risk"] is not (core_ok and delta_ok)
    _close(out["delta_t_c"], core - surface)


# TMS 402: Pa = (0.25 f'm An + 0.65 Ast Fs) R; R = 1 - (h / 140r)^2 for
# h/r <= 99, else (70r / h)^2; r = t / sqrt(12) when not given.
@pytest.mark.parametrize("kw,cap,R", [
    (dict(masonry_strength_mpa=10, net_area_mm2=190000, height_mm=3000, thickness_mm=190),
     402.5, 0.8474),      # h/r 54.70; 475 kN x 0.84736
    (dict(masonry_strength_mpa=12, net_area_mm2=140000, height_mm=6000, thickness_mm=140,
          steel_area_mm2=400, steel_stress_mpa=200),
     104.9, 0.2223),      # h/r 148.46; (420 + 52) kN x 0.22231
])
def test_masonry_wall_capacity_tms402(kw, cap, R):
    out = masonry_wall_capacity(**kw)
    _close(out["capacity_kn"], cap, tol=0.051)
    _close(out["slenderness_reduction_R"], R, tol=1e-4)


@pytest.mark.parametrize("h,branch_R", [
    (54.25 * 99.0, 1 - (99.0 / 140.0) ** 2),   # h/r = 99 exactly: the first form
    (54.25 * 100.0, (70.0 / 100.0) ** 2),      # h/r = 100: the second form
])
def test_masonry_wall_slenderness_switches_after_h_over_r_99(h, branch_R):
    out = masonry_wall_capacity(10, 100000, h, 190, radius_gyration_mm=54.25)
    _close(out["slenderness_reduction_R"], branch_R, tol=1e-4)


# EN 1996-1-1: NRd = Phi_m A fk / gM; Phi_m = A1 exp(-u^2 / 2);
# A1 = 1 - 2 e/t; u = (lam - 0.063) / (0.73 - 1.17 e/t); lam = (hef/tef) sqrt(fk/E).
@pytest.mark.parametrize("fk,cap,phi", [
    (5.0, 147.5, 0.7963),   # e = 0.05t; lam 0.3953; 0.79629 x 100000 x 1.85185
    (10.0, 294.9, 0.7963),  # E = KE fk, so lam and Phi_m do not change with fk
])
def test_masonry_wall_capacity_eurocode(fk, cap, phi):
    out = masonry_wall_capacity(fk, 100000, 2500, 200, code="eurocode")
    _close(out["phi_reduction"], phi, tol=1e-4)
    _close(out["capacity_kn"], cap, tol=0.051)


# P = w L^2 / (8 e), w = 0.75 x DL (DL = 25 x thickness when not given);
# e = ratio x thickness; strands = ceil(P / stress / (0.78 pi d^2 / 4)).
@pytest.mark.parametrize("kw,force,area,strands", [
    (dict(span_m=8, slab_thickness_m=0.2, live_load_kn_m2=3), 1000.0, 769.2, 8),
    (dict(span_m=10, slab_thickness_m=0.25, live_load_kn_m2=3, dead_load_kn_m2=6,
          tendon_stress_n_mm2=1400, tendon_diameter_mm=15.2, eccentricity_ratio=0.2),
     1125.0, 803.6, 6),
    # 769.23 / (0.78 x pi x 7.6^2 = 141.54) = 5.43 strands: rounded UP to 6
    (dict(span_m=8, slab_thickness_m=0.2, live_load_kn_m2=3, tendon_diameter_mm=15.2),
     1000.0, 769.2, 6),
])
def test_post_tensioning_force_balances_three_quarters_of_dead_load(kw, force, area, strands):
    out = post_tensioning_force(**kw)
    _close(out["tendon_force_kn"], force, tol=0.051)
    _close(out["tendon_area_mm2"], area, tol=0.051)
    assert out["num_strands"] == strands


# v = V / (b d), V in kN, b and d in mm.
@pytest.mark.parametrize("v,b,d,stress", [
    (150, 300, 500, 1.0),
    (90, 250, 450, 0.8),
])
def test_shear_stress_is_force_over_b_d(v, b, d, stress):
    _close(shear_stress_check(v, b, d)["shear_stress_n_mm2"], stress, tol=1e-3)


# intended = duty kPa x area; required = intended x safety factor (4:1).
@pytest.mark.parametrize("area,duty,sf,intended,required", [
    (10, "medium", 4.0, 24.0, 96.0),
    (5, "heavy", 4.0, 18.0, 72.0),
    (8, "Light", 3.0, 9.6, 28.8),     # the caller's 3:1 factor, not a fixed 4
])
def test_scaffold_capacity_is_four_times_intended_load(area, duty, sf, intended, required):
    out = scaffold_load_capacity(area, duty, sf)
    _close(out["intended_load_kn"], intended)
    _close(out["required_capacity_kn"], required)


@pytest.mark.parametrize("duty", ["extra heavy", ""])
def test_scaffold_capacity_asks_for_an_unknown_duty(duty):
    assert "error" in scaffold_load_capacity(10, duty)
