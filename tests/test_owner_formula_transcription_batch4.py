"""Transcription check, batch 4: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
No entry in ``app/knowledge/construction_kb.json`` states these ten, so the
stated form is each formula's own docstring. Every expected value below is a
hand calculation from that stated form, written as a literal; none is
obtained by running the calculator. All inputs are invented figures.

One difference is reported to the owner, not decided by a test: the stated
form of ``rebar_lap_length`` caps the confinement term at 2.5, but the public
route (``run_calculation``) declares that input's range as 1 to 2.5 and
refuses a larger value before the cap runs. The cap is pinned here on a
direct call only; the public route is pinned inside the declared range.
"""
from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas_commercial import productivity_rate
from app.agents.hats.planning.formulas.construction_formulas_planning import (
    productivity_manpower_duration,
    progress_quantity,
)
from app.agents.hats.design.formulas.construction_formulas_loads import (
    live_load_reduction,
    seismic_base_shear,
    wind_pressure,
)
from app.agents.hats.design.formulas.construction_formulas_structural_rc import (
    rebar_lap_length,
    slab_thickness_min,
)
from app.agents.hats.design.formulas.construction_formulas_earthwork import slope_fos_simple
from app.agents.hats.safety.formulas.construction_formulas_safety import fall_arrest_force

TOL = 0.011


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


# rate = output / man-hours; with a crew: crew-hours = man-hours / crew,
# per crew-hour = output / crew-hours.
@pytest.mark.parametrize("o,h,crew,rate,crew_rate", [
    (120, 40, 4, 3.0, 12.0),
    (90, 36, 3, 2.5, 7.5),
])
def test_productivity_rate_is_output_per_man_hour(o, h, crew, rate, crew_rate):
    out = productivity_rate(o, h, crew)
    _close(out["rate_per_worker_hour"], rate)
    _close(out["rate_per_crew_hour"], crew_rate)
    _close(out["crew_hours"], h / crew)


@pytest.mark.parametrize("o,h,crew,rate", [
    (90, 36, None, 2.5),
    (120, 30, "", 4.0),
])
def test_productivity_rate_without_crew_names_the_missing_input(o, h, crew, rate):
    out = productivity_rate(o, h, crew)
    _close(out["rate_per_worker_hour"], rate)
    assert out["needs_input"] == ["crew_size"]
    assert "rate_per_crew_hour" not in out


# PE sheet: productivity = qty / man-hours; man-hours = qty / productivity;
# manpower = man-hours / available hours; duration = qty / daily production;
# cost = duration x crew cost per day; daily required = remaining / days.
@pytest.mark.parametrize("kw,key,want", [
    (dict(quantity_executed=300, man_hours=120), "productivity", 2.5),
    (dict(quantity_executed=450, man_hours=90), "productivity", 5.0),
    (dict(quantity=500, productivity="2.5"), "manhours_required", 200.0),
    (dict(quantity=500, productivity="2.5", available_hours=40), "required_manpower", 5.0),
    (dict(quantity=360, productivity="4", available_hours=30), "required_manpower", 3.0),
    (dict(manpower=6, working_hours=10), "manhours", 60.0),
    (dict(manpower=4, working_hours=8), "manhours", 32.0),
    (dict(quantity=900, daily_production=60), "duration", 15.0),
    (dict(quantity=900, daily_production=60, crew_cost_per_day=2000), "total_cost", 30000.0),
    (dict(quantity=480, daily_production=40, crew_cost_per_day=1500), "total_cost", 18000.0),
    (dict(remaining_qty=450, remaining_days=9), "daily_required_production", 50.0),
    (dict(remaining_qty=120, remaining_days=8), "daily_required_production", 15.0),
])
def test_productivity_manpower_duration_is_the_pe_sheet(kw, key, want):
    _close(productivity_manpower_duration(**kw)[key], want)


# Planned % = planned/total x100; Actual % = actual/total x100;
# Remaining = total - actual; Variance = actual % - planned %.
@pytest.mark.parametrize("total,planned,actual,p_pct,a_pct,rem,var", [
    (1000, 400, 350, 40.0, 35.0, 650.0, -5.0),
    (800, 200, 260, 25.0, 32.5, 540.0, 7.5),
])
def test_progress_quantity_is_the_progress_sheet(total, planned, actual, p_pct, a_pct, rem, var):
    out = progress_quantity(total, planned, actual)
    _close(out["planned_percent"], p_pct)
    _close(out["actual_percent"], a_pct)
    _close(out["remaining_qty"], rem)
    _close(out["progress_variance_percent"], var)


def test_progress_quantity_actual_only_gives_no_variance():
    out = progress_quantity(250, actual_qty=100)
    _close(out["actual_percent"], 40.0)
    _close(out["remaining_qty"], 150.0)
    assert "planned_percent" not in out
    assert "progress_variance_percent" not in out


# ASCE: L = L0 (0.25 + 4.57/sqrt(KLL AT)), floor min_factor L0, no reduction
# when KLL AT < 37.2. EN: alphaA = (5/7) psi0 + 10/A <= 1.0.
@pytest.mark.parametrize("kw,factor,load", [
    (dict(base_live_load_kn_m2=4.79, tributary_area_m2=50), 0.573, 2.745),
    (dict(base_live_load_kn_m2=2.4, tributary_area_m2=20, kll=2.0), 0.973, 2.334),
    (dict(base_live_load_kn_m2=4.0, tributary_area_m2=1000), 0.5, 2.0),
    (dict(base_live_load_kn_m2=3.0, tributary_area_m2=9), 1.0, 3.0),
    # boundary: KLL AT = 36.8 is below 37.2 (no reduction); 37.2 reduces.
    (dict(base_live_load_kn_m2=10.0, tributary_area_m2=9.2), 1.0, 10.0),
    (dict(base_live_load_kn_m2=10.0, tributary_area_m2=9.3), 0.999, 9.993),
    (dict(base_live_load_kn_m2=3.0, tributary_area_m2=50, code="eurocode"), 0.7, 2.1),
    (dict(base_live_load_kn_m2=5.0, tributary_area_m2=25, code="EC"), 0.9, 4.5),
    (dict(base_live_load_kn_m2=5.0, tributary_area_m2=8, code="EC"), 1.0, 5.0),
])
def test_live_load_reduction_is_the_stated_factor(kw, factor, load):
    out = live_load_reduction(**kw)
    _close(out["reduction_factor"], factor, 0.0011)
    _close(out["reduced_live_load_kn_m2"], load, 0.0011)


# ASCE qz = 0.613 Kz Kzt Kd V^2; EN qb = 0.5 rho vb^2.
@pytest.mark.parametrize("kw,pa", [
    (dict(wind_speed_m_s=40), 980.8),
    (dict(wind_speed_m_s=30, kz=0.85, kd=0.85), 398.6),
    (dict(wind_speed_m_s=26, code="eurocode"), 422.5),
    (dict(wind_speed_m_s=30, code="EC"), 562.5),
])
def test_wind_pressure_is_the_stated_velocity_pressure(kw, pa):
    out = wind_pressure(**kw)
    _close(out["velocity_pressure_pa"], pa, 0.051)
    _close(out["velocity_pressure_kn_m2"], pa / 1000.0, 0.0001)


# V = Cs W; ASCE Cs = SDS Ie / R; EN Cs = (ag S 2.5 / q) lambda; a stated Cs governs.
@pytest.mark.parametrize("kw,cs,v", [
    (dict(seismic_weight_kn=10000), 0.125, 1250.0),
    (dict(seismic_weight_kn=8000, sds=0.6, ie=1.25, r=5.0), 0.15, 1200.0),
    (dict(seismic_weight_kn=10000, code="eurocode"), 0.1962, 1961.54),
    (dict(seismic_weight_kn=6000, code="EC", ag_g=0.2, soil_factor_s=1.0,
          behaviour_factor_q=1.5, lambda_factor=1.0), 0.3333, 2000.0),
    (dict(seismic_weight_kn=5000, seismic_coefficient=0.08), 0.08, 400.0),
    (dict(seismic_weight_kn=5000, code="EC", seismic_coefficient=0.05), 0.05, 250.0),
])
def test_seismic_base_shear_is_cs_times_w(kw, cs, v):
    out = seismic_base_shear(**kw)
    _close(out["seismic_response_coefficient_cs"], cs, 0.0001)
    _close(out["base_shear_kn"], v)


# ACI: ld = fy psi_s psi_g / (1.1 sqrt(fc) min(conf, 2.5)) db; lap = 1.3 ld;
#      psi_s 0.8 for db <= 19; psi_g 1.0 to fy 420, 1.15 to 550.
# EN:  lb = db/4 fyd/fbd, fbd = 2.25 eta2 fctd, fctd = 0.7 0.3 fck^(2/3)/1.5,
#      eta2 = (132 - db)/100 above 32 mm; lap = 1.5 lb.
@pytest.mark.parametrize("kw,lap", [
    (dict(bar_diameter_mm=16, fy_mpa=420, fc_mpa=28), 800.5),
    # boundary: psi_s 0.8 holds at db = 19 and is 1.0 at db = 20.
    (dict(bar_diameter_mm=19, fy_mpa=420, fc_mpa=28), 950.5),
    (dict(bar_diameter_mm=20, fy_mpa=420, fc_mpa=28), 1250.7),
    (dict(bar_diameter_mm=25, fy_mpa=500, fc_mpa=35, confinement_ratio=2.0), 1435.8),
    # direct call only: the cap of 2.5 (see the module note on the public route).
    (dict(bar_diameter_mm=25, fy_mpa=420, fc_mpa=35, confinement_ratio=3.0), 839.0),
    (dict(bar_diameter_mm=25, fy_mpa=421, fc_mpa=35, confinement_ratio=3.0), 967.2),
    (dict(bar_diameter_mm=16, fy_mpa=500, fc_mpa=30, code="eurocode"), 857.8),
    (dict(bar_diameter_mm=40, fy_mpa=500, fc_mpa=30, code="EC"), 2330.9),
])
def test_rebar_lap_length_is_the_stated_lap(kw, lap):
    _close(rebar_lap_length(**kw)["lap_length_mm"], lap, 0.051)


# ACI: h = L/ratio x (0.4 + fy/700); EN: h = L/ratio + cover + bar/2.
@pytest.mark.parametrize("kw,h", [
    (dict(span_mm=4800), 240.0),
    (dict(span_mm=6000, support_condition="both_ends_continuous", fy_mpa=500), 238.8),
    (dict(span_mm=2000, support_condition="cantilever"), 200.0),
    (dict(span_mm=4200, support_condition="one_end_continuous", fy_mpa=280), 140.0),
    (dict(span_mm=5000, code="eurocode"), 275.0),
    (dict(span_mm=6000, support_condition="both_ends_continuous", code="EC",
          cover_mm=30, bar_diameter_mm=12), 236.0),
])
def test_slab_thickness_min_is_the_stated_span_ratio(kw, h):
    _close(slab_thickness_min(**kw)["min_thickness_mm"], h, 0.051)


# FoS = tan(phi)/tan(beta) [+ c'/(gamma z sin(beta) cos(beta))].
@pytest.mark.parametrize("kw,fos", [
    (dict(friction_angle_deg=30, slope_angle_deg=20), 1.586),
    (dict(friction_angle_deg=35, slope_angle_deg=35), 1.0),
    (dict(friction_angle_deg=32, slope_angle_deg=25, cohesion_kpa=10,
          unit_weight_kn_m3=18, depth_m=3), 1.824),
])
def test_slope_fos_simple_is_the_infinite_slope(kw, fos):
    _close(slope_fos_simple(**kw)["factor_of_safety"], fos, 0.0011)


# F = m g (1 + H/d); within the limit when F <= 8 kN.
@pytest.mark.parametrize("kw,kn,within", [
    (dict(worker_mass_kg=100, free_fall_m=1.8, deceleration_distance_m=1.0), 2.75, True),
    (dict(worker_mass_kg=90, free_fall_m=2.0, deceleration_distance_m=0.2), 9.71, False),
    (dict(worker_mass_kg=100, free_fall_m=7.0, deceleration_distance_m=1.0, g=10.0), 8.0, True),
    (dict(worker_mass_kg=100, free_fall_m=7.1, deceleration_distance_m=1.0, g=10.0), 8.1, False),
])
def test_fall_arrest_force_is_the_energy_balance(kw, kn, within):
    out = fall_arrest_force(**kw)
    _close(out["max_arrest_force_kn"], kn)
    assert out["within_osha_limit"] is within


@pytest.mark.parametrize("name,params,key,want", [
    ("wind_pressure", {"wind_speed_m_s": 40}, "velocity_pressure_pa", 980.8),
    ("seismic_base_shear", {"seismic_weight_kn": 8000, "sds": 0.6, "ie": 1.25, "r": 5.0},
     "base_shear_kn", 1200.0),
    ("slab_thickness_min", {"span_mm": 4800}, "min_thickness_mm", 240.0),
    ("progress_quantity", {"total_qty": 800, "planned_qty": 200, "actual_qty": 260},
     "progress_variance_percent", 7.5),
    # fy 420 / 421 at the top of the declared confinement range.
    ("rebar_lap_length", {"bar_diameter_mm": 25, "fy_mpa": 420, "fc_mpa": 35,
                          "confinement_ratio": 2.5}, "lap_length_mm", 839.0),
    ("rebar_lap_length", {"bar_diameter_mm": 25, "fy_mpa": 421, "fc_mpa": 35,
                          "confinement_ratio": 2.5}, "lap_length_mm", 967.2),
    ("rebar_lap_length", {"bar_diameter_mm": 19, "fy_mpa": 420, "fc_mpa": 28},
     "lap_length_mm", 950.5),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "success"
    _close(env["result"][key], want, 0.051)


# The 8 kN limit on the public route, at a g it accepts: 256 x 9.75 x
# (1 + 8.6/3.9) = 8000.0 N exactly (within); H = 8.62 gives 8012.8 N (above).
@pytest.mark.parametrize("free_fall,kn,within", [
    (8.6, 8.0, True),
    (8.62, 8.013, False),
])
def test_public_route_fall_arrest_limit_is_inclusive(free_fall, kn, within):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation("fall_arrest_force", {
        "worker_mass_kg": 256, "free_fall_m": free_fall,
        "deceleration_distance_m": 3.9, "g": 9.75})
    assert env["status"] == "success"
    _close(env["result"]["max_arrest_force_kn"], kn, 0.0051)
    assert env["result"]["within_osha_limit"] is within


# Degenerate inputs are refused, never a confident wrong figure.
@pytest.mark.parametrize("fn,kw", [
    (productivity_rate, dict(output_quantity=10, labor_hours=0)),
    (productivity_rate, dict(output_quantity=10, labor_hours=5, crew_size=0)),
    (productivity_manpower_duration, dict()),
    (productivity_manpower_duration, dict(quantity_executed=10, man_hours=0)),
    (productivity_manpower_duration, dict(quantity=10, daily_production=0)),
    (productivity_manpower_duration, dict(remaining_qty=10, remaining_days=-1)),
    (progress_quantity, dict(total_qty=0, actual_qty=5)),
    (progress_quantity, dict(total_qty=100)),
    (fall_arrest_force, dict(worker_mass_kg=0)),
    (fall_arrest_force, dict(free_fall_m=-0.5)),
    (fall_arrest_force, dict(deceleration_distance_m=0)),
])
def test_degenerate_input_is_refused(fn, kw):
    assert "error" in fn(**kw)
