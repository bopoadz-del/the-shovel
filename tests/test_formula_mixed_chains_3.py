"""Formula mixed scenarios, part 3: CI chains, no model (exam 6).

From the formula test plan, part 3 (mixed scenarios). Each chain calls the
calculators in order through ``run_calculation`` (the entry point
``construction_calc`` uses). The named output of one step is typed into the
named input of the next. Every intermediate value in the plan's "Hand" line is
asserted, not only the final value.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
chain runs on two inputs: the plan's case and a second invented case worked
by hand the same way. Where the chain makes a decision (capacity OK, lift
passes, footing OK), one input passes and one fails. Plan tolerance: 0.5 %
on any number unless stated.

Chains here: M03, M09 (CI part), M10 (CI part), M15, M20, M22 and M23.
All inputs are invented numbers; no project, person or document is named.

M22 convention, read from the docstring before writing the test (plan rule
for a flagged convention): ``material_consumption`` computes
Material = Quantity of Work / Output per Unit, so its ``output_per_unit`` is
work per unit of material (metres of duct per litre), the reciprocal of the
grout step's litres per metre. The plan's line passes 7.854 L/m directly;
that would give 270 / 7.854 = 34.4, not litres of grout. The chain below
types in 1 / (L/m) and asserts the plan's own hand values (2,120.6 L,
2,332.6 L, 2.333 m3).
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# ------------------------------ M03 beam check -> beam volume -> carbon
# demand M = w L^2 / 8; ACI phi.Mn = 0.9 As fy (d - a/2), a = As fy / (0.85 fc b);
# the product's capacity is phi-reduced, so the check compares demand with
# phi.Mn on both sides. volume = L x b x h; carbon = volume x factor.

@pytest.mark.parametrize(
    "udl, span, n_bars, bar_mm, b, h, d, fy, fc, factor, "
    "demand, capacity, carries, volume, carbon_kg", [
        # plan: 30*36/8 = 135.0; As 3*314.16 = 942.5; a = 51.74; 0.9*942.5*420*(540-25.87)
        # = 183.16 kN.m; 135 <= 183.2 OK (ratio 0.737); 6*0.3*0.6 = 1.080 m3; *300 = 324.0
        (30, 6, 3, 20, 300, 600, 540, 420, 30, 300, 135.0, 183.16, True, 1.08, 324.0),
        # 20*25/8 = 62.5; As 2*201.06 = 402.12; a = 402.12*500/(0.85*25*250) = 37.85;
        # 0.9*402.12*500*(450-18.92) = 78.01; OK (ratio 0.801); 5*0.25*0.5 = 0.625; *250 = 156.25
        (20, 5, 2, 16, 250, 500, 450, 500, 25, 250, 62.5, 78.01, True, 0.625, 156.25),
        # plan section on an 8 m span: 30*64/8 = 240.0 > 183.16 NOT OK (ratio 1.310);
        # 8*0.3*0.6 = 1.440 m3; *300 = 432.0
        (30, 8, 3, 20, 300, 600, 540, 420, 30, 300, 240.0, 183.16, False, 1.44, 432.0),
    ])
def test_m03_beam_check_then_volume_then_carbon(udl, span, n_bars, bar_mm, b, h, d, fy, fc,
                                                factor, demand, capacity, carries, volume,
                                                carbon_kg):
    moment = _ok("beam_moment_simple", {"udl_w_kn_m": udl, "span_m": span})
    assert moment["max_moment_kn_m"] == pytest.approx(demand, rel=_REL)
    steel_area = n_bars * math.pi / 4 * bar_mm ** 2
    cap = _ok("rc_beam_moment_capacity", {"steel_area_mm2": steel_area, "fy_mpa": fy,
                                          "width_mm": b, "eff_depth_mm": d,
                                          "fc_mpa": fc, "code": "ACI"})
    assert cap["moment_capacity_kn_m"] == pytest.approx(capacity, rel=_REL)
    assert (moment["max_moment_kn_m"] <= cap["moment_capacity_kn_m"]) is carries
    vol = _ok("concrete_volume", {"length_m": span, "width_m": b / 1000,
                                  "thickness_m": h / 1000})
    assert vol["net_volume_m3"] == pytest.approx(volume, rel=_REL)
    co2 = _ok("carbon_footprint_concrete", {"volume_m3": vol["net_volume_m3"],
                                            "embodied_kgco2e_m3": factor})
    assert co2["total_kgco2e"] == pytest.approx(carbon_kg, rel=_REL)
    assert co2["total_tco2e"] == pytest.approx(carbon_kg / 1000, rel=_REL)


# --------------------- M09 (CI part) volume -> crews; missing rate is named
# crews = Q / (D x R) rounded UP; duration with N crews = Q / (R x N).

@pytest.mark.parametrize("dims, rate, days, volume, crews, crews_up, duration", [
    ((40, 20, 0.8), 28, 10, 640.0, 2.2857, 3, 7.619),  # 640; 640/280 = 2.286 -> 3; 640/84 = 7.62
    ((30, 15, 0.6), 25, 5, 270.0, 2.16, 3, 3.6),       # 270; 270/125 = 2.16 -> 3; 270/75 = 3.60
])
def test_m09_volume_then_crews_rounded_up(dims, rate, days, volume, crews, crews_up, duration):
    length, width, thick = dims
    vol = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick})
    assert vol["net_volume_m3"] == pytest.approx(volume, rel=_REL)
    sized = _ok("crew_sizing", {"quantity": vol["net_volume_m3"], "duration_days": days,
                                "rate_per_crew_day": rate})
    assert sized["crews_required"] == pytest.approx(crews, rel=_REL)
    assert sized["crews_required_rounded_up"] == crews_up
    again = _ok("crew_sizing", {"quantity": vol["net_volume_m3"], "duration_days": days,
                                "rate_per_crew_day": rate, "crews": crews_up})
    assert again["duration_for_crews"] == pytest.approx(duration, rel=_REL)


@pytest.mark.parametrize("given, missing", [
    ({"quantity": 640.0, "duration_days": 10}, "rate_per_crew_day"),  # the plan's turn 1
    ({"quantity": 270.0, "rate_per_crew_day": 25}, "duration_days"),
])
def test_m09_missing_input_mid_chain_is_named_not_defaulted(given, missing):
    env = run_calculation("crew_sizing", dict(given))
    assert env.get("status") != "success", env
    assert missing in repr(env), env


# ------------------ M10 (CI part) M01 rerun after each correction of input
# net = L x W x t; order = net x 1.05; cost = order x rate.

@pytest.mark.parametrize("thick, rate, net, order, cost", [
    (0.30, 310, 86.4, 90.72, 28_123.20),  # 24*12*0.30 = 86.4; *1.05 = 90.72; *310 = 28,123.20
    (0.30, 325, 86.4, 90.72, 29_484.00),  # volumes unchanged; 90.72*325 = 29,484.00
])
def test_m10_corrected_inputs_recompute_every_downstream_figure(thick, rate, net, order, cost):
    vol = _ok("concrete_volume", {"length_m": 24, "width_m": 12, "thickness_m": thick,
                                  "waste_factor": 0.05})
    assert vol["net_volume_m3"] == pytest.approx(net, rel=_REL)
    assert vol["volume_with_waste_m3"] == pytest.approx(order, rel=_REL)
    priced = _ok("unit_cost_total", {"quantity": vol["volume_with_waste_m3"], "unit_rate": rate})
    assert priced["total_cost"] == pytest.approx(cost, rel=_REL)


# --------------------- M15 precast erection -> crane lift check -> hire cost
# effective = (member + rigging) x dynamic; net = chart - deductions;
# util = effective / net, passes when <= limit; monthly = dry + operator +
# riggers; hire = monthly x cranes x months.

@pytest.mark.parametrize(
    "beam_t, rig_t, dyn, chart_t, deduct_t, limit, effective, erect_pct, net_t, util_pct, "
    "passes, months, dry, op, rig_rate, monthly, total", [
        # plan: (14+0.6)*1.15 = 16.79; 16.79/22 = 76.3 %; 22-1.2 = 20.8; 16.79/20.8 = 80.72 %
        # <= 85 % PASS; 55,000+9,000+12,000 = 76,000; *4 = 304,000
        (14, 0.6, 1.15, 22, 1.2, 0.85, 16.79, 76.3, 20.8, 80.72, True,
         4, 55_000, 9_000, 12_000, 76_000, 304_000),
        # (9+0.5)*1.2 = 11.4; 11.4/14 = 81.4 %; 14-0.8 = 13.2; 11.4/13.2 = 86.36 % > 85 % FAIL;
        # 40,000+8,000+10,000 = 58,000; *3 = 174,000
        (9, 0.5, 1.2, 14, 0.8, 0.85, 11.4, 81.4, 13.2, 86.36, False,
         3, 40_000, 8_000, 10_000, 58_000, 174_000),
    ])
def test_m15_erection_then_lift_capacity_then_crane_hire(beam_t, rig_t, dyn, chart_t, deduct_t,
                                                         limit, effective, erect_pct, net_t,
                                                         util_pct, passes, months, dry, op,
                                                         rig_rate, monthly, total):
    erect = _ok("precast_beam_erection_check", {
        "beam_weight_t": beam_t, "beam_length_m": 12, "crane_capacity_t": chart_t,
        "lift_radius_m": 15, "rigging_weight_t": rig_t, "dynamic_factor": dyn})
    assert erect["effective_lift_weight_t"] == pytest.approx(effective, rel=_REL)
    assert erect["utilization_pct"] == pytest.approx(erect_pct, rel=_REL)
    lift = _ok("crane_lift_capacity", {"chart_capacity_t": chart_t, "deductions_t": deduct_t,
                                       "load_t": erect["effective_lift_weight_t"],
                                       "max_utilization": limit})
    assert lift["net_capacity_t"] == pytest.approx(net_t, rel=_REL)
    assert lift["utilization_percent"] == pytest.approx(util_pct, rel=_REL)
    assert lift["passed"] is passes
    hire = _ok("crane_cost_estimate", {
        "num_cranes": 1, "crane_capacity_tons": chart_t, "duration_months": months,
        "dry_hire_rate_per_month": dry, "operator_rate_per_month": op,
        "riggers_rate_per_month": rig_rate})
    assert hire["per_crane_monthly"] == pytest.approx(monthly, rel=_REL)
    assert hire["total_project_cost"] == pytest.approx(total, rel=_REL)


# ------------------------------- M20 progress KPI -> earned value -> ROI
# variance = actual% - planned%; SPI = actual / planned; EV = actual% x BAC;
# PV = planned% x BAC; CV = EV - AC; SV = EV - PV; CPI = EV / AC;
# EAC = BAC / CPI; ROI = (price - EAC) / EAC x 100.

@pytest.mark.parametrize(
    "planned, actual, bac, ac, price, variance, spi, cv, sv, cpi, eac, net, roi", [
        # plan: -5.0; 35/40 = 0.875; EV 3.5 M, PV 4.0 M; CV -0.4 M; SV -0.5 M;
        # CPI 3.5/3.9 = 0.8974; EAC 10/0.8974 = 11,142,857; 10.8 M - EAC = -342,857; -3.08 %
        (40, 35, 10_000_000, 3_900_000, 10_800_000,
         -5.0, 0.875, -400_000, -500_000, 0.897, 11_142_857.14, -342_857.14, -3.08),
        # +5.0; 55/50 = 1.1; EV 1.1 M, PV 1.0 M; CV +0.1 M; SV +0.1 M; CPI 1.1/1.0 = 1.1;
        # EAC 2.0/1.1 = 1,818,181.82; 2.3 M - EAC = 481,818.18; 481,818.18/1,818,181.82 = 26.50 %
        (50, 55, 2_000_000, 1_000_000, 2_300_000,
         5.0, 1.1, 100_000, 100_000, 1.1, 1_818_181.82, 481_818.18, 26.5),
    ])
def test_m20_progress_then_evm_then_roi(planned, actual, bac, ac, price, variance, spi, cv, sv,
                                        cpi, eac, net, roi):
    kpi = _ok("progress_kpi", {"planned_cum": planned, "actual_cum": actual,
                               "planned_total": 100})
    assert kpi["variance_pct"] == pytest.approx(variance, rel=_REL)
    assert kpi["spi_progress"] == pytest.approx(spi, rel=_REL)
    ev = kpi["actual_pct"] / 100 * bac
    pv = kpi["planned_pct"] / 100 * bac
    evm = _ok("calculate_evm", {"bac": bac, "ev": ev, "pv": pv, "ac": ac})
    assert evm["CV"] == pytest.approx(cv, rel=_REL)
    assert evm["SV"] == pytest.approx(sv, rel=_REL)
    assert evm["CPI"] == pytest.approx(cpi, rel=_REL)
    assert evm["SPI"] == pytest.approx(spi, rel=_REL)
    assert evm["EAC"] == pytest.approx(eac, rel=_REL)
    ret = _ok("roi_calculator", {"gain": price, "cost": evm["EAC"]})
    assert ret["net_profit"] == pytest.approx(net, rel=_REL)
    assert ret["roi_percent"] == pytest.approx(roi, rel=_REL)


# ----------------- M22 grout: N/mm2 -> bar; litres per metre -> litres -> m3
# area = pi d^2 / 4; L/m = area_mm2 x 1000 mm / 1e6; 1 N/mm2 = 10 bar;
# litres = length / (m per litre) x (1 + waste%); m3 = L / 1000.

@pytest.mark.parametrize(
    "duct_mm, pressure, length_m, waste_pct, area, l_per_m, bar, base_l, with_waste_l, m3", [
        # plan: pi*100^2/4 = 7,853.98; 7.854 L/m; 1.2 N/mm2 = 12.00 bar; 6*45 = 270 m;
        # 270*7.854 = 2,120.6 L; *1.10 = 2,332.6 L = 2.333 m3
        (100, 1.2, 270, 10, 7_853.98, 7.854, 12.0, 2_120.6, 2_332.6, 2.333),
        # pi*80^2/4 = 5,026.55; 5.027 L/m; 0.8 N/mm2 = 8.00 bar; 4*30 = 120 m;
        # 120*5.027 = 603.2 L; *1.05 = 633.4 L = 0.633 m3
        (80, 0.8, 120, 5, 5_026.55, 5.027, 8.0, 603.2, 633.4, 0.6334),
    ])
def test_m22_grout_pressure_unit_and_volume_litres_to_m3(duct_mm, pressure, length_m, waste_pct,
                                                         area, l_per_m, bar, base_l,
                                                         with_waste_l, m3):
    grout = _ok("grout_pressure_calc", {"tendon_duct_diameter_mm": duct_mm,
                                        "required_pressure_n_mm2": pressure})
    assert grout["duct_area_mm2"] == pytest.approx(area, rel=_REL)
    assert grout["grout_volume_l_m"] == pytest.approx(l_per_m, rel=_REL)
    gauge = _ok("pe_unit_convert", {"value": grout["pressure_n_mm2"],
                                    "from_unit": "N/mm2", "to_unit": "bar"})
    assert gauge["value_out"] == pytest.approx(bar, rel=_REL)
    need = _ok("material_consumption", {"quantity_of_work": length_m,
                                        "output_per_unit": 1 / grout["grout_volume_l_m"],
                                        "waste_percent": waste_pct})
    assert need["base_without_waste"] == pytest.approx(base_l, rel=_REL)
    assert need["material_required"] == pytest.approx(with_waste_l, rel=_REL)
    in_m3 = _ok("pe_unit_convert", {"value": need["material_required"],
                                    "from_unit": "L", "to_unit": "m3"})
    assert in_m3["value_out"] == pytest.approx(m3, rel=_REL)


# ------------------- M23 live-load reduction -> column load -> footing
# ASCE 7 SI: factor = 0.25 + 4.57 / sqrt(KLL x AT); P = floors x AT x
# (dead + reduced live); q = P / (B x L); FoS = allowable / q; OK when FoS >= 1.

@pytest.mark.parametrize(
    "floors, at, live, dead, footing, allow, factor, reduced, load, q, fos, ok", [
        # plan: KLL*AT = 240; 0.25+4.57/15.492 = 0.545; 4.0*0.545 = 2.180; per floor
        # 60*5 + 60*2.180 = 430.8; *6 = 2,584.8 kN; /9 = 287.2; 300/287.2 = 1.045 OK
        (6, 60, 4.0, 5.0, 3, 300, 0.545, 2.180, 2_584.8, 287.2, 1.045, True),
        # KLL*AT = 160; 0.25+4.57/12.649 = 0.611; 3.0*0.611 = 1.834; per floor
        # 40*6 + 40*1.834 = 313.4; *4 = 1,253.4 kN; /4 = 313.4; 300/313.4 = 0.957 NOT OK
        (4, 40, 3.0, 6.0, 2, 300, 0.611, 1.834, 1_253.4, 313.4, 0.957, False),
    ])
def test_m23_live_load_reduction_then_column_load_then_footing(floors, at, live, dead, footing,
                                                               allow, factor, reduced, load, q,
                                                               fos, ok):
    llr = _ok("live_load_reduction", {"base_live_load_kn_m2": live, "tributary_area_m2": at,
                                      "code": "ASCE", "kll": 4})
    assert llr["reduction_factor"] == pytest.approx(factor, rel=_REL)
    assert llr["reduced_live_load_kn_m2"] == pytest.approx(reduced, rel=_REL)
    column = floors * at * (dead + llr["reduced_live_load_kn_m2"])
    assert column == pytest.approx(load, rel=_REL)
    bearing = _ok("foundation_bearing_pressure", {
        "foundation_width_m": footing, "foundation_length_m": footing,
        "column_load_kn": column, "soil_bearing_capacity_kn_m2": allow})
    assert bearing["bearing_pressure_kn_m2"] == pytest.approx(q, rel=_REL)
    assert bearing["fos_against_bearing"] == pytest.approx(fos, rel=_REL)
    assert bearing["bearing_ok"] is ok
