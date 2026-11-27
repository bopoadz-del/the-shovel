"""Transcription check, batch 2: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what the owner's stated form says.
No entry in ``app/knowledge/construction_kb.json`` states these ten, so the
stated form is each formula's own docstring. Every expected value below is a
hand calculation from that stated form, written as a literal; none is
obtained by running the calculator. All inputs are invented figures.
"""
from __future__ import annotations

import pytest

from app.agents.hats.design.formulas.construction_formulas_structural_rc import (
    rc_beam_moment_capacity,
    rc_beam_shear_capacity,
)
from app.agents.hats.design.formulas.construction_formulas_beam_analysis import (
    beam_moment_simple,
    beam_shear_simple,
)
from app.agents.hats.design.formulas.construction_formulas import (
    beam_deflection_ss_udl,
    foundation_bearing_pressure,
)
from app.agents.hats.commercial.formulas.construction_knowledge import calculate_evm
from app.agents.hats.planning.formulas.project_control import progress_kpi
from app.agents.hats.planning.formulas.construction_formulas_planning import critical_path_float
from app.agents.hats.contracts.formulas.construction_formulas_commercial import (
    delay_damages_daily,
)

TOL = 0.011


def _close(got, want):
    assert got is not None
    assert abs(float(got) - float(want)) <= TOL, (got, want)


# ACI: a = As*fy/(0.85*fc*b); phi.Mn = 0.9*As*fy*(d - a/2)
# EN:  fcd = fc/1.5, fyd = fy/1.15; x = As*fyd/(0.8*b*fcd); z = d - 0.4x; MRd = As*fyd*z
@pytest.mark.parametrize("kw,m", [
    (dict(steel_area_mm2=1500, fy_mpa=420, width_mm=300, eff_depth_mm=500, fc_mpa=30), 260.15),
    (dict(steel_area_mm2=1000, fy_mpa=500, width_mm=250, eff_depth_mm=450, fc_mpa=25), 181.32),
    (dict(steel_area_mm2=1500, fy_mpa=500, width_mm=300, eff_depth_mm=500, fc_mpa=30, code="eurocode"), 290.64),
    (dict(steel_area_mm2=800, fy_mpa=500, width_mm=250, eff_depth_mm=400, fc_mpa=25, code="EC"), 124.61),
])
def test_rc_beam_moment_capacity_is_the_stated_stress_block(kw, m):
    _close(rc_beam_moment_capacity(**kw)["moment_capacity_kn_m"], m)


# ACI: phi*0.66*lambda_s*rho^(1/3)*sqrt(fc)*bw*d, lambda_s = min(1, sqrt(2/(1+0.004d))),
#      capped at 0.42*sqrt(fc)*bw*d, phi = 0.75
# EN:  0.12*k*(100*rho*fck)^(1/3)*bw*d, k = min(1+sqrt(200/d), 2)
@pytest.mark.parametrize("kw,v", [
    (dict(width_mm=300, eff_depth_mm=500, fc_mpa=30, rho_l=0.01), 71.54),
    (dict(width_mm=250, eff_depth_mm=900, fc_mpa=40, rho_l=0.015), 114.55),
    (dict(width_mm=300, eff_depth_mm=500, fc_mpa=30, rho_l=0.01, code="eurocode"), 91.30),
    (dict(width_mm=250, eff_depth_mm=400, fc_mpa=40, rho_l=0.015, code="eurocode"), 80.20),
])
def test_rc_beam_shear_capacity_is_the_stated_concrete_term(kw, v):
    _close(rc_beam_shear_capacity(**kw)["shear_capacity_kn"], v)


# Mmax = w*L^2/8
@pytest.mark.parametrize("kw,m", [
    (dict(udl_w_kn_m=25, span_m=8), 200.0),
    (dict(udl_w_kn_m=12.5, span_m=6), 56.25),
])
def test_beam_moment_simple_is_wl2_over_8(kw, m):
    _close(beam_moment_simple(**kw)["max_moment_kn_m"], m)


# V = w*L/2 + P/2
@pytest.mark.parametrize("kw,v", [
    (dict(udl_w_kn_m=25, span_m=8), 100.0),
    (dict(udl_w_kn_m=10, span_m=6, central_point_load_kn=40), 50.0),
])
def test_beam_shear_simple_is_half_the_load(kw, v):
    _close(beam_shear_simple(**kw)["max_shear_kn"], v)


# delta = 5wL^4/(384EI), w in N/mm, L in mm
@pytest.mark.parametrize("kw,mm", [
    (dict(w_kn_m=20, span_m=6, ec_mpa=30000, i_mm4=5.4e9), 2.08),
    (dict(w_kn_m=10, span_m=5, ec_mpa=200000, i_mm4=1e8), 4.07),
])
def test_beam_deflection_is_5wl4_over_384ei(kw, mm):
    _close(beam_deflection_ss_udl(**kw), mm)


# q = P/A ; net = q - depth*soil unit weight (declared 18 kN/m3) ; FOS = allowable/net ; ok when FOS >= 1
@pytest.mark.parametrize("kw,q,net,fos,ok", [
    (dict(foundation_width_m=2, foundation_length_m=2, column_load_kn=800,
          soil_bearing_capacity_kn_m2=250), 200.0, 200.0, 1.25, True),
    (dict(foundation_width_m=2, foundation_length_m=2, column_load_kn=800,
          soil_bearing_capacity_kn_m2=250, foundation_depth_m=1.5), 200.0, 173.0, 1.45, True),
    (dict(foundation_width_m=2, foundation_length_m=2, column_load_kn=1200,
          soil_bearing_capacity_kn_m2=250), 300.0, 300.0, 0.83, False),
    (dict(foundation_width_m=2, foundation_length_m=2, column_load_kn=1000,
          soil_bearing_capacity_kn_m2=250), 250.0, 250.0, 1.0, True),
])
def test_foundation_bearing_pressure_is_load_over_area(kw, q, net, fos, ok):
    r = foundation_bearing_pressure(**kw)
    _close(r["bearing_pressure_kn_m2"], q)
    _close(r["net_pressure_kn_m2"], net)
    _close(r["fos_against_bearing"], fos)
    assert r["bearing_ok"] is ok


# CPI = EV/AC ; SPI = EV/PV ; CV = EV-AC ; SV = EV-PV ; EAC = BAC/CPI ; ETC = EAC-AC ; VAC = BAC-EAC
@pytest.mark.parametrize("kw,cpi,spi,cv,sv,eac,etc,vac", [
    (dict(pv=100000, ev=90000, ac=95000, bac=200000),
     0.947, 0.9, -5000.0, -10000.0, 211111.11, 116111.11, -11111.11),
    (dict(bcws=50000, bcwp=60000, acwp=48000, bac=120000),
     1.25, 1.2, 12000.0, 10000.0, 96000.0, 48000.0, 24000.0),
])
def test_calculate_evm_is_the_stated_ratios(kw, cpi, spi, cv, sv, eac, etc, vac):
    r = calculate_evm(**kw)
    for key, want in (("CPI", cpi), ("SPI", spi), ("CV", cv), ("SV", sv),
                      ("EAC", eac), ("ETC", etc), ("VAC", vac)):
        _close(r[key], want)


# SPI = actual_cum/planned_cum ; % = cum/total*100 ; variance = actual% - planned%
@pytest.mark.parametrize("kw,spi,pp,ap,var", [
    (dict(planned_cum=400, actual_cum=340, planned_total=1000), 0.85, 40.0, 34.0, -6.0),
    (dict(planned_cum=250, actual_cum=300, planned_total=500), 1.2, 50.0, 60.0, 10.0),
    (dict(planned_cum=250, actual_cum=300), 1.2, None, None, None),
])
def test_progress_kpi_is_actual_over_planned(kw, spi, pp, ap, var):
    r = progress_kpi(**kw)
    _close(r["spi_progress"], spi)
    for key, want in (("planned_pct", pp), ("actual_pct", ap), ("variance_pct", var)):
        if want is None:
            assert r[key] is None
        else:
            _close(r[key], want)


# TF = LS - ES = LF - EF ; critical when TF <= 0
@pytest.mark.parametrize("kw,tf,crit", [
    (dict(early_start=10, early_finish=15, late_start=14, late_finish=19), 4.0, False),
    (dict(early_start=5, early_finish=12, late_start=5, late_finish=12), 0.0, True),
    (dict(early_start=20, early_finish=30, late_start=12, late_finish=22), -8.0, True),
])
def test_critical_path_float_is_late_minus_early(kw, tf, crit):
    r = critical_path_float(**kw)
    _close(r["total_float"], tf)
    _close(r["consistency_check_lf_minus_ef"], tf)
    assert r["is_critical"] is crit


# daily = flat amount, or rate% * contract amount ; total = daily * days
@pytest.mark.parametrize("kw,daily,total", [
    (dict(rate_percent=0.1, contract_amount=5_000_000, days=30), 5000.0, 150000.0),
    (dict(amount_per_day=2500, days=12), 2500.0, 30000.0),
    (dict(rate_percent=0.05, contract_amount=12_000_000), 6000.0, None),
])
def test_delay_damages_daily_is_rate_times_amount_times_days(kw, daily, total):
    r = delay_damages_daily(**kw)
    _close(r["daily_amount"], daily)
    if total is None:
        assert "total_amount" not in r
    else:
        _close(r["total_amount"], total)


# The same figures through the public route every caller uses.
@pytest.mark.parametrize("name,params,key,want", [
    ("beam_moment_simple", {"udl_w_kn_m": 25, "span_m": 8}, "max_moment_kn_m", 200.0),
    ("progress_kpi", {"planned_cum": 400, "actual_cum": 340, "planned_total": 1000},
     "spi_progress", 0.85),
    ("delay_damages_daily", {"rate_percent": 0.1, "contract_amount": 5_000_000, "days": 30},
     "total_amount", 150000.0),
])
def test_public_route_returns_the_hand_value(name, params, key, want):
    from app.lib.construction_formulas import run_calculation
    env = run_calculation(name, params)
    assert env["status"] == "success"
    _close(env["result"][key], want)


# Degenerate inputs are refused, never a confident wrong figure.
@pytest.mark.parametrize("fn,kw", [
    (beam_moment_simple, dict(udl_w_kn_m=10, span_m=0)),
    (beam_shear_simple, dict(udl_w_kn_m=-1, span_m=5)),
    (foundation_bearing_pressure, dict(foundation_width_m=0, foundation_length_m=2,
                                       column_load_kn=100, soil_bearing_capacity_kn_m2=200)),
    (calculate_evm, dict(pv=100, ev=90)),
    (progress_kpi, dict(planned_cum=10, actual_cum=5, planned_total=0)),
    (critical_path_float, dict(early_start=10, early_finish=5, late_start=10, late_finish=12)),
    (delay_damages_daily, dict(rate_percent=0.1)),
])
def test_degenerate_input_is_refused(fn, kw):
    assert "error" in fn(**kw)


def test_beam_deflection_refuses_a_zero_modulus_and_an_m4_inertia():
    with pytest.raises(ValueError):
        beam_deflection_ss_udl(w_kn_m=10, span_m=5, ec_mpa=0, i_mm4=1e8)
    with pytest.raises(ValueError):
        beam_deflection_ss_udl(w_kn_m=10, span_m=5, ec_mpa=200000, i_mm4=0.0001)
