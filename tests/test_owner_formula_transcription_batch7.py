"""Transcription check, batch 7: the code computes the owner's formula.

The formulas are the owner's. These tests do not certify or re-derive them;
they check that each calculator's code does what its stated form says. The
stated form is the formula's own docstring and its named constants. Every
expected value below is a hand calculation from that stated form, written as
a literal; none is obtained by running the calculator. All inputs are invented.

The well-point calculator's stage count is pinned only to its own docstring
(5 m per stage, 3 stages at most). The owner's knowledge entry
``dewatering.method_selection`` states a single stage lowers the water table
5-6 m and gives a 1-3 m spacing; where the code differs from that entry the
difference is reported to the owner, not decided by a test.
"""
from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas import diaphragm_wall_panel_volume
from app.agents.hats.commercial.formulas.construction_formulas_commercial import roi_calculator
from app.agents.hats.design.formulas.construction_formulas import (
    concrete_mix_design_sg,
    dewatering_well_point_spacing,
)
from app.agents.hats.qaqc.formulas.construction_formulas import grout_pressure_calc
from app.agents.hats.quantities.formulas.construction_formulas import crane_cost_estimate
from app.agents.hats.safety.formulas.construction_formulas import wind_load_on_formwork

TOL = 0.011


def _close(got, want, tol=TOL):
    assert got is not None
    assert abs(float(got) - float(want)) <= tol, (got, want)


def _fields(out):
    """A qaqc calculator may return its fields bare or inside an envelope."""
    if isinstance(out, dict):
        return out.get("result", out) if "duct_area_mm2" not in out else out
    return getattr(out, "result", out)


# --- diaphragm_wall_panel_volume: L x t x D x n, then x 1.10 tremie waste ------
@pytest.mark.parametrize("length,thick,depth,n,per,total,wasted", [
    (6.0, 0.8, 20.0, 3, 96.0, 288.0, 316.8),
    (2.8, 0.6, 15.0, 1, 25.2, 25.2, 27.72),
])
def test_diaphragm_wall_volume(length, thick, depth, n, per, total, wasted):
    out = diaphragm_wall_panel_volume(length, thick, depth, n)
    _close(out["volume_per_panel_m3"], per)
    _close(out["total_volume_m3"], total)
    _close(out["volume_with_waste_m3"], wasted)
    assert out["waste_factor"] == 1.10


@pytest.mark.parametrize("length,thick,depth,n", [(0, 0.8, 20, 1), (6, 0.8, 20, 0), (6, -0.8, 20, 2)])
def test_diaphragm_wall_refuses_degenerate_input(length, thick, depth, n):
    assert "error" in diaphragm_wall_panel_volume(length, thick, depth, n)


# --- dewatering_well_point_spacing: stages = ceil(drawdown / 5), max 3 --------
@pytest.mark.parametrize("drawdown,stages,capacity", [
    (5.0, 1, 5.0),
    (5.01, 2, 10.0),
    (12.0, 3, 15.0),
    (16.0, 4, 20.0),
])
def test_well_point_stages_are_five_metres_each(drawdown, stages, capacity):
    out = dewatering_well_point_spacing(5e-4, drawdown)
    assert out["stages_needed"] == stages
    _close(out["max_depth_capacity_m"], capacity)


@pytest.mark.parametrize("drawdown,flagged", [(15.0, False), (15.5, True)])
def test_well_point_flags_more_than_three_stages(drawdown, flagged):
    notes = dewatering_well_point_spacing(5e-4, drawdown)["notes"]
    assert any("3-stage limit" in n for n in notes) is flagged


@pytest.mark.parametrize("k,drawdown", [(0.0, 4.0), (-1e-4, 4.0), (1e-4, -0.5)])
def test_well_point_refuses_degenerate_input(k, drawdown):
    with pytest.raises(ValueError):
        dewatering_well_point_spacing(k, drawdown)


# --- concrete_mix_design_sg: C = 1 / (w/c + 1/SGc + f/SGf + cr/SGcr) t/m3 -----
@pytest.mark.parametrize("wc,fine,coarse,cement,water,fkg,ckg,total", [
    (0.50, 2.0, 4.0, 318, 159, 636, 1272, 2385),   # 1/3.14132 t
    (0.42, 2.0, 2.6, 398, 167, 796, 1035, 2396),   # 1/2.51014 t
])
def test_mix_design_absolute_volume(wc, fine, coarse, cement, water, fkg, ckg, total):
    out = concrete_mix_design_sg(wc, fine_agg_ratio=fine, coarse_agg_ratio=coarse)
    assert (out["cement_kg_m3"], out["water_litres_m3"]) == (cement, water)
    assert (out["fine_aggregate_kg_m3"], out["coarse_aggregate_kg_m3"]) == (fkg, ckg)
    assert out["total_weight_kg_m3"] == total


@pytest.mark.parametrize("kw", [
    dict(w_c_ratio=0.0),
    dict(w_c_ratio=0.5, cement_sg=0.0),
    dict(w_c_ratio=0.5, fine_agg_ratio=-1.0),
    dict(w_c_ratio=0.5, dune_sand_pct=5.0),
])
def test_mix_design_refuses_degenerate_input(kw):
    with pytest.raises(ValueError):
        concrete_mix_design_sg(**kw)


# --- crane_cost_estimate: (dry + op + rig) x factor x n x months; mob 0.5, demob 0.75
@pytest.mark.parametrize("kw,per,total,mob,demob,grand", [
    (dict(num_cranes=2, crane_capacity_tons=50, duration_months=6, dry_hire_rate_per_month=30000,
          operator_rate_per_month=8000, riggers_rate_per_month=6000, remote_area_factor=1.1),
     48400, 580800, 30000, 22500, 633300),
    (dict(num_cranes=1, crane_capacity_tons=25, duration_months=3, dry_hire_rate_per_month=20000,
          include_operator=False, include_riggers=False),
     20000, 60000, 10000, 7500, 77500),
])
def test_crane_cost(kw, per, total, mob, demob, grand):
    out = crane_cost_estimate(**kw)
    assert (out["per_crane_monthly"], out["total_project_cost"]) == (per, total)
    assert (out["mobilization"], out["demobilization"], out["grand_total"]) == (mob, demob, grand)


@pytest.mark.parametrize("kw,missing", [
    (dict(operator_rate_per_month=None, riggers_rate_per_month=5000), ["operator_rate_per_month"]),
    (dict(operator_rate_per_month="", riggers_rate_per_month=""),
     ["operator_rate_per_month", "riggers_rate_per_month"]),
])
def test_crane_cost_asks_for_a_missing_crew_rate(kw, missing):
    out = crane_cost_estimate(1, 40, 2, 15000, **kw)
    assert out["needs_input"] == missing


@pytest.mark.parametrize("n,cap,months,dry,raf", [(-1, 40, 2, 1, 1), (1, 0, 2, 1, 1), (1, 40, 2, -1, 1), (1, 40, 2, 1, -0.1)])
def test_crane_cost_refuses_degenerate_input(n, cap, months, dry, raf):
    with pytest.raises(ValueError):
        crane_cost_estimate(n, cap, months, dry, include_operator=False, include_riggers=False,
                            remote_area_factor=raf)


# --- roi_calculator: (gain - cost) / cost x 100 --------------------------------
@pytest.mark.parametrize("gain,cost,net,roi", [(150, 100, 50, 50.0), (80, 100, -20, -20.0), (130, 40, 90, 225.0)])
def test_roi(gain, cost, net, roi):
    out = roi_calculator(gain, cost)
    _close(out["net_profit"], net)
    _close(out["roi_percent"], roi)


@pytest.mark.parametrize("gain", [0, 100])
def test_roi_refuses_zero_cost(gain):
    assert "error" in roi_calculator(gain, 0)


# --- grout_pressure_calc: duct area pi d^2/4; pressure in the converter's units
@pytest.mark.parametrize("duct,area,litres_per_m", [(50, 1963.50, 1.963), (80, 5026.55, 5.027)])
def test_grout_duct_area_and_volume(duct, area, litres_per_m):
    out = _fields(grout_pressure_calc(duct))
    _close(out["duct_area_mm2"], area)
    _close(out["grout_volume_l_m"], litres_per_m, tol=0.0011)


@pytest.mark.parametrize("mpa,kgf,psi", [(0.5, 5.1, 72.5), (1.0, 10.2, 145.0)])
def test_grout_pressure_units(mpa, kgf, psi):
    # 1 N/mm2 = 10.197 kgf/cm2 = 145.04 psi
    out = _fields(grout_pressure_calc(60, required_pressure_n_mm2=mpa))
    _close(out["pressure_kg_cm2"], kgf)
    _close(out["pressure_psi"], psi)


@pytest.mark.parametrize("duct", [0, -40])
def test_grout_refuses_degenerate_duct(duct):
    out = grout_pressure_calc(duct)
    assert "error" in (out if isinstance(out, dict) else _fields(out))


# --- wind_load_on_formwork: q = 0.613 V^2, F = q A s, M = F h/2, z = b h^2/6 ----
@pytest.mark.parametrize("kw,q,f,m,stress", [
    (dict(wind_velocity_m_s=20, formwork_area_m2=10, formwork_height_m=4, formwork_width_m=2.5),
     0.245, 2.452, 4.904, 735.6),
    (dict(wind_velocity_m_s=30, formwork_height_m=6, formwork_width_m=3, shape_factor=1.2),
     0.552, 11.917, 35.750, 1986.12),
])
def test_wind_on_formwork(kw, q, f, m, stress):
    out = wind_load_on_formwork(**kw)
    _close(out["wind_pressure_kpa"], q, tol=0.0011)
    _close(out["wind_force_kn"], f, tol=0.0011)
    _close(out["overturning_moment_kn_m"], m, tol=0.0011)
    _close(out["bending_stress_n_m2"], stress)


@pytest.mark.parametrize("kw", [
    dict(wind_velocity_m_s=20),
    dict(wind_velocity_m_s=20, formwork_area_m2=0, formwork_height_m=4),
])
def test_wind_on_formwork_asks_for_an_area(kw):
    assert "error" in wind_load_on_formwork(**kw)

