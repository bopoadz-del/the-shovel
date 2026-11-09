"""A4 crew sizing / productivity (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES).

Generic synthetic inputs; no project values.
"""
from __future__ import annotations

import pytest

from app.agents.hats.planning.formulas.project_control import crew_sizing


@pytest.mark.parametrize("qty,days,rate,hours,want_crews,up,crew_days,mh", [
    # blockwork: 3,000 m2 in 15 days at 12 m2/crew-day, 10-hour day
    (3000.0, 15.0, 12.0, 10.0, 16.667, 17, 250.0, 2500.0),
    # plaster: 350 m2 in 10 days at 20 m2/crew-day, 8-hour day
    (350.0, 10.0, 20.0, 8.0, 1.75, 2, 17.5, 140.0),
])
def test_crew_sizing_agrees(qty, days, rate, hours, want_crews, up, crew_days, mh):
    out = crew_sizing(qty, days, rate, hours)
    assert out["crews_required"] == pytest.approx(want_crews, abs=0.001)
    assert out["crews_required_rounded_up"] == up
    assert out["crew_days"] == pytest.approx(crew_days)
    assert out["man_hours"] == pytest.approx(mh)


@pytest.mark.parametrize("qty,rate,crew,days", [
    (3000.0, 12.0, 17.0, 14.71),
    (350.0, 20.0, 2.0, 8.75),
])
def test_crew_sizing_duration_for_crews(qty, rate, crew, days):
    out = crew_sizing(qty, 15.0, rate, 10.0, crew_size=crew)
    assert out["duration_for_crews"] == pytest.approx(days, abs=0.01)


from app.agents.hats.planning.formulas.project_control import progress_kpi


@pytest.mark.parametrize("planned,actual,total,want_ppct,want_apct,want_var,want_spi", [
    # on schedule: 50 planned, 50 actual, 100 total -> 50/50/0/1.0
    (50.0, 50.0, 100.0, 50.0, 50.0, 0.0, 1.0),
    # behind: 60 planned, 45 actual, 100 total -> 60/45/-15/0.75
    (60.0, 45.0, 100.0, 60.0, 45.0, -15.0, 0.75),
])
def test_progress_kpi_agrees(planned, actual, total, want_ppct, want_apct, want_var, want_spi):
    out = progress_kpi(planned, actual, total)
    assert out["planned_pct"] == pytest.approx(want_ppct)
    assert out["actual_pct"] == pytest.approx(want_apct)
    assert out["variance_pct"] == pytest.approx(want_var)
    assert out["spi_progress"] == pytest.approx(want_spi)


from app.agents.hats.planning.formulas.productivity_benchmarks import productivity_benchmark


@pytest.mark.parametrize("item,want_unit,want_rate", [
    ("blockwork walls", "m2", 12.0),
    ("internal wall plaster", "m2", 20.0),
])
def test_productivity_benchmark_lookup(item, want_unit, want_rate):
    out = productivity_benchmark(item)
    assert out["unit"] == want_unit
    assert out["rate_per_crew_day"] == want_rate
    assert "unverified" in out["provenance"]


def test_productivity_benchmark_unknown_item_is_not_invented():
    out = productivity_benchmark("widget extrusion")
    assert out["rate_per_crew_day"] is None
    assert "no benchmark entry" in out["flag"]


from app.agents.hats.planning.formulas.project_control import cash_flow


@pytest.mark.parametrize("value,cost,adv,ret,lag,want_cv,want_adv,want_ret,want_peak", [
    # 2 months, 10% advance, 5% retention, 15-day lag (1 month)
    ([100.0, 100.0], [90.0, 90.0], 0.10, 0.05, 15.0, 200.0, 20.0, 10.0, 75.0),
    # 1 month, no advance/retention, no lag
    ([1000.0], [800.0], 0.0, 0.0, 0.0, 1000.0, 0.0, 0.0, 0.0),
])
def test_cash_flow_agrees(value, cost, adv, ret, lag, want_cv, want_adv, want_ret, want_peak):
    out = cash_flow(value, cost, adv, ret, lag)
    assert out["contract_value"] == pytest.approx(want_cv)
    assert out["advance_payment"] == pytest.approx(want_adv)
    assert out["retention_total"] == pytest.approx(want_ret)
    assert out["peak_funding_requirement"] == pytest.approx(want_peak)
    # checks: recovery sums to -advance, deduction sums to -retention
    assert out["check_advance_recovery"] == pytest.approx(-want_adv)
    assert out["check_retention"] == pytest.approx(-want_ret)


from app.agents.hats.planning.formulas.project_control import manhours_scurve, floor_cycle_progress


@pytest.mark.parametrize("planned,actual,want_cp,want_ca,want_pi", [
    ([100.0, 100.0], [80.0, 80.0], 100.0, 80.0, 1.25),
    ([50.0, 50.0, 50.0], [50.0, 40.0], 66.7, 60.0, 1.111),
])
def test_manhours_scurve_agrees(planned, actual, want_cp, want_ca, want_pi):
    out = manhours_scurve(planned, actual)
    assert out["cum_planned_pct"] == pytest.approx(want_cp, abs=0.1)
    assert out["cum_actual_pct"] == pytest.approx(want_ca, abs=0.1)
    assert out["productivity_index"] == pytest.approx(want_pi, abs=0.001)


@pytest.mark.parametrize("comp,tot,wt,want", [
    ([6.0, 4.0], [8.0, 8.0], [1.0, 1.0], 0.625),
    ([9.0, 1.0], [10.0, 2.0], [2.0, 1.0], 0.767),
])
def test_floor_cycle_progress_agrees(comp, tot, wt, want):
    out = floor_cycle_progress(comp, tot, wt)
    assert out["weighted_progress"] == pytest.approx(want, abs=0.001)
