"""A4 crew sizing / productivity (SPEC_09NOV_PROJECT_CONTROL_FORMULAS_AND_RATES).

Generic synthetic inputs; no project values.
"""
from __future__ import annotations

import pytest

from app.agents.hats.planning.formulas.project_control import crew_sizing
from app.lib import construction_formulas as cf


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


@pytest.mark.parametrize("qty,days,rate,want_crews,want_unit", [
    # thousands separator in the quantity, "1,200 m3" at 8 m3/crew-day over 30 days
    ("1,200 m3", 30, 8.0, 5.0, "m3"),
    # decimal quantity, "350.5 m2" at 20 m2/crew-day over 10 days
    ("350.5 m2", 10, 20.0, 1.7525, "m2"),
])
def test_crew_sizing_accepts_a_unit_bearing_quantity(qty, days, rate, want_crews, want_unit):
    out = cf.run_calculation("crew_sizing", {
        "quantity": qty, "duration_days": days, "rate_per_crew_day": rate,
    })
    assert out["status"] == "success", out
    assert out["result"]["crews_required"] == pytest.approx(want_crews, abs=0.001)
    # the stated unit is carried through, not dropped
    conv = {c["parameter"]: c for c in out.get("conversions", [])}
    assert conv["quantity"]["from_unit"] == want_unit, out


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


@pytest.mark.parametrize("item,want_rate", [
    ("Blockwork (walls)", 12.0),
    ("Internal-wall plaster", 20.0),
    ("Ceramic tiles, floors", 40.0),
    ("Emulsion paint, walls (final coat)", 45.0),
])
def test_benchmark_display_name_reaches_its_row(item, want_rate):
    """A spec display name (parentheses, comma, hyphen, mixed case) reaches
    the same row as its stored key: case, punctuation, hyphens, parentheses
    and whitespace do not distinguish two names."""
    out = productivity_benchmark(item)
    assert out["rate_per_crew_day"] == want_rate


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


# Negative controls: each formula's guard must refuse invalid inputs (kills the
# drop_rule mutants that remove the first `if`).

@pytest.mark.parametrize("qty,days,rate,hours", [
    (0.0, 15.0, 12.0, 10.0),
    (3000.0, 0.0, 12.0, 10.0),
])
def test_crew_sizing_refuses_non_positive(qty, days, rate, hours):
    assert "error" in crew_sizing(qty, days, rate, hours)


@pytest.mark.parametrize("planned,actual,total", [
    (50.0, 50.0, 0.0),
    (50.0, 50.0, -1.0),
])
def test_progress_kpi_refuses_non_positive_total(planned, actual, total):
    assert "error" in progress_kpi(planned, actual, total)


@pytest.mark.parametrize("value,cost", [
    ([], []),
    ([100.0], []),
])
def test_cash_flow_refuses_bad_series(value, cost):
    assert "error" in cash_flow(value, cost)


@pytest.mark.parametrize("planned,actual", [
    ([0.0], [0.0]),
    ([], []),
])
def test_manhours_scurve_refuses_zero_total(planned, actual):
    assert "error" in manhours_scurve(planned, actual)


@pytest.mark.parametrize("comp,tot,wt", [
    ([1.0], [1.0, 2.0], [1.0, 1.0]),
    ([1.0], [0.0], [1.0]),
])
def test_floor_cycle_refuses_bad_input(comp, tot, wt):
    assert "error" in floor_cycle_progress(comp, tot, wt)


# ── 1b(a): a rate declared count-per-time carries its unit-per-unit ─────────

@pytest.mark.parametrize("rate", [
    "12 m2/crew-day",
    "12 m3/day",
])
def test_a_rate_with_a_unit_per_unit_is_accepted(rate):
    out = cf.run_calculation("crew_sizing", {
        "quantity": "3,000 m2", "duration_days": 15, "rate_per_crew_day": rate,
    })
    assert out["status"] == "success", out
    assert out["result"]["crews_required"] == pytest.approx(16.667, abs=0.001)


# ── 1b(c): SPI needs no planned_total ───────────────────────────────────────

@pytest.mark.parametrize("planned,actual,want", [
    (60, 45, 0.75),
    (50, 50, 1.0),
])
def test_spi_without_a_planned_total(planned, actual, want):
    out = cf.run_calculation("progress_kpi", {"planned_cum": planned, "actual_cum": actual})
    assert out["status"] == "success", out
    assert out["result"]["spi_progress"] == pytest.approx(want)
    assert out["result"]["planned_pct"] is None


# ── 1b(b): benchmark provenance carries tier / confidence ───────────────────

@pytest.mark.parametrize("item,want_rate", [
    ("blockwork walls", 12.0),
    ("internal wall plaster", 20.0),
])
def test_benchmark_provenance_is_unverified_general(item, want_rate):
    out = cf.run_calculation("productivity_benchmark", {"work_item": item})
    assert out["status"] == "success", out
    res = out["result"]
    assert res["rate_per_crew_day"] == pytest.approx(want_rate)
    assert "unverified" in res["provenance"]
    assert "owner" in res["provenance"] and "template" in res["provenance"]
    assert res["tier"] == "GENERAL"
    assert res["confidence"] == "MEDIUM"


# ── 1b residual (a): a working-days duration needs no hours per day ────────

@pytest.mark.parametrize("qty,days,rate,want_crews", [
    ("2,400 m2", "20 working days", "10 m2/crew-day", 12.0),
    ("3,150 m3", "30 workdays", "9 m3/crew-day", 11.667),
])
def test_a_working_days_duration_needs_no_hours_per_day(qty, days, rate, want_crews):
    """crews_required = qty / (days * rate) uses no hours-per-day, so a
    duration written in working days runs straight through, never asking for it."""
    out = cf.run_calculation("crew_sizing", {
        "quantity": qty, "duration_days": days, "rate_per_crew_day": rate,
    })
    assert out["status"] == "success", out
    assert out["result"]["crews_required"] == pytest.approx(want_crews, abs=0.001)
    # the run is not an ask for hours-per-day: it succeeds and reports no
    # question (the unused hours_per_day default is disclosed, not requested)
    assert "question" not in out and "rejected" not in out, out


# ── 1b residual (b): an hours-per-day input takes a time-per-time ratio ────

@pytest.mark.parametrize("hours,want_mh", [
    ("8 hours per day", 1920.0),
    ("10 working hours per day", 2400.0),
    ("9 h/day", 2160.0),
    ("11 h/d", 2640.0),
])
def test_an_hours_per_day_input_takes_a_time_per_time_ratio(hours, want_mh):
    """A per-day rate is the declared dimension of hours_per_day (h/d); any
    time-per-time spelling is that same quantity, never "does not convert"."""
    out = cf.run_calculation("crew_sizing", {
        "quantity": "2,400 m2", "duration_days": 20, "rate_per_crew_day": "10 m2/crew-day",
        "hours_per_day": hours,
    })
    assert out["status"] == "success", out
    assert out["result"]["man_hours"] == pytest.approx(want_mh)

