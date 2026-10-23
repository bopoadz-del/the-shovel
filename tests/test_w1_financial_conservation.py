"""W1 financial — conservation laws over the cash-flow forecast.

New test shape (2026-07-26): the forecast is deterministic, so we assert the
ACCOUNTING IDENTITIES that must hold for any inputs — money can't appear or
vanish between the components — plus linearity and single-release invariants.
Golden values can silently rot; conservation laws can't.
"""
import asyncio

import pytest


def _forecast(contract_value=60_000_000, months=18):
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    return asyncio.run(
        c.cash_flow_forecast(
            {"contract_value": contract_value},
            {"duration_months": months},
        )
    )


def test_revenue_conservation_and_s_curve_shape():
    r = _forecast()
    assert r["status"] == "success", r
    curve = r["s_curve_data"]
    assert len(curve) == 18

    # Revenue conservation: sum of monthly values == final cumulative value.
    total = sum(m["monthly_value"] for m in curve)
    assert total == pytest.approx(curve[-1]["cumulative_value"], abs=1.0)
    assert r["summary_metrics"]["total_planned_revenue"] == pytest.approx(total, abs=1.0)

    # S-curve: cumulative strictly non-decreasing, progress capped at
    # substantial completion (95%) — never a fabricated 100%.
    cums = [m["cumulative_value"] for m in curve]
    assert cums == sorted(cums)
    assert curve[-1]["planned_progress_percent"] == pytest.approx(95.0, abs=0.01)


def test_cash_conservation_identity():
    # Everything that flows in/out must reconcile:
    # sum(net_cash_in) == revenue - retention deducted + retention released
    #                     + advance paid - advance recovered
    r = _forecast()
    curve = r["s_curve_data"]
    terms = r["project_parameters"]["payment_terms"]
    advance_paid = r["project_parameters"]["contract_value"] * terms["advance_payment"]

    net = sum(m["net_cash_in"] for m in curve)
    revenue = sum(m["monthly_value"] for m in curve)
    deducted = sum(m["retention_deduction"] for m in curve)
    released = sum(m["retention_release"] for m in curve)
    recovered = r["summary_metrics"]["advance_recovered_total"]

    assert net == pytest.approx(revenue - deducted + released + advance_paid - recovered, abs=2.0)
    # The advance must be FULLY recovered by project end (recovery window
    # covers 80% of the duration by design).
    assert recovered == pytest.approx(advance_paid, abs=1.0)


def test_retention_released_exactly_once():
    r = _forecast()
    curve = r["s_curve_data"]
    release_months = [m["month"] for m in curve if m["retention_release"] > 0]
    assert len(release_months) == 1, release_months
    # The single release equals everything deducted up to that month.
    rm = release_months[0]
    deducted_to_release = sum(
        m["retention_deduction"] for m in curve if m["month"] <= rm
    )
    released = r["summary_metrics"]["retention_released_total"]
    assert released == pytest.approx(deducted_to_release, abs=2.0)


def test_linearity_in_contract_value():
    # Double the contract, every monetary series doubles exactly.
    a = _forecast(contract_value=30_000_000)
    b = _forecast(contract_value=60_000_000)
    for ma, mb in zip(a["s_curve_data"], b["s_curve_data"]):
        assert mb["monthly_value"] == pytest.approx(2 * ma["monthly_value"], abs=1.0)
        assert mb["net_cash_in"] == pytest.approx(2 * ma["net_cash_in"], abs=2.0)
    assert b["summary_metrics"]["total_planned_revenue"] == pytest.approx(
        2 * a["summary_metrics"]["total_planned_revenue"], abs=2.0
    )


def test_duration_change_conserves_total_revenue():
    short = _forecast(months=12)
    long = _forecast(months=24)
    assert short["summary_metrics"]["total_planned_revenue"] == pytest.approx(
        long["summary_metrics"]["total_planned_revenue"], rel=1e-6
    )


def test_no_fabricated_ev_ac_series():
    # EV/AC are measurements, not forecasts — must be nulls, never invented.
    r = _forecast()
    chart = r["chart_data"]
    assert set(chart["earned_value"]) == {None}
    assert set(chart["actual_cost"]) == {None}


def test_missing_contract_value_is_honest_error():
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    r = asyncio.run(c.cash_flow_forecast({}, {}))
    assert r["status"] == "error"
    assert "contract_value" in r["error"]
