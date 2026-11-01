"""Delay damages at a flat daily amount, through the live tool path.

Live c4f1605 (req 17860845-15f): an ask for a flat daily amount over a
number of days could only reach ``rate_percent``, so the amount was read
as a percentage and refused as out of range. A flat daily amount and the
days of delay are now inputs of their own; the percent-of-contract path
is unchanged.
"""
from __future__ import annotations

import json

import pytest

from app.agents.hats.contracts.formulas.construction_formulas_commercial import delay_damages_daily


def _call(params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc))


@pytest.mark.parametrize("amount,days,total", [(5000, 60, 300_000.0), (1200, 10, 12_000.0)])
def test_flat_daily_amount_over_days_through_the_tool(amount, days, total):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    # No currency in the ask -> currency is None and the figure stands on its own
    # (K-T5 ruling ii: a label, never an invented default; no leading space).
    assert res["total_amount"] == total
    assert res["daily_amount"] == amount
    assert res["days"] == days
    assert res["currency"] is None
    assert res["note"] == f"{amount:,.2f} per calendar day. Over {days} days: {total:,.2f}."


@pytest.mark.parametrize("amount", [750, 4000])
def test_flat_daily_amount_alone_needs_no_contract_amount(amount):
    out = delay_damages_daily(amount_per_day=amount)
    assert out["daily_amount"] == amount
    assert "total_amount" not in out and "rate_percent" not in out
    assert out.get("currency") is None
    assert out["note"] == f"{amount:,.2f} per calendar day."


@pytest.mark.parametrize("cur", ["EUR", " aed "])
def test_flat_daily_amount_keeps_a_given_currency(cur):
    out = delay_damages_daily(amount_per_day=100, days=3, currency=cur)
    code = cur.strip().upper()
    assert out["currency"] == code
    assert out["note"] == f"{code} 100.00 per calendar day. Over 3 days: {code} 300.00."


@pytest.mark.parametrize("amount,days", [(-1, 10), (500, -2)])
def test_negative_flat_amount_or_days_is_refused(amount, days):
    assert delay_damages_daily(amount_per_day=amount, days=days) == {"error": "amount_per_day and days must be >= 0."}


# The percent-of-contract path, through the same tool path.
@pytest.mark.parametrize("pct,base,days,daily,total", [
    (0.1, 50_000_000, 10, 50_000.0, 500_000.0),
    (0.2, 1_000_000, 30, 2_000.0, 60_000.0),
])
def test_percent_of_contract_path_through_the_tool(pct, base, days, daily, total):
    env = _call({"rate_percent": pct, "contract_amount": base, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["daily_amount"] == daily and res["total_amount"] == total
    assert res["currency"] is None
    assert res["note"].startswith(f"{pct:g}% of {base:,.2f} = {daily:,.2f} per calendar day.")


@pytest.mark.parametrize("pct,base", [(0.1, 1_000_000), (0.5, 2_000_000)])
def test_percent_path_without_days_has_no_total(pct, base):
    out = delay_damages_daily(rate_percent=pct, contract_amount=base)
    assert out["daily_amount"] == round(base * pct / 100, 2)
    assert "total_amount" not in out and "days" not in out


@pytest.mark.parametrize("pct", [0.1, 0.25])
def test_percent_without_contract_amount_still_asks_for_it(pct):
    env = _call({"rate_percent": pct})
    assert env["ok"] is False, env
    assert "missing required contract_amount" in json.dumps(env)


# With days of delay the answer leads with the total; without, the daily amount.
@pytest.mark.parametrize("params,key,value", [
    ({"amount_per_day": 5000, "days": 60}, "total_amount", 300_000.0),
    ({"amount_per_day": 5000}, "daily_amount", 5_000.0),
])
def test_the_headline_is_the_total_when_days_are_given(params, key, value):
    from app.lib.construction_formulas import calculation_headline

    assert calculation_headline("delay_damages_daily", delay_damages_daily(**params)) == (key, value)


# The registry declares both new inputs with their own units ...
@pytest.mark.parametrize("name,unit", [("days", "day"), ("amount_per_day", "currency/day")])
def test_new_inputs_are_declared_with_their_units(name, unit):
    from app.lib.formula_registry import get

    assert get("delay_damages_daily").inputs[name] == unit


# ... and the declared range of days is enforced on the tool path.
@pytest.mark.parametrize("days,ok", [(30, True), (2e9, False)])
def test_days_beyond_the_declared_range_are_refused(days, ok):
    env = _call({"amount_per_day": 100, "days": days})
    assert env["ok"] is ok, env
    if not ok:
        assert "days" in json.dumps(env)


@pytest.mark.parametrize("amount,daily", [(1234.567, 1234.57), (99.994, 99.99)])
def test_a_fractional_daily_amount_is_rounded_to_cents(amount, daily):
    out = delay_damages_daily(amount_per_day=amount)
    assert out["daily_amount"] == daily
    assert out["per"] == "calendar day"


@pytest.mark.parametrize("amount,days,total", [(33.33, 3, 99.99), (1000.5, 1.5, 1500.75)])
def test_a_fractional_total_is_rounded_to_cents(amount, days, total):
    out = delay_damages_daily(amount_per_day=amount, days=days)
    assert out["total_amount"] == total
    assert out["per"] == "calendar day"


# A flat daily amount under 1 is a real amount: accepted and totalled.
@pytest.mark.parametrize("amount,days,total", [(0.5, 2, 1.0), (0.25, 4, 1.0)])
def test_a_daily_amount_under_one_is_accepted_through_the_tool(amount, days, total):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["daily_amount"] == amount
    assert res["total_amount"] == total
