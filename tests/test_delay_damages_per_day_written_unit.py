"""Live b679cb6 (G3, rids badb4a80-d76 and 54dc3983-074): the asks
'Delay damages at a flat 5,000 per day for 60 days' and '45 days at 1,200
per day' reached construction_calc, which answered "I do not recognise the
unit" and asked for the daily amount again. The model passes the figure as
the user wrote it, '5,000 per day': a currency-per-time input whose currency
is implied. These drive the real tool path (_run_tool_call -> tools.py)."""
from __future__ import annotations

import json

import pytest


def _call(params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc))


@pytest.mark.parametrize("amount,days,total", [
    ("5,000 per day", 60, 300000.0),
    ("1,200 per day", "45 days", 54000.0),
    ("5,000 /day", 60, 300000.0),
    ("SAR 1,200 per day", 45, 54000.0),
    ("5000 SAR/day", 60, 300000.0),
    ("SAR 5,000/day", 60, 300000.0),
])
def test_a_daily_amount_written_per_day_is_totalled(amount, days, total):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["total_amount"] == total, env
    assert res["currency"] == "SAR"


@pytest.mark.parametrize("amount,days,total", [
    ("7,000 per week", 60, 60000.0),
    ("1,400 per week", 10, 2000.0),
])
def test_a_per_week_amount_converts_to_per_day(amount, days, total):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == total, env


@pytest.mark.parametrize("amount", ["5,000 per m3", "1,200 per kg"])
def test_a_per_amount_of_another_kind_is_still_refused(amount):
    env = _call({"amount_per_day": amount, "days": 60})
    assert "total_amount" not in json.dumps(env.get("result") or {}), env


def _chat(ask, params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc, user_message=ask))


@pytest.mark.parametrize("ask,params,total,shown", [
    ("Delay damages at a flat 5,000 per day for 60 days",
     {"amount_per_day": "5,000 per day", "days": 60}, 300000.0, "SAR 300,000.00"),
    ("45 days at 1,200 per day",
     {"amount_per_day": "1,200 per day", "days": "45 days"}, 54000.0, "SAR 54,000.00"),
])
def test_the_live_asks_give_an_amount_and_a_currency(ask, params, total, shown):
    env = _chat(ask, params)
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["total_amount"] == total
    assert shown in res["note"]
    assert "rate_percent" not in {k for k, v in res.items() if v}
