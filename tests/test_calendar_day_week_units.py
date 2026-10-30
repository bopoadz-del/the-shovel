"""'calendar day(s)' and 'calendar week(s)' are time units, singular and plural.

A daily delay-damages amount is quoted "per calendar day" (the calculator's
own note says so); the converter did not know the phrase and refused it.
"""
from __future__ import annotations

import json

import pytest

from app.agents.base.formulas.construction_formulas_planning import parse_unit


def _chat(ask, params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc, user_message=ask))


@pytest.mark.parametrize("amount,days,total", [
    ("5,000 SAR per calendar day", 60, 300000.0),
    ("5,000 SAR per calendar days", 60, 300000.0),
    ("7,000 SAR per calendar week", 60, 60000.0),
    ("7,000 SAR per calendar weeks", 60, 60000.0),
])
def test_a_calendar_rate_is_totalled(amount, days, total):
    env = _chat(f"Delay damages at {amount} for {days} days",
                {"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == pytest.approx(total)


@pytest.mark.parametrize("unit,hours", [
    ("calendar day", 24.0),
    ("calendar days", 24.0),
    ("calendar week", 168.0),
    ("calendar weeks", 168.0),
])
def test_each_calendar_alias_is_a_time_unit(unit, hours):
    u = parse_unit(unit)
    assert (u.dimension, u.scale) == ("time", hours)
