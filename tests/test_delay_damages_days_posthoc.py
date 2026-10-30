"""Post-hoc kills for #35 (G2 report on 0b978eb): one day and a fraction of
a day still give a total, and the total is rounded to cents."""
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
    (250, 1, 250.0),
    (10.01, 0.5, 5.0),
    (33.33, 1.5, 49.99),
    (12.34, 1.5, 18.51),
])
def test_short_and_fractional_delays_are_totalled_to_cents(amount, days, total):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["days"] == days
    assert res["total_amount"] == total
