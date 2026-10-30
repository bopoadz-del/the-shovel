"""Pins left open after #39 (G2 post-hoc review of b679cb6).

- A figure the extractor binds counts once as placed (B17: ``+ 1``).
- An integer figure the extractor binds is a figure too (B14).
- The days of delay stop at 1e9, inclusive (A20 boundary).
"""
from __future__ import annotations

import json

import pytest

import app.lib.construction_formulas as cf


@pytest.mark.parametrize("ask,figure,total", [
    ("Delay damages, amount per day 45, delay days 45", 45.0, 2_025.0),
    ("Delay damages, amount per day 30, delay days 30", 30.0, 900.0),
])
def test_one_figure_written_twice_fills_two_inputs(ask, figure, total):
    env = cf.run_calculation("delay_damages_daily", {"text": ask})
    assert env.get("status") == "success", env
    res = env["result"]
    assert (res["daily_amount"], res["days"], res["total_amount"]) == (figure, figure, total)
    assert f"SAR {total:,.2f}" in res["note"]


@pytest.mark.parametrize("given,extracted,text", [
    ({"a": 5}, {"b": 5}, "a 5"),            # an int figure, written once and placed: spent
    ({"a": 45}, {"b": 45}, "a 45"),
])
def test_an_integer_figure_already_placed_is_spent(given, extracted, text):
    assert cf._unspent_ask_figures(given, extracted, text) == {}


def _tool_days(days):
    from tests.test_construction_calc_tool import _agent, _run

    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": {"amount_per_day": 2, "days": days}})}}
    return _run(_agent(["construction"])._run_tool_call(tc))


@pytest.mark.parametrize("days", [1e9 + 1, 2e9])
def test_days_above_the_bound_are_refused_through_the_tool(days):
    env = _tool_days(days)
    assert env["ok"] is False, env
    assert "1,000,000,000 day" in env["result"]["error"]


@pytest.mark.parametrize("days,total", [(1e9, 2e9), (1e9 - 1, 2e9 - 2)])
def test_days_up_to_the_bound_are_accepted_through_the_tool(days, total):
    env = _tool_days(days)
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == total
