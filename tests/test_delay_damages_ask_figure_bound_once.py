"""One written figure fills one calculator input, through the chat tool path.

Live 0b978eb: an ask for 60 days at a daily rate of 5,000 was answered with
a 5,000 % rate refused as out of range. The model's call can carry
``amount_per_day`` and ``days``, but the ask text rides along with the call
and the text extractor bound the same written 5,000 again, by its label
"rate", onto ``rate_percent``. A figure the call already placed is spent.
"""
from __future__ import annotations

import json

import pytest

import app.lib.construction_formulas as cf


def _chat_call(ask, params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc, user_message=ask))


@pytest.mark.parametrize("ask,params,total", [
    ("What are the delay damages for 60 days at a daily rate of 5,000?",
     {"days": 60, "amount_per_day": 5000}, 300_000.0),
    ("Delay damages for 45 days at a daily rate of 1,200",
     {"days": "45 days", "amount_per_day": "1,200"}, 54_000.0),
])
def test_daily_rate_ask_with_amount_per_day_gives_the_total(ask, params, total):
    env = _chat_call(ask, params)
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["total_amount"] == total
    assert "rate_percent" not in res


@pytest.mark.parametrize("hat", ["contracts", None])
def test_the_schema_the_model_sees_offers_amount_per_day(hat):
    from app.agents.driver.tools import calculator_schema

    fn = calculator_schema(hat)["function"]
    line = next(l for l in fn["description"].splitlines() if l.startswith("- delay_damages_daily:"))
    assert "amount_per_day = flat daily delay damages amount" in line
    assert "delay_damages_daily" in fn["parameters"]["properties"]["calculation"]["enum"]


@pytest.mark.parametrize("given,extracted,text,kept", [
    ({"a": 5}, {"b": 5.0}, "a 5 and b 5", {"b": 5.0}),        # written twice: second is free
    ({"a": 5}, {"b": 5.0}, "a 5", {}),                       # written once: spent
    ({"a": "5,000"}, {"b": 5000.0}, "rate of 5,000", {}),    # string operand spends it
    ({"a": 7}, {"b": 1524.0}, "b 5 ft", {"b": 1524.0}),      # converted figure: not a written one
    ({}, {"b": 5.0, "c": 5.0}, "b 5", {"b": 5.0}),           # extractor cannot spend it twice
    ({"b": 1}, {"b": 9.0}, "b 9", {}),                       # a given key is never overwritten
    ({"a": None}, {"b": 3.0}, "3", {"b": 3.0}),              # an empty input spends nothing
    ({"a": True}, {"b": 1.0}, "1", {"b": 1.0}),              # a flag is not a figure
    ({}, {"code": "aci"}, "ACI 318", {"code": "aci"}),       # text values pass through
    ({"a": 5, "text": "a 5"}, {"b": 5.0}, "a 5", {}),        # the ask riding in params is no figure
    ({"text": "see the spec"}, {"b": 2.0}, "b 2", {"b": 2.0}),
])
def test_ask_figures_already_spent_are_not_bound_again(given, extracted, text, kept):
    assert cf._unspent_ask_figures(given, extracted, text) == kept


@pytest.mark.parametrize("amount,days,total", [(5000, 500, 2_500_000.0), (2, 1e9, 2e9)])
def test_large_day_counts_are_accepted_through_the_tool(amount, days, total):
    from tests.test_construction_calc_tool import _agent, _run

    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": {"amount_per_day": amount, "days": days}})}}
    env = _run(_agent(["construction"])._run_tool_call(tc))
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total


@pytest.mark.parametrize("ask,amount,days,total", [
    ("60 days at 5,000 per day", 5000, 60, 300_000.0),
    ("Delay damages: 45 days at 1,200 per day", 1200, 45, 54_000.0),
])
def test_days_written_as_the_unit_of_a_figure_bind_that_figure(ask, amount, days, total):
    env = _chat_call(ask, {"amount_per_day": amount})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total
    assert res["currency"] == "SAR"


@pytest.mark.parametrize("text,bound", [
    ("60 days at 5,000 per day", {"days": 60.0}),
    ("days of delay 12", {"days": 12.0}),           # label first, figure after: unchanged
])
def test_extractor_reads_a_figure_by_the_unit_after_it(text, bound):
    got = cf.extract_calculation_params_from_text(cf.CALCULATORS["delay_damages_daily"], text)
    assert {k: v for k, v in got.items() if k == "days"} == bound


@pytest.mark.parametrize("text,key", [
    ("18 currency/day", "currency"),     # a label inside a compound unit is not the unit
    ("6 days/week", "days"),
])
def test_a_label_inside_a_compound_unit_does_not_take_the_figure(text, key):
    got = cf.extract_calculation_params_from_text(cf.CALCULATORS["delay_damages_daily"], text)
    assert got.get(key) not in (18.0, 6.0)
