"""A numeric input the caller named but sent as null or blank ('days': None,
'days': '') is asked for, never defaulted. An unreadable one ('days3') is
asked for too. Drives the real tool path (_run_tool_call -> tools.py)."""
from __future__ import annotations

import json

import pytest

import app.lib.construction_formulas as cf


def _call(params):
    from tests.test_construction_calc_tool import _agent, _run

    agent = _agent(["construction"])
    tc = {"id": "c1", "function": {"name": "construction_calc", "arguments": json.dumps(
        {"calculation": "delay_damages_daily", "params": params})}}
    return _run(agent._run_tool_call(tc))


@pytest.mark.parametrize("params", [
    {"amount_per_day": 5000, "days": None},
    {"amount_per_day": 5000, "days": ""},
    {"amount_per_day": 5000, "days": "  "},
    {"params": {"amount_per_day": 5000, "days": None}},
    {"amount_per_day": 1200, "days": "days3"},
])
def test_a_blank_or_unreadable_figure_is_asked_for(params):
    env = _call(params)
    assert env["ok"] is False, env
    text = json.dumps(env["result"])
    assert "days of delay" in text
    assert "should I use" in text
    assert "total_amount" not in text
    assert "defaults_used" not in text


@pytest.mark.parametrize("params,total", [
    ({"amount_per_day": 5000, "days": 60}, 300000.0),
    ({"amount_per_day": 1200, "days": 45, "currency": None}, 54000.0),
])
def test_a_given_figure_still_runs_and_a_blank_text_input_does_not_block(params, total):
    env = _call(params)
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == total


@pytest.mark.parametrize("params,blank", [
    ({"days": None, "amount_per_day": 5}, ("days",)),
    ({"params": {"days": ""}, "x": 1}, ("days",)),
    ({"b": " ", "input": {"a": None}}, ("b", "a")),
    ({"z": None, "a": None}, ("z", "a")),
    ({"days": None, "params": {"days": None}}, ("days",)),
    ({"days": 3}, ()),
])
def test_blank_calc_kwargs_reads_top_level_and_nested(params, blank):
    assert cf._blank_calc_kwargs(params) == blank


@pytest.mark.parametrize("blank,params,missing,expected", [
    (("days",), {}, [], ["days"]),
    (("days",), {"days": 60}, [], []),
    (("days",), {}, ["days"], []),
    (("currency",), {}, [], []),
    (("not_a_param",), {}, [], []),
    (("days", "amount_per_day"), {}, [], ["days", "amount_per_day"]),
    (("amount_per_day", "days"), {}, [], ["amount_per_day", "days"]),
])
def test_named_but_blank_asks_only_for_unfilled_numeric_inputs(blank, params, missing, expected):
    assert cf._named_but_blank("delay_damages_daily", blank, params, missing) == expected


@pytest.mark.parametrize("value,blank", [(None, True), ("", True), (" \t", True),
                                         (0, False), ("0", False), ("days3", False)])
def test_is_blank(value, blank):
    assert cf._is_blank(value) is blank
