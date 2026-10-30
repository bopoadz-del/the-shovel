"""A label written straight after a figure is that figure's unit.

"60 days at 5,000 per day" names the 60 as the day count: the label "days"
labels the figure before it, never the next one. Run through the chat tool
path (``_run_tool_call`` -> tools.py -> construction_calc).
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


def _extract(text):
    return cf.extract_calculation_params_from_text(cf.CALCULATORS["delay_damages_daily"], text)


@pytest.mark.parametrize("ask,amount,days,total", [
    ("60 days at 5,000 per day", 5000, 60, 300_000.0),
    ("Delay damages: 45 days at 1,200 per day", 1200, 45, 54_000.0),
])
def test_days_written_as_the_unit_of_a_figure_bind_that_figure(ask, amount, days, total):
    env = _chat_call(ask, {"amount_per_day": amount})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total


@pytest.mark.parametrize("ask,amount,days,total", [
    ("60days at 5,000 per day", 5000, 60, 300_000.0),        # unit glued to the figure
    ("45days at 1,200 per day", 1200, 45, 54_000.0),
])
def test_days_glued_to_the_figure_bind_that_figure(ask, amount, days, total):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total


@pytest.mark.parametrize("ask,amount,days,total", [
    ("1,060 days at 5 per day", 5, 1060, 5_300.0),           # the 60 is not a figure on its own
    ("2,045 days at 2 per day", 2, 2045, 4_090.0),
])
def test_a_thousands_figure_binds_whole_not_its_tail(ask, amount, days, total):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total


@pytest.mark.parametrize("ask,amount", [
    ("2,5 days at 1,000 per day", 1000),     # a decimal comma: "5" is not the day count
    ("7,5 days at 200 per day", 200),
])
def test_the_tail_digits_of_a_figure_are_not_a_day_count(ask, amount):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    assert res.get("days") != 5.0 and res.get("total_amount") != 5.0 * amount


@pytest.mark.parametrize("text,bound", [
    ("60 days at 5,000 per day", {"days": 60.0}),
    ("days of delay 12", {"days": 12.0}),           # label first, figure after: unchanged
])
def test_extractor_reads_a_figure_by_the_unit_after_it(text, bound):
    assert {k: v for k, v in _extract(text).items() if k == "days"} == bound


@pytest.mark.parametrize("text,key", [
    ("18 currency/day", "currency"),     # a label inside a compound unit is not the unit
    ("6 days/week", "days"),
])
def test_a_label_inside_a_compound_unit_does_not_take_the_figure(text, key):
    assert _extract(text).get(key) not in (18.0, 6.0)


def _crew(concrete_m3: float = 0.0, structural_steel_t: float = 0.0) -> dict:
    return {"value": concrete_m3 / 1000 + structural_steel_t / 100}


@pytest.mark.parametrize("ask,value", [
    ("crew check for 120,000 m3 concrete and 6,000 t steel", 180.0),   # the 3 of m3 is no figure
    ("crew check for 60,000 m3 concrete and 3,000 t steel", 90.0),
])
def test_a_digit_inside_a_unit_is_not_a_figure(ask, value):
    from tests.lane5_core_support import Param, registered

    inputs = {
        "concrete_m3": Param("m3", lo=0, hi=1e9, label="concrete", required=False),
        "structural_steel_t": Param("t", lo=0, hi=1e9, label="structural steel", required=False),
    }
    with registered("lane5_fig_before_label_crew", _crew, "Crew check", inputs, {"value": "count"}):
        env = cf.run_calculation("lane5_fig_before_label_crew", {"text": ask})
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(value)


@pytest.mark.parametrize("text", ["60 days at 5,000 per day", "45 days at 1,200 per day"])
def test_an_ask_with_no_calculator_extracts_nothing(text):
    assert cf.extract_calculation_params_from_text(None, text) == {}


@pytest.mark.parametrize("ask,amount", [
    ("v.5 days at 1,000 per day", 1000),     # ".5" after a letter: the 5 is no figure of its own
    ("x.7 days at 200 per day", 200),
])
def test_digits_after_a_point_are_not_a_figure_on_their_own(ask, amount):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    tail = float(ask.split(".")[1].split()[0])
    assert res.get("days") != tail and res.get("total_amount") != tail * amount


@pytest.mark.parametrize("ask,amount,figure", [
    ("60 days3", 5000, 60.0),                # "days3" is not the unit "days"
    ("45 days2", 1200, 45.0),
])
def test_a_label_running_into_a_digit_is_not_the_unit(ask, amount, figure):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    assert res.get("days") != figure and res.get("total_amount") != figure * amount


@pytest.mark.parametrize("ask,amount,days,total", [
    ("60 DAYS at 5,000/day", 5000, 60, 300_000.0),     # the unit is read in any case
    ("45 Days at 1,200/day", 1200, 45, 54_000.0),
])
def test_the_unit_after_a_figure_binds_in_any_case(ask, amount, days, total):
    res = _chat_call(ask, {"amount_per_day": amount})["result"]["result"]
    assert res["days"] == days and res["total_amount"] == total
