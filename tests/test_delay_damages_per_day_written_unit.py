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


@pytest.mark.parametrize("amount,days,total,currency", [
    ("5,000 per day", 60, 300000.0, None),
    ("1,200 per day", "45 days", 54000.0, None),
    ("5,000 /day", 60, 300000.0, None),
    ("SAR 1,200 per day", 45, 54000.0, "SAR"),
    ("5000 SAR/day", 60, 300000.0, "SAR"),
    ("SAR 5,000/day", 60, 300000.0, "SAR"),
])
def test_a_daily_amount_written_per_day_is_totalled(amount, days, total, currency):
    env = _call({"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["total_amount"] == total, env
    # The currency flows from the amount the user wrote: SAR when the amount
    # carries it, none when it does not -- never an invented default (K-T5 ii).
    assert res.get("currency") == currency, env


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
     {"amount_per_day": "5,000 per day", "days": 60}, 300000.0, "300,000.00"),
    ("45 days at 1,200 per day",
     {"amount_per_day": "1,200 per day", "days": "45 days"}, 54000.0, "54,000.00"),
])
def test_the_live_asks_give_an_amount_and_total_without_an_invented_currency(ask, params, total, shown):
    # These asks carry no currency, so the total stands on its own -- no SAR is
    # invented (K-T5 ruling ii). A currency written in the amount still flows.
    env = _chat(ask, params)
    assert env["ok"] is True, env
    res = env["result"]["result"]
    assert res["total_amount"] == total
    assert shown in res["note"]
    assert "SAR" not in res["note"]
    assert res.get("currency") is None
    assert "rate_percent" not in {k for k, v in res.items() if v}


# The two functions the fix touches, read directly.
import app.lib.formula_registry as fr


@pytest.mark.parametrize("value,expected", [
    ("SAR 1,200 per day", (1200.0, "SAR/day")),
    ("AED 5,000", (5000.0, "AED")),
    ({"value": 20, "unit": "ft"}, (20.0, "ft")),
    ({"value": 3.5, "unit": "m"}, (3.5, "m")),
])
def test_value_and_unit_reads_a_trailing_per_unit(value, expected):
    assert fr.value_and_unit(value) == expected


def _spec(unit, hi=None):
    return fr.ParamSpec(name="x", kind="number", unit=unit, lo=0, hi=hi,
                        label="x", grade="", required=True)


@pytest.mark.parametrize("unit,number,written,expected", [
    ("currency/day", 5000.0, "per day", 5000.0),
    ("currency/day", 7000.0, "per week", 1000.0),
    ("count", 4.0, "bars", 4.0),
    ("count", 9.0, "pieces", 9.0),
])
def test_convert_figure_takes_an_implied_currency_and_leaves_counts(unit, number, written, expected):
    out = fr.convert_figure(_spec(unit), number, written)
    assert out["value"] == pytest.approx(expected), out


@pytest.mark.parametrize("written", ["per m3", "per kg"])
def test_convert_figure_refuses_a_per_unit_of_another_kind(written):
    assert fr.convert_figure(_spec("currency/day"), 5000.0, written)["rejection"]["reason"] == "unknown_unit"


@pytest.mark.parametrize("amount,days,total", [
    ("SAR 200 per man hour", 10, 48000.0),
    ("SAR 50 per man hour", 60, 72000.0),
])
def test_a_multi_word_per_unit_reaches_the_calculator_whole(amount, days, total):
    # "man hour" is a two-word time unit the converter already knows.
    env = _chat(f"Delay damages at {amount} for {days} days",
                {"amount_per_day": amount, "days": days})
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == pytest.approx(total)


@pytest.mark.parametrize("value,unit", [
    ("SAR 1,200 per working day", "SAR/working day"),
    ("AED 7,000 per working week", "AED/working week"),
])
def test_value_and_unit_keeps_a_multi_word_per_unit(value, unit):
    assert fr.value_and_unit(value)[1] == unit


@pytest.mark.parametrize("unit,written", [
    ("m3/h", "per h"),
    ("kg/m3", "per m3"),
])
def test_a_per_unit_on_a_non_currency_input_is_refused(unit, written):
    # The currency is implied only where the input is itself money per something.
    out = fr.convert_figure(_spec(unit), 5.0, written)
    assert out["rejection"]["reason"] == "unknown_unit", out


@pytest.mark.parametrize("written", ["per fortnightly cycle", "bananas"])
def test_convert_figure_refuses_a_unit_it_cannot_read_even_with_currency(written):
    out = fr.convert_figure(_spec("currency/day"), 5000.0, written)
    assert out["rejection"]["reason"] == "unknown_unit", out


@pytest.mark.parametrize("padded,plain", [
    ("SAR 1,200 per day  ", "SAR 1,200 per day"),
    ("AED 7,000 per working week \t", "AED 7,000 per working week"),
])
def test_trailing_space_after_a_per_unit_reads_as_without(padded, plain):
    assert fr.value_and_unit(padded) == fr.value_and_unit(plain)
    assert fr.value_and_unit(padded)[1] == plain.split(" ", 2)[0] + "/" + plain.split(" ", 3)[3]


# ---- One rule for how "per" is written (owner order, 30 Oct 16:39 Paris) ----
# Case, separator (space, hyphen, slash or none) and the one-word rate all
# normalise through planning.normalise_rate. G6's U3 (capitals) and U11
# (no space) now PARSE: the owner changed U11 from "refuse" to "normalise".
from app.agents.base.formulas import construction_formulas_planning as planning


@pytest.mark.parametrize("amount", [
    "SAR 5,000 per Day",            # U3
    "SAR 5,000 PER DAY",
    "SAR 5,000 perday",             # U11
    "SAR 5,000 per-day",
    "SAR 5,000 per/day",
    "SAR 5,000 daily",
    "SAR 5,000 Daily",
    "5,000 Per Day",
    "5,000 perday",
    "5,000 per-day",
    "5,000 daily",
    "SAR 5,000 Perday",
    "SAR 5,000 PERDAY",
])
def test_every_spelling_of_per_day_totals_the_same(amount):
    env = _call({"amount_per_day": amount, "days": 60})
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == 300000.0, env


@pytest.mark.parametrize("amount,total", [
    ("SAR 7,000 perweek", 60000.0),     # U11
    ("SAR 7,000 Per Week", 60000.0),
    ("SAR 7,000 weekly", 60000.0),
    ("SAR 7,000 Weekly", 60000.0),
    ("7,000 per-week", 60000.0),
])
def test_every_spelling_of_per_week_converts(amount, total):
    env = _call({"amount_per_day": amount, "days": 60})
    assert env["ok"] is True, env
    assert env["result"]["result"]["total_amount"] == total, env


@pytest.mark.parametrize("value,expected", [
    ("SAR 5,000 per Day", (5000.0, "SAR/Day")),                    # U3
    ("AED 7,000 per Working Week", (7000.0, "AED/Working Week")),  # U3
    ("SAR 5,000 perday", (5000.0, "SAR/day")),                     # U11
    ("SAR 1,200 perweek", (1200.0, "SAR/week")),                   # U11
    ("SAR 1,200 per/day", (1200.0, "SAR/day")),
    ("SAR 1,200 daily", (1200.0, "SAR/day")),
])
def test_value_and_unit_reads_any_spelling_of_per(value, expected):
    assert fr.value_and_unit(value) == expected


@pytest.mark.parametrize("value", ["SAR 5,000 approx", "SAR 5,000 lump sum"])
def test_a_currency_tail_that_is_not_a_rate_is_not_read(value):
    assert fr.value_and_unit(value) is None


@pytest.mark.parametrize("written,normal", [
    ("per day", "/day"), ("Per Day", "/Day"), ("perday", "/day"),
    ("per-day", "/day"), ("per/day", "/day"), ("daily", "/day"),
    ("hourly", "/hour"), ("monthly", "/month"), ("SAR perWeek", "SAR/Week"),
    ("kN per m", "kN/m"),
    ("  per day  ", "/day"), (" SAR daily ", "SAR/day"),
    ("Hourly", "/hour"), ("SAR Perweek", "SAR/week"),
])
def test_normalise_rate_gives_one_spelling(written, normal):
    assert planning.normalise_rate(written) == normal


@pytest.mark.parametrize("written,normal", [
    ("SAR 100 annually", "SAR 100/year"), ("SAR 100 yearly", "SAR 100/year"),
])
def test_normalise_rate_reads_a_yearly_word_as_per_year(written, normal):
    assert planning.normalise_rate(written) == normal


@pytest.mark.parametrize("word", ["percent", "perch", "Percent", "permit", "superday", "upperweek", "perc", "permph",
                                  "biweekly", "semimonthly", "dailyrate"])
def test_normalise_rate_leaves_a_word_that_is_not_per_a_unit(word):
    # "percent", "perch", "permit": the rest is not a unit. "superday":
    # "per" inside a word is not a "per".
    assert planning.normalise_rate(word) == word


@pytest.mark.parametrize("token,dimension", [
    ("SAR Per Day", "currency/time"), ("currency perweek", "currency/time"),
    ("SAR daily", "currency/time"), ("AED per Working Week", "currency/working time"),
    ("percent", "fraction"),
])
def test_parse_unit_reads_any_spelling_of_per(token, dimension):
    assert planning.unit_dimension(token) == dimension


@pytest.mark.parametrize("token,dimension,scale", [
    ("kg/cm2", "pressure", 98.0665),
    ("t/m2", "pressure", 9.80665),
    ("lb/ft2", "pressure", 0.04788025898033584),
])
def test_a_whole_unit_wins_over_splitting_it_as_a_ratio(token, dimension, scale):
    import app.agents.base.formulas.construction_formulas_planning as planning

    unit = planning.parse_unit(token)
    assert (unit.dimension, unit.scale) == (dimension, scale)
