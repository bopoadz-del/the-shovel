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
    ({"amount_per_day": 1200, "days": 45, "currency": "AED"}, 54000.0),
])
def test_a_given_figure_still_runs(params, total):
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
    (("currency",), {}, [], ["currency"]),
    (("currency",), {"currency": "AED"}, [], []),
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


@pytest.mark.parametrize("params", [
    {"amount_per_day": 1200, "days": 45, "currency": None},
    {"amount_per_day": 1200, "days": 45, "currency": " "},
])
def test_a_blank_text_input_is_asked_for_not_defaulted(params):
    env = _call(params)
    assert env["ok"] is False, env
    text = json.dumps(env["result"])
    assert "currency" in text and "should I use" in text
    assert "total_amount" not in text


def _declare_required(monkeypatch, name, key):
    import dataclasses

    import app.lib.formula_registry as fr

    spec = fr.get(name)
    params = dict(spec.params)
    params[key] = dataclasses.replace(params[key], required=True)
    monkeypatch.setitem(fr._REGISTRY, name, dataclasses.replace(spec, params=params))


@pytest.mark.parametrize("params", [
    {"amount_per_day": 5000},
    {"amount_per_day": 5000, "currency": "SAR"},
])
def test_an_input_declared_required_is_asked_for_when_omitted(monkeypatch, params):
    _declare_required(monkeypatch, "delay_damages_daily", "days")
    env = _call(params)
    assert env["ok"] is False, env
    text = json.dumps(env["result"])
    assert "days of delay" in text and "should I use" in text
    assert "daily_amount" not in text


@pytest.mark.parametrize("params,daily", [
    ({"amount_per_day": 5000}, 5000.0),
    ({"rate_percent": 0.1, "contract_amount": 1_000_000}, 1000.0),
])
def test_an_input_declared_optional_by_its_default_still_runs_without_it(params, daily):
    env = _call(params)
    assert env["ok"] is True, env
    assert env["result"]["result"]["daily_amount"] == daily


@pytest.mark.parametrize("bound,expected", [
    ({"amount_per_day": 5}, ["days"]),
    ({"amount_per_day": 5, "days": None}, ["days"]),
    ({"amount_per_day": 5, "days": 3}, []),
    ({}, ["rate_percent", "contract_amount", "days"]),
])
def test_missing_required_adds_declared_required_inputs(monkeypatch, bound, expected):
    _declare_required(monkeypatch, "delay_damages_daily", "days")
    import app.lib.formula_registry as fr

    assert cf._missing_required(fr.get("delay_damages_daily").fn, bound, name="delay_damages_daily") == expected


@pytest.mark.parametrize("declared,expected", [(True, True), (False, False), (None, False)])
def test_registry_required_honours_the_declaration_over_the_signature(monkeypatch, declared, expected):
    import dataclasses

    import app.lib.formula_registry as fr

    spec = fr.get("delay_damages_daily")
    params = dict(spec.params)
    params["days"] = dataclasses.replace(params["days"], required=declared)
    row = fr.parameters(dataclasses.replace(spec, params=params))["days"]
    assert (row.required, row.default) == (expected, 0.0)


@pytest.mark.parametrize("name,expected", [
    ("delay_damages_daily", []),
    ("no_such_formula", []),
    (None, []),
])
def test_declared_required_is_empty_without_a_declaration(name, expected):
    assert cf._declared_required(name) == expected


@pytest.mark.parametrize("name,blank", [
    ("beam_moment_point_load", ("distance_from_left_m",)),
    ("bolt_shear_capacity", ("grade",)),
])
def test_a_blank_input_whose_default_is_blank_is_not_asked_for(name, blank):
    assert cf._named_but_blank(name, blank, {}, []) == []


@pytest.mark.parametrize("declared,expected", [(None, True), (True, True), (False, False)])
def test_a_signature_required_input_follows_its_declaration(declared, expected):
    import dataclasses

    import app.lib.formula_registry as fr

    spec = fr.get("backfill_volume")
    params = dict(spec.params)
    params["excavation_bank_m3"] = dataclasses.replace(params["excavation_bank_m3"], required=declared)
    row = fr.parameters(dataclasses.replace(spec, params=params))["excavation_bank_m3"]
    assert (row.required, row.default) == (expected, None)


@pytest.mark.parametrize("bound,expected", [
    ({}, ["excavation_bank_m3", "structure_volume_m3"]),
    ({"excavation_bank_m3": 10}, ["structure_volume_m3"]),
])
def test_a_declared_required_input_is_listed_once(monkeypatch, bound, expected):
    import app.lib.formula_registry as fr

    _declare_required(monkeypatch, "backfill_volume", "excavation_bank_m3")
    assert cf._missing_required(fr.get("backfill_volume").fn, bound, name="backfill_volume") == expected


@pytest.mark.parametrize("name", [" delay_damages_daily ", "delay_damages_daily"])
def test_declared_required_reads_the_name_trimmed(monkeypatch, name):
    _declare_required(monkeypatch, "delay_damages_daily", "days")
    assert cf._declared_required(name) == ["days"]
