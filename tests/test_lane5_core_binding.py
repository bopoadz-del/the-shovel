"""CORE argument binding. Each case is a synthetic formula run by the real
driver through construction_calc. The figure the user wrote is what the
calculator must receive."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, call, drive, registered, tool_envelope


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _product(span_m: float, breadth_m: float) -> dict:
    return {"value": span_m * breadth_m}


def _envelope(monkeypatch, name, user, extra_args=None):
    args = {"calculation": name, "text": user}
    if extra_args:
        args.update(extra_args)
    _events, seen = drive(monkeypatch, user, [
        {"content": "", "tool_calls": [call("construction_calc", args, "c1")]},
        {"content": "noted"},
    ])
    return tool_envelope(seen)


def test_a_magnitude_word_scales_the_figure(monkeypatch):
    name = "lane5_bind_magnitude"

    def fn(reach_m: float, lanes: float) -> dict:
        return {"value": reach_m * lanes}

    user = "calculate the qxnet reach product reach 1.5 million m lanes 2"
    with registered(name, fn, "Qxnet reach product", {
        "reach_m": Param("m", lo=0, hi=1e12, label="reach"),
        "lanes": Param("-", lo=0, hi=1e6, label="lanes"),
    }, {"value": "m"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(3_000_000)


@pytest.mark.parametrize("phrase,expected", [
    ("purse amount USD 150000", 150000),
    ("purse amount SAR 5,000", 5000),
    ("purse amount $5,000", 5000),
])
def test_a_currency_prefix_binds_the_figure(monkeypatch, phrase, expected):
    name = "lane5_bind_currency"

    def fn(purse_amount: float) -> dict:
        return {"value": purse_amount}

    user = f"calculate the qxnet purse total {phrase}"
    with registered(name, fn, "Qxnet purse total", {
        "purse_amount": Param("USD", lo=0, hi=1e12, label="purse amount"),
    }, {"value": "USD"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(expected)


def test_a_plain_label_fills_the_parameter(monkeypatch):
    name = "lane5_bind_label"

    def fn(alpha_m: float, beta_m: float) -> dict:
        return {"value": alpha_m * beta_m}

    user = "calculate the qxnet opening product clear span 12 m side reach 3 m"
    with registered(name, fn, "Qxnet opening product", {
        "alpha_m": Param("m", lo=0, hi=1e6, label="clear span"),
        "beta_m": Param("m", lo=0, hi=1e6, label="side reach"),
    }, {"value": "m2"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(36)


def test_a_two_word_parameter_name_binds(monkeypatch):
    name = "lane5_bind_two_word"

    def fn(bay_width: float) -> dict:
        return {"value": bay_width}

    user = "calculate the qxnet bay product"
    with registered(name, fn, "Qxnet bay product", {
        "bay_width": Param("m", lo=0, hi=1e6, label="bay width"),
    }, {"value": "m"}):
        env = _envelope(monkeypatch, name, user, {"bay width": 7.5})
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(7.5)


def test_a_long_name_binds_when_the_short_name_is_also_a_parameter(monkeypatch):
    name = "lane5_bind_pair"

    def fn(pv: float, planned_value: float) -> dict:
        return {"value": planned_value, "pv": pv, "planned_value": planned_value}

    user = "calculate the qxnet value pair planned value 100 pv 10"
    with registered(name, fn, "Qxnet value pair", {
        "pv": Param("USD", lo=0, hi=1e12, label="pv"),
        "planned_value": Param("USD", lo=0, hi=1e12, label="planned value"),
    }, {"value": "USD"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["planned_value"] == pytest.approx(100)
    assert env["result"]["pv"] == pytest.approx(10)


@pytest.mark.parametrize("phrase", ["offset minus 3.5 m", "offset −3.5 m"])
def test_a_written_negative_keeps_its_sign(monkeypatch, phrase):
    name = "lane5_bind_sign"

    def fn(offset_m: float) -> dict:
        return {"value": offset_m}

    user = f"calculate the qxnet offset gate {phrase}"
    with registered(name, fn, "Qxnet offset gate", {
        "offset_m": Param("m", lo=-100, hi=100, label="offset"),
    }, {"value": "m"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(-3.5)


def test_a_natural_area_phrase_converts_into_the_declared_unit(monkeypatch):
    name = "lane5_bind_area"

    def fn(floor_m2: float) -> dict:
        return {"value": floor_m2}

    user = "calculate the qxnet floor product floor area at 130 sq ft"
    with registered(name, fn, "Qxnet floor product", {
        "floor_m2": Param("m2", lo=0, hi=1e9, label="floor area"),
    }, {"value": "m2"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(12.077, rel=1e-3)


def test_a_percent_written_as_150_binds_as_the_declared_fraction(monkeypatch):
    name = "lane5_bind_percent"

    def fn(allowance_percent: float) -> dict:
        return {"value": allowance_percent}

    user = "calculate the qxnet allowance gate allowance 150%"
    with registered(name, fn, "Qxnet allowance gate", {
        "allowance_percent": Param("fraction", lo=0, hi=2, label="allowance"),
    }, {"value": "fraction"}):
        env = _envelope(monkeypatch, name, user)
    assert env.get("status") == "success", env
    assert env["result"]["value"] == pytest.approx(1.5)
