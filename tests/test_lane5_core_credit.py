"""CORE credit line: display labels, the user's units, and no dumped defaults."""
from __future__ import annotations

import json

import pytest

from tests.lane5_core_support import Param, answer_text, call, drive, registered


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _echo_value(messages):
    try:
        payload = json.loads(messages[-1]["content"])
    except ValueError:  # the model reads a calculation as prose: take the run's own result
        from tests.lane5_core_support import _TOOL_RESULTS

        payload = _TOOL_RESULTS[-1]
    result = payload.get("result") or {}
    inner = result.get("result") if isinstance(result, dict) else None
    value = (inner or {}).get("value")
    return {"content": f"The stored total is {value}."}


def test_the_source_line_uses_the_display_label(monkeypatch):
    name = "lane5_credit_label"

    def fn(span_m: float) -> dict:
        return {"value": span_m}

    user = "calculate the qxnet credit reach clear span 12 m"
    with registered(name, fn, "Qxnet credit reach", {
        "span_m": Param("m", lo=0, hi=1e6, label="clear span"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 12,
            }, "c1")]},
            _echo_value,
        ])
    text = answer_text(events)
    assert "clear span" in text
    assert "span_m" not in text


@pytest.mark.xfail(strict=True, reason="CORE: 36 credit line dumps raw param labels and defaults")
def test_the_credit_line_does_not_dump_a_signature_default(monkeypatch):
    name = "lane5_credit_default"

    def fn(span_m: float, factor: float = 1.25) -> dict:
        return {"value": span_m * factor, "factor": factor}

    user = "calculate the qxnet credit scale clear span 4 m"
    with registered(name, fn, "Qxnet credit scale", {
        "span_m": Param("m", lo=0, hi=1e6, label="clear span"),
        "factor": Param("fraction", lo=0, hi=10, label="scale factor"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4,
            }, "c1")]},
            _echo_value,
        ])
    text = answer_text(events)
    assert "factor" not in text
    assert "default" not in text.lower()


@pytest.mark.xfail(strict=True, reason="CORE: 37 credit echoes converted internal units instead of the user's units")
def test_the_credit_line_keeps_the_users_unit(monkeypatch):
    name = "lane5_credit_unit"

    def fn(span_m: float) -> dict:
        return {"value": span_m}

    user = "calculate the qxnet credit length clear span 10 ft"
    with registered(name, fn, "Qxnet credit length", {
        "span_m": Param("m", lo=0, hi=1e6, label="clear span"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "text": user,
            }, "c1")]},
            _echo_value,
        ])
    text = answer_text(events)
    assert "10" in text and "ft" in text
