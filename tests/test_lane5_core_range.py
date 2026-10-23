"""CORE range rejection: an out-of-range input is asked, with the range
and the unit, and the driver does not replace that with its own figure."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, answer_text, call, drive, registered, tool_envelope


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _run(monkeypatch, value, closing):
    name = "lane5_range_gate"

    def fn(blend: float) -> dict:
        return {"value": blend * blend}

    user = "calculate the qxnet blend gate"
    with registered(name, fn, "Qxnet blend gate", {
        "blend": Param("ratio", lo=0, hi=1, label="blend"),
    }, {"value": "ratio"}):
        events, seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "blend": value,
            }, "c1")]},
            {"content": closing},
        ])
    return events, tool_envelope(seen)


def test_an_out_of_range_input_is_asked_not_replaced(monkeypatch):
    events, env = _run(monkeypatch, 4, "I multiplied the figures and the stored total is 16.")
    text = answer_text(events)
    assert env.get("needs_input") is True
    assert "16" not in text
    question = str(env.get("question") or "")
    assert question and question in text


def test_an_out_of_range_input_is_not_worked_by_hand(monkeypatch):
    events, env = _run(monkeypatch, 4, "4 times 4 is 16, so the stored total is 16.")
    text = answer_text(events)
    assert "4 times 4" not in text
    assert "16" not in text
    assert env.get("needs_input") is True


def test_the_rejection_names_the_range_and_the_unit(monkeypatch):
    _events, env = _run(monkeypatch, 4, "noted")
    question = str(env.get("question") or env.get("error") or "")
    assert "0" in question and "1" in question
    assert "ratio" in question
