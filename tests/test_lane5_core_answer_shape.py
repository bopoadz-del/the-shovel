"""CORE answer shape: tool rows and a finished section survive the driver."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import Param, answer_text, call, drive, registered


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _rows(span_m: float) -> dict:
    return {
        "value": span_m,
        "rows": ["north apron 4", "south apron 9"],
    }


def test_every_tool_row_reaches_the_answer(monkeypatch):
    name = "lane5_rows_kept"
    user = "calculate the qxnet apron table span 4 m"
    with registered(name, _rows, "Qxnet apron table", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4,
            }, "c1")]},
            {"content": "The rows are:\n| face | amount |\n"},
        ])
    text = answer_text(events)
    assert "north apron 4" in text
    assert "south apron 9" in text


def test_a_section_is_not_left_empty(monkeypatch):
    name = "lane5_section_kept"

    def fn(span_m: float) -> dict:
        return {"value": span_m}

    user = "calculate the qxnet section total span 4 m"
    with registered(name, fn, "Qxnet section total", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
    }, {"value": "m"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4,
            }, "c1")]},
            {"content": "## Result\n"},
        ])
    text = answer_text(events)
    assert "4" in text
    assert "## Result\n\n" not in text and not text.rstrip().endswith("## Result")
