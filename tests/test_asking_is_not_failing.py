"""A calculator that asks for an input has not failed: the answer never
adds "<tool> could not complete: it returned no result" to its question
(lane 4: the lap-length ask). Synthetic tool, scripted model."""
from __future__ import annotations

from app.agents.core.tool_registry import _TOOLS, tool
from tests.lane5_core_support import answer_text, call, drive, install_driver_stream


def test_a_tool_asking_for_input_draws_no_failure_line(monkeypatch):
    install_driver_stream(monkeypatch)
    name = "lane5_lap_ask"

    @tool(name, owner="base", display_name="Qx lap check")
    async def handle(_call):
        return {"name": name, "ok": False,
                "result": {"status": "error", "needs_input": True,
                           "question": "The Qx lap check needs the steel grade. What fy should I use?"}}

    try:
        events, _ = drive(monkeypatch, "Lap length for a 16 mm bar?", [
            {"content": "", "tool_calls": [call(name, {}, "c1")]},
            {"content": "What steel grade (fy) should I use?"},
        ])
    finally:
        _TOOLS.pop(name, None)
    text = answer_text(events)
    assert "could not complete" not in text


def test_a_failure_with_no_words_of_its_own_adds_no_line(monkeypatch):
    install_driver_stream(monkeypatch)
    name = "lane5_silent_fail"

    @tool(name, owner="base", display_name="Qx silent")
    async def handle(_call):
        return {"name": name, "ok": False, "result": {"status": "error"}}

    try:
        events, _ = drive(monkeypatch, "Run the qx silent check", [
            {"content": "", "tool_calls": [call(name, {}, "c1")]},
            {"content": "It did not run."},
        ])
    finally:
        _TOOLS.pop(name, None)
    assert "it returned no result" not in answer_text(events)
