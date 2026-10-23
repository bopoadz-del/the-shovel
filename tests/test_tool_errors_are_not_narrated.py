"""A tool's internal error text is never narrated to the user: the driver
records each tool result on the turn, so the shared exit check replaces a
copied internal error with a plain sentence. Synthetic tool, scripted model."""
from __future__ import annotations

from app.agents import answer_exit
from app.agents.core.tool_registry import _TOOLS, tool
from tests.lane5_core_support import answer_text, call, drive, install_driver_stream


def test_a_copied_internal_error_is_not_shown(monkeypatch):
    install_driver_stream(monkeypatch)
    name = "lane5_register_build"
    err = "auto_risks list is empty: provide risks or auto_risks parsed from a document"

    @tool(name, owner="base", display_name="Register build")
    async def handle(_call):
        return {"name": name, "ok": False, "error": err, "result": {"status": "error", "error": err}}

    turn, token = answer_exit.begin_turn()
    try:
        events, _ = drive(monkeypatch, "Build the qx register", [
            {"content": "", "tool_calls": [call(name, {}, "c1")]},
            {"content": f"The register could not be built: {err}."},
        ])
    finally:
        answer_exit.reset_turn(token)
        _TOOLS.pop(name, None)
    text = answer_text(events)
    assert "auto_risks" not in text
