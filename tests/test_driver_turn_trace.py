"""Every driver turn leaves one trace line: what the model was offered and
chose, and how the turn ended. Names of tools and workflows only, never the
user's words. Synthetic turn, scripted model."""
from __future__ import annotations

import json
import logging

from tests.lane5_core_support import call, drive, install_driver_stream


class _Lines(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_a_driver_turn_leaves_a_trace_without_the_users_words(monkeypatch):
    from app.agents.driver import loop, routes
    from app.core import predefined_reasoning

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(routes, "catalogue", lambda: ["widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {})

    async def fake_run_workflow(action, ctx, session):
        return {"handled": True, "status": "error", "answer": "No widget list supplied."}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    log = logging.getLogger(loop.__name__)
    monkeypatch.setattr(log, "disabled", False)
    seen = _Lines()
    log.addHandler(seen)
    log.setLevel(logging.INFO)
    try:
        drive(monkeypatch, "Register the qxsecret widgets please", [
            {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
            {"content": "{}"},  # reading the ask for the refused inputs: nothing stated
            {"content": "Which widgets?"},
            {"content": "Which widgets?"},  # after the one reminder to call the workflow again
        ])
    finally:
        log.removeHandler(seen)
    lines = [l for l in seen.lines if l.startswith("DRIVER_TURN ")]
    assert len(lines) == 1
    assert "qxsecret" not in lines[0]
    trace = json.loads(lines[0][len("DRIVER_TURN "):])
    assert trace["ended"] == "question"  # the refused workflow asks for its inputs
    assert trace["figures_removed"] == []
    first = trace["steps"][0]
    assert first["workflows"] == 1
    assert first["calls"] == [{"tool": "run_workflow", "workflow": "widget_register", "calculation": None,
                               "ok": False, "missing": True, "result": None, "given": None}]
