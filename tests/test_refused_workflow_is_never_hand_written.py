"""A workflow that refused for a missing input is never replaced by a
deliverable the model writes itself: if the model answers without running
it again, the turn asks for what the workflow needs. A successful re-call
answers as usual. Synthetic workflow, scripted model."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

_CARD = {"produces": "A widget register of the fitted widgets",
         "use_when": "The user asks to list the widgets fitted so far",
         "needs": "The widget schedule or a list of widgets",
         "inputs": {"widgets": "Widgets"}}


@pytest.fixture
def workflow(monkeypatch):
    from app.agents.driver import routes
    from app.core import predefined_reasoning

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(routes, "catalogue", lambda: ["widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {"widget_register": _CARD})
    calls = []

    async def fake_run_workflow(action, ctx, session):
        calls.append(ctx.get("params") or {})
        if not (ctx.get("params") or {}).get("widgets"):
            return {"handled": True, "status": "error", "answer": "No widget list supplied."}
        return {"handled": True, "status": "success", "answer": "Widget register: 3 widgets."}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    return calls


def test_an_answer_after_a_refusal_becomes_the_workflows_request(monkeypatch, workflow):
    events, _ = drive(monkeypatch, "Make the widget register", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
        {"content": "{}"},  # reading the ask for the refused inputs: nothing stated
        {"content": "", "tool_calls": [call("search_project_documents", {"query": "widgets"}, "s1")]},
        {"content": "Widget register\n| widget | location |\n| A | level 1 |"},
        {"content": "Widget register\n| widget | location |\n| A | level 1 |"},  # after the reminder
    ])
    text = answer_text(events)
    assert "| A | level 1 |" not in text
    assert "a widget register of the fitted widgets" in text and "widget schedule" in text


def test_a_successful_re_call_answers_as_usual(monkeypatch, workflow):
    events, _ = drive(monkeypatch, "Make the widget register", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
        {"content": "{}"},  # reading the ask for the refused inputs: nothing stated
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register",
                                                              "params": {"widgets": ["A", "B", "C"]}}, "w2")]},
        {"content": "Widget register: 3 widgets."},
    ])
    assert "Widget register: 3 widgets." in answer_text(events)


def test_a_model_reminded_once_can_still_re_call(monkeypatch, workflow):
    """The first plain answer after a refusal is not shown; the model is
    reminded once, and a re-call then answers as usual."""
    events, seen = drive(monkeypatch, "Make the widget register for widgets A, B and C", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
        {"content": "{}"},  # reading the ask for the refused inputs: nothing stated
        {"content": "Here is a register I wrote: A, B, C."},
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register",
                                                              "params": {"widgets": ["A", "B", "C"]}}, "w2")]},
        {"content": "Widget register: 3 widgets."},
    ])
    text = answer_text(events)
    assert "Here is a register I wrote" not in text
    assert "Widget register: 3 widgets." in text
    assert any("has not run yet" in m.get("content", "") for m in seen[3]["messages"] if m.get("role") == "system")
