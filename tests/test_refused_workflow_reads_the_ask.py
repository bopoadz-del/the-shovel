"""A workflow that refuses for a missing input that the user's own words
state is run again with it before any question is asked. Synthetic
workflow, scripted model."""
from __future__ import annotations

import pytest

from tests.lane5_core_support import answer_text, call, drive, install_driver_stream

_CARD = {"produces": "A widget register of the fitted widgets",
         "use_when": "The user asks to list the widgets fitted so far",
         "needs": "The widgets fitted",
         "inputs": {"widgets": "Widgets fitted (a list)"}}


@pytest.fixture
def runs(monkeypatch):
    from app.agents.driver import routes
    from app.core import predefined_reasoning

    install_driver_stream(monkeypatch)
    monkeypatch.setattr(routes, "catalogue", lambda: ["widget_register"])
    monkeypatch.setattr(routes, "cards", lambda: {"widget_register": _CARD})
    seen = []

    async def fake_run_workflow(action, ctx, session):
        params = ctx.get("params") or {}
        seen.append(params)
        if not params.get("widgets"):
            return {"handled": True, "status": "error", "answer": "No widgets supplied."}
        return {"handled": True, "status": "success",
                "answer": f"Widget register: {', '.join(params['widgets'])}."}

    monkeypatch.setattr(predefined_reasoning, "run_workflow", fake_run_workflow)
    return seen


def test_inputs_the_ask_states_are_filled_and_the_workflow_runs(monkeypatch, runs):
    events, _ = drive(monkeypatch, "Make the widget register for the lifts and the generators", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
        {"content": '{"widgets": ["lifts", "generators"]}'},          # the reading call
        {"content": "Widget register: lifts, generators."},
    ])
    assert runs[-1] == {"widgets": ["lifts", "generators"]}
    assert "Widget register: lifts, generators." in answer_text(events)


def test_nothing_stated_means_no_second_run(monkeypatch, runs):
    events, _ = drive(monkeypatch, "Make the widget register", [
        {"content": "", "tool_calls": [call("run_workflow", {"workflow": "widget_register"}, "w1")]},
        {"content": "{}"},                                             # the reading call: nothing stated
        {"content": "Which widgets?"},
        {"content": "Which widgets?"},
    ])
    assert len(runs) == 1
    assert "widget schedule" not in answer_text(events)
