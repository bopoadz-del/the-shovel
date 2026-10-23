"""CORE driver dispatch. A turn goes through the driver. The assertions are
what that turn selected and dispatched: the tool it ran, the workflow it
ran, and the answer it streamed."""
from __future__ import annotations

import pytest

from app.core.predefined_reasoning import DELIVER_REGISTRY, WORKFLOW_REGISTRY
from app.schemas.execution_plan import ExecutionPlan
from tests.lane5_core_support import (
    Param, agent, answer_text, call, drive, no_retrieval, registered, run_driver,
)


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _dispatched(events):
    calls = [event.get("tool") for event in events if event.get("type") == "tool_call"]
    end = events[-1] if events else {}
    return calls, list(end.get("tools") or [])


def _plan(_context):
    return ExecutionPlan(steps=[])


# CORE 10, fixed by "a stated percent is not divided by 100 twice".
def test_a_percent_outside_0_to_100_is_rejected(monkeypatch):
    name = "lane5_hold_gate"

    def fn(holdback_percent: float) -> dict:
        return {"value": holdback_percent}

    user = "calculate the qxnet hold gate holdback 150%"
    with registered(name, fn, "Qxnet hold gate", {
        "holdback_percent": Param("fraction", lo=0, hi=1, label="holdback"),
    }, {"value": "fraction"}):
        events, seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "text": user,
            }, "c1")]},
            {"content": "noted"},
        ])
    calls, tools = _dispatched(events)
    assert calls == ["construction_calc"], events
    assert tools == ["construction_calc"], events
    from tests.lane5_core_support import tool_envelope

    env = tool_envelope(seen)
    assert env.get("status") != "success", env
    assert env.get("needs_input") is True
    assert "0.015" not in answer_text(events)


@pytest.mark.xfail(strict=True, reason="CORE: 12 a workflow answer credits its display name and is not the ask")
def test_a_workflow_answer_credits_its_display_name_and_is_not_the_ask(monkeypatch):
    user = "Prepare the panel ledger."

    def deliver(_session, _deliverable, _message):
        return "The stored total is 17."

    deliver.display_name = "Panel ledger"
    saved = WORKFLOW_REGISTRY.get("lane5_panel_ledger")
    saved_d = DELIVER_REGISTRY.get("lane5_panel_ledger")
    WORKFLOW_REGISTRY["lane5_panel_ledger"] = _plan
    DELIVER_REGISTRY["lane5_panel_ledger"] = deliver
    try:
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("run_workflow", {
                "workflow": "lane5_panel_ledger", "deliverable": False,
            }, "c1")]},
            {"content": "The stored total is 17."},
        ])
    finally:
        if saved is None:
            WORKFLOW_REGISTRY.pop("lane5_panel_ledger", None)
        else:
            WORKFLOW_REGISTRY["lane5_panel_ledger"] = saved
        if saved_d is None:
            DELIVER_REGISTRY.pop("lane5_panel_ledger", None)
        else:
            DELIVER_REGISTRY["lane5_panel_ledger"] = saved_d
    calls, tools = _dispatched(events)
    assert calls == ["run_workflow"], events
    assert tools == ["run_workflow:lane5_panel_ledger"], events
    text = answer_text(events)
    assert text.strip() != user
    assert "Panel ledger" in text
    assert "lane5_panel_ledger" not in text


@pytest.mark.xfail(strict=True, reason="CORE: 23 a tool error shows raw parameter names")
def test_a_tool_error_uses_the_display_label(monkeypatch):
    name = "lane5_mode_label"

    def fn(alpha_m: float) -> dict:
        return {"value": alpha_m}

    user = "calculate the qxnet label gate"
    with registered(name, fn, "Qxnet label gate", {
        "alpha_m": Param("m", lo=0, hi=1e6, label="clear span"),
    }, {"value": "m"}):
        events, seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name,
            }, "c1")]},
            {"content": "noted"},
        ])
    calls, tools = _dispatched(events)
    assert calls == ["construction_calc"], events
    assert tools == ["construction_calc"], events
    from tests.lane5_core_support import tool_envelope

    env = tool_envelope(seen)
    blob = str(env.get("error") or env.get("question") or "") + answer_text(events)
    assert "clear span" in blob
    assert "alpha_m" not in blob


def test_an_empty_driver_answer_is_the_calculator_prose(monkeypatch):
    name = "lane5_mode_recover"

    def fn(span_m: float, breadth_m: float) -> dict:
        return {"value": span_m * breadth_m}

    user = "calculate the qxnet kestrel total span 4 m breadth 3 m"
    with registered(name, fn, "Qxnet kestrel total", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
        "breadth_m": Param("m", lo=0, hi=1e6, label="breadth"),
    }, {"value": "m2"}):
        events, _seen = drive(monkeypatch, user, [
            {"content": "", "tool_calls": [call("construction_calc", {
                "calculation": name, "span_m": 4, "breadth_m": 3,
            }, "c1")]},
            {"content": ""},
        ])
    calls, tools = _dispatched(events)
    assert calls == ["construction_calc"], events
    assert tools == ["construction_calc"], events
    text = answer_text(events)
    assert '"status"' not in text
    assert not text.lstrip().startswith("{")
    assert "12" in text


@pytest.mark.xfail(strict=True, reason="CORE: 25 a missing model key still dispatches a finished calculator")
def test_a_missing_model_key_still_dispatches_a_finished_calculator(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    name = "lane5_mode_nokey"

    def fn(span_m: float, breadth_m: float) -> dict:
        return {"value": span_m * breadth_m}

    user = "calculate the qxnet kestrel total span 4 m breadth 3 m"
    no_retrieval(monkeypatch)
    with registered(name, fn, "Qxnet kestrel total", {
        "span_m": Param("m", lo=0, hi=1e6, label="span"),
        "breadth_m": Param("m", lo=0, hi=1e6, label="breadth"),
    }, {"value": "m2"}):
        events = run_driver(agent(), user_message=user)
    calls, tools = _dispatched(events)
    assert calls == ["construction_calc"], events
    assert tools == ["construction_calc"], events
    assert not [event for event in events if event.get("type") == "error"]
    assert "12" in answer_text(events)
