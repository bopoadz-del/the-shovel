"""Default UI (project-assistant) gets hat kernels + SSE scores when live.

FORK_HATS_ENABLED=1 is on the live service. Kernels bind at load (#585);
the floor scorer reads ``hat_signals`` on the chat stream (event + end).
"""
from __future__ import annotations

import asyncio

import pytest
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, patch
from tests._fixture_values import STUB_TOKEN_01

from app.agents.runtime import (
    Agent,
    _apply_hat_activation,
    hat_scores_sse,
    hat_turn_system_note,
    hats_activation_enabled,
)


PLANNING_ASK = "What is the critical path for this project?"
SAFETY_ASK = "Check the scaffold load and trench shoring for this excavation"
CONTRACTS_ASK = "Draft a variation order under FIDIC clause"


def _collect(gen: AsyncIterator[Dict[str, Any]]) -> List[Dict[str, Any]]:
    async def _run():
        return [event async for event in gen]
    return asyncio.run(_run())


def _pa() -> Agent:
    return Agent(
        name="project-assistant",
        description="pa",
        system_prompt="you are project-assistant",
        allowed_blocks=[],
    )


def _hat_activation_recorded(events: List[Dict[str, Any]]) -> bool:
    """True when the stream recorded hat activation the floor scorer can see."""
    for event in events:
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind in ("hat_signals", "hats", "hat_scores"):
            if event.get("hat_signals") or event.get("scores") or event.get("hats"):
                return True
        if kind == "end" and (event.get("hat_signals") or event.get("hats")):
            return True
    return False


def _stream_with_mock(monkeypatch, message: str):
    monkeypatch.setenv("FORK_HATS_ENABLED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    agent = _pa()
    mock = AsyncMock(return_value={
        "status": "success",
        "choice": {"message": {"content": "ok", "tool_calls": []}},
    })
    with patch.object(agent, "_call_llm", mock):
        return _collect(agent.chat_stream(user_message=message)), mock


@pytest.mark.parametrize("value", [None, "0"])
def test_hat_router_on_without_the_removed_flag(monkeypatch, value):
    """K-T0a: FORK_HATS_ENABLED is removed; unset or 0 still routes."""
    if value is None:
        monkeypatch.delenv("FORK_HATS_ENABLED", raising=False)
    else:
        monkeypatch.setenv("FORK_HATS_ENABLED", value)
    assert hats_activation_enabled() is True
    assert hat_scores_sse(PLANNING_ASK) is not None


def test_hat_scores_sse_when_flag_on(monkeypatch):
    monkeypatch.setenv("FORK_HATS_ENABLED", "1")
    assert hats_activation_enabled() is True
    evt = hat_scores_sse(PLANNING_ASK)
    assert evt is not None
    assert evt["type"] == "hat_signals"
    assert evt["hat_signals"]
    assert evt["scores"]
    assert "planning" in evt["scores"]
    assert evt["selected"]
    assert evt["selected"].startswith("fork.hat.")
    assert any(row["discipline"] == "planning" for row in evt["hat_signals"])


def test_hat_turn_note_names_the_winning_hat(monkeypatch):
    monkeypatch.setenv("FORK_HATS_ENABLED", "1")
    note = hat_turn_system_note(PLANNING_ASK)
    assert note is not None
    assert note["role"] == "system"
    assert "Active discipline hat this turn" in note["content"]
    assert "fork.hat." in note["content"]
    assert "planning=" in note["content"]


def test_apply_hat_activation_steers_project_assistant_only(monkeypatch):
    monkeypatch.setenv("FORK_HATS_ENABLED", "1")
    pa_msgs = [{"role": "user", "content": PLANNING_ASK}]
    evt = _apply_hat_activation(pa_msgs, PLANNING_ASK, "project-assistant")
    assert evt and evt["type"] == "hat_signals"
    assert evt["hat_signals"]
    assert pa_msgs[0]["role"] == "system"
    assert pa_msgs[-1]["role"] == "user"

    qs_msgs = [{"role": "user", "content": PLANNING_ASK}]
    _apply_hat_activation(qs_msgs, PLANNING_ASK, "quantity-surveyor")
    assert qs_msgs == [{"role": "user", "content": PLANNING_ASK}]


def test_project_assistant_stream_emits_hat_signals(monkeypatch):
    events, mock = _stream_with_mock(monkeypatch, PLANNING_ASK)
    types = [e["type"] for e in events]
    assert "start" in types
    assert _hat_activation_recorded(events), (
        "hats enabled but the stream path never recorded hat activation metadata"
    )
    assert "hat_signals" in types
    assert types.index("hat_signals") == types.index("start") + 1
    hat = next(e for e in events if e["type"] == "hat_signals")
    assert hat["hat_signals"]
    assert hat["enabled"] is True
    end = next(e for e in events if e["type"] == "end")
    assert end.get("hat_signals"), (
        "hats enabled but the SSE end event has no hat_signals"
    )
    sent = mock.await_args.args[0]
    assert any(
        m.get("role") == "system" and "Active discipline hat this turn" in (m.get("content") or "")
        for m in sent
    )


def test_safety_question_produces_nonempty_hat_signals(monkeypatch):
    events, _mock = _stream_with_mock(monkeypatch, SAFETY_ASK)
    assert _hat_activation_recorded(events), (
        "hats enabled but the stream path never recorded hat activation metadata"
    )
    hat = next(e for e in events if e["type"] == "hat_signals")
    assert hat["hat_signals"]
    assert any(
        row["discipline"] == "safety" and float(row["score"]) > 0
        for row in hat["hat_signals"]
    )
    selected = [row for row in hat["hat_signals"] if row.get("selected")]
    assert selected and selected[0]["id"] == "fork.hat.safety"
    end = next(e for e in events if e["type"] == "end")
    assert any(
        row.get("id") == "fork.hat.safety" and float(row.get("score") or 0) > 0
        for row in (end.get("hat_signals") or [])
    )


def test_contracts_question_produces_nonempty_hat_signals(monkeypatch):
    events, _mock = _stream_with_mock(monkeypatch, CONTRACTS_ASK)
    assert _hat_activation_recorded(events), (
        "hats enabled but the stream path never recorded hat activation metadata"
    )
    hat = next(e for e in events if e["type"] == "hat_signals")
    assert hat["hat_signals"]
    assert any(
        row["discipline"] == "contracts" and float(row["score"]) > 0
        for row in hat["hat_signals"]
    )
    selected = [row for row in hat["hat_signals"] if row.get("selected")]
    assert selected and selected[0]["id"] == "fork.hat.contracts"
    end = next(e for e in events if e["type"] == "end")
    assert any(
        row.get("id") == "fork.hat.contracts" and float(row.get("score") or 0) > 0
        for row in (end.get("hat_signals") or [])
    )


def test_a_crash_in_hat_scoring_does_not_take_down_the_chat(monkeypatch):
    """Hat scoring runs before the stream's first yield, outside the safety
    net. It is telemetry; a bug in it must cost the turn its scores, never
    the user their answer."""
    from app.agents import runtime

    def boom(_message):
        raise RuntimeError("planted: catalog blew up")

    monkeypatch.setattr(runtime, "hat_scores_sse", boom)
    # hat_turn_system_note calls the same function from inside the impl;
    # neutralise it so this test isolates the unguarded call site.
    monkeypatch.setattr(runtime, "hat_turn_system_note", lambda _m: None)

    events, _mock = _stream_with_mock(monkeypatch, PLANNING_ASK)
    types = [e["type"] for e in events]

    assert "end" in types, f"the stream died instead of answering: {types}"
    assert "error" not in types
    assert "hat_signals" not in types
    assert any(e["type"] == "token" for e in events)


def test_multi_hat_selection_marks_both_parts_and_nothing_else(monkeypatch):
    """`selected` must match the parts of a merged "<a>__<b>" id exactly."""
    from types import SimpleNamespace

    from app.agents import runtime

    monkeypatch.setenv("FORK_HATS_ENABLED", "1")
    merged = SimpleNamespace(
        id="fork.hat.planning__fork.hat.contracts", name="planning+contracts"
    )
    monkeypatch.setattr(
        "app.agents.activation.HatActivationAdapter.select_hat_for_message",
        lambda self, _m: merged,
    )
    payload = runtime.hat_scores_sse(PLANNING_ASK)
    chosen = {row["id"] for row in payload["hat_signals"] if row["selected"]}

    assert chosen == {"fork.hat.planning", "fork.hat.contracts"}
