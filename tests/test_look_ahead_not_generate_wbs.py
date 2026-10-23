"""look_ahead asks must not route or dispatch to generate_wbs.

Live Phase 2: a 14-day look-ahead from a synthetic / project programme was
classified as generate_wbs (schedule-like keywords + UNDERSTAND 'schedule'
→ WBS). The product built a WBS instead of calling look_ahead (or returning
look_ahead's honest missing-.xer error).

Locks:
  1. Keyword router / best_action prefer look_ahead over generate_wbs.
  2. select_agent_for_message does not send look-ahead to generate_wbs.
  3. understand_intent does not map look-ahead to workflow=schedule.
  4. generate_wbs itself refuses a look-ahead ask (container mapping).
  5. Forced-tool still names look_ahead when that tool is available.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import (
    Agent,
    _forced_specific_tool,
    select_agent_for_message,
)
from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.containers.construction import ConstructionContainer
from app.core.action_router import best_action, needs_planning
from app.core.predefined_reasoning import lookup_question_hijack
from tests.conftest import requires_construction_kit

# Collision phrasings that currently let generate_wbs / parse_primavera win:
# "construction schedule", "master schedule", "project schedule", "programme".
LOOK_AHEAD_ASKS = [
    "Produce a 14-day look-ahead from the synthetic programme",
    "Give me a 14-day look ahead from the project schedule",
    "Build a 2-week look-ahead from the construction schedule",
    "Create a look-ahead programme for the next 14 days",
    "Generate a 14 day lookahead from the master schedule",
    "Can you produce a 14-day look-ahead schedule from the xer?",
    "I need a rolling look ahead from the baseline schedule",
]

TRUE_WBS_ASKS = [
    "Create L2 schedule with 200 activities for the data center.",
    "generate a WBS for a 10-floor tower",
]


def _run(coro):
    return asyncio.run(coro)


def _make_agent(name: str) -> Agent:
    return Agent(
        name=name,
        description=f"{name} test stub",
        system_prompt="(test stub)",
        allowed_blocks=[],
    )


# ── Forced-tool (both boot modes; no kit required) ─────────────────────────


@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
def test_forced_tool_is_look_ahead_not_generate_wbs(message):
    msgs = [{"role": "user", "content": message}]
    available = {"generate_wbs", "look_ahead", "primavera_parser", "construction_calc"}
    assert _forced_specific_tool(msgs, available) == "look_ahead"
    assert _forced_specific_tool(msgs, {"generate_wbs", "primavera_parser"}) != "generate_wbs"


# ── Dynamic UNDERSTAND: LLM saying "schedule" must not steal look-ahead ────


@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
def test_understand_intent_does_not_map_look_ahead_to_generate_wbs(message, monkeypatch):
    from app.core import dynamic_reasoning as dr

    called = {"n": 0}

    async def fake_schedule(*a, **k):
        called["n"] += 1
        return {"workflow": "schedule", "mode": "produce", "params": {"target_count": 200}}

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent(message))
    assert out["action"] != "generate_wbs", (
        f"understand_intent mapped look-ahead {message!r} to generate_wbs; got {out}"
    )
    assert out["action"] == "look_ahead", out
    assert called["n"] == 0, "look-ahead must short-circuit before the schedule LLM hop"


def test_understand_intent_still_maps_real_wbs_produce(monkeypatch):
    from app.core import dynamic_reasoning as dr

    async def fake_schedule(*a, **k):
        return {"workflow": "schedule", "mode": "produce", "params": {"target_count": 200}}

    monkeypatch.setattr(dr, "complete_json", fake_schedule)
    out = _run(dr.understand_intent("Create L2 schedule with 200 activities for the data center."))
    assert out["action"] == "generate_wbs"
    assert out["deliverable"] is True


# ── generate_wbs container must not build a WBS for a look-ahead ask ───────


@pytest.mark.asyncio
@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
async def test_generate_wbs_refuses_look_ahead_ask(message):
    result = await ConstructionContainer().generate_wbs(
        {},
        {"brief": message, "user_message": message, "target_count": 50},
    )
    assert result.get("status") == "error", result
    err = (result.get("error") or "").lower()
    assert "look-ahead" in err or "look ahead" in err or "lookahead" in err
    assert not result.get("activities"), "generate_wbs must not emit a WBS for a look-ahead ask"


@pytest.mark.asyncio
async def test_generate_wbs_still_builds_real_wbs_brief():
    result = await ConstructionContainer().generate_wbs(
        {},
        {
            "brief": "generate a WBS for a 10-floor tower",
            "user_message": "generate a WBS for a 10-floor tower",
            "target_count": 40,
        },
    )
    assert result.get("status") == "success"
    assert result.get("activities")


# ── Orchestrator / agent gate (construction kit) ───────────────────────────


@requires_construction_kit
@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
def test_orchestrator_does_not_classify_look_ahead_as_generate_wbs(message):
    block = SmartOrchestratorBlock()
    result = _run(block.process({"user_message": message}))
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" not in actions, (
        f"generate_wbs must not match look-ahead {message!r}; got {matched}"
    )
    action, confidence = best_action(result)
    assert action != "generate_wbs", (action, confidence, matched)
    assert action == "look_ahead", (action, confidence, matched)
    assert needs_planning(action, confidence) is True


@requires_construction_kit
@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
def test_select_agent_does_not_route_look_ahead_to_generate_wbs(message, monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(message, pa))
        assert routing["action"] != "generate_wbs", routing
        assert routing["action"] == "look_ahead", routing
        assert routing["reason"] == "needs_planning"
        assert final.name == "heavy-reasoning", routing
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_l2_schedule_still_routes_to_generate_wbs(monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(TRUE_WBS_ASKS[0], pa))
        assert final.name == "heavy-reasoning", routing
        assert routing["action"] == "generate_wbs"
        assert routing["reason"] == "needs_planning"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@pytest.mark.parametrize("message", LOOK_AHEAD_ASKS)
def test_look_ahead_is_not_a_lookup_hijack(message):
    assert lookup_question_hijack(message, 0.9) is False
