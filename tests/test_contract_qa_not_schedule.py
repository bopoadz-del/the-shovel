"""Contract Data TfC / milestone Q&A must not dispatch generate_wbs.

Live UI-PHYS: pinned project-assistant questions about Time for Completion
and Milestone 5 TfC were stolen to the predefined schedule workflow
("Schedule built: N activities…"). Those are RAG lookups, not WBS generate.
Catalog phrasing is the sanitized fixture set — no live client figures.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import (
    Agent,
    _forced_specific_tool,
    _user_intent_requires_tool,
    select_agent_for_message,
)
from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.core.action_router import best_action, needs_planning
from app.core.contract_lookup_intent import message_is_contract_data_lookup
from app.core.predefined_reasoning import lookup_question_hijack
from tests.conftest import requires_construction_kit

TFC_WHOLE_WORKS = "What is the Time for Completion for the whole of the Works?"
MILESTONE_5_TFC = (
    "How many Milestones are there and what is the Time for Completion "
    "for Milestone 5?"
)
MILESTONE_5_BARE = "Milestone 5 Time for Completion"
SCHEDULE_10 = "What does Schedule 10 of the contract contain?"
SCHEDULE_10_BARE = "What is Schedule 10?"
SCHEDULE_9_VOLUMES = "What does Schedule 9 of the contract volumes cover?"
PCG_VALUE = "What is the value of the Parent Company Guarantee?"
COMMENCEMENT_DATE = "What is the Commencement Date of the contract?"

CONTRACT_LOOKUP_ASKS = [
    TFC_WHOLE_WORKS, MILESTONE_5_TFC, MILESTONE_5_BARE,
    SCHEDULE_10, SCHEDULE_10_BARE, SCHEDULE_9_VOLUMES,
    PCG_VALUE, COMMENCEMENT_DATE,
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


@pytest.mark.parametrize(
    "message",
    CONTRACT_LOOKUP_ASKS,
)
def test_tfc_and_milestone_questions_are_contract_data_lookups(message):
    assert message_is_contract_data_lookup(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "Create L2 schedule with 200 activities for the data center.",
        "generate a WBS for a 10-floor tower",
        "extract the key milestones from the project programme with their dates",
        "give me a milestone report - what are the major completion dates?",
        "parse the xer for milestones",
        "Draft a claim for delay damages on Milestone 1",
        "Run a forensic delay analysis for the delay damages exposure",
        "Prepare an EOT claim for the 30-day delay to Milestone 1",
    ],
)
def test_real_schedule_asks_are_not_contract_data_lookups(message):
    assert message_is_contract_data_lookup(message) is False


@pytest.mark.parametrize("message", CONTRACT_LOOKUP_ASKS)
def test_tfc_does_not_force_wbs_or_primavera_tool(message):
    msgs = [{"role": "user", "content": message}]
    available = {"generate_wbs", "primavera_parser", "construction_calc"}
    assert _forced_specific_tool(msgs, available) is None
    assert _user_intent_requires_tool(msgs) is False


@pytest.mark.parametrize("message", CONTRACT_LOOKUP_ASKS)
def test_lookup_hijack_blocks_predefined_even_at_high_confidence(message):
    assert lookup_question_hijack(message, 0.9) is True
    assert lookup_question_hijack(message, 0.2) is True


@requires_construction_kit
@pytest.mark.parametrize("message", CONTRACT_LOOKUP_ASKS)
def test_orchestrator_does_not_classify_tfc_as_generate_wbs(message):
    block = SmartOrchestratorBlock()
    result = _run(block.process({"user_message": message}))
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "generate_wbs" not in actions, (
        f"generate_wbs must not match Contract Data Q&A {message!r}; got {matched}"
    )
    assert "parse_primavera_schedule" not in actions, (
        f"parse_primavera_schedule must not match {message!r}; got {matched}"
    )
    action, confidence = best_action(result)
    assert action not in ("generate_wbs", "parse_primavera_schedule")
    assert needs_planning(action, confidence) is False


@requires_construction_kit
@pytest.mark.parametrize("message", [TFC_WHOLE_WORKS, MILESTONE_5_TFC, SCHEDULE_10])
def test_select_agent_keeps_tfc_on_project_assistant(message, monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(select_agent_for_message(message, pa))
        assert final is pa, routing
        assert routing["final"] == "project-assistant"
        assert routing["action"] is None
        assert routing["reason"] == "contract_data_lookup"
        assert routing["action"] != "generate_wbs"
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
        final, routing = _run(select_agent_for_message(
            "Create L2 schedule with 200 activities for the data center.",
            pa,
        ))
        assert final.name == "heavy-reasoning", routing
        assert routing["action"] == "generate_wbs"
        assert routing["reason"] == "needs_planning"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@pytest.mark.asyncio
async def test_generate_wbs_tool_refuses_contract_tfc_lookup():
    agent = _make_agent("project-assistant")
    rec = await runtime_module.Agent.__dict__["_run_tool_call"](
        agent,
        {"function": {"name": "generate_wbs", "arguments": json.dumps({"brief": "x"})}},
        user_message=TFC_WHOLE_WORKS,
    )
    assert rec["ok"] is False
    assert rec["name"] == "generate_wbs"
    err = (rec.get("result") or {}).get("error") or ""
    assert "contract data" in err.lower()
    assert "wbs" in err.lower()


def test_understand_intent_skips_llm_for_tfc(monkeypatch):
    from app.core import dynamic_reasoning as dr

    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("LLM must not run for Contract Data TfC lookup")

    monkeypatch.setattr(dr, "complete_json", boom)
    out = _run(dr.understand_intent(TFC_WHOLE_WORKS))
    assert out["workflow"] == "none"
    assert out["action"] is None
    assert called["n"] == 0
