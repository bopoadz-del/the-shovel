"""S14 — EOT notice-period Q&A must not route to claims_builder.

Battery F-BAT-B `fresh_eot_notice_period` on live 0d9fd23 (#560):
"Within how many days must the Contractor give notice of an EOT claim
on this project?" was routed to claims_builder (needs delay_events /
schedule+baseline) on all 3 attempts. The answer is Contract Data
(fixture contract-eot-notice-period → 21 days), not a claim draft.

Shape-invariance: "EOT" / "claim" / "notice" in a days-to-give-notice
question is a lookup. Claim-build verbs + delay events stay on
claims_builder.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import (
    Agent,
    _forced_specific_tool,
    _message_wants_delay_claim,
    _user_intent_requires_tool,
    select_agent_for_message,
)
from app.blocks.smart_orchestrator import SmartOrchestratorBlock
from app.core.action_router import best_action, needs_planning
from app.core.contract_lookup_intent import (
    CONTRACT_LOOKUP_BLOCKED_ACTIONS,
    message_is_contract_data_lookup,
    message_is_eot_notice_period_lookup,
)
from app.core.delay_advice import DelayKind, claims_builder_permitted, classify_delay
from app.core.predefined_reasoning import lookup_question_hijack
from tests.conftest import requires_construction_kit

# Exact golden-set / F-BAT-B wording. Do not paraphrase.
FRESH_EOT_NOTICE_PERIOD = (
    "Within how many days must the Contractor give notice of an EOT "
    "claim on this project?"
)

NOTICE_PERIOD_ASKS = [
    FRESH_EOT_NOTICE_PERIOD,
    "What is the notice period for EOT claims?",
    "the notice period for EOT claims?",
    "How many days to give notice of an extension of time claim?",
    "What is the contractual EOT notice period on this project?",
    "What is the time-bar for EOT notices under this contract?",
]

CLAIM_BUILD_ASKS = [
    "Help me build a delay claim for the electrical package.",
    "Prepare an EOT claim for the 30-day delay to Milestone 1",
    "Draft a delay claim notice under Contract Data clause 8.8.1 "
    "for late access to Milestone 1.",
    "write the eot claim letter from the delay register",
    "build a loss and expense claim for the 3-week utility diversion delay",
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


@pytest.mark.parametrize("message", NOTICE_PERIOD_ASKS)
def test_s14_notice_period_asks_are_contract_lookups(message):
    assert message_is_eot_notice_period_lookup(message) is True
    assert message_is_contract_data_lookup(message) is True


@pytest.mark.parametrize("message", CLAIM_BUILD_ASKS)
def test_s14_claim_build_asks_are_not_lookups(message):
    assert message_is_eot_notice_period_lookup(message) is False
    assert message_is_contract_data_lookup(message) is False


def test_s14_battery_prompt_is_not_a_claim_build():
    assert classify_delay(FRESH_EOT_NOTICE_PERIOD) == DelayKind.NONE
    assert claims_builder_permitted(FRESH_EOT_NOTICE_PERIOD) is False
    assert _message_wants_delay_claim(FRESH_EOT_NOTICE_PERIOD) is False


@pytest.mark.parametrize("message", CLAIM_BUILD_ASKS[:3])
def test_s14_claim_build_still_permits_claims_builder(message):
    assert claims_builder_permitted(message) is True
    assert _message_wants_delay_claim(message) is True


@pytest.mark.parametrize("message", NOTICE_PERIOD_ASKS)
def test_s14_lookup_hijack_even_at_high_confidence(message):
    assert lookup_question_hijack(message, 0.9) is True
    assert lookup_question_hijack(message, 0.2) is True


def test_s14_claims_builder_stays_blocked_on_contract_lookup():
    assert "claims_builder" in CONTRACT_LOOKUP_BLOCKED_ACTIONS
    assert "forensic_delay_analysis" in CONTRACT_LOOKUP_BLOCKED_ACTIONS


@pytest.mark.parametrize("message", NOTICE_PERIOD_ASKS)
def test_s14_does_not_force_a_claim_tool(message):
    msgs = [{"role": "user", "content": message}]
    available = {"claims_builder", "forensic_delay_analysis", "generate_wbs"}
    assert _forced_specific_tool(msgs, available) is None
    assert _user_intent_requires_tool(msgs) is False


@requires_construction_kit
@pytest.mark.parametrize("message", NOTICE_PERIOD_ASKS)
def test_s14_orchestrator_does_not_classify_as_claims_builder(message):
    block = SmartOrchestratorBlock()
    result = _run(block.process({"user_message": message}))
    matched = result.get("matched_actions") or []
    actions = [m["action"] for m in matched]
    assert "claims_builder" not in actions, (
        f"claims_builder must not match EOT notice-period Q&A {message!r}; "
        f"got {matched}"
    )
    assert "forensic_delay_analysis" not in actions, (
        f"forensic_delay_analysis must not match {message!r}; got {matched}"
    )
    action, confidence = best_action(result)
    assert action != "claims_builder"
    assert needs_planning(action, confidence) is False or action not in {
        "claims_builder",
        "forensic_delay_analysis",
    }


@requires_construction_kit
def test_s14_select_agent_keeps_battery_prompt_on_project_assistant(monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(
            select_agent_for_message(FRESH_EOT_NOTICE_PERIOD, pa)
        )
        assert final is pa, routing
        assert routing["final"] == "project-assistant"
        assert routing["action"] is None
        assert routing["reason"] == "contract_data_lookup"
        assert routing["action"] != "claims_builder"
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
def test_s14_prepare_eot_claim_still_routes_to_claims_builder(monkeypatch):
    pa = _make_agent("project-assistant")
    heavy = _make_agent("heavy-reasoning")
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    runtime_module.AGENT_REGISTRY.clear()
    runtime_module.AGENT_REGISTRY["project-assistant"] = pa
    runtime_module.AGENT_REGISTRY["heavy-reasoning"] = heavy
    try:
        final, routing = _run(
            select_agent_for_message(
                "Prepare an EOT claim for the 30-day delay to Milestone 1",
                pa,
            )
        )
        assert routing["action"] == "claims_builder", routing
        assert routing["reason"] == "needs_planning", routing
        assert final.name == "heavy-reasoning", routing
    finally:
        runtime_module.AGENT_REGISTRY.clear()


@requires_construction_kit
@pytest.mark.asyncio
async def test_s14_claims_builder_refuses_notice_period_without_delay_events():
    from app.containers.construction import ConstructionContainer

    result = await ConstructionContainer().claims_builder(
        {"user_message": FRESH_EOT_NOTICE_PERIOD},
        {"user_message": FRESH_EOT_NOTICE_PERIOD},
    )
    assert result["status"] == "error"
    err = (result.get("error") or "").lower()
    assert "contract data" in err
    assert "delay_events" not in err or "do not require" in err
    assert "quantum" not in result and "total_claim" not in result


def test_s14_understand_intent_skips_llm_for_notice_period(monkeypatch):
    from app.core import dynamic_reasoning as dr

    called = {"n": 0}

    async def boom(*a, **k):
        called["n"] += 1
        raise AssertionError("LLM must not run for EOT notice-period lookup")

    monkeypatch.setattr(dr, "complete_json", boom)
    out = _run(dr.understand_intent(FRESH_EOT_NOTICE_PERIOD))
    assert out["workflow"] == "none"
    assert out["action"] is None
    assert called["n"] == 0
