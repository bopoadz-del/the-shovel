"""Tests for the smart-orchestrator → chat domain-hint integration (Fix #1).

The chat router now consults SmartOrchestratorBlock before invoking the chat
block. When the orchestrator finds a high-confidence intent match, a short
domain-aware sentence is prepended to the user message so the LLM answers
inside the right frame. When no match (or only weak matches), the message
flows through unchanged.
"""

from __future__ import annotations

import pytest

from app.routers.chat import _with_domain_hint
from tests.conftest import requires_construction_kit


@requires_construction_kit
@pytest.mark.asyncio
async def test_hint_prepended_when_message_clearly_about_boq():
    """A message with strong BOQ keywords should get a Bill-of-Quantities hint."""
    msg = "Process this Bill of Quantities and tell me the totals."
    out = await _with_domain_hint(msg)
    assert out != msg
    assert "[Context for your answer:" in out
    assert "Bill of Quantities" in out
    # The original message must still be present.
    assert msg in out


@pytest.mark.asyncio
async def test_no_hint_when_message_is_pure_smalltalk():
    """'Hello there' has no orchestrator match — message must pass through."""
    msg = "Hello, what's up?"
    out = await _with_domain_hint(msg)
    assert out == msg


@requires_construction_kit
@pytest.mark.asyncio
async def test_hint_for_specification_keywords():
    msg = "Check this specification against ACI 318 compliance."
    out = await _with_domain_hint(msg)
    assert "[Context for your answer:" in out
    assert "specification" in out.lower()


@pytest.mark.asyncio
async def test_domain_hint_failure_does_not_break_chat(monkeypatch):
    """If the orchestrator block raises, the message must still flow through."""
    from app.dependencies import block_instances

    # Inject a broken orchestrator so _with_domain_hint hits its except branch.
    class BoomOrchestrator:
        async def process(self, *args, **kwargs):
            raise RuntimeError("orchestrator broken")

    block_instances["smart_orchestrator"] = BoomOrchestrator()
    try:
        msg = "Process the Bill of Quantities."
        out = await _with_domain_hint(msg)
        assert out == msg  # unchanged
    finally:
        block_instances.pop("smart_orchestrator", None)


# ─────────────────────────────────────────────────────────────────────────────
# Intent classification → heavy-reasoning routing
# ─────────────────────────────────────────────────────────────────────────────
#
# The chat router's selective routing branch sends generative-intent requests
# (≥0.5 confidence, action in GENERATIVE_INTENTS) to the heavy-reasoning agent
# instead of the single-shot chat block. Q&A / smalltalk stays on the fast
# path. These tests pin both behaviors so a future orchestrator-tuning pass
# can't silently route "what is the total cost" through the slow agent loop.

from app.core.action_router import (
    GENERATIVE_INTENTS,
    GENERATIVE_ROUTING_THRESHOLD,
    ROUTING_CONFIDENCE_THRESHOLD,
    needs_planning,
)
from app.routers.chat import _classify_intent


def test_needs_planning_requires_action():
    assert needs_planning(None, 0.9) is False


def test_needs_planning_requires_confidence_above_threshold():
    # generate_wbs is a generative intent; it uses the relaxed gate-2 threshold.
    assert needs_planning("generate_wbs", GENERATIVE_ROUTING_THRESHOLD - 0.01) is False
    assert needs_planning("generate_wbs", GENERATIVE_ROUTING_THRESHOLD) is True
    # Non-generative actions still require the global threshold regardless of score.
    assert needs_planning("health_check", ROUTING_CONFIDENCE_THRESHOLD) is False


def test_needs_planning_requires_generative_action():
    # 'health_check' has a hint but is NOT a generative intent — must not route.
    assert needs_planning("health_check", 0.99) is False


def test_generative_intents_includes_wbs_and_workflow():
    # Spec-anchored: these are the keystone generative intents that the plan
    # explicitly approved. Lock them in so a typo can't quietly disable WBS.
    assert "generate_wbs" in GENERATIVE_INTENTS
    assert "intelligent_workflow" in GENERATIVE_INTENTS


@requires_construction_kit
@pytest.mark.asyncio
async def test_classify_wbs_message_routes_to_heavy_reasoning():
    """The user's 'create a 200 activities schedule' should classify as a
    generative intent above the routing threshold — i.e. heavy reasoning."""
    action, confidence = await _classify_intent(
        "Create a 200 activities schedule for this RFP."
    )
    # Orchestrator should pick generate_wbs (or another generative action) and
    # score it ≥ ROUTING_CONFIDENCE_THRESHOLD.
    assert action is not None
    assert needs_planning(action, confidence), (
        f"expected routing path for WBS phrasing, got action={action} confidence={confidence}"
    )


@pytest.mark.asyncio
async def test_classify_total_cost_stays_on_fast_path():
    """Plain Q&A like 'what is the total cost' should NOT route to the agent
    — either no classification or below threshold."""
    action, confidence = await _classify_intent("What is the total cost?")
    assert not needs_planning(action, confidence), (
        f"plain Q&A unexpectedly routed: action={action} confidence={confidence}"
    )


@pytest.mark.asyncio
async def test_classify_smalltalk_stays_on_fast_path():
    action, confidence = await _classify_intent("Hello, how are you today?")
    assert not needs_planning(action, confidence)


@pytest.mark.asyncio
async def test_classify_intent_robust_to_empty_prompt():
    action, confidence = await _classify_intent("")
    assert action is None
    assert confidence == 0.0
    assert not needs_planning(action, confidence)
