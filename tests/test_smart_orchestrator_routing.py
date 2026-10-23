"""PR #78 — smart_orchestrator routing gate in app/agents/runtime.py.

Pre-PR-#78, real user chat traffic bypassed smart_orchestrator entirely
(the React UI calls /v1/agents/project-assistant/chat/stream which lands
directly on Agent.chat_stream). PR #78 adds ``select_agent_for_message``
as a gate the agents-router calls before dispatch so generative-intent
queries redirect to the heavy-reasoning agent.

These tests exercise the gate directly (without the FastAPI router) so
the contract is locked even when the router's auth / project-ownership
plumbing changes around it.

smart_orchestrator ships with the construction kit (see
_KIT_BLOCK_SPECS in app/core/domain_kit_loader.py), so every test that
needs the real classifier registered is marked
``@requires_construction_kit`` — on the virgin CI profile they skip,
exactly like the rest of the kit-gated suite. The pass-through /
kill-switch / crash tests stay unmarked: they stub the block lookup or
short-circuit before it, so they must hold on a virgin boot too.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agents import runtime as runtime_module
from app.agents.runtime import Agent, AGENT_REGISTRY, select_agent_for_message
from tests.conftest import requires_construction_kit


def _run(coro):
    return asyncio.run(coro)


def _make_agent(name: str) -> Agent:
    """Minimal Agent instance for routing tests — we never call .chat() on
    these, only run them through the router gate."""
    return Agent(
        name=name,
        description=f"{name} test stub",
        system_prompt="(test stub)",
        allowed_blocks=[],
    )


@pytest.fixture
def project_assistant():
    return _make_agent("project-assistant")


@pytest.fixture
def heavy_reasoning():
    return _make_agent("heavy-reasoning")


@pytest.fixture
def registry_with_heavy(monkeypatch, project_assistant, heavy_reasoning):
    """Populate AGENT_REGISTRY with the two agents the gate consults so
    the redirect target exists. Also resets the smart_orchestrator block
    cache between tests so kill-switch checks observe a fresh state."""
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY["project-assistant"] = project_assistant
    AGENT_REGISTRY["heavy-reasoning"] = heavy_reasoning
    yield AGENT_REGISTRY
    AGENT_REGISTRY.clear()


# ── Generative-intent redirect ─────────────────────────────────────────────


@requires_construction_kit
def test_generative_intent_redirects_to_heavy_reasoning(registry_with_heavy, project_assistant):
    """A clear generative intent ("create L2 schedule with 200 activities")
    should classify above the routing threshold and re-target the
    project-assistant request to heavy-reasoning."""
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities for the data center.",
        project_assistant,
    ))
    assert final.name == "heavy-reasoning", routing
    assert routing["requested"] == "project-assistant"
    assert routing["final"] == "heavy-reasoning"
    assert routing["action"] == "generate_wbs"
    assert routing["confidence"] >= 0.4
    assert routing["reason"] == "needs_planning"


@requires_construction_kit
def test_drawing_qto_redirects_post_pr76(registry_with_heavy, project_assistant):
    """PR #76 added drawing_qto to GENERATIVE_INTENTS. Verify the routing
    gate honours that — a DXF/blueprint read should hit heavy-reasoning.
    Phrase deliberately avoids 'extract quantities' (which would route to
    the higher-scoring extract_quantities action, NOT in GENERATIVE_INTENTS)
    and instead uses drawing-specific keywords (dxf + blueprint + autocad)
    so drawing_qto is the unambiguous top match."""
    final, routing = _run(select_agent_for_message(
        "Read the DXF blueprint exported from AutoCAD and tell me the sheet layout.",
        project_assistant,
    ))
    assert final.name == "heavy-reasoning", routing
    assert routing["action"] == "drawing_qto"
    assert routing["reason"] == "needs_planning"


# ── Pass-through cases (no redirect, classifier metadata still returned) ─────


@requires_construction_kit
def test_small_talk_stays_on_requested_agent(registry_with_heavy, project_assistant):
    """Conversational openers never reach a generative intent — confidence
    is zero, the gate stays out of the way."""
    final, routing = _run(select_agent_for_message("hi there", project_assistant))
    assert final is project_assistant
    assert routing["final"] == "project-assistant"
    assert routing["reason"] in ("below_routing_gate", "no-op")
    assert routing["confidence"] < 0.4


@requires_construction_kit
def test_non_generative_action_stays_on_requested_agent(registry_with_heavy, project_assistant):
    """``boq_process`` is a routing action but NOT a GENERATIVE_INTENT
    (the operator's design: BOQ Q&A is RAG-based, not tool-dispatch). It
    should classify with confidence ≥ 0.4 but still pass through."""
    final, routing = _run(select_agent_for_message(
        "What is the BOQ total for the demolition section?",
        project_assistant,
    ))
    assert final is project_assistant
    # The classifier might match BOQ keywords; whichever action wins, the
    # gate must not redirect because the action isn't generative.
    assert routing["reason"] in ("below_routing_gate", "no-op")


@requires_construction_kit
def test_no_double_redirect_when_already_heavy(registry_with_heavy, heavy_reasoning):
    """When the caller already requested heavy-reasoning, the gate must
    pass through even on a generative intent — re-routing to itself
    would be a no-op at best and a confusing UX event at worst."""
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        heavy_reasoning,
    ))
    assert final is heavy_reasoning
    assert routing["final"] == "heavy-reasoning"
    assert routing["reason"] == "already_heavy_reasoning"
    # The classifier still ran and surfaced its decision so observability
    # is intact even on pass-through.
    assert routing["action"] == "generate_wbs"


def test_empty_message_passes_through(registry_with_heavy, project_assistant):
    final, routing = _run(select_agent_for_message("", project_assistant))
    assert final is project_assistant
    assert routing["reason"] == "empty_message"


def test_whitespace_message_passes_through(registry_with_heavy, project_assistant):
    final, routing = _run(select_agent_for_message("   \n\t  ", project_assistant))
    assert final is project_assistant
    assert routing["reason"] == "empty_message"


# ── Kill-switch ─────────────────────────────────────────────────────────────


def test_kill_switch_disables_routing(registry_with_heavy, project_assistant, monkeypatch):
    """``SMART_ORCH_ROUTING_DISABLED=true`` is the prod rollback knob —
    no classification runs, the requested agent passes through.
    Critical: even a textbook generative intent must not redirect when
    the kill-switch is on."""
    monkeypatch.setenv("SMART_ORCH_ROUTING_DISABLED", "true")
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        project_assistant,
    ))
    assert final is project_assistant
    assert routing["reason"] == "routing_disabled_env"
    # We deliberately do NOT populate action/confidence on the kill-switch
    # path so callers can't accidentally act on stale data.
    assert routing["action"] is None
    assert routing["confidence"] == 0.0


@pytest.mark.parametrize("flag", ["1", "true", "yes", "TRUE", "YES"])
def test_kill_switch_truthy_variants(registry_with_heavy, project_assistant, monkeypatch, flag):
    monkeypatch.setenv("SMART_ORCH_ROUTING_DISABLED", flag)
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        project_assistant,
    ))
    assert final is project_assistant
    assert routing["reason"] == "routing_disabled_env"


@requires_construction_kit
def test_kill_switch_unset_means_enabled(registry_with_heavy, project_assistant, monkeypatch):
    monkeypatch.delenv("SMART_ORCH_ROUTING_DISABLED", raising=False)
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        project_assistant,
    ))
    assert final.name == "heavy-reasoning"


# ── Heavy-reasoning missing from registry ───────────────────────────────────


@requires_construction_kit
def test_redirect_target_missing_passes_through(monkeypatch, project_assistant):
    """If the operator removes heavy-reasoning from AGENT_REGISTRY (e.g. a
    minimal deploy), the gate must NOT crash and must NOT redirect to a
    None target — it surfaces the gap in `reason` and passes through."""
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    AGENT_REGISTRY.clear()
    AGENT_REGISTRY["project-assistant"] = project_assistant
    try:
        final, routing = _run(select_agent_for_message(
            "Create L2 schedule with 200 activities.",
            project_assistant,
        ))
        assert final is project_assistant
        assert routing["reason"] == "heavy_reasoning_not_registered"
    finally:
        AGENT_REGISTRY.clear()


# ── smart_orchestrator not registered ───────────────────────────────────────


def test_smart_orchestrator_missing_passes_through(monkeypatch, project_assistant):
    """Without the construction kit loaded, smart_orchestrator isn't in
    BLOCK_REGISTRY. The gate must pass through cleanly — no AttributeError,
    no KeyError, no crash."""
    monkeypatch.setattr(runtime_module, "_SMART_ORCH_BLOCK_CACHE", None)
    monkeypatch.setattr(
        runtime_module, "_get_smart_orchestrator_block", lambda: None,
    )
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        project_assistant,
    ))
    assert final is project_assistant
    assert routing["reason"] == "smart_orchestrator_not_registered"


# ── Classifier crash ────────────────────────────────────────────────────────


def test_classifier_crash_passes_through(registry_with_heavy, project_assistant, monkeypatch):
    """If smart_orchestrator.process() raises, routing must never break
    the user's chat. Pass through with the error surfaced for logs."""

    class _Boom:
        async def process(self, *a, **k):
            raise RuntimeError("classifier exploded")

    monkeypatch.setattr(
        runtime_module, "_get_smart_orchestrator_block", lambda: _Boom(),
    )
    final, routing = _run(select_agent_for_message(
        "Create L2 schedule with 200 activities.",
        project_assistant,
    ))
    assert final is project_assistant
    assert routing["reason"] == "classifier_error"
    assert "classifier exploded" in routing["error"]
