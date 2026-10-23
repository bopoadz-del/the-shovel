"""Tests for agent-runtime capability tweaks — Phase C4 · Stream C.

Covers:
- MAX_TOOL_ITERATIONS raised to >= 12
- _run_tool_call returns ok=False + hint on malformed JSON args
- _run_tool_call returns ok=False + hint on unknown block name
"""

import asyncio

import pytest

from app.agents.runtime import Agent, MAX_TOOL_ITERATIONS


# ── MAX_TOOL_ITERATIONS ───────────────────────────────────────────────────────

def test_max_tool_iterations_at_least_12():
    assert MAX_TOOL_ITERATIONS >= 12


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_agent(allowed_blocks=None):
    return Agent(
        name="test-agent",
        description="Test agent for unit tests",
        system_prompt="You are a test agent.",
        allowed_blocks=allowed_blocks or [],
    )


def _run(coro):
    # asyncio.run isolates each call in its own loop, so this stays robust
    # even when pytest-asyncio has already closed the main-thread loop
    # while running async tests earlier in the suite. The previous
    # get_event_loop() form raised "no current event loop" under
    # pytest-asyncio >= 1.x after that teardown.
    return asyncio.run(coro)


# ── malformed JSON args ───────────────────────────────────────────────────────

def test_run_tool_call_malformed_json_returns_hint():
    agent = _make_agent()
    tool_call = {
        "id": "call-1",
        "function": {
            "name": "some_block",
            "arguments": "{ not valid json !!!",
        },
    }
    result = _run(agent._run_tool_call(tool_call))
    assert result["ok"] is False
    assert "hint" in result["result"]
    assert len(result["result"]["hint"]) > 0


# ── unknown block name ────────────────────────────────────────────────────────

def test_run_tool_call_unknown_block_returns_hint():
    agent = _make_agent(allowed_blocks=["nonexistent_block_xyz"])
    tool_call = {
        "id": "call-2",
        "function": {
            "name": "nonexistent_block_xyz",
            "arguments": "{}",
        },
    }
    result = _run(agent._run_tool_call(tool_call))
    assert result["ok"] is False
    assert "hint" in result["result"]
    assert len(result["result"]["hint"]) > 0


# ── not-allowed block ─────────────────────────────────────────────────────────

def test_run_tool_call_not_allowed_block_returns_hint():
    """A block that exists in the registry but is not in the agent's allowed_blocks."""
    from app.blocks import BLOCK_REGISTRY
    # Find any real registered block to use as the "exists but not allowed" case
    real_block = next(iter(BLOCK_REGISTRY), None)
    if real_block is None:
        pytest.skip("No blocks registered — skipping not-allowed test")

    agent = _make_agent(allowed_blocks=[])  # agent has no allowed blocks
    tool_call = {
        "id": "call-3",
        "function": {
            "name": real_block,
            "arguments": "{}",
        },
    }
    result = _run(agent._run_tool_call(tool_call))
    assert result["ok"] is False
    assert "hint" in result["result"]
