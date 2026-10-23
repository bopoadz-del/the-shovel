"""Tests for the zero-chunk project guardrail.

Chat must refuse to answer document-corpus questions for projects that
have no indexed chunks instead of hallucinating or silently falling back
on general knowledge.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import Agent, _UNINDEXED_PROJECT_MESSAGE
from tests._fixture_values import STUB_TOKEN_02


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    """Agent.chat() guards on the provider api key *before* _call_llm is
    reached. These tests monkeypatch _call_llm so no network call ever
    happens, but the guard still fires when the key is unset (e.g. in CI,
    where conftest's load_dotenv finds no .env key). Pin the provider to
    deepseek so the guard checks DEEPSEEK_API_KEY regardless
    of a developer's LLM_PROVIDER, then satisfy it with a placeholder.

    Deliberately a per-file fixture, not in conftest.py: a global key would
    un-skip the live DEEPSEEK acceptance tests (their skipif is evaluated at
    collection time) and make them run against the real API with a fake key.
    """
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def make_test_agent() -> Agent:
    return Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )


@pytest.mark.asyncio
async def test_chat_refuses_unindexed_project(monkeypatch):
    agent = make_test_agent()

    async def fake_call_llm(*args, **kwargs):
        raise AssertionError("LLM should not be called for an unindexed project")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready",
        lambda _pid: False,
    )

    result = await agent.chat(
        "What is this project about?",
        project_id="empty_proj",
        conversation_id="ws-empty_proj",
    )

    assert result["status"] == "success"
    assert result["answer"] == _UNINDEXED_PROJECT_MESSAGE
    assert result["iterations"] == 0
    assert result["tool_calls"] == []


@pytest.mark.asyncio
async def test_chat_stream_refuses_unindexed_project(monkeypatch):
    agent = make_test_agent()

    async def fake_call_llm(*args, **kwargs):
        raise AssertionError("LLM should not be called for an unindexed project")

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready",
        lambda _pid: False,
    )

    events = []
    async for event in agent.chat_stream(
        "What is this project about?",
        project_id="empty_proj",
        conversation_id="ws-empty_proj",
    ):
        events.append(event)

    types = [e["type"] for e in events]
    assert "start" in types
    assert "token" in types
    assert "end" in types
    assert "error" not in types

    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert token_text == _UNINDEXED_PROJECT_MESSAGE

    end_event = [e for e in events if e["type"] == "end"][0]
    assert end_event.get("iterations") == 0
    assert end_event.get("sources") == []


@pytest.mark.asyncio
async def test_chat_allows_indexed_project(monkeypatch):
    """When the project has chunks, the guardrail should not fire."""
    agent = make_test_agent()

    called = False

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None,
                            with_tools=True, **kwargs):
        # **kwargs tolerates _call_llm's evolving keyword args (e.g.
        # exclude_tools, added for the forced tool-free synthesis path) so
        # this stub doesn't break every time the real signature grows.
        nonlocal called
        called = True
        return {
            "status": "success",
            "choice": {
                "message": {"role": "assistant", "content": "Here is the answer.", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready",
        lambda _pid: True,
    )

    result = await agent.chat(
        "What is this project about?",
        project_id="indexed_proj",
    )

    assert called is True
    # Coverage honesty stamps "N of M project documents indexed" when the
    # project_id is known; the guardrail contract is that the LLM answer
    # is returned, not that the string is byte-identical.
    assert result["answer"].startswith("Here is the answer.")
