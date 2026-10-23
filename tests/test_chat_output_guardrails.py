"""Tests for chat output guardrails:

* raw internal tool-call JSON must not reach the user
* empty final answers must be replaced by a controlled fallback
* DSML markup must still be stripped
"""
from __future__ import annotations

import json

import pytest


from app.agents.runtime import (
    Agent,
    _looks_like_internal_tool_json,
    _sanitize_final_text,
    _EMPTY_RESPONSE_FALLBACK,
    _TOOL_FORMAT_FALLBACK,
    _TOOL_FORMAT_RETRY_NUDGE,
    _final_text_needs_forced_retry,
)


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


def test_looks_like_internal_tool_json_detects_tool_call_shape():
    raw = json.dumps({"name": "search_project_documents", "arguments": {"query": "XYZ-501"}})
    assert _looks_like_internal_tool_json(raw) is True

    raw_list = json.dumps([{"type": "function", "function": {"name": "foo", "arguments": "{}"}}])
    assert _looks_like_internal_tool_json(raw_list) is True

    assert _looks_like_internal_tool_json("Plain text answer") is False
    assert _looks_like_internal_tool_json('{"answer": "plain json"}') is False


def test_sanitize_final_text_strips_dsml_and_tool_json():
    assert _sanitize_final_text("Hello world") == "Hello world"
    assert _sanitize_final_text("  Hello world  ") == "Hello world"
    assert _sanitize_final_text("") == ""

    dsml = "Answer here <｜｜DSML｜｜tool_calls>...</｜｜DSML｜｜tool_calls>"
    assert _sanitize_final_text(dsml) == "Answer here"

    raw_tool = json.dumps({"name": "search_project_documents", "arguments": {"query": "x"}})
    assert _sanitize_final_text(raw_tool) == _TOOL_FORMAT_FALLBACK

    # Search argument leak observed in production: plain {"query": ..., "top_k": ...}
    search_args = json.dumps({"query": "registered capital", "top_k": 5})
    assert _sanitize_final_text(search_args) == _TOOL_FORMAT_FALLBACK

    # Markdown code block containing only search args.
    md_search = "```json\n{\"query\": \"x\", \"project_id\": \"p\"}\n```"
    assert _sanitize_final_text(md_search) == _TOOL_FORMAT_FALLBACK

    # User-requested JSON example should not be sanitized.
    user_json = '{"answer": "example", "result": "ok"}'
    assert _sanitize_final_text(user_json) == user_json

    xml_leak = (
        'I need to re-run the formula executor with valid JSON.\n'
        '<function_calls>\n'
        '<invoke name="formula_executor_v2">\n'
        '<parameter name="input">{"task":"x"}</parameter>\n'
        '</invoke>\n'
        '</function_calls>'
    )
    assert _sanitize_final_text(xml_leak) == _TOOL_FORMAT_FALLBACK


def test_final_text_needs_forced_retry_on_tool_format_fallback():
    assert _final_text_needs_forced_retry("") is True
    assert _final_text_needs_forced_retry("   ") is True
    assert _final_text_needs_forced_retry(_TOOL_FORMAT_FALLBACK) is True
    assert _final_text_needs_forced_retry("Real answer") is False


@pytest.mark.asyncio
async def test_chat_stream_retries_when_sanitizer_would_emit_tool_format_fallback(monkeypatch):
    """After search tools, a leaked search-args JSON final must trigger a forced
    no-tools synthesis retry instead of returning the 120-char fallback."""
    from app.agents import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_recover_tool_calls_from_content", lambda text: [])

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    search_args = json.dumps({"query": "landscaping RFP scope", "top_k": 5})
    calls = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "success",
                "choice": {
                    "message": {"role": "assistant", "content": search_args, "tool_calls": []},
                    "finish_reason": "stop",
                },
            }
        assert with_tools is False
        assert any(m.get("content") == _TOOL_FORMAT_RETRY_NUDGE for m in messages if m.get("role") == "user")
        return {
            "status": "success",
            "choice": {
                "message": {
                    "role": "assistant",
                    "content": "# RFP\n\n## Scope\nLandscaping subcontract package.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    events = []
    async for event in agent.chat_stream("Prepare an RFP for landscaping."):
        events.append(event)

    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert search_args not in token_text
    assert "internal search formatting issue" not in token_text.lower()
    assert "RFP" in token_text
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_chat_replaces_raw_tool_json_with_fallback(monkeypatch):
    """If the LLM returns a raw tool-call JSON object as 'content', the
    user-facing answer must be the safe fallback, not the JSON."""
    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    raw_tool = json.dumps({"name": "search_project_documents", "arguments": {"query": "XYZ-501", "top_k": 20}})

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {"role": "assistant", "content": raw_tool, "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.chat("What is XYZ-501?")
    assert result["status"] == "success"
    assert raw_tool not in result["answer"]
    assert result["answer"] in (_EMPTY_RESPONSE_FALLBACK, _TOOL_FORMAT_FALLBACK)


@pytest.mark.asyncio
async def test_chat_replaces_empty_final_with_fallback(monkeypatch):
    """If the LLM returns an empty final answer, the user sees the fallback."""
    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {"role": "assistant", "content": "", "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.chat("Hello?")
    assert result["status"] == "success"
    assert result["answer"] == _EMPTY_RESPONSE_FALLBACK


# ── P0B: harden raw-tool-JSON detection ───────────────────────────────────────

from app.agents.runtime import _looks_like_internal_tool_json


def test_looks_like_internal_tool_json_detects_embedded_json_in_prose():
    raw = (
        "We need to emit the tool call. We should output the JSON tool call. "
        '{"name": "search_project_documents", "arguments": {"query": "foo"}}'
    )
    assert _looks_like_internal_tool_json(raw) is True


def test_looks_like_internal_tool_json_ignores_plain_json_without_tool_keys():
    raw = 'The answer is {"answer": "plain json"} and that is fine.'
    assert _looks_like_internal_tool_json(raw) is False


# ── P0C: block raw search-tool argument leakage ───────────────────────────────

from app.agents.runtime import _is_search_tool_args_obj
from tests._fixture_values import STUB_TOKEN_02


def test_is_search_tool_args_obj_detects_plain_search_payload():
    assert _is_search_tool_args_obj({"query": "x", "top_k": 5}) is True
    assert _is_search_tool_args_obj({"query": "x", "project_id": "p"}) is True
    # UI-PHYS A5: bare {"query": "..."} is the live Scout/Kimi leak shape.
    assert _is_search_tool_args_obj({"query": "x"}) is True
    assert _is_search_tool_args_obj({"query": "x", "answer": "y"}) is False


def test_is_search_tool_args_obj_detects_tool_envelopes():
    assert _is_search_tool_args_obj({"tool": "search_project_documents", "arguments": {"query": "x"}}) is True
    assert _is_search_tool_args_obj({"name": "search_project_documents", "arguments": {"query": "x"}}) is True
    assert _is_search_tool_args_obj({"function": "search_project_documents", "arguments": {"query": "x"}}) is True


def test_is_search_tool_args_obj_ignores_user_data_json():
    assert _is_search_tool_args_obj({"answer": "example"}) is False
    assert _is_search_tool_args_obj({"items": [{"qty": 1}]}) is False
    assert _is_search_tool_args_obj({"query": "x", "answer": "y"}) is False


def test_sanitize_final_text_allows_user_requested_json():
    boq = '{"items": [{"description": "concrete", "quantity": 10, "unit": "m3"}]}'
    assert _sanitize_final_text(boq) == boq

    code_example = 'Here is an example JSON: `{"status": "ok"}`'
    assert _sanitize_final_text(code_example) == code_example


@pytest.mark.asyncio
async def test_chat_blocks_raw_search_args_leak(monkeypatch):
    """A final answer that is only {"query": ..., "top_k": ...} must not reach the user."""
    from app.agents import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_recover_tool_calls_from_content", lambda text: [])

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    search_args = json.dumps({"query": "registered capital", "top_k": 5})
    calls = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "success",
                "choice": {
                    "message": {"role": "assistant", "content": search_args, "tool_calls": []},
                    "finish_reason": "stop",
                },
            }
        return {
            "status": "success",
            "choice": {
                "message": {
                    "role": "assistant",
                    "content": "Registered capital is 10 million SAR per the excerpts.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.chat("What is the registered capital?")
    assert result["status"] == "success"
    assert search_args not in result["answer"]
    assert "registered capital" in result["answer"].lower()
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_chat_stream_blocks_raw_search_args_leak(monkeypatch):
    """Streaming must also hide raw search argument payloads."""
    from app.agents import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_recover_tool_calls_from_content", lambda text: [])

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    search_args = json.dumps({"query": "registered capital", "top_k": 5})
    calls = {"n": 0}

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "success",
                "choice": {
                    "message": {"role": "assistant", "content": search_args, "tool_calls": []},
                    "finish_reason": "stop",
                },
            }
        return {
            "status": "success",
            "choice": {
                "message": {
                    "role": "assistant",
                    "content": "Registered capital is 10 million SAR per the excerpts.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    events = []
    async for event in agent.chat_stream("What is the registered capital?"):
        events.append(event)

    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert search_args not in token_text
    assert "registered capital" in token_text.lower()
    assert calls["n"] == 2
    end_events = [e for e in events if e["type"] == "end"]
    assert len(end_events) == 1
    assert end_events[0].get("sources") == []


@pytest.mark.asyncio
async def test_chat_preserves_user_requested_json(monkeypatch):
    """If the model returns legitimate user-requested JSON data, it is not sanitized."""
    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    user_json = '{"items": [{"description": "concrete", "quantity": 10, "unit": "m3"}]}'

    async def fake_call_llm(messages, api_key, *, project_id=None, user_id=None, with_tools=True, exclude_tools=None, **kwargs):
        return {
            "status": "success",
            "choice": {
                "message": {"role": "assistant", "content": user_json, "tool_calls": []},
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    result = await agent.chat("Return a BOQ JSON example.")
    assert result["status"] == "success"
    assert user_json in result["answer"]


# ── UI-PHYS A5: Delay Damages tool-call envelope must never reach the user ────


def test_looks_like_internal_tool_json_detects_delay_damages_envelope():
    """Exact production leak from Master Corpus Delay Damages smoke."""
    raw = json.dumps({
        "name": "search_project_documents",
        "arguments": {"query": "Delay Damages Contract Data"},
    })
    assert _looks_like_internal_tool_json(raw) is True
    assert _sanitize_final_text(raw) == _TOOL_FORMAT_FALLBACK


def test_looks_like_internal_tool_json_detects_bare_query_leak():
    bare = json.dumps({"query": "Delay Damages Contract Data"})
    assert _looks_like_internal_tool_json(bare) is True
    assert _sanitize_final_text(bare) == _TOOL_FORMAT_FALLBACK


@pytest.mark.asyncio
async def test_chat_stream_never_emits_search_tool_envelope(monkeypatch):
    """UI-PHYS A5 regression: leaked search envelope is executed / synthesised,
    never streamed as the assistant bubble body."""
    from app.agents import runtime as runtime_module

    monkeypatch.setenv("AGENT_SEARCH_TOOL_CAP", "2")
    monkeypatch.setattr(
        runtime_module, "rag_inject",
        lambda **kw: (None, {"chunks": [], "fallback_used": False}),
    )
    monkeypatch.setattr(runtime_module, "project_is_rag_ready", lambda pid: True)

    agent = Agent(
        name="project-assistant",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    leaked = json.dumps({
        "name": "search_project_documents",
        "arguments": {"query": "Delay Damages Contract Data"},
    })
    calls = {"n": 0}
    tool_runs = {"n": 0}

    async def fake_call_llm(
        messages, api_key, *, project_id=None, user_id=None,
        with_tools=True, exclude_tools=None, **kwargs,
    ):
        calls["n"] += 1
        if with_tools:
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "role": "assistant",
                        "content": leaked,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            }
        return {
            "status": "success",
            "choice": {
                "message": {
                    "role": "assistant",
                    "content": (
                        "Delay Damages for the whole of the Works are "
                        "0.1% of the Contract Price per calendar day."
                    ),
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    async def fake_run(tc, **kw):
        tool_runs["n"] += 1
        return {
            "name": "search_project_documents",
            "ok": True,
            "result": {"hits": [{"text": "0.1% of the Contract Price"}]},
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(agent, "_run_tool_call", fake_run)

    prompt = (
        "Answer only from the client project documents. "
        "What are the Delay Damages for the whole of the Works?"
    )
    events = []
    async for event in agent.chat_stream(prompt):
        events.append(event)

    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert leaked not in token_text
    assert '"name": "search_project_documents"' not in token_text
    assert '{"name":"search_project_documents"' not in token_text
    assert "0.1%" in token_text
    assert tool_runs["n"] >= 1
    # Cap stops the 12-iteration search spin; synthesis happens promptly.
    assert tool_runs["n"] <= 2
    assert calls["n"] >= 2
    end_events = [e for e in events if e["type"] == "end"]
    assert len(end_events) == 1
    assert end_events[0].get("content")
    assert leaked not in (end_events[0].get("content") or "")


@pytest.mark.asyncio
async def test_chat_stream_blocks_bare_query_leak(monkeypatch):
    """Bare {"query": "..."} without top_k must not reach the user."""
    from app.agents import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_recover_tool_calls_from_content", lambda text: [])

    agent = Agent(
        name="test-agent",
        description="test",
        system_prompt="You are a test assistant. Be concise.",
        allowed_blocks=[],
    )

    bare = json.dumps({"query": "Delay Damages Contract Data"})
    calls = {"n": 0}

    async def fake_call_llm(
        messages, api_key, *, project_id=None, user_id=None,
        with_tools=True, exclude_tools=None, **kwargs,
    ):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "status": "success",
                "choice": {
                    "message": {
                        "role": "assistant",
                        "content": bare,
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                },
            }
        return {
            "status": "success",
            "choice": {
                "message": {
                    "role": "assistant",
                    "content": "Delay Damages are 0.1% per calendar day.",
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            },
        }

    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)

    events = []
    async for event in agent.chat_stream(
        "What are the Delay Damages for the whole of the Works?"
    ):
        events.append(event)

    token_text = "".join(e.get("content", "") for e in events if e["type"] == "token")
    assert bare not in token_text
    assert "0.1%" in token_text
    assert calls["n"] == 2
