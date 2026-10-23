"""Outbound message whitelist-sanitisation (Groq 400 reasoning-field root cause).

Reasoning models (gpt-oss / GLM) attach a non-standard `reasoning` field to the
assistant message that carries a tool call. Replayed to Groq it 400s
("messages[N].reasoning: reasoning is not supported with this model") and the
whole tool-calling turn errors. _call_llm whitelist-sanitises every outbound
message at a single chokepoint. These tests lock that in.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from app.agents.runtime import Agent, _sanitize_messages_for_provider
from tests._fixture_values import ENV_SECRET_08


def test_strips_reasoning_and_extras_keeps_standard():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "long chain of thought...",   # gpt-oss/GLM field
            "reasoning_content": "another variant",     # future provider field
            "extra_provider_field": {"x": 1},
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "commissioning_checklist",
                        "arguments": '{"systems":["electrical"]}',
                        "extra_fn_key": "nope",
                    },
                    "bogus_tc_key": 123,
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "commissioning_checklist",
            "content": "{...}",
            "reasoning": "leaked here too",
        },
    ]
    out = _sanitize_messages_for_provider(messages)

    # standard messages survive byte-for-byte
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "hi"}

    a = out[2]
    assert a["role"] == "assistant"
    assert a["content"] == ""
    assert "reasoning" not in a
    assert "reasoning_content" not in a
    assert "extra_provider_field" not in a
    tc = a["tool_calls"][0]
    assert tc == {
        "id": "call_1",
        "type": "function",
        "function": {"name": "commissioning_checklist", "arguments": '{"systems":["electrical"]}'},
    }

    t = out[3]
    assert t == {"role": "tool", "tool_call_id": "call_1", "name": "commissioning_checklist", "content": "{...}"}
    assert "reasoning" not in t

    # input list is NOT mutated
    assert "reasoning" in messages[2]
    assert "extra_fn_key" in messages[2]["tool_calls"][0]["function"]


def test_clean_messages_pass_through_unchanged():
    messages = [{"role": "user", "content": "x"}]
    out = _sanitize_messages_for_provider(messages)
    assert out == [{"role": "user", "content": "x"}]
    assert out is not messages  # a copy


def test_call_llm_sends_sanitized_payload(monkeypatch):
    """End-to-end: the actual JSON _call_llm POSTs must carry no `reasoning`
    (or other extra) field, so Groq never 400s again."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", ENV_SECRET_08)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    agent = Agent(name="t", description="", system_prompt="x", allowed_blocks=[])

    client = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    contaminated = [
        {"role": "user", "content": "generate a commissioning checklist"},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "should never reach the provider",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "commissioning_checklist", "arguments": "{}"},
                            "junk": 1}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "commissioning_checklist", "content": "{}"},
    ]
    with patch("app.agents.runtime.httpx.AsyncClient", return_value=client):
        asyncio.run(agent._call_llm(contaminated, ENV_SECRET_08, with_tools=False))

    sent = client.post.await_args_list[0].kwargs["json"]["messages"]
    assert all("reasoning" not in m for m in sent), sent
    assert "junk" not in sent[1]["tool_calls"][0]
    assert sent[1]["tool_calls"][0]["function"] == {"name": "commissioning_checklist", "arguments": "{}"}
    # caller's original list is untouched (still contaminated)
    assert "reasoning" in contaminated[1]


def _fetch_tc(n):
    return {
        "id": f"fetch_document:{n}",
        "type": "function",
        "function": {"name": "fetch_document", "arguments": "{}"},
    }


def test_interleaved_user_nudge_is_moved_after_all_tool_results():
    """Live IPC_NEG: Kimi 400'd Missing fetch_document:3,:4 because a user
    nudge landed after the second of four tool results."""
    messages = [
        {"role": "user", "content": "issue the payment certificate"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_fetch_tc(1), _fetch_tc(2), _fetch_tc(3), _fetch_tc(4)],
        },
        {"role": "tool", "tool_call_id": "fetch_document:1", "name": "fetch_document", "content": "{}"},
        {"role": "tool", "tool_call_id": "fetch_document:2", "name": "fetch_document", "content": "{}"},
        {"role": "user", "content": "The router matched NO action. Dispatch a calculator."},
        {"role": "tool", "tool_call_id": "fetch_document:3", "name": "fetch_document", "content": "{}"},
        {"role": "tool", "tool_call_id": "fetch_document:4", "name": "fetch_document", "content": "{}"},
    ]
    out = _sanitize_messages_for_provider(messages)
    roles_ids = [
        (m.get("role"), m.get("tool_call_id")) for m in out
    ]
    assert roles_ids == [
        ("user", None),
        ("assistant", None),
        ("tool", "fetch_document:1"),
        ("tool", "fetch_document:2"),
        ("tool", "fetch_document:3"),
        ("tool", "fetch_document:4"),
        ("user", None),
    ]
    assert "router matched NO action" in out[-1]["content"]


def test_missing_tool_results_are_stubbed_so_the_payload_is_legal():
    messages = [
        {"role": "assistant", "content": "",
         "tool_calls": [_fetch_tc(3), _fetch_tc(4)]},
        {"role": "user", "content": "please continue"},
    ]
    out = _sanitize_messages_for_provider(messages)
    assert out[0]["role"] == "assistant"
    assert out[1]["role"] == "tool" and out[1]["tool_call_id"] == "fetch_document:3"
    assert out[2]["role"] == "tool" and out[2]["tool_call_id"] == "fetch_document:4"
    assert out[3]["role"] == "user"
    assert "missing from history" in out[1]["content"]
