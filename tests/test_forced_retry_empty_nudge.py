"""D1: the ``empty`` rule's forced retry carries an instruction, never replays.

Live 19 Nov (three delegated turns, one shape): the draft sanitized to
nothing, the ``empty`` rule fired, and the no-tools retry was the first
request again (``empty`` had no nudge). At temperature 0 the model wrote the
same draft (raw=274c twice) and the user got the bare generic line.

The fake provider below is deterministic like temperature 0: the same
request gets the same reply. It answers only when the request differs from
the draft's. Every input is invented by the test.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import patch

import pytest
from tests._fixture_values import STUB_TOKEN_01

from app.agents import runtime as rt
from app.agents.runtime import _EMPTY_RESPONSE_FALLBACK, Agent

QUESTIONS = [
    "what is the retention on the Velmor pump skid order?",
    "which sheet shows the Ostrand culvert headwall?",
]
# Two drafts that sanitize to nothing: nothing written, and reasoning only
# (stripped before the user sees it). Tool-call markup is not used here: the
# stream path runs it as a tool call, so it never reaches the empty rule.
EMPTY_DRAFTS = [
    "",
    "<think>The headwall may sit on a drainage sheet; check the register.</think>",
]
ANSWERS = [
    "The Velmor order states no retention figure; none of the documents names one.",
    "No drawing in the documents shows an Ostrand culvert headwall.",
]


def _ok(content: str) -> Dict[str, Any]:
    return {"status": "success", "choice": {"message": {"content": content, "role": "assistant"}},
            "raw": {"model": "test-model"}}


def _key(messages: List[Dict[str, Any]]) -> str:
    return json.dumps([(m.get("role"), m.get("content")) for m in messages], sort_keys=True)


def _run(mode: str, question: str, draft: str, answer: str | None, monkeypatch):
    """Drive one turn. ``answer`` None: the model never answers (every
    request gets the draft back)."""
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_01)
    monkeypatch.setenv("CHAT_STREAM_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("CHAT_STREAM_HEARTBEAT_SECONDS", "20")
    monkeypatch.setenv("FORCED_RETRY_MIN_SECONDS", "1")
    agent = Agent(name="d1-test", description="d", system_prompt="x", allowed_blocks=[])
    sent: List[str] = []
    replies: Dict[str, str] = {}

    async def _llm(messages, *_a, **_kw):
        k = _key(messages)
        sent.append(k)
        if k not in replies:  # temperature 0: a request seen before gets the same reply
            replies[k] = draft if (len(replies) == 0 or answer is None) else answer
        return _ok(replies[k])

    async def drain():
        if mode == "chat":
            out = await agent.chat(user_message=question, history=[], project_id=None,
                                   conversation_id=None, user_id=None)
            return (out or {}).get("content") or (out or {}).get("response") or str(out)
        events = [e async for e in agent.chat_stream(user_message=question)]
        return "".join(e.get("content", "") for e in events if e["type"] == "token")

    with patch.object(agent, "_call_llm", _llm):
        text = asyncio.run(drain())
    return text, sent


@pytest.mark.parametrize("draft", EMPTY_DRAFTS)
def test_the_empty_rule_has_an_instruction(draft):
    clean = rt._sanitize_final_text(draft)
    assert rt.forced_retry_reason(clean) == "empty"
    assert rt._forced_retry_nudge(clean) == rt._EMPTY_RETRY_NUDGE


def test_every_rule_that_forces_a_retry_has_an_instruction():
    # A rule with no instruction re-sends the first request unchanged.
    for text in ["", rt._TOOL_FORMAT_FALLBACK, "Let me search the drawings register next.",
                 "What can I help you with today?"]:
        assert rt.forced_retry_reason(text) is not None
        assert rt._forced_retry_nudge(text), text


@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize("i", [0, 1])
def test_the_retry_is_never_the_first_request_again(mode, i, monkeypatch):
    text, sent = _run(mode, QUESTIONS[i], EMPTY_DRAFTS[i], ANSWERS[i], monkeypatch)
    assert len(sent) == 2
    assert sent[0] != sent[1], "the retry replayed the first request"
    assert rt._EMPTY_RETRY_NUDGE in sent[1]
    # Before this change the same request got the same draft and the user
    # got the fallback; now the instructed retry's answer is shown.
    assert ANSWERS[i].split(";")[0] in text
    assert _EMPTY_RESPONSE_FALLBACK not in text


@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize("i", [0, 1])
def test_a_retry_still_unusable_gets_a_plain_statement(mode, i, monkeypatch):
    text, sent = _run(mode, QUESTIONS[i], EMPTY_DRAFTS[i], None, monkeypatch)
    assert len(sent) == 2 and sent[0] != sent[1]
    assert _EMPTY_RESPONSE_FALLBACK not in text
    assert "unable to generate" not in text.lower()
    assert "did not yield an answer" in text
    assert "Searched" in text


@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize("i", [0, 1])
def test_a_usable_first_reply_sends_no_retry(mode, i, monkeypatch):
    text, sent = _run(mode, QUESTIONS[i], ANSWERS[i], ANSWERS[1 - i], monkeypatch)
    assert len(sent) == 1
    assert ANSWERS[i].split(";")[0] in text
    assert not any(rt._EMPTY_RETRY_NUDGE in s for s in sent)


# The statement reads this turn's own messages: what was searched for, and
# only after an instructed retry.

def _turn(question: str, queries: List[str], nudged: bool) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    for n, q in enumerate(queries):
        msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{n}", "function": {"name": "search_project",
                                         "arguments": json.dumps({"query": q})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{n}", "content": "[]"})
    if nudged:
        msgs.append({"role": "user", "content": rt._EMPTY_RETRY_NUDGE})
    return msgs


@pytest.mark.parametrize("question,queries", [
    (QUESTIONS[0], ["Velmor pump skid retention", "retention percentage"]),
    (QUESTIONS[1], ["Ostrand culvert headwall drawing"]),
])
def test_the_statement_names_what_was_searched(question, queries):
    out = rt._unanswered_after_forced_retry(_turn(question, queries, nudged=True))
    for q in queries:
        assert f'"{q}"' in out
    assert "did not yield an answer" in out


@pytest.mark.parametrize("question", QUESTIONS)
def test_no_instructed_retry_no_statement(question):
    assert rt._unanswered_after_forced_retry(_turn(question, ["x y"], nudged=False)) == ""
