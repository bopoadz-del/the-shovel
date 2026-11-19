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
import uuid
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


# ── The live shape: a delegated sub-turn, through POST /v1/chat/stream ──────
# Live 19 Nov: the router redirected project-assistant to heavy-reasoning; the
# delegated chat_stream ran one deliverable tool (iter 0), the tool-free
# synthesis call (iter 1, force_synthesis) wrote a draft that sanitized to
# nothing, and the retry was that request again. The fake provider below
# answers per agent and, like temperature 0, gives a request it has seen the
# same reply. Two tools, two drafts, two questions; all invented here.

DELEGATED = [
    ("commissioning_checklist", {"systems": ["chilled water"]}),
    ("construction", {"action": "drawing_qto"}),
]


def _delegated_llm(name: str, tool: tuple, draft: str, answer: str | None, sent: List[str]):
    replies: Dict[str, Dict[str, Any]] = {}

    async def _llm(self, messages, *_a, **kw):
        if self.name != name:  # the speculative stream the redirect cancels
            return _ok("speculative")
        k = _key(messages) + f"|tools={kw.get('with_tools', True)}"
        sent.append(k)
        if k in replies:  # temperature 0: the same request, the same reply
            return replies[k]
        if not any(m.get("role") == "tool" for m in messages):
            replies[k] = {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": "d1", "type": "function", "function": {
                        "name": tool[0], "arguments": json.dumps(tool[1])}}]}}}
        elif len(replies) == 1 or answer is None:
            replies[k] = _ok(draft)
        else:
            replies[k] = _ok(answer)
        return replies[k]

    async def _tool(self, tool_call, **_kw):
        return {"name": tool[0], "ok": True,
                "result": {"status": "success", "items": [{"ref": "A-1", "qty": 3}]}}

    return _llm, _tool


def _route_turn(i: int, answer: str | None, monkeypatch):
    from fastapi.testclient import TestClient

    import app.agents as agents
    import app.routers.chat as chat
    from app.main import app
    from tests._fixture_values import PW_06

    for k, v in {"LLM_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": STUB_TOKEN_01,
                 "FORCED_RETRY_MIN_SECONDS": "1",
                 "AGENT_COMMISSIONING_PREDISPATCH": "0"}.items():
        monkeypatch.setenv(k, v)
    first = Agent(name="project-assistant", description="pa", system_prompt="x",
                  allowed_blocks=["construction"])
    target = Agent(name="heavy-reasoning", description="hr", system_prompt="x",
                   allowed_blocks=["construction"])

    async def _select(_message, current):
        return target, {"requested": current.name, "final": target.name,
                        "action": DELEGATED[i][1].get("action"), "confidence": 0.6,
                        "reason": "test"}

    sent: List[str] = []
    llm, tool = _delegated_llm(target.name, DELEGATED[i], EMPTY_DRAFTS[i], answer, sent)
    monkeypatch.setattr(agents, "get_agent", lambda _n: first)
    monkeypatch.setattr(rt, "select_agent_for_message", _select)
    monkeypatch.setattr(chat, "_predefined_enabled", lambda: False)
    with patch.object(Agent, "_call_llm", llm), patch.object(Agent, "_run_tool_call", tool):
        with TestClient(app) as c:
            email = f"d1-{i}-{uuid.uuid4().hex[:8]}@x.com"
            c.post("/v1/users/register", json={"email": email, "password": PW_06})
            token = c.post("/v1/users/login",
                           json={"email": email, "password": PW_06}).json()["token"]
            r = c.post("/v1/chat/stream", headers={"Authorization": f"Bearer {token}"},
                       json={"message": QUESTIONS[i], "history": []})
    events = []
    for line in r.text.splitlines():
        if line.startswith("data:"):
            try:
                events.append(json.loads(line[5:]))
            except ValueError:
                pass
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    return events, text, sent


@pytest.mark.parametrize("i", [0, 1])
def test_route_delegated_turn_gets_the_nudged_answer(i, monkeypatch):
    events, text, sent = _route_turn(i, ANSWERS[i], monkeypatch)
    assert any(e.get("type") == "route" and e.get("final") == "heavy-reasoning" for e in events)
    # tools call, tool-free synthesis draft, instructed retry
    assert len(sent) == 3, sent
    assert sent[1] != sent[2], "the retry replayed the synthesis request"
    assert rt._EMPTY_RETRY_NUDGE in sent[2]
    assert ANSWERS[i].split(";")[0] in text
    assert _EMPTY_RESPONSE_FALLBACK not in text


@pytest.mark.parametrize("i", [0, 1])
def test_route_delegated_turn_still_unusable_is_stated_plainly(i, monkeypatch):
    _events, text, sent = _route_turn(i, None, monkeypatch)
    assert len(sent) == 3 and sent[1] != sent[2]
    assert _EMPTY_RESPONSE_FALLBACK not in text
    assert "unable to generate" not in text.lower()
    assert "did not yield an answer" in text


@pytest.mark.parametrize("i", [0, 1])
def test_route_delegated_usable_synthesis_sends_no_retry(i, monkeypatch):
    # The synthesis draft is already an answer: no retry, the answer is shown.
    import tests.test_forced_retry_empty_nudge as me
    monkeypatch.setattr(me, "EMPTY_DRAFTS", [ANSWERS[0], ANSWERS[1]])
    _events, text, sent = _route_turn(i, ANSWERS[1 - i], monkeypatch)
    assert len(sent) == 2, sent
    assert not any(rt._EMPTY_RETRY_NUDGE in s for s in sent)
    assert ANSWERS[i].split(";")[0] in text
