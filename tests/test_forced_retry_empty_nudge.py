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


# The provider call itself fails (not a reply): the retry's answer in tests
# of the error path.
PROVIDER_ERROR = {"status": "error", "error": "upstream 503"}


def _run(mode: str, question: str, draft: str, answer: str | None, monkeypatch,
         retry_error: bool = False):
    """Drive one turn. ``answer`` None: the model never answers (every
    request gets the draft back). ``retry_error``: the retry's provider call
    fails."""
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
        if retry_error and len(replies) == 1 and k not in replies:
            return PROVIDER_ERROR
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
    # No search ran in this turn, so the statement claims none.
    assert "Searched" not in text


# A provider ERROR on the instructed retry is not a not-found: the user is
# told the turn failed and to try again, never that the documents lack it.

@pytest.mark.parametrize("mode", ["stream", "chat"])
@pytest.mark.parametrize("i", [0, 1])
def test_a_provider_error_on_the_instructed_retry_is_not_a_not_found(mode, i, monkeypatch):
    text, sent = _run(mode, QUESTIONS[i], EMPTY_DRAFTS[i], ANSWERS[i], monkeypatch,
                      retry_error=True)
    assert len(sent) == 2 and rt._EMPTY_RETRY_NUDGE in sent[1]
    assert _EMPTY_RESPONSE_FALLBACK in text
    assert "did not yield an answer" not in text
    assert ANSWERS[i].split(";")[0] not in text


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
    out = _statement(_turn(question, queries, nudged=True), "unusable")
    for q in queries:
        assert f'"{q}"' in out
    assert "did not yield an answer" in out


@pytest.mark.parametrize("question", QUESTIONS)
def test_no_instructed_retry_no_statement(question):
    assert _statement(_turn(question, ["x y"], nudged=False), "unusable") == ""


@pytest.mark.parametrize("question,outcome", [
    (QUESTIONS[0], "provider_error"), (QUESTIONS[1], "provider_error"),
    (QUESTIONS[0], ""), (QUESTIONS[1], ""),
])
def test_only_a_returned_unusable_retry_is_stated(question, outcome):
    # The same instructed turn: stated only when the retry RETURNED a reply.
    msgs = _turn(question, ["a b"], nudged=True)
    assert _statement(msgs, outcome) == ""
    assert "did not yield an answer" in _statement(msgs, "unusable")


@pytest.mark.parametrize("question", QUESTIONS)
def test_no_recorded_search_no_searched_line(question):
    out = _statement(_turn(question, [], nudged=True), "unusable")
    assert "did not yield an answer" in out
    assert "Searched" not in out


@pytest.mark.parametrize("outcome", ["unusable", "provider_error"])
def test_the_retry_outcome_is_recorded_on_the_turn(outcome):
    from app.agents import answer_exit

    turn, token = answer_exit.begin_turn("q")
    try:
        assert turn.forced_retry_outcome == ""
        rt.record_forced_retry_outcome(outcome)
        assert turn.forced_retry_outcome == outcome
    finally:
        answer_exit.reset_turn(token)
    rt.record_forced_retry_outcome(outcome)  # no turn open: nothing to record, no raise


def _statement(msgs: List[Dict[str, Any]], outcome: str) -> str:
    from app.agents import answer_exit

    turn, token = answer_exit.begin_turn(msgs[0]["content"])
    try:
        turn.forced_retry_outcome = outcome
        return rt._unanswered_after_forced_retry(msgs)
    finally:
        answer_exit.reset_turn(token)


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


def _delegated_llm(name: str, tool: tuple, draft: str, answer: str | dict | None,
                   sent: List[str]):
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
            replies[k] = answer if isinstance(answer, dict) else _ok(answer)
        return replies[k]

    async def _tool(self, tool_call, **_kw):
        return {"name": tool[0], "ok": True,
                "result": {"status": "success", "items": [{"ref": "A-1", "qty": 3}]}}

    return _llm, _tool


def _route_turn(i: int, answer: str | dict | None, monkeypatch):
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
def test_route_delegated_provider_error_is_not_a_not_found(i, monkeypatch):
    _events, text, sent = _route_turn(i, PROVIDER_ERROR, monkeypatch)
    assert len(sent) == 3 and rt._EMPTY_RETRY_NUDGE in sent[2]
    assert _EMPTY_RESPONSE_FALLBACK in text
    assert "did not yield an answer" not in text


@pytest.mark.parametrize("i", [0, 1])
def test_route_delegated_usable_synthesis_sends_no_retry(i, monkeypatch):
    # The synthesis draft is already an answer: no retry, the answer is shown.
    import tests.test_forced_retry_empty_nudge as me
    monkeypatch.setattr(me, "EMPTY_DRAFTS", [ANSWERS[0], ANSWERS[1]])
    _events, text, sent = _route_turn(i, ANSWERS[1 - i], monkeypatch)
    assert len(sent) == 2, sent
    assert not any(rt._EMPTY_RETRY_NUDGE in s for s in sent)
    assert ANSWERS[i].split(";")[0] in text


# ── The instruction is the platform's, not the operator's question ──────────
# It is appended after the turn's tool results. A reader that took it for the
# question cut those results out of the turn: CI on this PR, a computed bill
# total recovered from the tool result was lost and the bare line shown.

@pytest.mark.parametrize("i,rule", [(0, "empty"), (1, "search_promise")])
def test_a_retry_instruction_is_never_the_operators_question(i, rule):
    msgs = _turn(QUESTIONS[i], ["query one", "query two"], nudged=False)
    before = (rt._operator_turn_start(msgs), rt._latest_user_text(msgs))
    msgs.append({"role": "user", "content": rt._FORCED_RETRY_NUDGES[rule]})
    assert rt._is_retry_instruction(msgs[-1]["content"])
    assert (rt._operator_turn_start(msgs), rt._latest_user_text(msgs)) == before
    assert rt._latest_user_text(msgs) == QUESTIONS[i]
    assert rt._cg_is_platform_bubble(rt._FORCED_RETRY_NUDGES[rule])


@pytest.mark.parametrize("q", QUESTIONS)
def test_an_operator_question_is_not_a_retry_instruction(q):
    assert not rt._is_retry_instruction(q)
    assert not rt._cg_is_platform_bubble(q)


# A usable reply is never rewritten by the empty-turn path: it is returned
# exactly as written, even on a turn whose retry outcome says "unusable".

@pytest.mark.parametrize("i,reply", [
    (0, ANSWERS[0] + "\n"),
    (1, "  " + ANSWERS[1] + "  "),
])
def test_a_usable_reply_is_returned_as_written(i, reply):
    from app.agents import answer_exit

    msgs = _turn(QUESTIONS[i], ["c d"], nudged=True)
    turn, token = answer_exit.begin_turn(QUESTIONS[i])
    try:
        turn.forced_retry_outcome = "unusable"
        assert rt._nonblank_after_empty_synthesis(reply, msgs) == reply
    finally:
        answer_exit.reset_turn(token)


# The handoff hat is read from the operator's words, never from the
# platform's retry instruction appended after them.

@pytest.mark.parametrize("question,hat", [
    ("Ingest the survey model north_block_survey.ifc please", "bim-analyst"),
    ("Load the baseline programme tower_baseline.xer", "construction-pm"),
])
def test_the_handoff_hat_skips_the_retry_instruction(question, hat):
    msgs = [{"role": "user", "content": question},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": rt._EMPTY_RETRY_NUDGE}]
    assert rt._next_agent_from_turn(msgs) == hat
    assert rt._next_agent_from_turn(msgs[:1]) == hat


@pytest.mark.parametrize("question", [
    "Open the BIM model for the podium levels",
    "Check the IFC export of the east stair core",
])
def test_a_named_model_without_a_file_goes_to_the_bim_hat(question):
    msgs = [{"role": "user", "content": question},
            {"role": "user", "content": rt._EMPTY_RETRY_NUDGE}]
    assert rt._next_agent_from_turn(msgs) == "bim-analyst"
