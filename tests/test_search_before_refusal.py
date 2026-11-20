"""S-R8: a first-pass miss runs a project search instead of ending in a refusal.

Measured 14 Nov: 21 of 24 turns were one model call with no tool call. The
contract told the model to OFFER a search, and a model that promised one was
retried with its tools removed. The guard under test reads only turn state
(search budget, indexed project, one round per turn) plus a pure text test.

Every test builds its own question, excerpt and rows; two inputs each.
"""
from __future__ import annotations

import json

import pytest

from app.agents.core.context import (  # noqa: F401 - the mutation gate ties
    SEARCH_FIRST_NUDGE,                    # these helpers to this file; the
    SEARCH_TOOL_NAME,                      # stream/chat tests below run them
    _intent_hard_excludes,
    _search_before_refusal,
    _search_first_nudges,
    search_before_refusal_call,
    search_before_refusal_due,
)
from app.agents.missing_input import declines_for_lack, hands_search_to_user
from app.agents.runtime import Agent, _UNINDEXED_PROJECT_MESSAGE
from tests._fixture_values import STUB_TOKEN_02


def _is_nudge(msg: dict) -> bool:
    """The guard's follow-up bubble: platform-marked, ending in the instruction."""
    from app.agents.runtime import _PREDISPATCH_PREFIX

    content = str(msg.get("content") or "")
    return (
        msg.get("role") == "user"
        and content.startswith(_PREDISPATCH_PREFIX)
        and content.endswith(SEARCH_FIRST_NUDGE)
    )


@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))


# ── Two-input cases, each built here ─────────────────────────────────────
# (question, excerpt handed first, draft that lacks it, row the search finds,
#  answer written from that row, a word the search query must carry)
MISS_CASES = [
    (
        "What curing period does the specification require for the deck slab?",
        "Section 3 covers formwork striking times for columns.",
        "The retrieved excerpts do not state the curing period for the deck "
        "slab. If you give me the exact section number I will search for it.",
        "Deck slab concrete shall be cured for not less than 7 days.",
        "The deck slab must be cured for at least 7 days (Specification, "
        "concrete section).",
        "curing",
    ),
    (
        "Which pipe material is specified for the foul drainage outfall?",
        "Storm water gullies shall be precast concrete.",
        "I could not find the pipe material for the foul drainage outfall in "
        "the excerpts provided.",
        "Foul drainage outfall pipes shall be vitrified clay to the stated "
        "standard.",
        "The foul drainage outfall is specified in vitrified clay (Drainage "
        "specification).",
        "outfall",
    ),
]

HELD_CASES = [
    (
        "What is the retention percentage?",
        "Retention: 5% of each interim payment, released half at completion.",
        "Retention is 5% of each interim payment (Contract Data).",
    ),
    (
        "How many days notice does the contractor give before a pour?",
        "The Contractor shall give 2 days notice before any concrete pour.",
        "The contractor gives 2 days notice before a pour (Specification).",
    ),
]


def _agent() -> Agent:
    return Agent(
        name="project-assistant", description="pa",
        system_prompt="You answer from project documents.", allowed_blocks=[],
    )


def _ready(monkeypatch, ready: bool, non_rag: bool = False):
    async def _is_ready(_pid):
        return ready
    monkeypatch.setattr("app.agents.runtime._project_is_rag_ready_off_process", _is_ready)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: ready)
    monkeypatch.setattr(
        "app.agents.runtime._project_has_non_rag_context", lambda pid, msg: non_rag,
    )


def _inject(monkeypatch, excerpt: str):
    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": f"Relevant project context.\n{excerpt}"},
            {"project_id": "proj_sr8", "chunks": [
                {"doc_id": "doc_1", "chunk_index": 0, "chunk_id": "proj_sr8:doc_1:0", "score": 0.5},
            ]},
        )
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr("app.core.projects.get_document", lambda did: {"original_name": "Spec.pdf"})


def _scripted_llm(replies: list[str], seen: list[dict]):
    """Each call returns the next reply as a final answer and records what
    it was given (tools on/off, the last message)."""
    async def fake(messages, api_key, **kwargs):
        seen.append({
            "with_tools": kwargs.get("with_tools", True),
            "exclude_tools": set(kwargs.get("exclude_tools") or ()),
            "messages": list(messages),
        })
        text = replies[min(len(seen) - 1, len(replies) - 1)]
        return {"status": "success", "choice": {
            "message": {"role": "assistant", "content": text, "tool_calls": []},
            "finish_reason": "stop"}}
    return fake


def _search_tool(row: str, ran: list[dict]):
    async def fake(tool_call, **kwargs):
        fn = tool_call.get("function") or {}
        ran.append({"name": fn.get("name"), "args": json.loads(fn.get("arguments") or "{}")})
        return {"name": fn.get("name"), "ok": True, "result": {
            "status": "success", "results": [
                {"document_id": "doc_2", "filename": "Spec.pdf", "snippet": row, "score": 0.6},
            ]}}
    return fake


async def _stream(agent, question):
    return [e async for e in agent.chat_stream(question, project_id="proj_sr8")]


def _end(events):
    ends = [e for e in events if e.get("type") == "end"]
    assert len(ends) == 1, events
    return ends[0]


# ── Pure text test ───────────────────────────────────────────────────────
@pytest.mark.parametrize("draft", [c[2] for c in MISS_CASES])
def test_a_draft_that_lacks_the_asked_thing_declines(draft):
    assert declines_for_lack(draft)


@pytest.mark.parametrize("answer", [c[2] for c in HELD_CASES] + [
    "Works shall be carried out without delay once the permit is issued.",
])
def test_an_answer_that_states_the_thing_does_not_decline(answer):
    assert not declines_for_lack(answer)


# Drafts that never say the thing is missing, only hand the search back.
OFFER_ONLY = [
    "The excerpts cover formwork for columns. Give me the exact section "
    "number and I can look it up for the deck slab.",
    "Here is what the drainage excerpts cover. If you share the drawing "
    "reference, I can locate the outfall detail.",
]


@pytest.mark.parametrize("draft", OFFER_ONLY)
def test_handing_the_search_to_the_user_is_a_decline(draft):
    assert hands_search_to_user(draft)
    assert declines_for_lack(draft)


@pytest.mark.parametrize("answer", [
    "If you need the formwork striking times too, they are 7 days for soffits.",
    "Let me know if anything else is needed.",
])
def test_a_courtesy_line_is_not_handing_the_search_back(answer):
    assert not hands_search_to_user(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("draft", OFFER_ONLY)
async def test_stream_offer_only_draft_searches(monkeypatch, draft):
    question, excerpt, _d, row, answer, _w = MISS_CASES[0]
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool(row, ran))

    events = await _stream(agent, question)

    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME]
    assert _is_nudge(seen[1]["messages"][-1])
    assert _end(events)["content"].startswith(answer)


# ── Turn-state decision ──────────────────────────────────────────────────
_DUE = dict(has_project=True, search_calls=0, search_cap=2, excluded_tools=set(),
            force_synthesis=False, already_ran=False, iterations_left=5)


@pytest.mark.parametrize("draft", [MISS_CASES[0][2], "Let me search the specification for that."])
def test_due_when_budget_unused(draft):
    assert search_before_refusal_due(draft, **_DUE)


@pytest.mark.parametrize("change", [
    {"search_calls": 2},
    {"excluded_tools": {SEARCH_TOOL_NAME}},
    {"already_ran": True},
    {"force_synthesis": True},
    {"has_project": False},
    {"iterations_left": 1},
])
def test_not_due_when_turn_state_forbids(change):
    assert not search_before_refusal_due(MISS_CASES[1][2], **{**_DUE, **change})


# ── Streaming path (the live UI path) ────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_stream_first_pass_miss_searches_and_answers_from_rows(monkeypatch, case):
    question, excerpt, draft, row, answer, query_word = case
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool(row, ran))

    events = await _stream(agent, question)

    calls = [e for e in events if e.get("type") == "tool_call"]
    assert [c["tool"] for c in calls] == [SEARCH_TOOL_NAME]
    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME]
    assert query_word in ran[0]["args"]["query"].lower()
    # The answer call saw the rows and the search-first instruction, with
    # tools still offered (not the no-tools retry).
    second = seen[1]
    assert second["with_tools"] is True
    assert any(m.get("role") == "tool" and row in (m.get("content") or "") for m in second["messages"])
    assert _is_nudge(second["messages"][-1])
    # The declining draft never reaches the user or the next call.
    assert not any(draft == (m.get("content") or "") for m in second["messages"])
    end = _end(events)
    assert end["content"].startswith(answer)
    assert SEARCH_TOOL_NAME in end["tools"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", HELD_CASES)
async def test_stream_first_pass_holding_the_fact_makes_no_extra_call(monkeypatch, case):
    question, excerpt, answer = case
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool("unused", ran))

    events = await _stream(agent, question)

    assert len(seen) == 1
    assert ran == []
    assert not [e for e in events if e.get("type") == "tool_call"]
    assert _end(events)["content"].startswith(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_stream_project_without_documents_still_refuses_plainly(monkeypatch, case):
    question = case[0]
    _ready(monkeypatch, False)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([case[2]], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool("unused", ran))

    events = await _stream(agent, question)

    assert seen == [] and ran == []
    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    assert text == _UNINDEXED_PROJECT_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_stream_unindexed_project_with_facts_does_not_search(monkeypatch, case):
    """Non-document context lets the turn run, but with nothing indexed the
    guard has nothing to search: the draft stands as written."""
    question, excerpt, draft = case[0], case[1], case[2]
    _ready(monkeypatch, False, non_rag=True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool("unused", ran))

    events = await _stream(agent, question)

    assert ran == []
    assert not [e for e in events if e.get("type") == "tool_call"]
    # Any later call is the pre-existing no-tools retry, never a search.
    assert all(s["with_tools"] is False for s in seen[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("promise", [
    "Let me search the specification for the curing period.",
    "I'll pull the drainage specification to check the outfall material.",
])
async def test_stream_search_promise_keeps_tools_and_runs_the_search(monkeypatch, promise):
    """A promise to search with the budget unused is answered with a search,
    not with a no-tools retry that says 'Do not search again'."""
    question, excerpt, _draft, row, answer, _w = MISS_CASES[0]
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([promise, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool(row, ran))

    events = await _stream(agent, question)

    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME]
    assert all(s["with_tools"] for s in seen)
    assert not any("Do not search again" in (m.get("content") or "")
                   for s in seen for m in s["messages"])
    assert _end(events)["content"].startswith(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_stream_search_round_runs_once_per_turn(monkeypatch, case):
    """Bounded: if the rows do not carry it either, the second decline stands."""
    question, excerpt, draft, row = case[0], case[1], case[2], case[3]
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, draft], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool("unrelated row", ran))

    events = await _stream(agent, question)

    assert len(ran) == 1
    # After the one round, a repeated decline may only meet the no-tools
    # retry: no second search is offered or run.
    assert seen[1]["with_tools"] is True
    assert all(s["with_tools"] is False for s in seen[2:])
    assert _end(events)["content"]


# ── Non-streaming path shares the same guard ─────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_chat_first_pass_miss_searches_through_the_same_guard(monkeypatch, case):
    question, excerpt, draft, row, answer, query_word = case
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    monkeypatch.setenv("MISSING_INPUT_FETCH", "0")
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool(row, ran))

    result = await agent.chat(question, project_id="proj_sr8")

    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME]
    assert query_word in ran[0]["args"]["query"].lower()
    assert result["answer"].startswith(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", MISS_CASES)
async def test_stream_search_first_instruction_is_given_once(monkeypatch, case):
    """After the guard's round the model may search again on its own; the
    instruction that followed the guard's result is not repeated."""
    question, excerpt, draft, row, answer, _w = case
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent()
    seen: list[dict] = []
    ran: list[dict] = []

    async def fake(messages, api_key, **kwargs):
        seen.append({"messages": list(messages)})
        if len(seen) == 1:
            msg = {"role": "assistant", "content": draft, "tool_calls": []}
        elif len(seen) == 2:
            msg = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "own1", "type": "function", "function": {
                    "name": SEARCH_TOOL_NAME,
                    "arguments": json.dumps({"query": question})}}]}
        else:
            msg = {"role": "assistant", "content": answer, "tool_calls": []}
        return {"status": "success", "choice": {"message": msg, "finish_reason": "stop"}}

    monkeypatch.setattr(agent, "_call_llm", fake)
    monkeypatch.setattr(agent, "_run_tool_call", _search_tool(row, ran))

    events = await _stream(agent, question)

    assert len(ran) == 2
    last = seen[-1]["messages"]
    assert sum(1 for m in last if _is_nudge(m)) == 1
    assert _end(events)["content"].startswith(answer)


@pytest.mark.parametrize("question", [
    "what curing period is given for the ground floor slab pour",
    "which gauge of mesh is listed for the perimeter fence panels",
])
def test_operator_ask_survives_the_search_first_nudge(question):
    """The nudge is the platform's bubble: after it is appended the
    operator's question is still the latest operator ask (readers that
    graft or gate on the ask must not read the nudge instead)."""
    from app.agents.runtime import _latest_operator_ask, _operator_user_text

    state = {"nudge": True}
    messages = [{"role": "user", "content": question},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "content": "row"}]
    messages.extend(_search_first_nudges(None, state))
    assert _is_nudge(messages[-1])
    assert _latest_operator_ask(messages) == question
    assert _operator_user_text(messages) == question
    assert _search_first_nudges(None, state) == []
