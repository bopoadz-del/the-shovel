"""D5: a refusal stands only after the encoded knowledge has been consulted.

Live 24 Nov (exam 7, 40 course asks): the draft said "not in the project
documents", the search-before-refusal guard then searched the project
documents only, and the refusal stood. The encoded construction knowledge
is not in the project index, so that search could never find it. The guard
under test reads only turn state (tool offered, tool already called, one
round per turn) plus the same pure decline test as the project round.

Every test builds its own question, draft, rows and knowledge entry; two
inputs each. The knowledge tool is faked: what is tested is that the turn
reaches it, with the asked thing as its query, and that the instruction
after it asks for the entry's source.
"""
from __future__ import annotations

import json

import pytest

from app.agents.core.context import (
    KNOWLEDGE_FIRST_NUDGE,
    KNOWLEDGE_TOOL_NAME,
    SEARCH_FIRST_NUDGE,
    SEARCH_TOOL_NAME,
    _refusal_query,
    _search_first_nudges,
    _tool_offered,
    knowledge_before_refusal_call,
    knowledge_before_refusal_due,
    search_before_refusal_call,
)
from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_02

# (question, excerpt handed first, declining draft, unrelated project row,
#  knowledge entry statement, its declared source, answer from the entry,
#  a word the consultation query must carry)
COURSE_CASES = [
    (
        "What minimum factor of safety against uplift applies to a basement slab?",
        "Basement walls shall be tanked with a bonded membrane.",
        ("The project documents do not state the minimum factor of safety "
         "against uplift for the basement slab."),
        "Membrane laps shall be not less than 100 mm.",
        "Uplift resistance shall give a factor of safety of not less than 1.4.",
        "Course notes, foundations module, p3",
        ("The minimum factor of safety against uplift is 1.4 (encoded "
         "knowledge: Course notes, foundations module, p3)."),
        "uplift",
    ),
    (
        "What maximum slump is allowed for slipform kerb concrete?",
        "Kerbs shall be laid to the line and level shown.",
        ("I could not find the maximum slump for slipform kerb concrete in "
         "the project documents."),
        "Kerb joints shall be at 3 m centres.",
        "Slipform kerb concrete shall have a slump not exceeding 25 mm.",
        "Course notes, pavements module, p8",
        ("Slipform kerb concrete may have a slump of at most 25 mm (encoded "
         "knowledge: Course notes, pavements module, p8)."),
        "slump",
    ),
]

ANSWER_DRAFTS = [
    "The minimum factor of safety against uplift is 1.4 (Course notes).",
    "The slump for slipform kerb concrete must not exceed 25 mm.",
]


def _due(draft: str, **change) -> bool:
    state = {
        "knowledge_offered": True, "knowledge_consulted": False,
        "force_synthesis": False, "already_ran": False, "iterations_left": 5,
    }
    state.update(change)
    return knowledge_before_refusal_due(draft, **state)


# ── The decision, pure ───────────────────────────────────────────────────
@pytest.mark.parametrize("draft", [c[2] for c in COURSE_CASES])
def test_due_when_a_decline_has_not_met_the_knowledge(draft):
    assert _due(draft) is True


@pytest.mark.parametrize("answer", ANSWER_DRAFTS)
def test_not_due_when_the_draft_answers(answer):
    assert _due(answer) is False


@pytest.mark.parametrize("change", [
    {"knowledge_offered": False},
    {"knowledge_consulted": True},
    {"force_synthesis": True},
    {"already_ran": True},
    {"iterations_left": 1},
])
@pytest.mark.parametrize("draft", [c[2] for c in COURSE_CASES])
def test_not_due_when_turn_state_forbids(draft, change):
    assert _due(draft, **change) is False


@pytest.mark.parametrize("case", COURSE_CASES)
def test_consultation_queries_what_the_draft_found_missing(case):
    question, draft, word = case[0], case[2], case[7]
    call = knowledge_before_refusal_call(draft, question, 3)
    assert call["function"]["name"] == KNOWLEDGE_TOOL_NAME
    assert call["id"] == "call_knowledge_first_3"
    query = json.loads(call["function"]["arguments"])["query"]
    assert word in query.lower()
    # One query rule for both rounds: they look for the same missing thing.
    search = json.loads(search_before_refusal_call(draft, question, 3)["function"]["arguments"])
    assert query == search["query"] == _refusal_query(draft, question)


class _Roster:
    def __init__(self, names):
        self.names = names

    def tool_definitions(self, project_id=None):
        return [{"type": "function", "function": {"name": n}} for n in self.names]


@pytest.mark.parametrize("names, excluded, offered", [
    ([SEARCH_TOOL_NAME, KNOWLEDGE_TOOL_NAME], set(), True),
    ([KNOWLEDGE_TOOL_NAME], set(), True),
    ([SEARCH_TOOL_NAME, "remember_fact"], set(), False),
    ([SEARCH_TOOL_NAME, KNOWLEDGE_TOOL_NAME], {KNOWLEDGE_TOOL_NAME}, False),
])
def test_tool_offered_reads_the_roster_and_the_exclusions(names, excluded, offered):
    assert _tool_offered(_Roster(names), KNOWLEDGE_TOOL_NAME, "p1", excluded) is offered


@pytest.mark.parametrize("flags, tools", [
    ({"nudge": True}, [SEARCH_TOOL_NAME]),
    ({"kb_nudge": True}, [KNOWLEDGE_TOOL_NAME]),
    ({"nudge": True, "kb_nudge": True}, [SEARCH_TOOL_NAME, KNOWLEDGE_TOOL_NAME]),
    ({}, []),
])
def test_each_round_owes_its_instruction_once(flags, tools):
    state = dict(flags)
    first = _search_first_nudges(None, state)
    assert [m["content"].split(" ran. ")[0].split()[-1] for m in first] == tools
    for msg, name in zip(first, tools):
        nudge = SEARCH_FIRST_NUDGE if name == SEARCH_TOOL_NAME else KNOWLEDGE_FIRST_NUDGE
        assert msg["role"] == "user" and msg["content"].endswith(nudge)
    assert _search_first_nudges(None, state) == []


# ── The live UI path (streaming) ─────────────────────────────────────────
@pytest.fixture(autouse=True)
def _stub_llm_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


@pytest.fixture(autouse=True)
def _names_follow_the_document_list(monkeypatch):
    from app.core import projects as _p
    monkeypatch.setattr(_p, "list_document_names", lambda pid: _p.list_documents(pid))


def _agent(monkeypatch, offers_knowledge: bool) -> Agent:
    agent = Agent(
        name="project-assistant", description="pa",
        system_prompt="You answer from project documents.", allowed_blocks=[],
    )
    names = [SEARCH_TOOL_NAME] + ([KNOWLEDGE_TOOL_NAME] if offers_knowledge else [])
    monkeypatch.setattr(agent, "tool_definitions", _Roster(names).tool_definitions)
    return agent


def _ready(monkeypatch, ready: bool, non_rag: bool = False):
    async def _is_ready(_pid):
        return ready
    monkeypatch.setattr("app.agents.runtime._project_is_rag_ready_off_process", _is_ready)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: ready)
    monkeypatch.setattr("app.agents.runtime._project_has_non_rag_context", lambda pid, msg: non_rag)


def _inject(monkeypatch, excerpt: str):
    def fake_rag_inject(**kwargs):
        return (
            {"role": "system", "content": f"Relevant project context.\n{excerpt}"},
            {"project_id": "proj_d5", "chunks": [
                {"doc_id": "doc_1", "chunk_index": 0, "chunk_id": "proj_d5:doc_1:0", "score": 0.5},
            ]},
        )
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr("app.core.projects.get_document", lambda did: {"original_name": "Spec.pdf"})


def _scripted_llm(replies: list[str], seen: list[dict]):
    async def fake(messages, api_key, **kwargs):
        seen.append({"with_tools": kwargs.get("with_tools", True), "messages": list(messages)})
        text = replies[min(len(seen) - 1, len(replies) - 1)]
        return {"status": "success", "choice": {
            "message": {"role": "assistant", "content": text, "tool_calls": []},
            "finish_reason": "stop"}}
    return fake


def _tools(row: str, statement: str, source: str, ran: list[dict]):
    """The project search returns ``row``; the knowledge tool returns one
    entry stating ``statement`` with ``source`` as its provenance."""
    async def fake(tool_call, **kwargs):
        fn = tool_call.get("function") or {}
        name = fn.get("name")
        ran.append({"name": name, "args": json.loads(fn.get("arguments") or "{}")})
        if name == KNOWLEDGE_TOOL_NAME:
            return {"name": name, "ok": True, "result": {"status": "success", "matches": [
                {"id": "kb.case", "statement": statement,
                 "provenance": {"course_reference": source}}], "count": 1}}
        return {"name": name, "ok": True, "result": {"status": "success", "results": [
            {"document_id": "doc_2", "filename": "Spec.pdf", "snippet": row, "score": 0.6}]}}
    return fake


def _is_kb_nudge(msg: dict) -> bool:
    content = str(msg.get("content") or "")
    return msg.get("role") == "user" and content.endswith(KNOWLEDGE_FIRST_NUDGE)


def _end(events):
    ends = [e for e in events if e.get("type") == "end"]
    assert len(ends) == 1, events
    return ends[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COURSE_CASES)
async def test_stream_project_search_miss_then_knowledge_answers(monkeypatch, case):
    question, excerpt, draft, row, statement, source, answer, word = case
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent(monkeypatch, offers_knowledge=True)
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _tools(row, statement, source, ran))

    events = [e async for e in agent.chat_stream(question, project_id="proj_d5")]

    # The project round first (unchanged), then the knowledge round.
    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME, KNOWLEDGE_TOOL_NAME]
    assert word in ran[1]["args"]["query"].lower()
    # A successful knowledge result is synthesised without tools (the
    # existing force-synthesis rule); that call sees the entry and the
    # instruction to name its source.
    third = seen[2]
    assert any(m.get("role") == "tool" and source in (m.get("content") or "")
               for m in third["messages"])
    assert _is_kb_nudge(third["messages"][-1])
    end = _end(events)
    assert end["content"].startswith(answer)
    assert KNOWLEDGE_TOOL_NAME in end["tools"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COURSE_CASES)
async def test_stream_unindexed_project_still_consults_knowledge(monkeypatch, case):
    """Nothing indexed, so no project round; the knowledge round still runs
    (the project's own non-document facts let the turn reach the model)."""
    question, _excerpt, draft, row, statement, source, answer, _w = case
    _ready(monkeypatch, False, non_rag=True)
    agent = _agent(monkeypatch, offers_knowledge=True)
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _tools(row, statement, source, ran))

    events = [e async for e in agent.chat_stream(question, project_id="proj_d5")]

    assert [r["name"] for r in ran] == [KNOWLEDGE_TOOL_NAME]
    assert _is_kb_nudge(seen[1]["messages"][-1])
    assert _end(events)["content"].startswith(answer)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COURSE_CASES)
async def test_stream_knowledge_not_offered_leaves_the_project_round_alone(monkeypatch, case):
    question, excerpt, draft, row, statement, source = case[:6]
    _ready(monkeypatch, True)
    _inject(monkeypatch, excerpt)
    agent = _agent(monkeypatch, offers_knowledge=False)
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, draft], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _tools(row, statement, source, ran))

    events = [e async for e in agent.chat_stream(question, project_id="proj_d5")]

    assert [r["name"] for r in ran] == [SEARCH_TOOL_NAME]
    assert not any(_is_kb_nudge(m) for s in seen for m in s["messages"])
    assert _end(events)["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COURSE_CASES)
async def test_stream_knowledge_round_runs_once_per_turn(monkeypatch, case):
    """Bounded: if the entry does not state it either, the decline stands."""
    question, _excerpt, draft, row = case[:4]
    _ready(monkeypatch, False, non_rag=True)
    agent = _agent(monkeypatch, offers_knowledge=True)
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, draft, draft], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _tools(row, "unrelated entry", "Other notes", ran))

    events = [e async for e in agent.chat_stream(question, project_id="proj_d5")]

    assert [r["name"] for r in ran] == [KNOWLEDGE_TOOL_NAME]
    assert all(s["with_tools"] is False for s in seen[2:])
    assert _end(events)["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", COURSE_CASES)
async def test_chat_consults_knowledge_through_the_same_guard(monkeypatch, case):
    question, _excerpt, draft, row, statement, source, answer, word = case
    _ready(monkeypatch, False, non_rag=True)
    monkeypatch.setenv("MISSING_INPUT_FETCH", "0")
    agent = _agent(monkeypatch, offers_knowledge=True)
    seen: list[dict] = []
    ran: list[dict] = []
    monkeypatch.setattr(agent, "_call_llm", _scripted_llm([draft, answer], seen))
    monkeypatch.setattr(agent, "_run_tool_call", _tools(row, statement, source, ran))

    result = await agent.chat(question, project_id="proj_d5")

    assert [r["name"] for r in ran] == [KNOWLEDGE_TOOL_NAME]
    assert word in ran[0]["args"]["query"].lower()
    assert result["answer"].startswith(answer)
