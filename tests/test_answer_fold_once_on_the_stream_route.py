"""Rule 5 on the real stream route: the fold is decided once per answer.

On the chat route one end event passes the exit twice: the agent's own end
check (``Agent.chat_stream``) and then the route's (``with_answer_exit``).
On the default route the agent also runs in its own task, so the route's
pass reads a turn that does not know the answer was cut. These tests drive
the real agent stream (a scripted model reply, nothing else stubbed in the
exit) inside its own task, through the real route wrapper, and read the
event the client gets. Every input is built here.
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents import answer_exit
from app.core.rag.inject import format_chunks_as_system_message
from app.core.rag.vector_store import Chunk
from app.routers.exit_frames import with_answer_exit
from app.routers.hat_frames import _decode, _encode
from tests._fixture_values import STUB_TOKEN_02


def _docs(*named: tuple[str, int]) -> list[dict]:
    return [{"doc_id": f"{i + 1:032x}", "name": name, "chunk": i, "page": page}
            for i, (name, page) in enumerate(named)]


def _agent(monkeypatch, reply: str, docs: list[dict], cut: bool):
    from app.agents import runtime
    from app.agents.runtime import Agent

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    agent = Agent(name="fold-route-agent", description="t", system_prompt="t", allowed_blocks=[])

    async def fake_call_llm(messages, api_key, **kwargs):
        # A reply the provider stopped at its output limit is ended the way
        # the real call ends it (which marks this turn cut, in this task).
        content = runtime._end_cut_answer(reply) if cut else reply
        return {"status": "success",
                "choice": {"message": {"content": content, "tool_calls": []},
                           "finish_reason": "length" if cut else "stop"}}

    chunks = []
    for d in docs:
        c = Chunk(chunk_id=f"p:{d['doc_id']}:{d['chunk']}", project_id="p", doc_id=d["doc_id"],
                  chunk_index=d["chunk"], text="The stated figures are in this document.", score=0.8)
        c.page, c.source_name, c.revision, c.superseded = d["page"], d["name"], "B", False
        chunks.append(c)

    def fake_rag_inject(**kwargs):
        return (format_chunks_as_system_message(chunks, total_candidates=len(chunks), query="q"),
                {"project_id": "proj_x", "chunks": [
                    {"doc_id": d["doc_id"], "chunk_index": d["chunk"], "chunk_id": f"x:{d['doc_id']}:{d['chunk']}",
                     "score": 0.8, "page": d["page"]} for d in docs]})

    by_id = {d["doc_id"]: {"original_name": d["name"]} for d in docs}
    monkeypatch.setattr(agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr("app.agents.runtime.rag_inject", fake_rag_inject)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.core.projects.get_document", lambda did: by_id.get(did))
    return agent


async def _client_end(agent, question: str) -> dict:
    """The end event the client reads: the agent's stream runs in its own
    task (as the chat route runs it) and its frames pass the route's exit."""
    async def frames():
        q: asyncio.Queue = asyncio.Queue()
        done = object()

        async def produce():
            try:
                async for event in agent.chat_stream(question, project_id="proj_x"):
                    await q.put(_encode(event))
            finally:
                await q.put(done)

        task = asyncio.create_task(produce())
        while (frame := await q.get()) is not done:
            yield frame
        await task

    out = [f async for f in with_answer_exit(frames(), question)]
    ends = [e for e in map(_decode, out) if e and e.get("type") == "end"]
    assert ends, "the route sent no end event"
    return ends[-1]


def _shown(end: dict) -> str:
    """Everything the bubble can show: the lead and the fold."""
    return (end.get("content") or "") + "\n" + (end.get("details") or "")


# ── a list whose items carry a marker from a later sentence: nothing is lost ──

_LISTS = [
    ("The pumps are listed below:\n\n"
     "- Pump P1 is rated at 40 l/s (Plant handbook.pdf, p. 4). It sits in the basement plant room "
     "(Pump schedule.pdf, p. 9).\n"
     "- Pump P2 is rated at 25 l/s. It sits on level 3.",
     [("Plant handbook.pdf", 4), ("Pump schedule.pdf", 9)],
     ["basement plant room", "It sits on level 3"]),
    ("Two hoists serve the core:\n\n"
     "- Hoist H1 lifts 2000 kg (Hoist data.pdf, p. 2). It reaches the roof slab (Core sections.pdf, p. 7).\n"
     "- Hoist H2 lifts 1200 kg. It stops at level 14.",
     [("Hoist data.pdf", 2), ("Core sections.pdf", 7)],
     ["reaches the roof slab", "It stops at level 14"]),
]


@pytest.mark.parametrize("reply,named,moved", _LISTS)
def test_every_moved_sentence_reaches_the_client_on_the_route(monkeypatch, reply, named, moved):
    agent = _agent(monkeypatch, reply, _docs(*named), cut=False)
    end = asyncio.run(_client_end(agent, "Which units are there?"))
    assert end.get("details"), "a finished document answer with a list is folded"
    for sentence in moved:
        assert sentence in _shown(end)
        assert sentence not in end["content"]
    assert end["folded"] is True


# ── an answer cut part-way is read whole, its notice in the visible text ─────

_TWO_PARS = [
    ("The slab is 250 mm thick (Slab spec.pdf, p. 3).\n\nThe topping is 50 mm (Slab spec.pdf, p. 4).",
     [("Slab spec.pdf", 3)], "The topping is 50 mm"),
    ("The wall is 200 mm thick (Wall spec.pdf, p. 1).\n\nIt is fair-faced on both sides.",
     [("Wall spec.pdf", 1)], "It is fair-faced on both sides."),
]


@pytest.mark.parametrize("reply,named,later", _TWO_PARS)
def test_a_cut_answer_is_read_whole_on_the_route(monkeypatch, reply, named, later):
    from app.agents.runtime import _LENGTH_CUT_NOTICE

    agent = _agent(monkeypatch, reply, _docs(*named), cut=True)
    end = asyncio.run(_client_end(agent, "How thick is it?"))
    assert later in end["content"]
    assert _LENGTH_CUT_NOTICE in end["content"]
    assert not end.get("details")
    assert end["folded"] is False


@pytest.mark.parametrize("reply,named,later", _TWO_PARS)
def test_the_same_answer_finished_is_folded_on_the_route(monkeypatch, reply, named, later):
    agent = _agent(monkeypatch, reply, _docs(*named), cut=False)
    end = asyncio.run(_client_end(agent, "How thick is it?"))
    assert later not in end["content"]
    assert later in end["details"]
    assert end["folded"] is True


# ── the decision recorded on the event is what a later pass keeps ──────────

_ROW = {"doc_name": "Plant handbook.pdf", "page_or_section": "p. 4"}


@pytest.mark.parametrize("answer", [
    "The gate is grey.\n\nThe fence is green.\n\nThe kerb is precast.",
    "Level 3 has 42 bays.\n\nLevel 2 has 38.",
])
def test_a_second_exit_pass_keeps_the_first_fold(answer):
    once = answer_exit.check_end_event({"type": "end", "content": answer, "sources": [dict(_ROW)]})
    twice = answer_exit.check_end_event(dict(once), "", None)
    assert twice["content"] == once["content"]
    assert twice["details"] == once["details"]
    assert twice["folded"] is True


@pytest.mark.parametrize("answer", [
    "The gate is grey.\n\nThe fence is green.",
    "Level 3 has 42 bays.\n\nLevel 2 has 38.",
])
def test_an_event_recorded_as_read_whole_is_not_folded_by_a_later_pass(answer):
    out = answer_exit.check_end_event({"type": "end", "content": answer, "sources": [dict(_ROW)],
                                       "folded": False})
    assert out["content"] == answer
    assert "details" not in out
    assert out["folded"] is False


@pytest.mark.parametrize("marker", ["yes", 1, None])
def test_a_folded_field_that_is_not_a_decision_is_ignored(marker):
    # Only a True/False the exit wrote counts as a decision; anything else on
    # the event is not one, and the answer is folded as usual.
    out = answer_exit.check_end_event({"type": "end", "content": "The gate is grey.\n\nThe fence is green.",
                                       "sources": [dict(_ROW)], "folded": marker})
    assert out["content"] == "The gate is grey."
    assert out["details"] == "The fence is green."
    assert out["folded"] is True


@pytest.mark.parametrize("answer", ["The gate is grey.", "Level 3 has 42 bays."])
def test_a_one_paragraph_answer_is_recorded_as_read_whole(answer):
    out = answer_exit.check_end_event({"type": "end", "content": answer, "sources": [dict(_ROW)]})
    assert out["content"] == answer
    assert out["folded"] is False


@pytest.mark.parametrize("answer", ["The gate is grey.\n\nThe fence is green.", "Level 3 has 42 bays.\n\nLevel 2 has 38."])
def test_a_reader_with_no_fold_gets_no_decision(answer):
    # A whole-turn result is read with no fold; it carries no folded field.
    out = answer_exit.check_result({"answer": answer, "sources": [dict(_ROW)]})
    assert "folded" not in out and "details" not in out
