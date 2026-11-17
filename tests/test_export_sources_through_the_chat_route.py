"""The produced document lists the sources the answer left with, on the REAL
chat route (advisor 21:38Z, tester 21:09Z).

Live: an answer cited three documents inline and the side panel showed ten
sources, yet the Word file pressed under it ended with "No document was cited
in this answer". The stored assistant message carried no citations.

Why (the function under test is
``app.routers.chat_watchdog.guarantee_terminal``): the SSE watchdog that
wraps every chat stream pulled each frame with
``asyncio.wait_for(iterator.__anext__(), ...)``. On Python 3.11 that runs
every step as its own task, and a task runs in a COPY of the caller's
context. ``exit_frames.with_answer_exit`` puts the turn's stored-answer
holder in a context variable while the stream produces its FIRST frame
(``start``); the agent stores its answer frames later, saw no holder, and
the citations were never attached. The W2-a test drove ``with_answer_exit``
directly, storing before its first frame, so it never crossed the watchdog:
that is the boundary it did not cross.

The rule held here: one wrapped stream runs in one context, so a context
variable it sets while producing one frame is there for the next. And
through ``POST /v1/chat/stream`` (both the automatic route and a pinned
agent): ask, then export the answer as a document; its Sources list names
the documents the stream's ``end`` carried, never the "none cited" line.

Every user, project, answer and document name is invented by the test; the
expected source lines are read off the test's own ``end`` frame.
"""
from __future__ import annotations

import asyncio
import contextvars
import io
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import chat_watchdog, exports
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]

TURNS = [
    {
        "ask": "Put the culvert headwall details in a document",
        "answer": "## Culvert Headwalls\n\nHeadwalls are cast in C32/40 concrete "
                  "(Structures Register.pdf, p. 4).\nWing walls splay at 45 degrees.",
        "sources": [
            {"doc_name": "Structures Register.pdf", "page_or_section": "p. 4"},
            {"doc_name": "Headwall Sections.dwg", "page_or_section": "Sheet 3"},
        ],
    },
    {
        "ask": "Summarise the fencing specification as a downloadable note",
        "answer": "## Boundary Fencing\n\nPosts are set at 2.4 m centres "
                  "(Fencing Spec Volume 2.docx, clause 7.1).\nGates open inward.",
        "sources": [
            {"doc_name": "Fencing Spec Volume 2.docx", "page_or_section": "clause 7.1"},
            {"doc_name": "Gate Schedule.xlsx", "page_or_section": "Sheet 1"},
            {"doc_name": "Site Boundary Plan.pdf", "page_or_section": "p. 2"},
        ],
    },
]


def _source_line(row: dict) -> str:
    return f"{row['doc_name']}, {row['page_or_section']}"


# ── the mechanism: one stream, one context ──────────────────────────────────

_SEEN: contextvars.ContextVar = contextvars.ContextVar("seen", default=None)


@pytest.mark.parametrize("frames", [3, 6])
def test_a_variable_set_while_producing_one_frame_is_there_for_the_next(frames):
    async def producer():
        for n in range(frames):
            if n == 0:
                _SEEN.set({"set_at": 0})
            held = _SEEN.get()
            yield f"data: {json.dumps({'type': 'token', 'content': str(held)})}\n\n"
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    async def run():
        return [f async for f in chat_watchdog.guarantee_terminal(producer(), timeout_s=30)]

    out = asyncio.run(run())
    tokens = [json.loads(f[6:])["content"] for f in out if '"token"' in f]
    assert tokens == [str({"set_at": 0})] * frames, tokens


@pytest.mark.parametrize("frames", [2, 5])
def test_the_watchdog_still_ends_a_stream_that_runs_past_its_deadline(frames):
    async def producer():
        for n in range(frames):
            yield f"data: {json.dumps({'type': 'token', 'content': str(n)})}\n\n"
        await asyncio.sleep(5)
        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    async def run():
        return [f async for f in chat_watchdog.guarantee_terminal(producer(), timeout_s=0.3)]

    out = [json.loads(f[6:]) for f in asyncio.run(run())]
    assert [e["content"] for e in out[:frames]] == [str(n) for n in range(frames)]
    assert [e["type"] for e in out[frames:]] == ["error", "end"], out


# ── through the real route: ask, then export ────────────────────────────────

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> tuple[dict, str]:
    email = f"srcroute-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    body = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()
    return {"Authorization": f"Bearer {body['token']}"}, body.get("user_id") or body.get("user", {}).get("id")


class _StoringAgent:
    """Streams like the real agent: the answer is stored AFTER the first
    frames have left (through the same off-loop store call), then ``end``
    carries the turn's sources."""

    name = "project-assistant"

    def __init__(self, turn: dict):
        self.turn = turn

    async def chat_stream(self, prompt, history=None, project_id=None,
                          conversation_id=None, user_id=None, attached_documents=None, **_kw):
        from app.core import agent_memory
        from app.core.offload import off_loop

        await off_loop(agent_memory.get_or_create_conversation, conversation_id, self.name,
                       project_id, owner_id=user_id)
        await off_loop(agent_memory.append_message, conversation_id, "user", prompt)
        yield {"type": "start"}
        yield {"type": "token", "content": self.turn["answer"]}
        await off_loop(agent_memory.append_message, conversation_id, "assistant", self.turn["answer"])
        yield {"type": "end", "content": self.turn["answer"], "sources": self.turn["sources"]}


def _docx_texts(raw: bytes) -> list[str]:
    from docx import Document

    return [p.text.strip() for p in Document(io.BytesIO(raw)).paragraphs if p.text.strip()]


@pytest.mark.parametrize("pin", ["", "project-assistant"])
@pytest.mark.parametrize("case", range(len(TURNS)))
def test_the_exported_document_lists_the_sources_the_route_sent(client, monkeypatch, pin, case):
    from app.agents import runtime

    turn = TURNS[case]
    agent = _StoringAgent(turn)
    monkeypatch.setenv("CHAT_SSE_WATCHDOG", "1")
    monkeypatch.setattr("app.agents.get_agent", lambda name: agent)

    async def _same_agent(message, a):
        return a, {"requested": a.name, "final": a.name, "action": None,
                   "confidence": 1.0, "reason": "test"}

    monkeypatch.setattr(runtime, "select_agent_for_message", _same_agent)
    monkeypatch.setattr("app.routers.chat._predefined_enabled", lambda: False)

    headers, _uid = _register(client, f"{case}-{pin or 'auto'}")
    res = client.post("/v1/projects", json={"name": f"SrcRoute-{case}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    pid = res.json()["id"]
    cid = f"ws-{pid}-{int(time.time() * 1000)}"

    body = {"message": turn["ask"], "project_id": pid, "conversation_id": cid}
    if pin:
        body["agent"] = pin
    with client.stream("POST", "/v1/chat/stream", json=body, headers=headers) as stream:
        assert stream.status_code == 200
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    end = [e for e in events if e.get("type") == "end"]
    assert end and end[-1].get("sources"), events
    panel = [_source_line(r) for r in end[-1]["sources"]]

    path = app.url_path_for(
        exports.export_conversation_message.__name__, project_id=pid, conversation_id=cid,
    )
    got = client.post(f"{path}?format=docx&scope=document", headers=headers)
    assert got.status_code == 200, got.text
    texts = _docx_texts(got.content)

    assert exports._NO_SOURCES_LINE not in texts, texts
    tail = texts[-len(panel):]
    assert sorted(tail) == sorted(panel), (tail, panel)
