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
through ``POST /v1/chat/stream``, with the REAL agent and only the model
(``Agent._call_llm``, the last hop) and the retrieval store stubbed, so the
answer is stored where the agent really stores it: ask, then export the
answer as a document.

* its Sources list names the documents the answer cited, in the order the
  answer cited them, then the turn's other sources in the order ``end``
  carried them -- never the "none cited" line;
* an answer that truly cites nothing (no source on the turn) ends with the
  "none cited" line;
* two turns interleaved on one loop each export only their own sources;
* an assistant message stored without structured citations (as every
  message stored before W2-a was) exports through the text fallback, in the
  order its text cites them.

Every user, project, answer and document name is invented by the test; the
expected source lines are derived from the test's own answer and chunks.
"""
from __future__ import annotations

import asyncio
import contextvars
import io
import json
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.core.rag.vector_store import Chunk
from app.main import app
from app.routers import chat_watchdog, exports
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]


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


# ── a citation's file name stays whole ──────────────────────────────────────
# Found by the ordered-sources test below: "[source: Headwall Sections.dwg]"
# was read as document "Headwall", section "Sections.dwg", because the
# locator words (section, page, clause) were matched inside the file name.
# A locator follows the name; it never sits inside it.

@pytest.mark.parametrize("name", [
    "Headwall Sections.dwg", "Spec Section 03.pdf", "Pages From Survey.pdf",
    "Clause Map v2.xlsx", "Kerb pp. Notes.docx",
])
@pytest.mark.parametrize("locator", ["", "p. 4", "section 2.1", "clause 7.3"])
def test_a_locator_word_inside_a_file_name_is_part_of_the_name(name, locator):
    from app.agents import answer_contract

    cite = f"{name}, {locator}" if locator else name
    done = answer_contract.apply(f"Posts are set at 2.4 m centres [source: {cite}].", sources=[])
    assert [(c["document"], c["section"]) for c in done.citations] == [(name, locator)], done.citations


# ── through the real route: ask, then export ────────────────────────────────

# Each turn: the question, the invented documents retrieval hands the model
# (name, text), and the order in which the model's answer cites them. Every
# turn leaves one retrieved document uncited: it reaches the export only
# through the citations stored with the answer, never through the text, so
# a turn whose citations were not stored cannot pass.
TURNS = [
    {
        "ask": "What do the culvert documents say about the headwalls?",
        "title": "Culvert Headwalls",
        "docs": [("Structures Register.pdf", "Headwalls are cast in C32/40 concrete."),
                 ("Headwall Sections.dwg", "Wing walls splay at 45 degrees."),
                 ("Apron Notes.txt", "The apron is 300 mm thick.")],
        "cites": [1, 0],
    },
    {
        "ask": "Summarise the boundary fencing requirements",
        "title": "Boundary Fencing",
        "docs": [("Fencing Spec Volume 2.docx", "Posts are set at 2.4 m centres."),
                 ("Gate Schedule.xlsx", "Gates open inward."),
                 ("Site Boundary Plan.pdf", "The boundary runs 640 m.")],
        "cites": [2, 0],
    },
]


def _answer(turn: dict) -> str:
    """The model's answer: one sentence per cited document, in ``cites`` order."""
    lines = [f"{turn['docs'][i][1]} [source: {turn['docs'][i][0]}]" for i in turn["cites"]]
    return f"## {turn['title']}\n\n" + "\n".join(lines)


def _expected_sources(turn: dict, panel: list[str]) -> list[str]:
    cited = [turn["docs"][i][0] for i in turn["cites"]]
    return cited + [name for name in panel if name not in cited]


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


def _project(client, headers: dict, name: str) -> str:
    res = client.post("/v1/projects", json={"name": f"{name}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


@pytest.fixture
def real_agent(monkeypatch):
    """The real agent; only the model and the retrieval store are stubbed.

    Each turn is told apart by its own question: retrieval returns that
    turn's invented chunks and the model answers with that turn's answer.
    With ``hold`` set, every model call waits until all the named turns have
    reached the model, so their streams are interleaved on the one loop.
    """
    by_ask = {t["ask"]: t for t in TURNS}
    state = {"answers": {}, "chunks": {}, "hold": set(), "arrived": set(),
             "lock": threading.Lock()}

    def _turn_of(text: str) -> dict | None:
        return next((t for ask, t in by_ask.items() if ask in text), None)

    def fake_retrieve(query, project_id, k=5, **_kw):
        turn = _turn_of(query)
        docs = state["chunks"].get(turn["ask"], turn["docs"]) if turn else []
        return [Chunk(chunk_id=f"{project_id}:d{i}:0", project_id=project_id, doc_id=f"d{i}",
                      chunk_index=0, text=text, score=0.9 - i * 0.1, source_name=name)
                for i, (name, text) in enumerate(docs)][:k], 0

    async def fake_llm(self, messages, api_key, project_id=None, **_kw):
        asks = " ".join(str(m.get("content") or "") for m in messages if m.get("role") == "user")
        turn = _turn_of(asks)
        if turn and state["hold"]:
            with state["lock"]:
                state["arrived"].add(turn["ask"])
            deadline = time.monotonic() + 20
            while not state["hold"] <= state["arrived"] and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
        text = state["answers"].get(turn["ask"], _answer(turn)) if turn else "No answer."
        return {"status": "success", "choice": {"message": {"content": text, "tool_calls": []}},
                "raw": {}}

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setenv("CHAT_SSE_WATCHDOG", "1")
    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.core.rag.inject.retrieve_with_filter", fake_retrieve)
    monkeypatch.setattr("app.routers.chat._predefined_enabled", lambda: False)
    return state


def _ask(client, headers: dict, pid: str, ask: str, pin: str = "") -> tuple[str, dict]:
    cid = f"ws-{pid}-{int(time.time() * 1000)}"
    body = {"message": ask, "project_id": pid, "conversation_id": cid}
    if pin:
        body["agent"] = pin
    with client.stream("POST", "/v1/chat/stream", json=body, headers=headers) as stream:
        assert stream.status_code == 200
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    end = [e for e in events if e.get("type") == "end"]
    assert end, events
    return cid, end[-1]


def _export_sources(client, headers: dict, pid: str, cid: str) -> list[str]:
    """The lines under the produced document's last 'Sources' heading, in order."""
    from docx import Document

    path = app.url_path_for(
        exports.export_conversation_message.__name__, project_id=pid, conversation_id=cid,
    )
    got = client.post(f"{path}?format=docx&scope=document", headers=headers)
    assert got.status_code == 200, got.text
    texts = [p.text.strip() for p in Document(io.BytesIO(got.content)).paragraphs if p.text.strip()]
    assert "Sources" in texts, texts
    return texts[len(texts) - texts[::-1].index("Sources"):]


def _names(lines: list[str]) -> list[str]:
    return [line.split(",")[0].strip() for line in lines]


@pytest.mark.parametrize("pin", ["", "project-assistant"])
@pytest.mark.parametrize("case", range(len(TURNS)))
def test_the_exported_document_lists_the_cited_sources_in_order(client, real_agent, pin, case):
    turn = TURNS[case]
    headers, _uid = _register(client, f"{case}-{pin or 'auto'}")
    pid = _project(client, headers, f"SrcRoute-{case}")

    cid, end = _ask(client, headers, pid, turn["ask"], pin)
    panel = [row["doc_name"] for row in end.get("sources") or []]
    assert panel, end

    got = _export_sources(client, headers, pid, cid)
    assert exports._NO_SOURCES_LINE not in got, got
    assert _names(got) == _expected_sources(turn, panel), (got, panel)


@pytest.mark.parametrize("case", range(len(TURNS)))
def test_an_answer_that_cites_nothing_ends_with_the_none_cited_line(client, real_agent, case):
    turn = TURNS[case]
    real_agent["chunks"][turn["ask"]] = []
    real_agent["answers"][turn["ask"]] = (
        f"## {turn['title']}\n\nNo project document covers this; it is general practice."
    )
    headers, _uid = _register(client, f"none-{case}")
    pid = _project(client, headers, f"SrcNone-{case}")

    cid, end = _ask(client, headers, pid, turn["ask"], "project-assistant")
    assert not end.get("sources"), end
    assert _export_sources(client, headers, pid, cid) == [exports._NO_SOURCES_LINE]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_two_interleaved_turns_each_export_only_their_own_sources(client, real_agent, order):
    real_agent["hold"] = {t["ask"] for t in TURNS}
    tag = "".join(map(str, order))
    users = [_register(client, f"pair{n}-{tag}") for n in range(len(TURNS))]
    pids = [_project(client, users[n][0], f"SrcPair{n}-{tag}") for n in range(len(TURNS))]
    results: dict[int, tuple[str, dict]] = {}
    errors: list[BaseException] = []

    def run(n: int) -> None:
        try:
            results[n] = _ask(client, users[n][0], pids[n], TURNS[n]["ask"], "project-assistant")
        except BaseException as exc:  # noqa: BLE001 -- re-raised by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(n,)) for n in order]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    assert real_agent["arrived"] == real_agent["hold"], "the two turns did not overlap"

    for n, turn in enumerate(TURNS):
        cid, end = results[n]
        panel = [row["doc_name"] for row in end.get("sources") or []]
        got = _names(_export_sources(client, users[n][0], pids[n], cid))
        assert got == _expected_sources(turn, panel), (n, got, panel)
        others = {name for m, t in enumerate(TURNS) if m != n for name, _ in t["docs"]}
        assert not others & set(got), (n, got)


@pytest.mark.parametrize("case", range(len(TURNS)))
def test_a_message_stored_without_citations_exports_through_the_text(client, case):
    """A message stored without structured citations: the export reads the
    sources off the answer's own text, in the order the text cites them."""
    from app.core import agent_memory

    turn = TURNS[case]
    headers, uid = _register(client, f"old-{case}")
    pid = _project(client, headers, f"SrcOld-{case}")
    cid = f"ws-{pid}-{int(time.time() * 1000)}"
    agent_memory.get_or_create_conversation(cid, "project-assistant", pid, owner_id=uid)
    agent_memory.append_message(cid, "user", turn["ask"])
    agent_memory.append_message(cid, "assistant", _answer(turn))
    stored = [m for m in agent_memory.get_messages(cid) if m.get("role") == "assistant"]
    assert stored and not stored[-1].get("citations"), stored

    got = _export_sources(client, headers, pid, cid)
    assert _names(got) == [turn["docs"][i][0] for i in turn["cites"]], got
