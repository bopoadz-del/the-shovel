"""A chat session's files are not project evidence anywhere (grader, #400).

A file attached in (or produced by) chat session X belongs to X and to the
user who made it there. Outside X it is not listed, not counted and not
searched: not in the project's document list, not in another session of the
same user, not for another user of the same project. Inside X it is still
the user's to read.

Driven through the routes the browser uses: files are uploaded through
``POST /v1/projects/{pid}/documents``, the
question goes through ``POST /v1/chat/stream`` with the REAL agent and the
REAL retriever; only the model (``Agent._call_llm``, the last hop) is
stubbed, and it records what retrieval put in front of it.

Every user, project, session, file name and file text is invented here.
"""
from __future__ import annotations

import io
import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.main import app
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]

# One rare word every file shares, so the question reaches all of them, and
# one word per file, so the model's input shows which of them it was given.
_SHARED = "quillback"
_FILES = {
    "project": ("invented_cofferdam_method.txt", "zephyrine"),
    "x": ("invented_x_session_note.txt", "marrowind"),
    "stranger": ("invented_stranger_note.txt", "tolquessa"),
}


def _text(word: str) -> bytes:
    return (f"The {_SHARED} sheet piles are driven to refusal. Marker {word} applies "
            f"to the {_SHARED} cofferdam sequence.\n").encode()


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, tag: str) -> tuple[dict, str]:
    email = f"sessev-{tag}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    body = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()
    return ({"Authorization": f"Bearer {body['token']}"},
            body.get("user_id") or body.get("user", {}).get("id"))


def _upload(client, headers, pid, key, session_id=None, uploader_id=None) -> str:
    """Upload through the route; give the row chunks with the real indexer.

    A session file never enters the index today (the upload route stamps it
    not indexed and ``doc_index`` refuses a row naming a session). Rows that
    were chunked BEFORE that rule still exist, and they are what this exam's
    retrieval half is about: such a row is built here as it came about --
    indexed while it named no session -- then given its session and uploader.
    """
    name, word = _FILES[key]
    res = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (name, io.BytesIO(_text(word)), "text/plain")},
        headers=headers,
    )
    assert res.status_code in (200, 201), res.text
    body = res.json()
    did = (body.get("document") or body)["id"]
    from app.core import doc_index
    from app.core import projects as store

    assert doc_index.index_document(pid, did).get("indexed") == 1
    if session_id:
        assert uploader_id
        store.update_document_metadata(
            did, {"session_id": session_id, "uploader_id": uploader_id})
    return did


@pytest.fixture
def model(monkeypatch):
    """The real agent; the model only records its input and answers plainly."""
    seen: list[str] = []

    async def fake_llm(self, messages, api_key, project_id=None, **_kw):
        seen.append("\n".join(str(m.get("content") or "") for m in messages))
        return {"status": "success",
                "choice": {"message": {"content": "The piles are driven to refusal.",
                                       "tool_calls": []}},
                "raw": {}}

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    # The test embedder's cosines are arbitrary; the relevance cut is not
    # under test, which candidates may be retrieved at all is.
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "-1")
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setattr("app.routers.chat._predefined_enabled", lambda: False)
    return seen


def _ask(client, headers, pid, cid, seen) -> tuple[str, set]:
    """Ask in session ``cid``; the model's input and the turn's source names."""
    start = len(seen)
    body = {"message": f"How are the {_SHARED} sheet piles driven?",
            "project_id": pid, "conversation_id": cid, "agent": "project-assistant"}
    with client.stream("POST", "/v1/chat/stream", json=body, headers=headers) as stream:
        assert stream.status_code == 200
        events = [json.loads(line[6:]) for line in stream.iter_lines() if line.startswith("data: ")]
    end = [e for e in events if e.get("type") == "end"]
    assert end, events
    names = {row.get("doc_name") for row in end[-1].get("sources") or []}
    return "\n".join(seen[start:]), names


def _listed(client, headers, pid) -> set:
    res = client.get(f"/v1/projects/{pid}", headers=headers)
    assert res.status_code == 200, res.text
    return {d["id"] for d in res.json()["documents"]}


@pytest.mark.parametrize("y_suffix", ["2000", "mwiqvj9k28n987unh726"])
def test_a_sessions_file_is_not_listed_or_retrieved_outside_it(client, model, y_suffix):
    from app.core import projects as store

    owner, owner_id = _register(client, f"own{y_suffix[:4]}")
    stranger, stranger_id = _register(client, f"oth{y_suffix[:4]}")
    pid = client.post("/v1/projects", json={"name": f"SessEv-{y_suffix[:4]}-{_RUN}"},
                      headers=owner).json()["id"]
    store.approve_project(pid)  # shared: the stranger opens it too
    t0 = int(time.time() * 1000)
    x, y, z = f"ws-{pid}-{t0}", f"ws-{pid}-{y_suffix}", f"ws-{pid}-{t0 + 7}"

    project_doc = _upload(client, owner, pid, "project")
    x_doc = _upload(client, owner, pid, "x", session_id=x, uploader_id=owner_id)
    stranger_doc = _upload(client, stranger, pid, "stranger", session_id=z,
                           uploader_id=stranger_id)

    # Listed: the project's own document, for both users; no session's file.
    for viewer in (owner, stranger):
        listed = _listed(client, viewer, pid)
        assert project_doc in listed
        assert not {x_doc, stranger_doc} & listed, listed

    word = {k: w for k, (_n, w) in _FILES.items()}
    name = {k: n for k, (n, _w) in _FILES.items()}

    # Retrieved in session Y of the same user: the project's file only.
    seen, sources = _ask(client, owner, pid, y, model)
    assert word["project"] in seen, "retrieval reached nothing: the exam proves nothing"
    assert word["x"] not in seen and word["stranger"] not in seen, seen
    assert not {name["x"], name["stranger"]} & sources, sources

    # Retrieved in session X, where the user attached it: X's file is theirs
    # to read; the stranger's session file is still not.
    seen, sources = _ask(client, owner, pid, x, model)
    assert word["x"] in seen, seen
    assert word["stranger"] not in seen and name["stranger"] not in sources

    # The other user, in a session of their own: neither the owner's session
    # file nor (outside Z) their own.
    seen, sources = _ask(client, stranger, pid, f"ws-{pid}-{t0 + 11}", model)
    assert word["project"] in seen
    assert word["x"] not in seen and word["stranger"] not in seen, seen

    # A client may not pin another user's session file by id either.
    res = client.post("/v1/chat/stream", headers=stranger, json={
        "message": "Read this.", "project_id": pid,
        "conversation_id": f"ws-{pid}-{t0 + 13}", "document_ids": [x_doc]})
    assert res.status_code == 404, res.text


@pytest.mark.parametrize("scoped", ["none", "own", "other-session", "other-user"])
def test_the_store_lists_counts_and_searches_by_one_rule(client, scoped):
    """``list_documents``, ``count_documents`` and retrieval's hidden set
    agree for every scope: the project's own rows always, a session's rows
    only inside that session for the user who made them."""
    from app.core import projects as store

    pid = store.create_project(f"SessEvStore-{scoped}-{_RUN}")["id"]
    own = store.add_document(pid, "invented_own.md")["id"]
    sess = store.add_document(pid, "invented_sess.md", metadata={
        "session_id": "ws-a", "uploader_id": "u-1", "origin": "uploaded"})["id"]
    scope = {"none": None, "own": ("ws-a", "u-1"),
             "other-session": ("ws-b", "u-1"), "other-user": ("ws-a", "u-2")}[scoped]
    token = store._SESSION_SCOPE.set(scope)
    try:
        listed = {d["id"] for d in store.list_documents(pid)}
        counted = store.count_documents(pid)
        hidden = store.session_hidden_document_ids(pid)
        every = {d["id"] for d in store.list_documents(pid, all_sessions=True)}
    finally:
        store._SESSION_SCOPE.reset(token)
    want = {own, sess} if scoped == "own" else {own}
    assert listed == want and counted == len(want)
    assert hidden == {own, sess} - want
    assert every == {own, sess}
