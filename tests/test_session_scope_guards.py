"""The guards on the paths the session-scope change (#400) runs through.

Each test states one behaviour of a function the change touched, through
the function itself or the route that calls it:

* ``privileges.raise_if_inaccessible_document_ids``: a missing id, an id on
  a project the caller cannot read, and another user's session file are the
  same 404; the caller's own session file is not refused.
* ``projects.session_hidden_document_ids`` reads through the caller's own
  database session, so retrieval sees the rows of the transaction it runs in.
* ``coverage_honesty.live_coverage`` counts neither another session's file
  nor its chunks; an empty project id has no coverage to report.
* ``VectorStore._hidden_doc_ids_among``: a store that cannot compute the
  session scope (the documents table cannot gate it, or reading it fails)
  hides every candidate -- fails closed. This corrects the rule as first
  written here ("hides nothing"), which let another session's file through.
* ``agents.agent_chat`` / ``agents.agent_chat_stream``: an unknown agent is a
  404; the non-streaming route answers and keeps another session's file out
  of the model's input; the streaming route honours the per-user driver
  switch in request mode.
* ``knowledge_seed.seed_knowledge``: no configured general-knowledge project
  means nothing is created or written.

Every project, user, session, file and answer here is invented by the test.
"""
from __future__ import annotations

import io
import json
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.core import privileges
from app.core import projects as store
from app.core.rag import coverage_honesty
from app.main import app
from app.routers import agents as agents_router
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _register(client, tag: str) -> tuple[dict, str]:
    email = f"scopeguard-{tag}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    body = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()
    return ({"Authorization": f"Bearer {body['token']}"},
            body.get("user_id") or body.get("user", {}).get("id"))


def _user(tag: str) -> str:
    """A user row only: these tests call the store, not a route."""
    from app.core.users import ensure_user_exists

    uid = f"scopeguard-{tag}-{_RUN}"
    ensure_user_exists(uid)
    return uid


def _session_doc(pid: str, name: str, session_id: str, uploader: str) -> str:
    return store.add_document(pid, name, metadata={
        "session_id": session_id, "uploader_id": uploader, "origin": "uploaded"})["id"]


# ── privileges.raise_if_inaccessible_document_ids ───────────────────────────

@pytest.mark.parametrize("case", ["missing", "foreign-project", "other-users-session"])
def test_an_unreadable_pinned_document_is_a_404(case):
    owner_id, other_id = _user(f"pin-own-{case}"), _user(f"pin-oth-{case}")
    pid = store.create_project(f"PinGuard-{case}-{_RUN}", user_id=owner_id)["id"]
    store.approve_project(pid)
    if case == "missing":
        did, caller = f"no-such-{uuid.uuid4().hex[:6]}", owner_id
    elif case == "foreign-project":
        private = store.create_project(
            f"PinPrivate-{_RUN}", user_id=other_id, is_approved=False)["id"]
        did, caller = store.add_document(private, "invented_private.md")["id"], owner_id
    else:
        did, caller = _session_doc(pid, "invented_theirs.md", "ws-a", other_id), owner_id
    with pytest.raises(HTTPException) as err:
        privileges.raise_if_inaccessible_document_ids(
            {"user_id": caller}, {"document_ids": [did]})
    assert err.value.status_code == 404


@pytest.mark.parametrize("own", ["session-file", "project-file"])
def test_the_callers_own_document_is_not_refused(own):
    """From the chat the turn runs in (the bound session scope): the caller's
    own file of THAT chat, and any project document, are readable by id."""
    uid = _user(f"pin-mine-{own}")
    pid = store.create_project(f"PinMine-{own}-{_RUN}", user_id=uid)["id"]
    did = (_session_doc(pid, "invented_mine.md", "ws-b", uid) if own == "session-file"
           else store.add_document(pid, "invented_project.md")["id"])
    token = store.bind_session_scope("ws-b", uid)
    try:
        assert privileges.raise_if_inaccessible_document_ids(
            {"user_id": uid}, {"document_ids": [did]}) is None
    finally:
        store.reset_session_scope(token)


@pytest.mark.parametrize("from_chat", ["ws-other", None])
def test_the_callers_own_session_file_from_another_chat_is_a_404(from_chat):
    """Advisor ruling (12:39Z, #451): a file attached to a chat is reachable
    by id only from that chat; from another chat, or from no chat, it is not
    found -- its maker included."""
    uid = _user(f"pin-elsewhere-{from_chat}")
    pid = store.create_project(f"PinElse-{from_chat}-{_RUN}", user_id=uid)["id"]
    did = _session_doc(pid, "invented_mine_elsewhere.md", "ws-b", uid)
    token = store.bind_session_scope(from_chat, uid)
    try:
        with pytest.raises(HTTPException) as err:
            privileges.raise_if_inaccessible_document_ids(
                {"user_id": uid}, {"document_ids": [did]})
        assert err.value.status_code == 404
    finally:
        store.reset_session_scope(token)


# ── projects.session_hidden_document_ids ────────────────────────────────────

@pytest.mark.parametrize("session_id", ["ws-x-1000", "ws-x-mwiqvj9k28n987unh726"])
def test_the_hidden_set_reads_through_the_callers_database_session(session_id):
    from app.core.db import SessionLocal
    from app.core.models import Document

    pid = store.create_project(f"HiddenTxn-{session_id[-4:]}-{_RUN}")["id"]
    did = store.add_document(pid, "invented_txn.md")["id"]
    with SessionLocal() as s:
        row = s.get(Document, did)
        row.metadata_ = {"session_id": session_id, "uploader_id": "u-1"}
        s.flush()  # not committed: only this transaction sees it
        assert did in store.session_hidden_document_ids(pid, db_session=s)
        s.rollback()
    assert did not in store.session_hidden_document_ids(pid)


# ── coverage_honesty.live_coverage ──────────────────────────────────────────

@pytest.mark.parametrize("chunks", [(3, 2), (1, 7)])
def test_live_coverage_counts_no_other_sessions_file(monkeypatch, chunks):
    from app.core.rag import vector_store

    pid = store.create_project(f"Coverage-{chunks[0]}{chunks[1]}-{_RUN}")["id"]
    own = store.add_document(pid, "invented_own.md")["id"]
    sess = _session_doc(pid, "invented_sess.md", "ws-c", "u-1")
    counts = {own: chunks[0], sess: chunks[1]}

    class _Store:
        def count_by_doc(self, project_id):
            return counts if project_id == pid else {}

    monkeypatch.setattr(vector_store, "get_store", lambda: _Store())
    assert coverage_honesty.live_coverage(pid) == (1, 1)
    token = store.bind_session_scope("ws-c", "u-1")
    try:
        assert coverage_honesty.live_coverage(pid) == (2, 2)
    finally:
        store.reset_session_scope(token)
    assert coverage_honesty.live_coverage("  ") is None


# ── VectorStore._hidden_doc_ids_among ───────────────────────────────────────

@pytest.mark.parametrize("cause", ["visibility-not-ready", "database-error"])
@pytest.mark.parametrize("hidden_by", ["excluded", "other-session"])
def test_a_store_that_cannot_compute_the_scope_hides_every_candidate(
        monkeypatch, hidden_by, cause):
    from sqlalchemy.exc import OperationalError

    from app.core.rag.vector_store import get_store

    pid = store.create_project(f"Ungated-{hidden_by}-{_RUN}")["id"]
    if hidden_by == "excluded":
        from app.core.db import SessionLocal
        from app.core.models import Document

        did = store.add_document(pid, "invented_retired.md")["id"]
        with SessionLocal() as s:
            s.get(Document, did).retrieval_visible = False
            s.commit()
    else:
        did = _session_doc(pid, "invented_sess.md", "ws-d", "u-1")
    own = store.add_document(pid, "invented_own_project_file.md")["id"]
    vs = get_store()
    with vs._session_factory()() as s:
        assert vs._hidden_doc_ids_among(s, pid, [did, own]) == {did}
        if cause == "visibility-not-ready":
            monkeypatch.setattr(vs, "_docs_visibility_ready", lambda _s=None: False)
        else:
            def _unreadable(*_a, **_k):
                raise OperationalError("SELECT documents", {}, Exception("invented"))
            monkeypatch.setattr(store, "session_hidden_document_ids", _unreadable)
        # Nothing can be shown safely: the project's own file is held back too.
        assert vs._hidden_doc_ids_among(s, pid, [did, own]) == {did, own}


# ── agents.agent_chat / agents.agent_chat_stream ────────────────────────────

@pytest.mark.parametrize("route", ["agent_chat", "agent_chat_stream"])
def test_an_unknown_agent_is_a_404(client, route):
    headers, _uid = _register(client, f"noagent-{route}")
    name = f"no-such-agent-{uuid.uuid4().hex[:6]}"
    path = app.url_path_for(getattr(agents_router, route).__name__, name=name)
    res = client.post(path, json={"message": "Hello."}, headers=headers)
    assert res.status_code == 404, res.text


@pytest.fixture
def recording_model(monkeypatch):
    seen: list[str] = []

    async def fake_llm(self, messages, api_key, project_id=None, **_kw):
        seen.append("\n".join(str(m.get("content") or "") for m in messages))
        return {"status": "success",
                "choice": {"message": {"content": "Piles are driven to refusal.",
                                       "tool_calls": []}}, "raw": {}}

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "-1")
    monkeypatch.setattr(Agent, "_call_llm", fake_llm)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    return seen


@pytest.mark.parametrize("y_suffix", ["2000", "mwiqvj9k28n987unh726"])
def test_agent_chat_answers_and_keeps_another_sessions_file_out(client, recording_model, y_suffix):
    from app.core import doc_index

    headers, uid = _register(client, f"agchat-{y_suffix[:4]}")
    pid = client.post("/v1/projects", json={"name": f"AgChat-{y_suffix[:4]}-{_RUN}"},
                      headers=headers).json()["id"]
    words = {"own": "brindlecap", "x": "sorrelmane"}
    ids = {}
    for key, sid in (("own", None), ("x", f"ws-{pid}-1000")):
        res = client.post(
            f"/v1/projects/{pid}/documents", headers=headers,
            data={"session_id": sid} if sid else {},
            files={"file": (f"invented_{key}.txt", io.BytesIO(
                f"The halvard piles are driven to refusal. Marker {words[key]}.\n".encode()),
                "text/plain")})
        assert res.status_code in (200, 201), res.text
        ids[key] = (res.json().get("document") or res.json())["id"]
        doc_index.index_document(pid, ids[key])

    path = app.url_path_for(agents_router.agent_chat.__name__, name="project-assistant")
    res = client.post(path, headers=headers, json={
        "message": "How are the halvard piles driven?", "project_id": pid,
        "conversation_id": f"ws-{pid}-{y_suffix}"})
    assert res.status_code == 200, res.text
    assert "refusal" in json.dumps(res.json())
    seen = "\n".join(recording_model)
    assert words["own"] in seen, "retrieval reached nothing: the exam proves nothing"
    assert words["x"] not in seen, seen


@pytest.mark.parametrize("switched_on", [True, False])
def test_agent_chat_stream_honours_the_users_driver_switch(client, monkeypatch, switched_on):
    from app.agents import driver
    from app.core import users

    headers, _uid = _register(client, f"drv-{int(switched_on)}")
    monkeypatch.setenv("DRIVER_MODE", "request")
    monkeypatch.setattr(users, "driver_mode_for", lambda _uid: switched_on)
    seen: list[bool] = []

    async def fake_stream(self, *a, **kw):
        seen.append(driver.enabled())
        yield {"type": "token", "content": "ok"}
        yield {"type": "end"}

    monkeypatch.setattr(Agent, "chat_stream", fake_stream)
    path = app.url_path_for(agents_router.agent_chat_stream.__name__, name="project-assistant")
    with client.stream("POST", path, json={"message": "Hello."}, headers=headers) as res:
        assert res.status_code == 200
        list(res.iter_lines())
    assert seen == [switched_on], seen


# ── knowledge_seed.seed_knowledge ───────────────────────────────────────────

@pytest.mark.parametrize("gk_id", ["", None])
def test_seed_knowledge_without_a_gk_project_writes_nothing(monkeypatch, tmp_path, gk_id):
    from app.core import knowledge_seed

    (tmp_path / "invented_note.md").write_text("# Invented note\n", encoding="utf-8")
    monkeypatch.setattr(knowledge_seed, "KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr(knowledge_seed, "_gk_project_id", lambda: gk_id)
    calls: list[str] = []
    for fn in ("create_project", "add_document", "get_project"):
        monkeypatch.setattr(store, fn, lambda *a, _fn=fn, **k: calls.append(_fn))
    knowledge_seed.seed_knowledge()
    assert calls == []
