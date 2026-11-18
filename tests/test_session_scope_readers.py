"""Every project-wide reader keeps a chat session's file to its own session.

One test per reader the session-scope rule (``projects.document_in_scope``)
was missing from: a file attached in a chat session is absent when the
reader runs project-wide (no session in scope) and in another session, and
present when the reader runs inside its own session, for its own user.

Readers: ``list_document_names``, ``documents_matching_filename_terms``,
``documents_matching_title_phrase``, ``compute_readiness``;
``VectorStore.term_chunk_counts`` (and its cache), ``count_by_doc``,
``count``, ``has_chunks``; the project route's ``searchable_document_count``
and ``indexed_chunks``.

Every project, user, session, file name and chunk text is invented here.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from app.core import projects as store
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]
# (session, user) of the session that owns the file, in two shapes of the
# stored ids: a short session id, and the UI's long one. Unique per run.
OWN = [(f"ws-own-1000-{_RUN}", f"u-own-{_RUN}"),
       (f"ws-own-mwiqvj9k28n987unh726-{_RUN}", f"usr-0007-own-{_RUN}")]


def _other_scopes(own):
    sid, uid = own
    return {
        "project-wide": None,
        "other session": (f"{sid}-other", uid),
        "other user": (sid, f"{uid}-other"),
    }


@contextmanager
def _scope(scope):
    token = store.bind_session_scope(*scope) if scope else store.bind_session_scope(None, None)
    try:
        yield
    finally:
        store.reset_session_scope(token)


def _session_file(pid: str, name: str, own, **kw) -> str:
    return store.add_document(pid, name, metadata={
        "session_id": own[0], "uploader_id": own[1], "origin": "uploaded"}, **kw)["id"]


def _project(tag: str) -> str:
    return store.create_project(f"ScopeReader-{tag}-{uuid.uuid4().hex[:6]}")["id"]


def _absent_outside_present_inside(read, did, own):
    for label, scope in _other_scopes(own).items():
        with _scope(scope):
            assert did not in read(), label
    with _scope(own):
        assert did in read(), "own session"


# ── app/core/projects.py ────────────────────────────────────────────────────

@pytest.mark.parametrize("own", OWN)
def test_list_document_names(own):
    pid = _project("names")
    project_doc = store.add_document(pid, "zephyr_project_register.md")["id"]
    did = _session_file(pid, "zephyr_session_register.md", own)
    read = lambda: {d["id"] for d in store.list_document_names(pid)}  # noqa: E731
    _absent_outside_present_inside(read, did, own)
    with _scope(None):
        assert project_doc in read()


@pytest.mark.parametrize("own", OWN)
def test_documents_matching_filename_terms(own):
    pid = _project("terms")
    project_doc = store.add_document(pid, "quillon_marbury_project.md")["id"]
    did = _session_file(pid, "quillon_marbury_session.md", own)
    read = lambda: {d["id"] for d in store.documents_matching_filename_terms(  # noqa: E731
        pid, ["quillon", "marbury"])}
    _absent_outside_present_inside(read, did, own)
    with _scope(None):
        assert project_doc in read()


@pytest.mark.parametrize("own", OWN)
def test_documents_matching_title_phrase(own):
    pid = _project("title")
    project_doc = store.add_document(pid, "Ostrel Vantage Procedure rev1.md")["id"]
    did = _session_file(pid, "Ostrel Vantage Procedure rev2.md", own)
    read = lambda: {d["id"] for d in store.documents_matching_title_phrase(  # noqa: E731
        pid, "ostrel vantage procedure")}
    _absent_outside_present_inside(read, did, own)
    with _scope(None):
        assert project_doc in read()


@pytest.mark.parametrize("own", OWN)
def test_compute_readiness(own):
    pid = _project("ready")
    _session_file(pid, "invented_baseline.xer", own, role=store.ROLE_BASELINE)
    _session_file(pid, "invented_daily.pdf", own, role=store.ROLE_DAILY)
    for label, scope in _other_scopes(own).items():
        with _scope(scope):
            r = store.compute_readiness(pid)
        assert (r["baseline_schedule"], r["daily_reports"]) == (False, 0), label
    with _scope(own):
        r = store.compute_readiness(pid)
    assert (r["baseline_schedule"], r["daily_reports"]) == (True, 1)


# ── app/core/rag/vector_store.py ────────────────────────────────────────────

@pytest.fixture()
def chunked(own):
    """A project with one project file and one session file, both chunked."""
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    pid = _project("chunks")
    own_doc = store.add_document(pid, "invented_project_notes.md")["id"]
    did = _session_file(pid, "invented_session_notes.md", own)
    emb = get_embedder()
    st = get_store(dim=emb.dim)
    own_texts = ["the gantry crane rests on plinth four"]
    sess_texts = ["the gantry crane carries a velquorine counterweight",
                  "velquorine ballast is checked weekly"]
    st.upsert_chunks(pid, own_doc, own_texts, emb.encode(own_texts))
    st.upsert_chunks(pid, did, sess_texts, emb.encode(sess_texts))
    return st, pid, own_doc, did, own


@pytest.mark.parametrize("own", OWN)
def test_count_by_doc(chunked):
    st, pid, own_doc, did, own = chunked
    _absent_outside_present_inside(lambda: st.count_by_doc(pid), did, own)
    with _scope(None):
        assert st.count_by_doc(pid).get(own_doc) == 1
        # Maintenance still sees every row.
        assert st.count_by_doc(pid, all_sessions=True).get(did) == 2


@pytest.mark.parametrize("own", OWN)
def test_count_and_has_chunks(chunked):
    st, pid, _own_doc, _did, own = chunked
    for label, scope in _other_scopes(own).items():
        with _scope(scope):
            assert st.count(pid) == 1, label
    with _scope(own):
        assert st.count(pid) == 3
    with _scope(None):
        assert st.count(pid, all_sessions=True) == 3

    only = _project("only-session")
    _session_file(only, "invented_only.md", own)
    did = store.list_session_documents(only, *own)[0]["id"]
    from app.core.rag.embeddings import get_embedder
    emb = get_embedder()
    st.upsert_chunks(only, did, ["an invented line"], emb.encode(["an invented line"]))
    with _scope(None):
        assert st.has_chunks(only) is False
    with _scope(own):
        assert st.has_chunks(only) is True


@pytest.mark.parametrize("own", OWN)
def test_term_chunk_counts_and_its_cache_are_scoped(chunked):
    st, pid, _own_doc, _did, own = chunked
    for label, scope in _other_scopes(own).items():
        with _scope(scope):
            assert st.term_chunk_counts(pid, ["velquorine", "gantry"]) == {
                "velquorine": 0, "gantry": 1}, label
    # Same project, inside the TTL: the session's own counts, not the cache
    # the project-wide reads above filled.
    with _scope(own):
        assert st.term_chunk_counts(pid, ["velquorine", "gantry"]) == {
            "velquorine": 2, "gantry": 2}
    with _scope(None):
        assert st.term_chunk_counts(pid, ["velquorine"]) == {"velquorine": 0}


# ── app/routers/projects.py (searchable_document_count, indexed_chunks) ─────

@pytest.mark.parametrize("own", OWN)
def test_project_route_counts_no_session_file(chunked):
    st, pid, own_doc, did, own = chunked
    email = f"scopereader-{own[0][-12:]}-{_RUN}@x.com"
    with TestClient(__import__("app.main", fromlist=["app"]).app) as client:
        client.post("/v1/users/register", json={"email": email, "password": PW_06})
        body = client.post("/v1/users/login",
                           json={"email": email, "password": PW_06}).json()
        headers = {"Authorization": f"Bearer {body['token']}"}
        uid = body.get("user_id") or body.get("user", {}).get("id")
        from app.core.db import SessionLocal
        from app.core.models import Project
        with SessionLocal() as s:
            s.get(Project, pid).user_id = uid
            s.commit()
        proj = client.get(f"/v1/projects/{pid}", headers=headers).json()
    assert proj["searchable_document_count"] == 1
    assert proj["indexed_chunks"] == 1
    assert {d["id"] for d in proj["documents"]} == {own_doc}
