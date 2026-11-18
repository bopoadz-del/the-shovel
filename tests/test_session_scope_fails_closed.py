"""The session scope fails CLOSED, its count cache is bounded, and the
content-hash lookup a project walk uses sees no other session's file.

1. ``VectorStore._session_hidden_ids`` cannot always compute the set of
   chat-session files out of scope: the documents table may not gate the
   store (the visibility probe failed), or reading it may fail. In both
   states every project-wide read of the store returns NO candidate -- never
   every session's files. (A correction to the rule as first written, which
   hid nothing in that state.) One failed read decides that call only: the
   next call, with the table readable, sees the project's own file again.
2. ``term_chunk_counts`` caches one entry per (project, session scope). A
   write drops the expired entries and keeps at most the cap, oldest first.
3. ``projects.find_document_by_sha`` counts only rows in the current session
   scope, so a chat-session upload of the same bytes does not make a project
   walk skip, or refuse, the project's file.

Every project, user, session, file name and chunk text is invented here.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager

import pytest
from sqlalchemy.exc import OperationalError

from app.core import projects as store

_RUN = uuid.uuid4().hex[:8]
OWN = (f"ws-closed-{_RUN}", f"u-closed-{_RUN}")
CAUSES = ["visibility-not-ready", "database-error"]


@contextmanager
def _scope(scope):
    token = store.bind_session_scope(*scope) if scope else store.bind_session_scope(None, None)
    try:
        yield
    finally:
        store.reset_session_scope(token)


def _project(tag: str) -> str:
    return store.create_project(f"ScopeClosed-{tag}-{uuid.uuid4().hex[:6]}")["id"]


def _session_file(pid: str, name: str, own=OWN, **kw) -> str:
    return store.add_document(pid, name, metadata={
        "session_id": own[0], "uploader_id": own[1], "origin": "uploaded"}, **kw)["id"]


@pytest.fixture()
def chunked():
    """A project with one project file and one session file, both chunked."""
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    pid = _project("chunks")
    own_doc = store.add_document(pid, "invented_tarnwick_notes.md")["id"]
    did = _session_file(pid, "invented_tarnwick_session.md")
    emb = get_embedder()
    st = get_store(dim=emb.dim)
    own_texts = ["the tarnwick hoist rests on plinth nine"]
    sess_texts = ["the tarnwick hoist carries a brellomite counterweight"]
    st.upsert_chunks(pid, own_doc, own_texts, emb.encode(own_texts))
    st.upsert_chunks(pid, did, sess_texts, emb.encode(sess_texts))
    return st, emb, pid, own_doc, did


def _break_scope(monkeypatch, st, cause):
    """Put the store in a state where the session scope cannot be computed."""
    if cause == "visibility-not-ready":
        monkeypatch.setattr(st, "_docs_visibility_ready", lambda _s=None: False)
    else:
        def _unreadable(*_a, **_k):
            raise OperationalError("SELECT documents", {}, Exception("invented outage"))
        monkeypatch.setattr(store, "session_hidden_document_ids", _unreadable)


# ── 1. fail closed ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("cause", CAUSES)
def test_no_project_wide_read_returns_a_candidate_when_the_scope_is_unknown(
        chunked, monkeypatch, cause):
    st, emb, pid, own_doc, did = chunked
    with _scope(None):
        # Readable: the project's file only.
        assert st.count(pid) == 1
        assert set(st.count_by_doc(pid)) == {own_doc}
        _break_scope(monkeypatch, st, cause)
        reads = {
            "count": st.count(pid),
            "has_chunks": st.has_chunks(pid),
            "count_by_doc": st.count_by_doc(pid),
            "term_chunk_counts": st.term_chunk_counts(pid, ["brellomite"]),
            "search": [c.doc_id for c in st.search(
                pid, emb.encode(["tarnwick hoist"])[0], k=5,
                query_text="tarnwick hoist brellomite")],
            "bm25_search": [c.doc_id for c in st.bm25_search(pid, "tarnwick hoist", k=5)],
            "identifier_search": [c.doc_id for c in st.identifier_search(pid, ["tarnwick"])],
            "chunks_containing_all": [
                c.doc_id for c in st.chunks_containing_all(pid, ["tarnwick", "hoist"])],
        }
        # Maintenance that opts out of the scope still sees every row.
        assert st.count(pid, all_sessions=True) == 2
        assert set(st.count_by_doc(pid, all_sessions=True)) == {own_doc, did}
    assert reads == {
        "count": 0, "has_chunks": False, "count_by_doc": {},
        "term_chunk_counts": {"brellomite": 0}, "search": [], "bm25_search": [],
        "identifier_search": [], "chunks_containing_all": [],
    }


@pytest.mark.parametrize("cause", CAUSES)
def test_the_hidden_sql_clause_and_id_sets_hide_everything_when_the_scope_is_unknown(
        chunked, monkeypatch, cause):
    st, _emb, pid, own_doc, did = chunked
    _break_scope(monkeypatch, st, cause)
    with _scope(None), st._session_factory()() as s:
        params: dict = {}
        assert st._session_hidden_ids(s, pid) is None
        assert st._session_hidden_sql(s, pid, "c.doc_id", params) == " AND 1 = 0"
        assert params == {}
        assert st._hidden_doc_ids(s, pid) is None
        assert st._hidden_doc_ids_among(s, pid, [own_doc, did]) == {own_doc, did}
        assert st._hidden_doc_ids_among(s, pid, []) == set()


@pytest.mark.parametrize("cause", ["database-error", "probe-failed"])
def test_one_failed_read_does_not_close_the_store_for_good(chunked, monkeypatch, cause):
    st, _emb, pid, own_doc, _did = chunked
    with _scope(None):
        if cause == "database-error":
            with monkeypatch.context() as m:
                _break_scope(m, st, cause)
                assert st.count(pid) == 0
                assert st.term_chunk_counts(pid, ["tarnwick"]) == {"tarnwick": 0}
            assert st._visibility_ready is True
        else:
            # A probe that failed long enough ago is made again.
            monkeypatch.setattr(st, "_visibility_ready", False)
            monkeypatch.setattr(st, "_visibility_probed_at",
                                time.monotonic() - st._VISIBILITY_REPROBE_S - 1,
                                raising=False)
        assert st.count(pid) == 1
        assert set(st.count_by_doc(pid)) == {own_doc}
        # A closed answer was not remembered: the count is read again.
        assert st.term_chunk_counts(pid, ["tarnwick"]) == {"tarnwick": 1}


@pytest.mark.parametrize("age", [0.0, 5.0])
def test_a_recently_failed_probe_is_not_repeated_on_every_read(monkeypatch, age):
    from app.core.rag.vector_store import get_store

    st = get_store()
    probes: list = []

    class _Session:
        def execute(self, *_a, **_k):
            probes.append(1)

        def close(self):
            pass

    monkeypatch.setattr(st, "_visibility_ready", False)
    monkeypatch.setattr(st, "_visibility_probed_at", time.monotonic() - age,
                        raising=False)
    assert st._docs_visibility_ready(_Session()) is False
    assert probes == []
    # And a cached "ready" is never probed again.
    monkeypatch.setattr(st, "_visibility_ready", True)
    assert st._docs_visibility_ready(_Session()) is True
    assert probes == []


# ── 2. the term-count cache is bounded ──────────────────────────────────────

@pytest.mark.parametrize("n_scopes", [3, 5])
def test_the_term_count_cache_keeps_at_most_the_cap(chunked, monkeypatch, n_scopes):
    st, _emb, pid, _own_doc, _did = chunked
    monkeypatch.setattr(st, "_term_count_cache", {}, raising=False)
    monkeypatch.setattr(st, "_TERM_COUNT_MAX_ENTRIES", 2)
    keys = [(f"ws-cap-{i}-{_RUN}", f"u-cap-{_RUN}") for i in range(n_scopes)]
    for scope in keys:
        with _scope(scope):
            st.term_chunk_counts(pid, ["tarnwick"])
    assert list(st._term_count_cache) == [(pid, k) for k in keys[-2:]]


@pytest.mark.parametrize("ttl_past", [1.0, 300.0])
def test_a_write_drops_the_expired_entries(chunked, monkeypatch, ttl_past):
    st, _emb, pid, _own_doc, _did = chunked
    now = time.monotonic()
    stale = now - st._TERM_COUNT_TTL_S - ttl_past
    monkeypatch.setattr(st, "_term_count_cache", {
        ("invented-gone-1", None): (stale, {"x": 1}),
        ("invented-gone-2", ("ws-old", "u-old")): (stale, {"x": 2}),
        ("invented-fresh", None): (now, {"x": 3}),
    }, raising=False)
    with _scope(None):
        assert st.term_chunk_counts(pid, ["tarnwick"]) == {"tarnwick": 1}
    assert set(st._term_count_cache) == {("invented-fresh", None), (pid, None)}


# ── 3. the content-hash lookup a project walk uses ──────────────────────────

@pytest.mark.parametrize("who", ["same user", "other user"])
def test_a_session_copy_does_not_make_the_project_walk_skip_the_file(who):
    uploader = OWN[1] if who == "same user" else f"u-other-{_RUN}"
    pid = _project("sha")
    body = f"invented drawing register {uuid.uuid4().hex}".encode()
    sha = hashlib.sha256(body).hexdigest()
    sess = _session_file(pid, "invented_register.pdf", (OWN[0], uploader),
                         content_sha256=sha, size=len(body))
    # The walk (no session in scope): the session copy is not the project's.
    with _scope(None):
        assert store.find_document_by_sha(pid, sha) is None
        doc = store.add_document(pid, "invented_register.pdf", content_sha256=sha,
                                 size=len(body), metadata={"source": "drive_oauth_folder"})
        assert doc["id"] != sess
        assert store.find_document_by_sha(pid, sha)["id"] == doc["id"]
    names = [d["id"] for d in store.list_documents(pid)]
    assert names == [doc["id"]]


@pytest.mark.parametrize("which", ["own session", "other session"])
def test_inside_a_session_its_own_copy_counts_and_another_does_not(which):
    pid = _project("sha-sess")
    body = f"invented method statement {uuid.uuid4().hex}".encode()
    sha = hashlib.sha256(body).hexdigest()
    sess = _session_file(pid, "invented_method.docx", content_sha256=sha, size=len(body))
    scope = OWN if which == "own session" else (f"{OWN[0]}-x", OWN[1])
    with _scope(scope):
        found = store.find_document_by_sha(pid, sha)
    assert (found or {}).get("id") == (sess if which == "own session" else None)


@pytest.mark.parametrize("doc_ids", [[], ()])
def test_no_ids_asks_the_database_nothing(monkeypatch, doc_ids):
    from app.core.rag.vector_store import get_store

    st = get_store()
    monkeypatch.setattr(st, "_visibility_ready", True)

    class _NoQueries:
        def execute(self, *_a, **_k):
            raise AssertionError("no id was asked about; nothing may be read")

    assert st._hidden_doc_ids_among(_NoQueries(), "invented-project", doc_ids) == set()


@pytest.mark.parametrize("missing", [None, ""])
def test_a_missing_hash_matches_no_row(missing):
    """A row stored without a hash is not 'the same bytes' as another."""
    pid = _project("nosha")
    store.add_document(pid, "invented_unhashed.md", content_sha256=missing)
    with _scope(None):
        assert store.find_document_by_sha(pid, missing) is None
