"""The per-message checks cost a row, not the corpus.

Live 2026-10-07, 15 users: the loop watchdog caught the chat access gate
listing every document of the master corpus and checking each file on disk
(1-2 s), the "is there any corpus" guard counting every chunk (12 s, waiting on
a process-wide store lock), and the file-name detectors listing every document
again. Synthetic projects and documents.
"""
from __future__ import annotations

import os

import pytest

from app.core import projects
from tests._fixture_values import PW_02

_ON_POSTGRES = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")


@pytest.fixture
def two_projects(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import users
    users.init_db()
    u1 = users.create_user("owner-one@example.invalid", PW_02)["id"]
    u2 = users.create_user("owner-two@example.invalid", PW_02)["id"]
    own = projects.create_project("Synthetic Harbour Works", user_id=u1)
    other = projects.create_project("Synthetic Ridge Road", user_id=u2)
    return own["id"], other["id"], u1, u2


def test_the_access_check_agrees_with_get_project(two_projects):
    own, other, u1, u2 = two_projects
    for pid, user in ((own, u1), (own, u2), (other, u2), (other, u1),
                      (own, None), ("no-such-project", u1)):
        for shared in (False, True):
            expected = projects.get_project(pid, user_id=user, include_admin_approved=shared) is not None
            assert projects.can_access_project(pid, user_id=user, include_admin_approved=shared) is expected, (pid, user)


def test_an_archived_project_is_not_accessible(two_projects):
    own, _, u1, _ = two_projects
    assert projects.can_access_project(own, user_id=u1) is True
    projects.archive_project(own)
    assert projects.can_access_project(own, user_id=u1) is False


def test_the_conversation_gate_never_lists_documents(two_projects, monkeypatch):
    from app.routers import agents as agents_router

    own, _, u1, u2 = two_projects

    def boom(*a, **k):
        raise AssertionError("the access gate listed documents")
    monkeypatch.setattr(projects, "list_documents", boom)
    monkeypatch.setattr(projects, "compute_readiness", boom)
    agents_router._enforce_conversation_access(f"ws-{own}", {"user_id": u1})
    with pytest.raises(Exception):
        agents_router._enforce_conversation_access(f"ws-{own}", {"user_id": u2})


def test_names_are_listed_without_checking_any_file(two_projects, monkeypatch, tmp_path):
    own, _, _, _ = two_projects
    p = tmp_path / "spec_fenwick_reach.txt"
    p.write_text("invented specification", encoding="utf-8")
    projects.add_document(own, original_name="spec_fenwick_reach.txt", stored_as=p.name,
                          file_path=str(p), size=p.stat().st_size)
    monkeypatch.setattr(projects, "_path_looks_present",
                        lambda path: (_ for _ in ()).throw(AssertionError("stat'ed a file")))
    names = projects.list_document_names(own)
    assert [n["original_name"] for n in names] == ["spec_fenwick_reach.txt"]
    assert set(names[0]) == {"id", "original_name", "doc_type", "file_path"}


def test_the_corpus_guard_probes_instead_of_counting(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as emb, retriever, vector_store as vs
    emb.reset_embedder_cache()
    vs.reset_store_cache()
    e = emb.Embedder(model_name="fake")
    st = vs.get_store(dim=e.dim)
    st.upsert_chunks("synthetic-full", "d1", ["an invented clause"], e.encode(["an invented clause"]))
    monkeypatch.setattr(type(st), "count", lambda *a, **k: (_ for _ in ()).throw(AssertionError("counted")))
    assert st.has_chunks("synthetic-full") is True
    assert st.has_chunks("synthetic-empty") is False
    assert retriever._project_has_any_chunks(st, "synthetic-full") is True
    assert retriever._project_has_any_chunks(st, "synthetic-empty") is False
    emb.reset_embedder_cache()
    vs.reset_store_cache()


@pytest.mark.skipif(not _ON_POSTGRES, reason="the lock is SQLite-only (test-postgres job checks PostgreSQL)")
def test_postgres_operations_do_not_take_turns():
    from contextlib import nullcontext

    from app.core.rag import vector_store as vs
    from app.core.rag.embeddings import get_embedder

    st = vs.get_store(dim=get_embedder().dim)
    assert isinstance(st._lock, nullcontext)
