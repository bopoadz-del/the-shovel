"""Project-own chunks must not be returned to another project.

Enforced by the retrieval query's project-id set (active + configured
general-knowledge), not by a name list. Shared general-knowledge rows
stay retrievable by every project.

Synthetic project ids and tokens only — no client, project or person names.
"""
from __future__ import annotations

import pytest


OWN_A = "scope_syn_proj_a"
OWN_B = "scope_syn_proj_b"
GK = "scope_syn_gk"
TOKEN_A = "zxq9quorum-alpha-own-row"
TOKEN_B = "zxq9quorum-beta-own-row"
TOKEN_GK = "zxq9quorum-gamma-shared-row"


@pytest.fixture
def scoped_store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    # The other project's own documents are pointed at as a fallback corpus.
    # Leakage on main is exactly "retrieve(A) returns B".
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", OWN_B)
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.99")
    from app.core.rag import embeddings as _emb, vector_store as _vs
    from app.core.rag import retriever as _ret

    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    yield _ret
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()


def _index(ret, project_id: str, doc_id: str, text: str) -> None:
    n = ret.index_chunks(project_id, doc_id, [text])
    assert n == 1


def test_retrieve_does_not_return_another_projects_own_chunks(scoped_store):
    """Fails on main if an empty/thin project can be served another
    project's own documents (the former master-corpus fallback)."""
    ret = scoped_store
    _index(ret, OWN_B, "doc_b", TOKEN_B)
    _index(ret, GK, "doc_g", TOKEN_GK)

    empty = "scope_syn_empty"
    hits = ret.retrieve(TOKEN_B, empty, k=5)
    leaked = [c.project_id for c in hits if c.project_id == OWN_B]
    assert leaked == [], f"project-own rows leaked across projects: {leaked}"
    assert all(c.project_id != OWN_B for c in hits)


def test_shared_general_knowledge_stays_retrievable(scoped_store):
    ret = scoped_store
    _index(ret, OWN_A, "doc_a", TOKEN_A)
    _index(ret, OWN_B, "doc_b", TOKEN_B)
    _index(ret, GK, "doc_g", TOKEN_GK)

    hits = ret.retrieve(TOKEN_GK, OWN_A, k=5)
    assert any(c.project_id == GK for c in hits), "shared general-knowledge must stay readable"
    assert all(c.project_id != OWN_B for c in hits)


def test_store_search_scopes_the_query(scoped_store):
    """The SQL/numpy search itself is keyed by project_id."""
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    ret = scoped_store
    _index(ret, OWN_A, "doc_a", TOKEN_A)
    _index(ret, OWN_B, "doc_b", TOKEN_B)
    emb = get_embedder()
    store = get_store(dim=emb.dim)
    hits = store.search(OWN_A, emb.encode_queries([TOKEN_B])[0], k=10)
    assert all(c.project_id == OWN_A for c in hits)
    assert all(c.project_id != OWN_B for c in hits)


def test_late_scan_does_not_add_a_foreign_project(monkeypatch):
    monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", OWN_B)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    from app.core.rag.retriever import _late_scan_project_ids, retrievable_project_ids

    pids = _late_scan_project_ids(OWN_A)
    assert OWN_B not in pids
    assert OWN_A in pids
    allowed = retrievable_project_ids(OWN_A)
    assert OWN_A in allowed
    assert GK in allowed
    assert OWN_B not in allowed
