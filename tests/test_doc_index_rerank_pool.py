"""doc_index hybrid search requests the top-50 pool only when RERANK_ENABLED.

Flag OFF must keep today's over_fetch (max(top_k * 4, 20)). Flag ON raises
the retrieve_with_filter k to the reranker's candidate_depth (default 50).
"""
from __future__ import annotations

import pytest

from app.core import doc_index


@pytest.fixture
def capture_k(monkeypatch):
    seen: dict[str, int] = {}

    def fake_retrieve(query, project_id, k=5, **_kw):
        seen["k"] = k
        return [], 0

    import app.core.rag.retriever as retr
    monkeypatch.setattr(retr, "retrieve_with_filter", fake_retrieve)
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda _d: "")
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})
    return seen


async def test_flag_off_keeps_legacy_over_fetch(monkeypatch, capture_k):
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    await doc_index.search_project_documents("p1", "concrete cover", top_k=5)
    assert capture_k["k"] == 20


async def test_rerank_enabled_requests_hybrid_top_50(monkeypatch, capture_k):
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    monkeypatch.delenv("RAG_RERANK_CANDIDATES", raising=False)
    await doc_index.search_project_documents("p1", "concrete cover", top_k=5)
    assert capture_k["k"] == 50


async def test_rerank_disabled_false_string_is_off(monkeypatch, capture_k):
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("RAG_RERANKER", "1")
    await doc_index.search_project_documents("p1", "concrete cover", top_k=5)
    assert capture_k["k"] == 20
