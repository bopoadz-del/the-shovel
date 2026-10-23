"""Production k=5 must over-fetch 60 candidates (parked pool-stability).

UI-PHYS A5/E1 sat at ranks 21/29 on the live particulars family. The old
floor of 20 dropped those rows before #430's label bonus could promote
them. Raising the floor first (without #430) broke A2/A6 by supplying
more near-tie competitors. After #430 the floor is safe to unpark.

The SHA 7efeadb was never pushed; this pins the reconstructed contract:
k=5 → pool 60, result still cut to 5.
"""
from __future__ import annotations

from app.core.rag.retriever import candidate_overfetch
from app.core.rag.vector_store import Chunk


def test_production_k5_overfetch_is_60():
    assert candidate_overfetch(5) == 60
    assert candidate_overfetch(1) == 60
    assert candidate_overfetch(20) == 80  # multiplier still wins above the floor


def test_retrieve_with_filter_asks_the_store_for_sixty_at_k5(monkeypatch):
    from app.core.rag import retriever as ret

    seen_k: list[int] = []

    def fake_search(self, project_id, qvec, k, query_text=None):
        seen_k.append(k)
        return [
            Chunk(
                chunk_id=f"c{i}",
                project_id=project_id,
                doc_id="doc",
                chunk_index=i,
                text=f"chunk {i} filler content",
                score=0.90 - i * 0.001,
            )
            for i in range(k)
        ]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setenv("RAG_DUAL_QUERY", "0")
    for var in (
        "RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
        "RAG_GK_LEXICAL_FOLD", "RAG_CD_PARTICULARS_BOOST",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("RAG_CD_PARTICULARS_BOOST", "0")

    chunks, _ = ret.retrieve_with_filter("neutral query about concrete", "p_active", k=5)
    assert seen_k, "store.search was never called"
    assert 60 in seen_k, f"production k=5 must over-fetch 60, saw {seen_k}"
    assert len(chunks) == 5, f"over-fetch must still cut to k=5, got {len(chunks)}"
