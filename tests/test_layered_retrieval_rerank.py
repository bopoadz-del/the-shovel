"""Stage 3 integration — authority precedence reorders retrieve_with_filter.

Two active-project chunks at EQUAL cosine, returned historical-first. With
RAG_LAYERED on, the project_record/contractual chunk is re-ranked ahead of the
shared_domain/historical one. With the flag off, insertion order is preserved
(byte-for-byte today's behaviour) — proving the bonus is the sole cause.
"""
import pytest


def _chunk(cid, pid, score, layer, authority):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=cid, project_id=pid, doc_id=f"doc-{cid}",
                 chunk_index=0, text="filler", score=score,
                 knowledge_layer=layer, authority=authority)


def _install(monkeypatch, chunks):
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in chunks if c.project_id == project_id]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    for v in ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
              "RAG_GK_LEXICAL_FOLD", "RAG_AUTHORITY_WEIGHT", "RAG_LAYER_WEIGHT"):
        monkeypatch.delenv(v, raising=False)
    return ret


ACTIVE = "p_active"


def test_precedence_reorders_when_enabled(monkeypatch):
    ret = _install(monkeypatch, [
        # returned historical-first, both at equal cosine
        _chunk("hist", ACTIVE, 0.80, "shared_domain", "historical"),
        _chunk("contract", ACTIVE, 0.80, "project_record", "contractual"),
    ])
    monkeypatch.setenv("RAG_LAYERED", "1")
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert [c.chunk_id for c in chunks][0] == "contract", \
        "L2B contractual must outrank L1 historical at equal cosine"


def test_order_unchanged_when_disabled(monkeypatch):
    ret = _install(monkeypatch, [
        _chunk("hist", ACTIVE, 0.80, "shared_domain", "historical"),
        _chunk("contract", ACTIVE, 0.80, "project_record", "contractual"),
    ])
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    # flag off: stable insertion order (historical first) is preserved
    assert [c.chunk_id for c in chunks][0] == "hist"
