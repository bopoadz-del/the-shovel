"""Stage 1 / Task 3 — knowledge_layer + authority round-trip through the store.

upsert_chunks(..., knowledge_layer=, authority=) persists both columns; search
hydrates them back onto the Chunk. Uses the default prod namespace (v2).
"""
import numpy as np
from app.core.rag.vector_store import VectorStore


def test_upsert_persists_layer_and_authority(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    # in-memory-ish: default namespace v2, SQLite dev DB
    store = VectorStore(db_path=str(tmp_path / "rag.db"))
    vecs = np.ones((1, store.dim), dtype=np.float32)
    store.upsert_chunks("p_layer_rt", "d1", ["hello layered world"], vecs,
                        knowledge_layer="project_record", authority="contractual")
    got = store.search("p_layer_rt", np.ones(store.dim, dtype=np.float32), k=1)
    assert got, "expected the upserted chunk back"
    assert got[0].knowledge_layer == "project_record"
    assert got[0].authority == "contractual"
    # STEP 0 isolation tag is untouched by hydration (still its default)
    assert got[0].layer == "own"


def test_unlayered_upsert_reads_as_none(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    store = VectorStore(db_path=str(tmp_path / "rag.db"))
    vecs = np.ones((1, store.dim), dtype=np.float32)
    store.upsert_chunks("p_unlayered_rt", "d1", ["plain chunk"], vecs)
    got = store.search("p_unlayered_rt", np.ones(store.dim, dtype=np.float32), k=1)
    assert got
    assert got[0].knowledge_layer is None
    assert got[0].authority is None
