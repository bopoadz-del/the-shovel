"""A document outside the indexed statuses holds no chunks and is never retrieved.

Synthetic documents. Failing a document, tombstoning it or deleting it
removes its chunks in the same transaction; and retrieval ignores any chunk
whose document has a no-chunk status, even if a chunk were left behind.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.rag.embeddings import get_embedder, reset_embedder_cache
from app.core.rag.vector_store import VectorStore, reset_store_cache

TEXT = "PROBEWORD formwork striking sequence for transfer slab"


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_embedder_cache()
    reset_store_cache()
    projects.init_db()
    users_mod.ensure_user_exists("u1")
    pid = projects.create_project(name="NoChunk", client="C", user_id="u1")["id"]

    url = db_mod.get_database_url()
    embedder = get_embedder()
    store = VectorStore(db_path=url.replace("sqlite:///", ""), dim=embedder.dim)
    docs = {}
    for name in ("kept.docx", "failed.docx", "gone.docx", "deleted.docx"):
        doc = projects.add_document(project_id=pid, original_name=name, size=10)
        store.upsert_chunks(pid, doc["id"], [f"{TEXT} {name}"], embedder.encode([f"{TEXT} {name}"]))
        projects.stamp_document_index(doc["id"], chunk_count=1, ingest_status="INDEXED")
        docs[name] = doc["id"]
    store._visibility_ready = None
    yield projects, store, embedder, pid, docs
    store.close()


def _count(store, pid, doc_id):
    return store.count_by_doc(pid).get(doc_id, 0)


def test_failing_tombstoning_or_deleting_a_document_drops_its_chunks(world):
    projects, store, _emb, pid, docs = world

    projects.stamp_document_index(docs["failed.docx"], chunk_count=0, ingest_status="EXTRACT_FAILED")
    projects.tombstone_document(docs["gone.docx"])
    projects.delete_document(docs["deleted.docx"])

    assert _count(store, pid, docs["kept.docx"]) == 1
    assert _count(store, pid, docs["failed.docx"]) == 0
    assert _count(store, pid, docs["gone.docx"]) == 0
    assert _count(store, pid, docs["deleted.docx"]) == 0
    assert projects.get_document(docs["failed.docx"])["chunk_count"] == 0


def test_retrieval_ignores_chunks_of_a_document_in_a_no_chunk_status(world):
    """Even a chunk left behind (written before this rule) is not retrieved."""
    projects, store, embedder, pid, docs = world
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:  # status changed without the ledger writer
        session.get(Document, docs["failed.docx"]).ingest_status = "ZERO_CHUNK"
        session.commit()
    store._visibility_ready = None

    vec_ids = [c.doc_id for c in store._semantic_search(pid, embedder.encode([TEXT])[0], k=10)]
    bm25_ids = [c.doc_id for c in store.bm25_search(pid, "PROBEWORD", k=10)]

    for ids in (vec_ids, bm25_ids):
        assert docs["kept.docx"] in ids
        assert docs["failed.docx"] not in ids


def test_existing_rows_are_brought_into_line_once(world):
    """The data correction migration 0020 runs: idempotent, on synthetic rows."""
    projects, store, embedder, pid, docs = world
    from app.core import ingest_status as ist
    from app.core.db import SessionLocal, get_engine
    from app.core.models import Document

    pack = projects.add_document(project_id=pid, original_name="pack.zip", size=10)
    store.upsert_chunks(pid, pack["id"], ["archived text"], embedder.encode(["archived text"]))
    pending = projects.add_document(project_id=pid, original_name="pending.docx", size=10)
    store.upsert_chunks(pid, pending["id"], [TEXT], embedder.encode([TEXT]))
    with SessionLocal() as session:  # states written before the ledger kept the rule
        session.get(Document, docs["failed.docx"]).ingest_status = ist.EXTRACT_FAILED
        session.get(Document, docs["gone.docx"]).ingest_status = ist.INDEXED
        session.commit()
    store.delete_doc(pid, docs["gone.docx"])  # INDEXED, yet holds no chunk

    with get_engine().begin() as conn:
        first = ist.enforce_chunk_invariants(conn)
    with get_engine().begin() as conn:
        second = ist.enforce_chunk_invariants(conn)

    assert first["removed_documents"] == 1 and projects.get_document(pack["id"]) is None
    assert _count(store, pid, pack["id"]) == 0
    assert _count(store, pid, docs["failed.docx"]) == 0
    assert projects.get_document(docs["gone.docx"])["ingest_status"] == ist.ZERO_CHUNK
    assert projects.get_document(pending["id"])["ingest_status"] in ist.INDEXED_STATUSES
    assert _count(store, pid, docs["kept.docx"]) == 1
    assert second == {"removed_documents": 0, "removed_chunks": 0, "reclassified": 0}
