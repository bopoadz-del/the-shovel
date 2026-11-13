"""A re-extraction replaces a document's chunks only when it succeeds.

Synthetic documents. Stamping a no-chunk status deletes a document's chunks
in the same transaction, so a reindex whose extraction comes back empty used
to wipe a working index (a failed force-OCR of a long scan would delete the
OCR chunks it was meant to extend). Each decision is exercised from both
sides: a row that already serves chunks keeps them and its status; a row
that serves none is stamped ZERO_CHUNK exactly as before; a successful
re-extraction still swaps the chunks.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.rag.embeddings import reset_embedder_cache
from app.core.rag.vector_store import get_store, reset_store_cache


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
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
    pid = projects.create_project(name="Swap", client="C", user_id="u1")["id"]
    from app.core import doc_index

    yield projects, doc_index, pid, tmp_path
    reset_store_cache()


def _doc(projects, pid, tmp_path, name, body: bytes):
    from app.core import file_crypto

    path = str(tmp_path / name)
    file_crypto.write_document(path, body)
    return projects.add_document(pid, name, file_path=path, size=len(body)), path


def _rewrite(path, body: bytes):
    from app.core import file_crypto

    file_crypto.write_document(path, body)


def _stored(pid, doc_id):
    return get_store().count_by_doc(pid).get(doc_id, 0)


def _text(word: str, n: int) -> bytes:
    return (" ".join(f"{word}{i} slab pour sequence" for i in range(n))).encode()


@pytest.mark.parametrize("stamp_as_indexed", [True, False])
def test_failed_reextraction_keeps_a_working_index(world, stamp_as_indexed):
    projects, doc_index, pid, tmp_path = world
    doc, path = _doc(projects, pid, tmp_path, "bill.txt", _text("alpha", 900))
    first = doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)
    assert first.get("status") != "error"
    before = projects.get_document(doc["id"])
    held = _stored(pid, doc["id"])
    assert held > 1 and before["chunk_count"] == held

    _rewrite(path, b"")  # the re-extraction now yields nothing
    result = doc_index.index_document(
        pid, doc["id"], stamp_as_indexed=stamp_as_indexed,
    )

    assert result["status"] == "error" and result["error"] == "ZERO_CHUNK"
    assert result["chunks_kept"] is True
    assert result["prior_chunk_count"] == held
    after = projects.get_document(doc["id"])
    assert after["ingest_status"] == before["ingest_status"]
    assert after["chunk_count"] == held
    assert _stored(pid, doc["id"]) == held


@pytest.mark.parametrize("body", [b"", b"   \n\t  \n\n   "])
def test_failed_extraction_without_an_index_is_still_zero_chunk(world, body):
    projects, doc_index, pid, tmp_path = world
    doc, _path = _doc(projects, pid, tmp_path, "empty.txt", body)

    result = doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)

    assert result["error"] == "ZERO_CHUNK"
    assert "chunks_kept" not in result
    row = projects.get_document(doc["id"])
    assert row["ingest_status"] == "ZERO_CHUNK"
    assert row["chunk_count"] == 0
    assert _stored(pid, doc["id"]) == 0


@pytest.mark.parametrize(
    ("failed_status", "stale_count"),
    [("EXTRACT_FAILED", 7), ("UNSUPPORTED_TYPE", 42), ("ZERO_CHUNK", 1)],
)
def test_a_row_in_a_no_chunk_status_does_not_count_as_an_index(
    world, failed_status, stale_count,
):
    from app.core.ingest_status import NO_CHUNK_STATUSES

    assert failed_status in NO_CHUNK_STATUSES
    projects, doc_index, pid, tmp_path = world
    doc, path = _doc(projects, pid, tmp_path, "gone.txt", _text("beta", 300))
    doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)
    # A stale chunk_count on a failed row is not a working index.
    projects.stamp_document_index(
        doc["id"], chunk_count=0, ingest_status=failed_status,
    )
    assert _stored(pid, doc["id"]) == 0
    stale = dict(projects.get_document(doc["id"]), chunk_count=stale_count)
    assert doc_index._prior_retrievable_chunks(stale) == 0  # noqa: SLF001
    indexed = dict(stale, ingest_status="INDEXED")
    assert doc_index._prior_retrievable_chunks(indexed) == stale_count  # noqa: SLF001

    _rewrite(path, b"")
    result = doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)
    assert "chunks_kept" not in result
    assert projects.get_document(doc["id"])["ingest_status"] == "ZERO_CHUNK"


@pytest.mark.parametrize(("first_n", "second_n"), [(900, 150), (150, 900)])
def test_successful_reextraction_still_swaps_the_chunks(world, first_n, second_n):
    projects, doc_index, pid, tmp_path = world
    doc, path = _doc(projects, pid, tmp_path, "spec.txt", _text("gamma", first_n))
    doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)
    held = _stored(pid, doc["id"])
    assert held > 0

    _rewrite(path, _text("delta", second_n))
    result = doc_index.index_document(pid, doc["id"], stamp_as_indexed=True)

    assert result.get("status") != "error"
    assert "chunks_kept" not in result
    now = _stored(pid, doc["id"])
    # The new extraction replaced the old chunks: the count follows the new
    # text in whichever direction it moved, and the row agrees with the store.
    assert now > 0
    assert (now < held) if second_n < first_n else (now > held)
    row = projects.get_document(doc["id"])
    assert row["chunk_count"] == now
    assert row["ingest_status"] not in ("ZERO_CHUNK", "EXTRACT_FAILED")
