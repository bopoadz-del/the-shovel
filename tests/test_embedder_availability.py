"""A document that cannot be embedded must not report itself as indexed.

THE PRODUCTION FAILURE THIS PINS
--------------------------------
Only the embedding LIBRARIES were baked into the image; the model WEIGHTS were
resolved against the model host at RUNTIME. With ephemeral containers and no
persistent cache that happened on every deploy and restart, which put a
third-party host in the boot path of retrieval.

The way it failed is what made it expensive:

* ``Embedder.available()`` only checks that the libraries IMPORT. They always
  do — they are in the image — so it reported True while every embed call
  failed.
* ``doc_index`` caught the resulting exception, logged a WARNING, and returned
  ``status: ok``.

So a document was stored, registered, listed in the sidebar, and reported as a
successful upload — while being indexed with zero chunks and therefore
permanently unsearchable. Observed live as ``RAG indexing skipped for
63771d02: 403 Forbidden``.

The weights are now baked at build time with the Hub disabled at runtime
(Dockerfile + scripts/prefetch_embedder.py), so this should not recur — and
if it ever does, these tests require it to be LOUD.
"""
from __future__ import annotations

import importlib
import os

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.core import projects as projects_mod


def _write_doc(tmp_path, name: str, body: bytes) -> str:
    from app.core import file_crypto

    path = str(tmp_path / name)
    file_crypto.write_document(path, body)
    return path


@pytest.fixture
def indexed_doc(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import doc_index

    importlib.reload(doc_index)
    from app.core.rag import vector_store as _vs

    _vs.reset_store_cache()

    proj = projects_mod.create_project("Embedder Availability")
    pid = proj["id"]
    body = b"Concrete for suspended slabs shall be grade C40/50 throughout."
    path = _write_doc(tmp_path, "spec.txt", body)
    doc = projects_mod.add_document(
        pid, "spec.txt", file_path=path, size=len(body),
    )
    return {"doc_index": doc_index, "pid": pid, "doc_id": doc["id"]}


def test_an_embedding_failure_is_reported_not_swallowed(indexed_doc, monkeypatch):
    """The exact live shape: the embedder raises, text extracted fine."""
    doc_index = indexed_doc["doc_index"]
    from app.core.rag import retriever as _rag

    def _boom(*_a, **_kw):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(_rag, "index_chunks", _boom)

    result = doc_index.index_document(indexed_doc["pid"], indexed_doc["doc_id"])

    assert result.get("rag_error"), (
        "an embedding failure returned no rag_error — the caller cannot tell "
        "this document apart from one that indexed successfully"
    )
    assert "403" in result["rag_error"]
    assert result.get("rag_indexed", 0) == 0


def test_embedding_zero_chunks_is_also_a_failure(indexed_doc, monkeypatch):
    """Text extracted, nothing embedded, no exception raised. Same end state —
    an unsearchable document — and it used to look exactly like success."""
    doc_index = indexed_doc["doc_index"]
    from app.core.rag import retriever as _rag

    monkeypatch.setattr(_rag, "index_chunks", lambda *_a, **_kw: 0)

    result = doc_index.index_document(indexed_doc["pid"], indexed_doc["doc_id"])

    assert result.get("rag_error"), "embedding 0 chunks was reported as success"
    assert "0 of" in result["rag_error"]


def test_an_unavailable_embedding_stack_is_reported(indexed_doc, monkeypatch):
    doc_index = indexed_doc["doc_index"]
    from app.core.rag import retriever as _rag

    monkeypatch.setattr(_rag, "available", lambda: False)

    result = doc_index.index_document(indexed_doc["pid"], indexed_doc["doc_id"])

    assert result.get("rag_error") == "embedding stack unavailable"


def test_the_failure_is_recorded_on_the_stored_index(indexed_doc, monkeypatch):
    """Queryable after the fact, not only findable in log scrollback — the
    warning-and-drop is why three projects sat unsearchable for weeks."""
    doc_index = indexed_doc["doc_index"]
    from app.core.rag import retriever as _rag

    monkeypatch.setattr(
        _rag, "index_chunks",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("503 unavailable")),
    )
    doc_index.index_document(indexed_doc["pid"], indexed_doc["doc_id"])

    saved = doc_index._load_index(indexed_doc["pid"])
    entry = saved["documents"][0]
    assert "rag_error" in entry and "503" in entry["rag_error"]


def test_a_healthy_index_reports_no_error(indexed_doc):
    """The guard must not fire on the happy path."""
    doc_index = indexed_doc["doc_index"]
    result = doc_index.index_document(indexed_doc["pid"], indexed_doc["doc_id"])

    assert result["status"] == "ok"
    assert "rag_error" not in result
    assert result["rag_indexed"] > 0


# ── the probe that tells health surfaces the truth ──────────────────────────

def test_embedder_health_reports_ok_for_a_loadable_model(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core.rag import embeddings

    embeddings.reset_embedder_cache()
    health = embeddings.embedder_health()

    assert health["ok"] is True
    assert health["dim"] > 0
    assert health["error"] is None


def test_embedder_health_reports_the_reason_a_load_failed(monkeypatch):
    """available() cannot distinguish these cases; this must."""
    from app.core.rag import embeddings

    embeddings.reset_embedder_cache()

    def _boom(*_a, **_kw):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(embeddings, "get_embedder", _boom)
    health = embeddings.embedder_health()

    assert health["ok"] is False
    assert "403" in health["error"]
    embeddings.reset_embedder_cache()


def test_available_is_documented_as_an_import_check_only(monkeypatch):
    """available() stays cheap (hot paths + readiness probes call it), so it
    must NOT be mistaken for proof that embedding works."""
    from app.core.rag.embeddings import Embedder

    assert Embedder.available() is True
    doc = Embedder.available.__doc__ or ""
    assert "NOT a statement that embedding works" in doc
