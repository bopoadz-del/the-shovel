"""Scanned-PDF rule is configurable; image-only pages are counted, never dropped silently.

``_mostly_without_text`` decides whether a PDF is a scan (body missing) or a
text document with some figure/map pages. The share that tips the verdict
comes from ``PDF_SCANNED_EMPTY_PAGE_SHARE`` (default one half). Pages with no
text layer in an otherwise-text document are recorded on the document row as
``metadata.indexing.pages_without_text`` and show in the document listing.

Synthetic data only.
"""
from __future__ import annotations

import uuid

import pytest

from app.core import doc_index


def test_default_share_is_one_half(monkeypatch):
    monkeypatch.delenv("PDF_SCANNED_EMPTY_PAGE_SHARE", raising=False)
    assert doc_index.pdf_scanned_empty_page_share() == 0.5
    assert not doc_index._mostly_without_text(3, 10)
    assert doc_index._mostly_without_text(5, 10)


def test_configured_share_changes_the_verdict(monkeypatch):
    monkeypatch.setenv("PDF_SCANNED_EMPTY_PAGE_SHARE", "0.3")
    assert doc_index._mostly_without_text(3, 10)
    monkeypatch.setenv("PDF_SCANNED_EMPTY_PAGE_SHARE", "0.9")
    assert not doc_index._mostly_without_text(5, 10)
    # The scanned-PDF verdict on the index path follows the same share.
    meta = {"pages_read": 10, "empty_text_pages": 5, "ocr_required": True}
    assert not doc_index._scanned_pdf_missing_ocr(".pdf", meta)
    monkeypatch.setenv("PDF_SCANNED_EMPTY_PAGE_SHARE", "0.5")
    assert doc_index._scanned_pdf_missing_ocr(".pdf", meta)


@pytest.mark.parametrize("raw", ["0", "-1", "1.5", "half"])
def test_an_invalid_share_falls_back_to_the_default(monkeypatch, raw):
    monkeypatch.setenv("PDF_SCANNED_EMPTY_PAGE_SHARE", raw)
    assert doc_index.pdf_scanned_empty_page_share() == 0.5


@pytest.fixture
def pdf_doc(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import projects as store
    from app.core.users import ensure_user_exists

    store.init_db()
    ensure_user_exists("u-pages")
    pid = store.create_project(name=f"Pages {uuid.uuid4().hex[:6]}", user_id="u-pages")["id"]
    path = tmp_path / "synthetic_atlas.pdf"
    path.write_bytes(b"%PDF-1.4 synthetic")
    doc = store.add_document(pid, "synthetic_atlas.pdf", path.name, str(path), 18)
    return store, pid, doc


def _fake_production(monkeypatch, empty_pages, pages_read=40):
    chunks = [f"synthetic atlas text block {i}" for i in range(6)]
    meta = {"pages_read": pages_read, "empty_text_pages": empty_pages,
            "chunk_pages": [1, 2, 3, 5, 6, 7]}
    monkeypatch.setattr(doc_index, "_produce_chunks_isolated",
                        lambda *a, **k: (list(chunks), dict(meta)))
    from app.core.rag import retriever
    monkeypatch.setattr(retriever, "available", lambda: True)
    monkeypatch.setattr(retriever, "index_chunks", lambda pid, did, ch, **k: len(ch))


def test_pages_without_text_land_on_the_row_and_in_the_listing(pdf_doc, monkeypatch):
    store, pid, doc = pdf_doc
    _fake_production(monkeypatch, empty_pages=4)

    result = doc_index.index_document(pid, doc["id"])

    assert result["status"] == "ok" and result["pages_without_text"] == 4
    row = store.get_document(doc["id"])
    assert row["metadata"]["indexing"]["pages_without_text"] == 4
    listed = {d["id"]: d for d in store.list_documents(pid)}[doc["id"]]
    assert listed["metadata"]["indexing"]["pages_without_text"] == 4


def test_the_upload_status_record_keeps_the_count(pdf_doc, monkeypatch):
    store, pid, doc = pdf_doc
    _fake_production(monkeypatch, empty_pages=2)
    monkeypatch.setenv("INDEX_ON_UPLOAD", "true")

    doc_index.maybe_eager_index(pid, doc["id"])

    indexing = store.get_document(doc["id"])["metadata"]["indexing"]
    assert indexing["status"] == "ok" and indexing["pages_without_text"] == 2


def test_a_fully_text_pdf_writes_nothing_extra(pdf_doc, monkeypatch):
    store, pid, doc = pdf_doc
    _fake_production(monkeypatch, empty_pages=0)
    doc_index.index_document(pid, doc["id"])
    assert "indexing" not in (store.get_document(doc["id"]).get("metadata") or {})
