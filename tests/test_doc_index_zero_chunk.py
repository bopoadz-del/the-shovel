"""Fail-loud zero-chunk indexing (Task T4).

* index_document returns an explicit error status for supported files that
  produce no chunks.
* admin_project_reindex surfaces the ZERO_CHUNK condition as a 422 with a
  banner instead of silently returning a green summary.
"""
from __future__ import annotations

import importlib
import os

import pytest
from fastapi.testclient import TestClient

from app.dependencies import require_api_key
from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def _admin_auth():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "system",
        "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    from app.core import projects as projects_mod

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _write_txt(tmp_path, filename: str, content: bytes) -> str:
    from app.core import file_crypto

    p = str(tmp_path / filename)
    file_crypto.write_document(p, content)
    return p


def test_index_document_zero_chunk_for_empty_supported_file(
    fresh_db, tmp_path, monkeypatch
):
    """A supported .txt file with no extractable text must fail loud."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import doc_index, projects as projects_mod

    importlib.reload(doc_index)

    proj = projects_mod.create_project("Zero Doc Project")
    doc_path = _write_txt(tmp_path, "empty.txt", b"")
    doc = projects_mod.add_document(
        proj["id"], "empty.txt", file_path=doc_path, size=0
    )

    result = doc_index.index_document(proj["id"], doc["id"])

    assert result["status"] == "error"
    assert result["error"] == "ZERO_CHUNK"
    assert "0 chunks" in result["banner"]
    assert result["document_id"] == doc["id"]


def test_admin_project_reindex_returns_422_when_all_docs_zero_chunk(
    client, fresh_db, tmp_path, monkeypatch
):
    """The admin reindex endpoint must reject a project whose supported docs
    all produce zero chunks."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import doc_index, projects as projects_mod

    importlib.reload(doc_index)

    proj = projects_mod.create_project("Zero Project")
    doc_path = _write_txt(tmp_path, "empty.txt", b"")
    projects_mod.add_document(proj["id"], "empty.txt", file_path=doc_path, size=0)

    resp = client.post(
        "/v1/admin/debug/project-reindex",
        params={"project_id": proj["id"]},
    )

    assert resp.status_code == 422, resp.text
    text = resp.text
    assert "ZERO_CHUNK" in text
    assert "0 chunks" in text


def test_extract_exception_is_named_not_swallowed(tmp_path, monkeypatch):
    """An extractor exception must land in meta, not come back as empty {}."""
    from app.core import doc_index

    def boom(*_a, **_k):
        raise ValueError("bad xref")

    monkeypatch.setattr(doc_index, "_extract_pdf", boom)
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    text, meta = doc_index._extract_with_meta_impl(str(pdf), "broken.pdf")

    assert text == ""
    assert meta["extract_failed"] == "ValueError"
    assert "bad xref" in meta["extract_failed_detail"]


def test_zero_chunk_names_the_cause(fresh_db, tmp_path, monkeypatch):
    """ZERO_CHUNK must carry the extractor exception, not just the marker."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import doc_index, projects as projects_mod

    importlib.reload(doc_index)

    def boom(*_a, **_k):
        raise ValueError("bad xref")

    monkeypatch.setattr(doc_index, "_extract_pdf", boom)

    proj = projects_mod.create_project("Zero Cause Project")
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    doc = projects_mod.add_document(
        proj["id"], "broken.pdf", file_path=str(pdf), size=pdf.stat().st_size
    )

    result = doc_index.index_document(proj["id"], doc["id"])

    assert result["status"] == "error"
    assert result["error"] == "ZERO_CHUNK"
    assert "ValueError: bad xref" in result["extract_error"]


def test_maybe_eager_index_persists_indexing_status(fresh_db, tmp_path, monkeypatch):
    """Background indexing must write status onto the document row."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("INDEX_ON_UPLOAD", "true")
    from app.core import doc_index, projects as projects_mod

    importlib.reload(doc_index)

    def boom(*_a, **_k):
        raise ValueError("bad xref")

    monkeypatch.setattr(doc_index, "_extract_pdf", boom)

    proj = projects_mod.create_project("Index Persist Project")
    pdf = tmp_path / "broken.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    bad = projects_mod.add_document(
        proj["id"], "broken.pdf", file_path=str(pdf), size=pdf.stat().st_size
    )

    err_result = doc_index.maybe_eager_index(proj["id"], bad["id"])
    assert err_result["error"] == "ZERO_CHUNK"
    stored_bad = projects_mod.get_document(bad["id"])
    idx_bad = (stored_bad.get("metadata") or {}).get("indexing")
    assert idx_bad["status"] == "error"
    assert idx_bad["error"] == "ZERO_CHUNK"
    assert idx_bad["chunks"] == 0
    assert "ValueError: bad xref" in (idx_bad.get("detail") or "")

    monkeypatch.setattr(doc_index, "_extract_pdf", lambda *_a, **_k: ("ok page", {}))
    good_path = _write_txt(tmp_path, "ok.txt", b"hello world " * 80)
    good = projects_mod.add_document(
        proj["id"], "ok.txt", file_path=good_path, size=os.path.getsize(good_path)
    )
    ok_result = doc_index.maybe_eager_index(proj["id"], good["id"])
    assert ok_result["status"] == "ok"
    assert ok_result["total_chunks"] > 0
    stored_ok = projects_mod.get_document(good["id"])
    idx_ok = (stored_ok.get("metadata") or {}).get("indexing")
    assert idx_ok["status"] == "ok"
    assert idx_ok["chunks"] > 0


def test_admin_doc_reindex_does_not_ok_a_scan_without_ocr(
    client, fresh_db, tmp_path, monkeypatch
):
    """POST /doc-reindex must not return status=ok for a cover-only scan."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import doc_index, projects as projects_mod
    from app.routers import admin as admin_mod

    importlib.reload(doc_index)
    monkeypatch.setattr(admin_mod, "_pdf_needs_async_ocr", lambda *a, **k: False)
    monkeypatch.setattr(
        doc_index,
        "_extract_with_meta",
        lambda *a, **k: (
            "Cover page only. VERIFIED-TOTAL GUARD.",
            {
                "ocr_required": True,
                "empty_text_pages": 8,
                "ocr_pages": 0,
                "ocr_skipped_too_large": True,
            },
        ),
    )
    monkeypatch.setattr(doc_index, "_boq_chunks_for_document", lambda *a, **k: [])
    monkeypatch.setattr(doc_index, "_drawing_chunks_for_document", lambda *a, **k: [])

    proj = projects_mod.create_project("Admin Scan Reindex")
    path = _write_txt(tmp_path, "Priced BOQ.pdf", b"%PDF-1.4 cover")
    doc = projects_mod.add_document(
        proj["id"], "Bill of Quantities (Priced).pdf", file_path=path, size=20
    )

    r = client.post(
        "/v1/admin/debug/doc-reindex",
        params={
            "project_id": proj["id"],
            "document_id": doc["id"],
            "chunker": "finer",
            "force_ocr": "false",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] != "ok", body
    assert body["error"] == "OCR_REQUIRED"


def test_admin_doc_reindex_queues_long_scan(client, fresh_db, tmp_path, monkeypatch):
    """A 370-page scan must not pretend to finish in one sync 15s POST."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core import projects as projects_mod
    from app.routers import admin as admin_mod

    monkeypatch.setattr(admin_mod, "_pdf_needs_async_ocr", lambda *a, **k: True)
    monkeypatch.setattr(
        "app.core.doc_index._pdf_page_count", lambda *_a, **_k: 370
    )
    monkeypatch.setattr(
        "app.core.doc_index.index_document",
        lambda *a, **k: {
            "status": "ok",
            "indexed": 1,
            "ocr_pages": 40,
            "total_chunks": 80,
        },
    )

    proj = projects_mod.create_project("Async OCR Reindex")
    path = _write_txt(tmp_path, "Priced BOQ.pdf", b"%PDF-1.4 cover")
    doc = projects_mod.add_document(
        proj["id"], "Bill of Quantities (Priced).pdf", file_path=path, size=20
    )

    r = client.post(
        "/v1/admin/debug/doc-reindex",
        params={"project_id": proj["id"], "document_id": doc["id"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ocr_started", body
    assert body["page_count"] == 370
    assert body.get("job_id")
    poll = client.get(f"/v1/admin/debug/doc-reindex/job/{body['job_id']}")
    assert poll.status_code == 200, poll.text
