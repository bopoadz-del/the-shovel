"""Terminal scanned-image outcome in the p1b in-place retry path.

A zero-chunk/failed PDF that carries no digital text layer is an image-only
scan: extracting it again can only reproduce the same empty result, and OCR is
not a remedy for these sheets. The retry path therefore stamps the row's
ingest status as a terminal scanned-image outcome (reusing
``classify_source_document``) and skips extraction, so the file is never
OCR-rescued and never retried. Two inputs per decision: one scan PDF (no text
layer) and one text PDF (text layer).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.core import ingest_status as ist
from scripts.p1b_ingest_drive_server import scanned_pdf_terminal_status


def _write_text_pdf(path, pages: list[str | None]) -> None:
    """Create a real PDF at ``path``. A page entry of ``None`` is image-only
    (no text layer); a non-empty string is inserted as that page's text layer."""
    import fitz

    doc = fitz.open()
    try:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            if text:
                page.insert_text(fitz.Point(72, 72), text, fontsize=12)
        doc.save(path)
    finally:
        doc.close()


class _Drive:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def download_file_to_path(self, fid, dest):
        Path(dest).write_bytes(self.payload)
        return len(self.payload), hashlib.sha256(self.payload).hexdigest(), None


def _install_retry_fakes(monkeypatch):
    indexed = []
    stamped = []

    monkeypatch.setattr("app.core.projects.find_document_by_sha", lambda *a, **k: None)
    monkeypatch.setattr("app.core.projects.update_document_metadata", lambda *a, **k: None)
    monkeypatch.setattr("app.core.projects.set_document_file_path", lambda *a, **k: None)
    monkeypatch.setattr("app.core.projects.set_document_drive_md5", lambda *a, **k: None)
    monkeypatch.setattr(
        "app.core.projects.stamp_document_index",
        lambda doc_id, **kw: stamped.append({"doc_id": doc_id, **kw}) or {"id": doc_id},
    )
    monkeypatch.setattr(
        "app.core.doc_index.index_document",
        lambda *a, **k: indexed.append((a, k)) or {"status": "ok", "rag_indexed": 2},
    )
    return indexed, stamped


@pytest.mark.parametrize("extension", [".pdf", ".PDF"])
def test_scanned_pdf_is_a_terminal_outcome(extension):
    status, reason = scanned_pdf_terminal_status(
        extension=extension, has_text_layer=False,
    )
    assert status == ist.UNSUPPORTED_TYPE
    assert reason == f"scanned_image:{ist.TERMINAL}"


@pytest.mark.parametrize("has_text_layer", [True, None])
def test_text_pdf_is_not_a_scanned_image_outcome(has_text_layer):
    assert scanned_pdf_terminal_status(
        extension=".pdf", has_text_layer=has_text_layer,
    ) is None


@pytest.mark.parametrize("extension", [".docx", ".txt"])
def test_non_pdf_is_never_a_scanned_image_outcome(extension):
    assert scanned_pdf_terminal_status(
        extension=extension, has_text_layer=False,
    ) is None


@pytest.mark.parametrize("fid", ["drive-scan-a", "drive-scan-b"])
def test_scanned_pdf_retry_stamps_terminal_and_skips_extraction(tmp_path, monkeypatch, fid):
    """An image-only PDF retried in place is closed as terminal, never extracted."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from scripts.p1b_ingest_drive_server import _ingest_file

    indexed, stamped = _install_retry_fakes(monkeypatch)
    pdf = tmp_path / f"{fid}.pdf"
    _write_text_pdf(str(pdf), [None, None])

    existing = {
        "id": f"doc-{fid}",
        "original_name": "scan.pdf",
        "metadata": {"drive_file_id": fid, "drive_path": "Tier1/scan.pdf"},
    }
    _rel, result = _ingest_file(
        {"id": fid, "name": "scan.pdf", "_drive_path": "Tier1/scan.pdf",
         "mimeType": "application/pdf", "size": pdf.stat().st_size},
        "proj", tmp_path, "run-scan", _Drive(pdf.read_bytes()),
        existing_doc=existing,
    )

    assert indexed == []  # never extracted / OCR'd
    assert result["status"] == "error"
    assert result["error"] == "SCANNED_IMAGE"
    assert len(stamped) == 1
    assert stamped[0]["doc_id"] == f"doc-{fid}"
    assert stamped[0]["chunk_count"] == 0
    assert stamped[0]["ingest_status"] == ist.UNSUPPORTED_TYPE
    assert stamped[0]["ingest_status_reason"] == f"scanned_image:{ist.TERMINAL}"


@pytest.mark.parametrize("fid", ["drive-text-a", "drive-text-b"])
def test_text_pdf_retry_still_indexes(tmp_path, monkeypatch, fid):
    """A PDF with a text layer retried in place is re-indexed, not stamped terminal."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from scripts.p1b_ingest_drive_server import _ingest_file

    indexed, stamped = _install_retry_fakes(monkeypatch)
    pdf = tmp_path / f"{fid}.pdf"
    _write_text_pdf(str(pdf), ["Specification text layer"])

    existing = {
        "id": f"doc-{fid}",
        "original_name": "text.pdf",
        "metadata": {"drive_file_id": fid, "drive_path": "Tier1/text.pdf"},
    }
    _ingest_file(
        {"id": fid, "name": "text.pdf", "_drive_path": "Tier1/text.pdf",
         "mimeType": "application/pdf", "size": pdf.stat().st_size},
        "proj", tmp_path, "run-text", _Drive(pdf.read_bytes()),
        existing_doc=existing,
    )

    assert stamped == []
    assert len(indexed) == 1
    assert indexed[0][0][1] == f"doc-{fid}"  # re-indexed the SAME row
