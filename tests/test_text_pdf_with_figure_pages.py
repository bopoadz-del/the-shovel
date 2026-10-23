"""A text-layer PDF with some figure-only pages is indexed, not called a scan.

Reference books (codes, standards, handbooks) carry full-page figures and maps
with no text layer. When OCR is skipped (a large file) those pages are empty;
that must not throw away the hundreds of pages that DO carry text. A document
is a scan missing OCR only when most of its pages have no text layer.

Synthetic pages only.
"""
from __future__ import annotations

import importlib

from app.core import file_crypto


class _Page:
    def __init__(self, txt):
        self._t = txt

    def get_text(self):
        return self._t


def _extract(tmp_path, monkeypatch, pages):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    doc_path = str(tmp_path / "reference.pdf")
    file_crypto.write_document(doc_path, b"%PDF-1.4 text")

    class _Doc:
        def __iter__(self):
            return iter([_Page(t) for t in pages])

        def close(self):
            pass

    import fitz as real_fitz
    monkeypatch.setattr(real_fitz, "open", lambda path: _Doc())
    # Any file is "too large" for OCR here, so OCR never runs.
    monkeypatch.setenv("PDF_OCR_MAX_SIZE_MB", "0.000000001")
    from app.core import doc_index
    importlib.reload(doc_index)
    monkeypatch.setattr(doc_index, "_ocr_pdf_page", lambda page: "")
    monkeypatch.setattr(doc_index, "_pdf_tables_enabled", lambda *a, **k: False)
    text, meta = doc_index._extract_pdf(doc_path, "synthetic_reference.pdf")
    return doc_index, text, meta


def test_text_pages_with_a_few_figure_pages_index(tmp_path, monkeypatch):
    pages = [f"Section {i}.1 Synthetic design requirement number {i} applies to all members." for i in range(1, 11)]
    pages[3] = ""   # a full-page figure
    pages[7] = ""   # a map
    doc_index, text, meta = _extract(tmp_path, monkeypatch, pages)
    assert "Section 10.1" in text
    assert meta.get("ocr_required") is not True
    assert not doc_index._scanned_pdf_missing_ocr(".pdf", meta)


def test_a_mostly_blank_scan_is_still_a_scan(tmp_path, monkeypatch):
    pages = ["Cover sheet"] + [""] * 8
    doc_index, _text, meta = _extract(tmp_path, monkeypatch, pages)
    assert doc_index._scanned_pdf_missing_ocr(".pdf", meta)
