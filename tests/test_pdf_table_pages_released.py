"""Every pdfplumber page is released after its tables are read.

pdfplumber keeps a page's parsed layout until the page is closed. Left open,
a long text PDF accumulates every page's layout: live, a 161-page code book
held ~1.3 GB and the container memory guard stopped the indexing. Synthetic
pages only.
"""
from __future__ import annotations

import importlib

from app.core import file_crypto


class _Page:
    def __init__(self, txt):
        self._t = txt

    def get_text(self):
        return self._t


class _PlumberPage:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_each_table_page_is_closed_after_use(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    doc_path = str(tmp_path / "long.pdf")
    file_crypto.write_document(doc_path, b"%PDF-1.4 text")
    texts = [f"Section {i} synthetic requirement text long enough to count." for i in range(6)]

    class _Doc:
        def __iter__(self):
            return iter([_Page(t) for t in texts])

        def close(self):
            pass

    plumber_pages = [_PlumberPage() for _ in texts]

    class _Plumber:
        pages = plumber_pages

        def close(self):
            pass

    import fitz as real_fitz
    import pdfplumber
    monkeypatch.setattr(real_fitz, "open", lambda path: _Doc())
    monkeypatch.setattr(pdfplumber, "open", lambda path: _Plumber())
    from app.core import doc_index
    importlib.reload(doc_index)
    monkeypatch.setattr(doc_index, "_pdf_tables_enabled", lambda *a, **k: True)
    monkeypatch.setattr(doc_index, "_pdf_tables_markdown", lambda page: "")
    monkeypatch.setattr(doc_index, "_ocr_pdf_page", lambda page: "")

    text, _meta = doc_index._extract_pdf(doc_path, "synthetic_long.pdf")
    assert "Section 5" in text
    assert all(p.closed for p in plumber_pages)
