"""OCR in the indexer: image-only / scanned PDFs, and the rule that images
are NOT ingested.

The RAG takes text formats only (docs/RAG_GAPS_REVIEW_2026-09-12.md section
E; docs/INGEST_EXCLUSION_RULE.md). Images used to be OCR'd into the index
("Stream F"); that path is removed. OCR still runs where it belongs -- on the
image-only pages of a PDF.

Tesseract is not guaranteed in the test environment, so OCR is mocked.

Covered:
  * image-only .pdf (empty fitz text) -> falls back to OCR
  * a .pdf with a text layer keeps its text and is not OCR'd
  * an image is not ingested: no OCR call, empty text, skipped by the indexer
  * the async->sync bridge works with and without a running event loop
"""

import importlib
import json
import os

import pytest

from app.core import file_crypto
from app.core import projects as projects_mod


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_ocr_stub(result):
    """Build an async stand-in for OCRBlock.process returning ``result``."""
    async def _fake_process(self, input_data, params=None):
        return result
    return _fake_process


def _make_ocr_raiser():
    async def _fake_process(self, input_data, params=None):
        raise RuntimeError("tesseract exploded")
    return _fake_process


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    # Disable GK so training_material chunks from other tests don't outrank
    # the OCR-indexed documents in search results on the shared PostgreSQL DB.
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    from app.core.rag import vector_store as _vs, embeddings as _emb
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    return tmp_path


# ────────────────────────────────────────────────────────────────────────────
# Scanned PDFs are OCR'd
# ────────────────────────────────────────────────────────────────────────────

def test_extract_scanned_pdf_falls_back_to_ocr(tmp_path, monkeypatch):
    """An image-only PDF (empty fitz text) falls back to OCR."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index
    importlib.reload(doc_index)

    # fitz yields effectively empty text → triggers OCR fallback
    class FakePage:
        def get_text(self):
            return "   \n  "

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage()])

        def close(self):
            pass

    import fitz as real_fitz
    monkeypatch.setattr(real_fitz, "open", lambda path: FakeDoc())

    # _extract_pdf OCRs each thin-text page via doc_index._ocr_pdf_page
    # (pytesseract on a rendered pixmap), NOT OCRBlock.process. Mock that
    # per-page entry point — FakePage has no pixmap, so the real function
    # would return "" and the fallback text would never appear.
    monkeypatch.setattr(
        doc_index, "_ocr_pdf_page",
        lambda page: "SCANNED SITE PLAN extracted via OCR fallback",
    )

    pdf_path = str(tmp_path / "scan.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 image-only")

    text = doc_index.extract_document_text(pdf_path, "scan.pdf")
    assert "SCANNED SITE PLAN" in text


def test_extract_text_pdf_does_not_ocr(tmp_path, monkeypatch):
    """A PDF with a real text layer keeps the fitz text — no OCR fallback."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.blocks.ocr import OCRBlock
    from app.core import doc_index
    importlib.reload(doc_index)

    class FakePage:
        def get_text(self):
            return "This PDF has a genuine, sufficiently long text layer. "

    class FakeDoc:
        def __iter__(self):
            return iter([FakePage(), FakePage()])

        def close(self):
            pass

    import fitz as real_fitz
    monkeypatch.setattr(real_fitz, "open", lambda path: FakeDoc())

    # If OCR were (wrongly) invoked, it would return this sentinel.
    monkeypatch.setattr(OCRBlock, "process", _make_ocr_stub({
        "status": "success", "text": "OCR-SENTINEL-SHOULD-NOT-APPEAR",
        "quality": {"low_quality": False},
    }))

    pdf_path = str(tmp_path / "textlayer.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 text layer")

    text = doc_index.extract_document_text(pdf_path, "textlayer.pdf")
    assert "genuine, sufficiently long text layer" in text
    assert "OCR-SENTINEL" not in text


# ────────────────────────────────────────────────────────────────────────────
# Images are not ingested (text formats only)
# ────────────────────────────────────────────────────────────────────────────

def _ocr_spy(monkeypatch):
    from app.blocks.ocr import OCRBlock

    calls = []

    async def _process(self, input_data, params=None):
        calls.append(input_data)
        return {"status": "success", "text": "SHOULD NOT BE EXTRACTED",
                "quality": {"low_quality": False}}

    monkeypatch.setattr(OCRBlock, "process", _process)
    return calls


@pytest.mark.parametrize("name", ["drawing.png", "site.jpg", "scan.tiff"])
def test_an_image_is_not_extracted_or_ocrd(tmp_path, monkeypatch, name):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    from app.core import doc_index
    importlib.reload(doc_index)
    calls = _ocr_spy(monkeypatch)

    path = str(tmp_path / name)
    file_crypto.write_document(path, b"\x89PNG\r\n\x1a\n image bytes")

    assert doc_index.extract_document_text(path, name) == ""
    assert calls == []


def test_an_image_document_is_skipped_by_the_indexer(fresh_db, tmp_path, monkeypatch):
    """Even if an image row exists (pre-rule data), the indexer skips it as an
    unsupported type -- it never reaches OCR, chunks or the vector store."""
    from app.core import doc_index
    importlib.reload(doc_index)
    calls = _ocr_spy(monkeypatch)

    pid = projects_mod.create_project("Image Skip Project")["id"]
    img_path = str(tmp_path / "rebar.jpg")
    file_crypto.write_document(img_path, b"\xff\xd8 jpeg")
    projects_mod.add_document(pid, "rebar.jpg", file_path=img_path, size=8)

    result = doc_index.index_project(pid)

    assert result["indexed"] == 0
    assert result["skipped_unsupported"] == 1
    saved = doc_index._load_index(pid)
    assert saved["documents"] == []
    assert [s["reason"] for s in saved["skipped"]] == ["unsupported_type"]
    assert calls == []


# ────────────────────────────────────────────────────────────────────────────
# Bridge — async OCR/BOQ blocks from sync indexing code
# ────────────────────────────────────────────────────────────────────────────

def test_run_sync_works_with_no_running_loop():
    from app.core import doc_index

    async def _work():
        return "sync-context result"

    assert doc_index._run_sync(_work()) == "sync-context result"


@pytest.mark.asyncio
async def test_run_sync_works_inside_a_running_loop():
    """search_project_documents can trigger indexing from inside a running
    loop; the bridge must not call asyncio.run() there."""
    from app.core import doc_index

    async def _work():
        return "running-loop result"

    assert doc_index._run_sync(_work()) == "running-loop result"


# ────────────────────────────────────────────────────────────────────────────
# Scanned-PDF OCR: guards that went with the image tests in #773
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("low", [True, False])
async def test_pdf_ocr_quality_flag_reaches_entry_and_hit(fresh_db, tmp_path, monkeypatch, low):
    from app.core import doc_index
    importlib.reload(doc_index)
    text = "blurry steel beam connection detail drawing scan " * 4
    meta = {"ocr_low_quality": True} if low else {}
    monkeypatch.setattr(doc_index, "_extract_with_meta", lambda fp, fn, *a, **k: (text, dict(meta)))
    monkeypatch.setenv("DOC_EXTRACT_ISOLATE_MIN_MB", "1000000")

    pid = projects_mod.create_project("PDF OCR quality")["id"]
    p = str(tmp_path / "scan.pdf")
    file_crypto.write_document(p, b"%PDF-1.4 scan")
    doc = projects_mod.add_document(pid, "scan.pdf", file_path=p, size=13)
    doc_index.index_project(pid)

    entry = next(d for d in doc_index._load_index(pid)["documents"] if d["document_id"] == doc["id"])
    assert bool(entry.get("ocr_low_quality", False)) is low
    hits = await doc_index.search_project_documents(pid, "steel beam connection")
    hit = next(h for h in hits if h["document_id"] == doc["id"])
    assert bool(hit.get("ocr_low_quality", False)) is low


def test_pdf_page_ocr_never_raises():
    from app.core import doc_index
    assert doc_index._ocr_pdf_page(object()) == ""


def test_scanned_pdf_with_failed_ocr_is_indexed_not_skipped(fresh_db, tmp_path, monkeypatch):
    from app.core import doc_index
    importlib.reload(doc_index)
    monkeypatch.setattr(doc_index, "_extract_with_meta", lambda fp, fn, *a, **k: ("", {}))
    monkeypatch.setenv("DOC_EXTRACT_ISOLATE_MIN_MB", "1000000")

    pid = projects_mod.create_project("OCR fail PDF")["id"]
    p = str(tmp_path / "fail.pdf")
    file_crypto.write_document(p, b"%PDF-1.4 x")
    projects_mod.add_document(pid, "fail.pdf", file_path=p, size=10)
    result = doc_index.index_project(pid)

    assert result["skipped_unsupported"] == 0
