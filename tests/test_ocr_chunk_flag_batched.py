"""P-G3: OCR provenance survives the batched (force-OCR) PDF path, and a
doc-reindex-once run stores the headings it found on the document row.

The batched extractor summed each batch's ``ocr_pages`` but dropped its
``ocr_page_numbers``, so ``_ocr_flags_for`` had nothing to map and every chunk
of a force-OCR'd scan was written with ``ocr=false``. These tests build their
own synthetic PDFs (a text-layer page vs an image-only page) and mock per-page
OCR at ``doc_index._ocr_pdf_page`` (Tesseract is not guaranteed here).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from app.core import doc_index
from app.core.doc_index import _extract_pdf_batched, _ocr_flags_for
from scripts import doc_reindex_once as dro


def _pdf(path: Path, layout: List[bool]) -> str:
    """One page per entry: True = text-layer page, False = blank (image-only)."""
    import fitz

    doc = fitz.open()
    for n, has_text in enumerate(layout, start=1):
        page = doc.new_page()
        if has_text:
            body = " ".join(f"txt{n}word{i}" for i in range(400))
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), body, fontsize=6)
    doc.save(str(path))
    doc.close()
    return str(path)


def _fake_ocr(page: Any) -> str:
    n = int(page.number) + 1
    return " ".join(f"ocr{n}word{i}" for i in range(400))


def _first_page_of(chunk: str) -> int:
    tok = chunk.split()[0]
    return int(tok[3:tok.index("word")])


@pytest.mark.parametrize(
    "layout, batch, ocr_pages",
    [
        ([True, False, True], "1", [2]),
        ([False, True, True, False], "2", [1, 4]),
    ],
    ids=["ocr_middle_page_batch1", "ocr_first_and_last_batch2"],
)
def test_batched_force_ocr_flags_ocr_chunks_and_not_text_layer_chunks(
    tmp_path, monkeypatch, layout, batch, ocr_pages,
):
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", batch)
    monkeypatch.setattr(doc_index, "_ocr_pdf_page", _fake_ocr)
    path = _pdf(tmp_path / "scan_mix.pdf", layout)

    text, meta = _extract_pdf_batched(path, "scan_mix.pdf", force_ocr=True)

    # Absolute page numbers, across batches, exactly the OCR'd pages.
    assert meta["ocr_page_numbers"] == ocr_pages
    assert meta["pages_ocrd"] == len(ocr_pages)

    chunks, pages = doc_index._text_chunks_with_pages(text, meta, "default", "scan_mix.pdf")
    flags = _ocr_flags_for(chunks, {**meta, "chunk_pages": pages})
    assert flags is not None and len(flags) == len(chunks)
    # Each chunk's flag is true exactly when its text came from OCR.
    for chunk, flag in zip(chunks, flags):
        assert flag is chunk.startswith("ocr"), (chunk[:20], flag)
        assert (_first_page_of(chunk) in ocr_pages) is flag
    assert True in flags and False in flags


@pytest.mark.parametrize("layout, batch", [([True, True], "1"), ([True, True, True], "2")])
def test_batched_text_layer_only_pdf_has_no_ocr_pages(tmp_path, monkeypatch, layout, batch):
    monkeypatch.setenv("PDF_OCR_BATCH_PAGES", batch)
    monkeypatch.setattr(doc_index, "_ocr_pdf_page", _fake_ocr)
    path = _pdf(tmp_path / "digital.pdf", layout)
    text, meta = _extract_pdf_batched(path, "digital.pdf", force_ocr=True)
    assert "ocr_page_numbers" not in meta
    assert meta["pages_ocrd"] == 0
    chunks, pages = doc_index._text_chunks_with_pages(text, meta, "default", "digital.pdf")
    assert chunks and _ocr_flags_for(chunks, {**meta, "chunk_pages": pages}) is None


# --- doc-reindex-once: headings stored on the row -----------------------------


class _World:
    def __init__(self, new_chunks: List[str], index_result: Dict[str, Any]):
        self.doc = {"id": "d1", "project_id": "p1", "original_name": "b.pdf",
                    "drive_file_id": "drv", "chunk_count": 1, "metadata": {}}
        self.chunks = ["old"]
        self.new_chunks = new_chunks
        self.index_result = index_result
        self.writes: List[tuple] = []

    def deps(self, tmp_path: Path) -> dro.Deps:
        def index(pid: str, did: str, **kw: Any) -> Dict[str, Any]:
            if self.index_result.get("status") != "error":
                self.chunks = list(self.new_chunks)
            return dict(self.index_result)

        def write(raw: bytes, name: str) -> Path:
            p = tmp_path / name
            p.write_bytes(raw)
            return p

        def update_metadata(doc_id: str, md: Dict[str, Any]) -> None:
            self.writes.append((doc_id, md))
            self.doc["metadata"].update(md)

        return dro.Deps(
            get_document=lambda d: dict(self.doc) if d == "d1" else None,
            drive_file_id_of=lambda d: d.get("drive_file_id") or "",
            download=lambda fid: (b"%PDF", None),
            write_transient=write,
            discard_transient=lambda p: Path(p).unlink(missing_ok=True),
            set_file_path=lambda d, fp: None,
            index_document=index,
            chunk_texts=lambda pid, did: list(self.chunks),
            update_metadata=update_metadata,
            log=lambda line: None,
        )


@pytest.mark.parametrize(
    "new_chunks, pattern, want",
    [
        (["SECTION 1 - A\nrow", "row\nSECTION 2 - B"], r"^SECTION \d", ["SECTION 1 - A", "SECTION 2 - B"]),
        (["Part X\nPart Y", "body"], r"^Part ", ["Part X", "Part Y"]),
    ],
)
def test_successful_run_stores_its_headings_on_the_row(tmp_path, new_chunks, pattern, want):
    w = _World(new_chunks, {"status": "ok", "indexed": 1})
    rep = dro.reindex_one("p1", "d1", force_ocr=True, heading_re=pattern, deps=w.deps(tmp_path))
    assert rep["ok"] is True and rep["headings_stored"] is True
    assert len(w.writes) == 1
    stored = w.doc["metadata"][dro.HEADINGS_KEY]
    assert stored["lines"] == want == rep["headings"]
    assert stored["pattern"] == pattern
    assert stored["at"].endswith("Z")


@pytest.mark.parametrize(
    "index_result, pattern",
    [
        ({"status": "error", "error": "extract failed"}, r"^SECTION"),
        ({"status": "ok", "indexed": 1}, None),
    ],
    ids=["failed_run", "no_pattern"],
)
def test_failed_run_or_no_pattern_stores_nothing(tmp_path, index_result, pattern):
    w = _World(["SECTION 9 - Z"], index_result)
    rep = dro.reindex_one("p1", "d1", force_ocr=True, heading_re=pattern, deps=w.deps(tmp_path))
    assert rep["headings_stored"] is False
    assert w.writes == []
    assert dro.HEADINGS_KEY not in w.doc["metadata"]
