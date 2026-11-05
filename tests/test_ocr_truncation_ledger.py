"""OCR truncation ledger: a scan cut short by the page cap is never INDEXED.

S-C of the corpus-indexing order. The owner's rule: a photo/scan PDF is OCR'd
into readable retrievable chunks, never silently truncated. These tests pin
the mechanism, not any document:

  * a fixture scan of N pages with the page cap at N-2 records
    ocr_truncated=True, pages_ocrd=N-2, pages_in_file=N on the extraction
    metadata (parametrized over two page counts);
  * the ledger stamps PARTIAL_OCR (not in INDEXED_STATUSES) with the pages
    columns, and the panel/chat phrasing says "partially read, pages 1-N of
    M" (parametrized over two pairs);
  * an OCR'd page below the quality floor is never emitted as text and is
    counted in ocr_unusable_pages (parametrized over two inputs).

Tesseract is not guaranteed in the test environment, so per-page OCR is
mocked at the entry point ``doc_index._ocr_pdf_page``.

The mutation gate (CI-M) runs the tests that name the functions it mutates;
the imports below use the full module path so this file is selected for every
changed function, and the tests assert the exact behaviours the mutants
remove.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from app.agents.base.tools import handle_fetch_document
from app.agents.runtime import (
    _fetch_document_content,
    _partial_read_note_for_doc,
)
from app.core import doc_index
from app.core.doc_index import (
    _extract_pdf,
    _extract_pdf_batched,
    _index_document,
    _stamp_index_ledger,
    ocr_page_usable,
)
from app.core.ingest_status import (
    INDEXED_STATUSES,
    NO_CHUNK_STATUSES,
    PARTIAL_OCR,
    partial_read_note,
)
from app.core.projects import panel_status_for
from app.core import projects as projects_mod


def _fake_doc_of_pages(pages: int) -> tuple:
    """A fake fitz doc with ``pages`` image-only pages and a page_count."""

    class FakePage:
        def get_text(self):
            return "   \n  "

    class FakeDoc:
        def __init__(self):
            self.page_count = pages

        def __iter__(self):
            return iter(FakePage() for _ in range(pages))

        def close(self):
            pass

    return FakeDoc()


def _usable_ocr_text() -> str:
    return "Measured quantity item code A-1234 with rate and total value noted"


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _reset_doc_index(monkeypatch):
    """Reload doc_index so env-read ceilings are the test's own."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    importlib.reload(doc_index)
    return doc_index


# --- extraction metadata -----------------------------------------------------


@pytest.mark.parametrize("pages", [5, 8])
def test_scan_truncated_by_page_cap_records_the_ledger_fields(
    tmp_path, monkeypatch, pages,
):
    """N image-only pages with the OCR cap at N-2: truncation is explicit."""
    di = _reset_doc_index(monkeypatch)
    monkeypatch.setenv("PDF_OCR_PAGE_CAP", str(pages - 2))

    import fitz as real_fitz
    monkeypatch.setattr(real_fitz, "open", lambda path: _fake_doc_of_pages(pages))
    monkeypatch.setattr(di, "_ocr_pdf_page", lambda page: _usable_ocr_text())

    from app.core import file_crypto
    pdf_path = str(tmp_path / "scan.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 image-only")

    text, meta = di._extract_pdf(pdf_path, "scan.pdf")
    assert meta["ocr_truncated"] is True
    assert meta["pages_in_file"] == pages
    assert meta["pages_ocrd"] == pages - 2
    assert meta["ocr_unusable_pages"] == 0
    # The OCR'd prefix is real text; the tail is what truncation names.
    assert text.count(_usable_ocr_text().split()[0]) == pages - 2


@pytest.mark.parametrize(
    "noise",
    ["xx", "---- ---- ---- ----"],
    ids=["below_char_floor", "no_alnum_density"],
)
def test_ocr_page_below_the_quality_floor_is_not_emitted(tmp_path, monkeypatch, noise):
    """OCR output below the floor is counted, never indexed as a chunk."""
    di = _reset_doc_index(monkeypatch)
    pages = 3
    monkeypatch.setenv("PDF_OCR_PAGE_CAP", str(pages))

    import fitz as real_fitz
    monkeypatch.setattr(real_fitz, "open", lambda path: _fake_doc_of_pages(pages))
    monkeypatch.setattr(di, "_ocr_pdf_page", lambda page: noise)

    from app.core import file_crypto
    pdf_path = str(tmp_path / "noise.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 image-only")

    text, meta = di._extract_pdf(pdf_path, "noise.pdf")
    assert noise.strip() not in text
    assert meta["pages_ocrd"] == 0
    assert meta["ocr_unusable_pages"] == pages


@pytest.mark.parametrize("reject_input,accept_input", [
    ("xx", "measured item code A-1234 with rate and total"),
    ("---- ---- ---- ----", "ocr text page one of the scanned schedule"),
])
def test_ocr_quality_floor_rejects_and_accepts_by_mechanism(
    reject_input, accept_input,
):
    """The floor itself: below it rejected, above it accepted."""
    assert not ocr_page_usable(reject_input)
    assert ocr_page_usable(accept_input)


@pytest.mark.parametrize("reason", ["suspect head", "orphaned source"])
def test_extract_pdf_keeps_quarantined_documents_blocked(tmp_path, monkeypatch, reason):
    """A quarantined file never reaches extraction: the blocked meta is the
    result, not an extract attempt (kills the drop of that first rule)."""
    di = _reset_doc_index(monkeypatch)
    monkeypatch.setattr(
        di, "_quarantine_meta",
        lambda path, name: {"quarantined": True, "reason": reason},
    )
    from app.core import file_crypto
    pdf_path = str(tmp_path / "q.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 dummy")

    text, meta = di._extract_pdf(pdf_path, "q.pdf")
    assert text == ""
    assert meta.get("quarantined") is True
    assert meta.get("reason") == reason


@pytest.mark.parametrize("reason", ["suspect head", "orphaned source"])
def test_extract_pdf_batched_keeps_quarantined_documents_blocked(tmp_path, monkeypatch, reason):
    """Same first rule on the batched path (kills its drop)."""
    di = _reset_doc_index(monkeypatch)
    monkeypatch.setattr(
        di, "_quarantine_meta",
        lambda path, name: {"quarantined": True, "reason": reason},
    )
    from app.core import file_crypto
    pdf_path = str(tmp_path / "q.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 dummy")

    text, meta = di._extract_pdf_batched(pdf_path, "q.pdf")
    assert text == ""
    assert meta.get("quarantined") is True
    assert meta.get("reason") == reason


def test_extract_pdf_propagates_memory_exhaustion(tmp_path, monkeypatch):
    """A genuine MemoryError is surfaced, never absorbed into an empty
    document (kills the swallow-exception mutant on the re-raise)."""
    di = _reset_doc_index(monkeypatch)
    import fitz as real_fitz

    def _oom(path):
        raise MemoryError("PDF extraction exhausted memory")

    monkeypatch.setattr(real_fitz, "open", _oom)
    from app.core import file_crypto
    pdf_path = str(tmp_path / "oom.pdf")
    file_crypto.write_document(pdf_path, b"%PDF-1.4 dummy")

    with pytest.raises(MemoryError):
        di._extract_pdf(pdf_path, "oom.pdf")


@pytest.mark.parametrize("missing_id", ["no-such-document", "ghost-row-2"])
def test_index_document_reports_a_missing_document(tmp_path, monkeypatch, missing_id):
    """A missing documents row is the named 'document not found' outcome,
    never an attribute walk (kills the drop of that first rule)."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    out = _index_document("demo_proj", missing_id)
    assert out.get("error") == "document not found"


# --- ledger stamp + panel phrasing -------------------------------------------


@pytest.mark.parametrize("ocrd,in_file,expected", [
    (3, 9, "partially read, pages 1-3 of 9"),
    (0, 9, "partially read, no usable OCR text from its 9 pages"),
])
def test_partial_ocr_ledger_and_panel_say_partially_read(
    fresh_db, ocrd, in_file, expected,
):
    """A truncated stamp lands PARTIAL_OCR with the pages columns, and the
    panel phrasing carries exactly the pages that were read."""
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(pid, "scan.pdf", file_path="scan.pdf", size=8)
    _stamp_index_ledger(
        doc["id"], "scan.pdf", 1,
        stamp_as_indexed=True,
        ocr_truncated=True,
        pages_in_file=in_file,
        pages_ocrd=ocrd,
        ocr_unusable_pages=0,
    )
    row = projects_mod.get_document(doc["id"])
    assert row["ingest_status"] == PARTIAL_OCR
    assert row["ingest_status"] not in INDEXED_STATUSES
    assert row["ingest_status"] not in NO_CHUNK_STATUSES
    assert row["ocr_truncated"] is True
    assert row["pages_in_file"] == in_file
    assert row["pages_ocrd"] == ocrd
    note = panel_status_for(row)
    assert note is not None
    assert note["label"] == "Partially read"
    assert note["detail"] == expected


@pytest.mark.parametrize("ocrd,in_file,expected", [
    (3, 9, "partially read, pages 1-3 of 9"),
    (0, 9, "partially read, no usable OCR text from its 9 pages"),
])
def test_partial_read_note_says_the_pages_that_were_read(ocrd, in_file, expected):
    assert partial_read_note(ocrd, in_file) == expected


@pytest.mark.parametrize("chunk_count", [2, 5])
def test_non_truncated_reindex_still_stamps_indexed(fresh_db, chunk_count):
    """The PARTIAL_OCR override must not leak onto complete documents."""
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(pid, "fine.pdf", file_path="fine.pdf", size=8)
    _stamp_index_ledger(
        doc["id"], "fine.pdf", chunk_count,
        stamp_as_indexed=True,
        ocr_truncated=False,
        pages_in_file=4,
        pages_ocrd=0,
    )
    row = projects_mod.get_document(doc["id"])
    assert row["ingest_status"] == "INDEXED"
    assert row["ocr_truncated"] is False


# --- chat-side note ----------------------------------------------------------


@pytest.mark.parametrize("ocrd,in_file,expected", [
    (3, 9, "partially read, pages 1-3 of 9"),
    (0, 9, "partially read, no usable OCR text from its 9 pages"),
])
def test_fetch_document_content_carries_the_partial_read_note(
    monkeypatch, ocrd, in_file, expected,
):
    """The answer path reads the ledger and says the document was partially
    read, never presents the OCR'd prefix as the whole file."""
    doc = {
        "id": "doc-1",
        "original_name": "scan.pdf",
        "file_path": "scan.pdf",
        "ingest_status": PARTIAL_OCR,
        "pages_ocrd": ocrd,
        "pages_in_file": in_file,
    }
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: [doc])
    import app.core.rag.retriever as _retriever
    monkeypatch.setattr(_retriever, "available", lambda: False)
    monkeypatch.setattr(
        doc_index, "extract_document_text",
        lambda path, name: "OCR'd prefix of the scanned document",
    )

    content, fetched, err = _fetch_document_content(
        "demo_proj", "doc-1", "scan.pdf",
    )
    assert err is None
    assert fetched is doc
    assert content["partial_read"] == expected

    # With no extractable text the error itself carries the honest note.
    monkeypatch.setattr(doc_index, "extract_document_text", lambda path, name: "")
    content, fetched, err = _fetch_document_content(
        "demo_proj", "doc-1", "scan.pdf",
    )
    assert content is None
    assert err == expected


@pytest.mark.parametrize("doc,expected_note", [
    ({"ingest_status": PARTIAL_OCR, "pages_ocrd": 3, "pages_in_file": 9},
     "partially read, pages 1-3 of 9"),
    ({"ingest_status": "INDEXED", "pages_ocrd": None, "pages_in_file": None},
     None),
])
def test_partial_read_note_for_doc_reads_the_ledger(doc, expected_note):
    assert _partial_read_note_for_doc(doc) == expected_note


@pytest.mark.parametrize("call,expected_field,expected_value", [
    (SimpleNamespace(
        args={"document_id": "doc-1"}, name="fetch_document",
        project_id="demo_proj",
    ), "partial_read", "partially read, pages 1-3 of 9"),
    (SimpleNamespace(
        args={"document_id": "doc-1"}, name="fetch_document",
        project_id=None,
    ), "error", "no project in scope"),
])
def test_fetch_document_tool_passes_the_partial_read_note(
    monkeypatch, call, expected_field, expected_value,
):
    """The tool surface carries the ledger note; a missing project is the
    named error (kills the tool-path mutants)."""
    import app.agents.runtime as rt

    doc = {"id": "doc-1", "original_name": "scan.pdf", "doc_type": "pdf"}
    content = {
        "text": "prefix text",
        "truncated": False,
        "source": "chunks",
        "partial_read": "partially read, pages 1-3 of 9",
    }
    async def _fake_off_loop(fn, *a, **k):
        return fn(*a, **k)

    monkeypatch.setattr(rt, "_off_loop", _fake_off_loop)
    monkeypatch.setattr(
        rt, "_fetch_document_content",
        lambda pid, doc_id, filename: (content, doc, None),
    )
    out = asyncio.run(handle_fetch_document(call))
    if call.project_id:
        assert out["ok"] is True
        assert out["result"][expected_field] == expected_value
    else:
        assert out["ok"] is False
        assert out["result"][expected_field] == expected_value
