"""Table-stage chunks carry the page they were read from (P-G5).

The drawing stage reads a sheet page by page, so the page of every table
chunk and of the title-block chunk is known when it is emitted. ``chunk_pages``
used to pad every structured chunk with None, so a drawing volume indexed
with a page on its text chunks only. Each test builds its own PDF.
"""

from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the drawing reader")

from app.core import doc_index

DRAWING = "set-DWG-sheets.pdf"


def _pdf(path, n_pages: int, table_pages: set[int], *, title: bool = True) -> str:
    """``n_pages`` pages of prose; a ruled schedule on each 1-based page in
    ``table_pages``; a title block on page 1 when ``title``."""
    doc = fitz.open()
    for page_no in range(1, n_pages + 1):
        page = doc.new_page(width=842, height=595)
        if page_no == 1 and title:
            page.insert_text(fitz.Point(500, 400), "DRAWING NO: S-100", fontsize=8)
            page.insert_text(fitz.Point(500, 414), "REV: C2", fontsize=8)
        page.insert_text(
            fitz.Point(40, 60), f"General arrangement notes for sheet {page_no}.",
            fontsize=9,
        )
        if page_no in table_pages:
            xs, ys = [40, 170, 300, 430], [140, 175, 210]
            for y in ys:
                page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
            for x in xs:
                page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]))
            cells = [["BAR MARK", "DIAMETER", "QTY"], [f"B{page_no}", "16", "48"]]
            for r, row in enumerate(cells):
                for c, value in enumerate(row):
                    page.insert_text(fitz.Point(xs[c] + 6, ys[r] + 22), value, fontsize=8)
    doc.save(str(path))
    doc.close()
    return str(path)


def _table_pages(chunks: list[str], pages: list[int | None]) -> list[int | None]:
    return [p for c, p in zip(chunks, pages) if c.startswith("DRAWING TABLE")]


@pytest.mark.parametrize(
    "table_pages, expected",
    [({3, 7}, [3, 7]), ({2, 5}, [2, 5])],
)
def test_drawing_stage_reports_the_page_each_table_was_read_from(
    tmp_path, monkeypatch, table_pages, expected,
):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _pdf(tmp_path / DRAWING, 8, table_pages)
    pages: list[int | None] = []
    chunks = doc_index._drawing_chunks_for_document(
        path, DRAWING, ".pdf", "p1", None, pages,
    )
    assert len(pages) == len(chunks)
    assert _table_pages(chunks, pages) == expected
    # each table chunk holds the values printed on the page it is credited to
    for chunk, page in zip(chunks, pages):
        if chunk.startswith("DRAWING TABLE"):
            assert f"B{page}" in chunk
    titles = [p for c, p in zip(chunks, pages) if c.startswith("DRAWING TITLE BLOCK")]
    assert titles == [1]


@pytest.mark.parametrize(
    "table_pages, expected",
    [({3, 7}, [3, 7]), ({4}, [4])],
)
def test_produce_chunks_pages_structured_chunks(
    tmp_path, monkeypatch, table_pages, expected,
):
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _pdf(tmp_path / DRAWING, 8, table_pages)
    chunks, meta = doc_index._produce_chunks(
        path, DRAWING, ".pdf", "p1", "default", False,
    )
    pages = meta["chunk_pages"]
    assert len(pages) == len(chunks)
    assert _table_pages(chunks, pages) == expected
    structured = [p for c, p in zip(chunks, pages) if c.startswith("DRAWING ")]
    assert structured and None not in structured


@pytest.mark.parametrize("name", [DRAWING, "plain-report.pdf"])
def test_document_without_tables_is_unchanged(tmp_path, monkeypatch, name):
    """No table on any page: no structured chunk, and ``chunk_pages`` is the
    text stage's pages exactly as before."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _pdf(tmp_path / name, 4, set(), title=False)
    chunks, meta = doc_index._produce_chunks(path, name, ".pdf", "p1", "default", False)
    assert not any(c.startswith("DRAWING ") for c in chunks)
    text, tmeta = doc_index._extract_with_meta(path, name)
    text_chunks, text_pages = doc_index._text_chunks_with_pages(
        text, dict(tmeta), "default", name,
    )
    assert chunks == doc_index._normalize_cesmm_in_chunks(text_chunks)
    assert meta["chunk_pages"] == text_pages


@pytest.mark.parametrize(
    "table_pages, expected",
    [({3, 7}, [3, 7]), ({6}, [6])],
)
def test_whole_document_path_pages_structured_chunks(
    tmp_path, monkeypatch, table_pages, expected,
):
    """The corpus_build / batched path runs the table stages in their own
    child; the pages must survive that hop."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _pdf(tmp_path / DRAWING, 8, table_pages)
    chunks, meta = doc_index._produce_whole_document_chunks(
        path, DRAWING, ".pdf", "p1", "default", False,
        max_size_mb=0.0, ocr_size_gate=False,
    )
    pages = meta["chunk_pages"]
    assert len(pages) == len(chunks)
    assert _table_pages(chunks, pages) == expected


@pytest.mark.parametrize(
    "pages, n, expected",
    [
        ([3, 7], 2, [3, 7]),
        ([3], 2, [None, None]),
        ([], 3, [None, None, None]),
        ([4, 5, 6], 2, [None, None]),
    ],
)
def test_stage_pages_of_the_wrong_length_give_no_attribution(pages, n, expected):
    assert doc_index._aligned_pages(pages, n) == expected


@pytest.mark.parametrize(
    "stage_chunks",
    [["DRAWING TABLE x", "DRAWING TABLE y"], ["DRAWING TABLE z"]],
)
def test_stage_that_reports_no_pages_pads_with_none(tmp_path, monkeypatch, stage_chunks):
    """A drawing stage replaced by one that fills no page list (as the BOQ and
    spreadsheet stages have no page) keeps its chunks with page None."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(
        doc_index, "_drawing_chunks_for_document", lambda *a, **k: list(stage_chunks),
    )
    path = _pdf(tmp_path / DRAWING, 2, set(), title=False)
    chunks, _attrs, pages = doc_index._structured_stage(path, DRAWING, ".pdf", "p1")
    assert chunks == stage_chunks
    assert pages == [None] * len(stage_chunks)


@pytest.mark.parametrize(
    "table_pages, expected",
    [({3, 7}, [3, 7]), ({5}, [5])],
)
def test_index_project_hands_structured_pages_to_the_chunk_store(
    tmp_path, monkeypatch, table_pages, expected,
):
    """``index_project`` appends the drawing chunks itself; the pages it hands
    the chunk store must credit each table chunk to its page."""
    from app.core import projects as projects_mod
    from app.core.rag import retriever as rag_retriever

    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    path = _pdf(tmp_path / DRAWING, 8, table_pages)
    doc = {
        "id": "d1", "original_name": DRAWING, "file_path": path,
        "uploaded_at": "2026-11-14T00:00:00Z", "size": 1, "bytes_state": "present",
    }
    seen: dict = {}
    monkeypatch.setattr(projects_mod, "list_documents", lambda pid: [doc])
    monkeypatch.setattr(doc_index, "_load_index", lambda pid: None)
    monkeypatch.setattr(doc_index, "_write_index", lambda pid, data: None)
    monkeypatch.setattr(doc_index, "_record_document_attributes", lambda *a: None)
    monkeypatch.setattr(doc_index, "_stamp_index_ledger", lambda *a, **k: None)
    monkeypatch.setattr(rag_retriever, "available", lambda: True)

    def _index_chunks(project_id, document_id, chunks, *, pages=None, ocr_flags=None):
        seen["chunks"], seen["pages"] = list(chunks), pages

    monkeypatch.setattr(rag_retriever, "index_chunks", _index_chunks)
    doc_index.index_project("p")
    chunks, pages = seen["chunks"], seen["pages"]
    assert pages is not None and len(pages) == len(chunks)
    assert _table_pages(chunks, pages) == expected


@pytest.mark.parametrize(
    "stage_chunks, stage_pages, kept, kept_pages",
    [
        (
            ["DRAWING TABLE a", "0 | 0 | 0", "DRAWING TABLE c"],
            [2, 4, 9],
            ["DRAWING TABLE a", "DRAWING TABLE c"],
            [2, 9],
        ),
        (
            ["0 | 0", "DRAWING TABLE b", "DRAWING TABLE c"],
            [1, 6, 8],
            ["DRAWING TABLE b", "DRAWING TABLE c"],
            [6, 8],
        ),
    ],
)
def test_a_refused_noise_chunk_takes_its_own_page_with_it(
    tmp_path, monkeypatch, stage_chunks, stage_pages, kept, kept_pages,
):
    """The noise refusal (RQ-1) drops a chunk; the pages after it never shift."""
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)

    def _stage(file_path, filename, ext, project_id, attrs=None, pages=None):
        if pages is not None:
            pages.extend(stage_pages)
        return list(stage_chunks)

    monkeypatch.setattr(doc_index, "_drawing_chunks_for_document", _stage)
    path = _pdf(tmp_path / DRAWING, 2, set(), title=False)
    chunks, _attrs, pages = doc_index._structured_stage(path, DRAWING, ".pdf", "p1")
    assert chunks == kept
    assert pages == kept_pages
