"""The document engine's PDF parser must return the tables it declares.

`PDFDocument.tables: List[List[List[str]]]` was populated by
`_blocks_to_tables(blocks)`, whose whole body was:

    # Simplified — return empty; real table extraction handled by pdfplumber
    return []

The comment was wrong in a way that mattered. `parse()` tries
`_parse_with_pymupdf` FIRST and it succeeds for essentially every PDF, so the
pdfplumber branch that does real extraction never ran. Every specification,
BOQ and schedule that went through the document engine came out with
`tables == []`, and the comment made that look intentional.

These tests fail against the old body: it returned `[]` for every input,
including the ruled grid below.
"""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("fitz", reason="PyMuPDF is the primary PDF backend")

from app.document_engine.parsers.pdf_parser import PDFParser


@pytest.fixture
def spec_pdf(tmp_path):
    """A page with a ruled 3x3 grid — a specification's material table."""
    path = tmp_path / "spec.pdf"
    doc = fitz.open()
    page = doc.new_page(width=400, height=300)
    xs, ys = [40, 160, 270, 370], [40, 80, 120, 160]
    for y in ys:
        page.draw_line(fitz.Point(xs[0], y), fitz.Point(xs[-1], y))
    for x in xs:
        page.draw_line(fitz.Point(x, ys[0]), fitz.Point(x, ys[-1]))
    cells = [
        ["ITEM", "GRADE", "COVER"],
        ["Slab", "C40", "25"],
        ["Column", "C50", "40"],
    ]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text(fitz.Point(xs[c] + 5, ys[r] + 25), val, fontsize=9)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_a_ruled_table_reaches_the_parsed_document(spec_pdf):
    """The load-bearing one — this is what retrieval never got to see."""
    doc = PDFParser().parse(spec_pdf)

    assert doc.tables, "parsed a PDF containing a ruled table and got tables == []"
    flat = " ".join(" ".join(row) for table in doc.tables for row in table)
    assert "C40" in flat and "Slab" in flat, f"table content lost: {doc.tables}"


def test_the_shape_matches_what_the_dataclass_declares(spec_pdf):
    """`List[List[List[str]]]` — tables of rows of cell strings, the same
    shape `pdfplumber.extract_tables()` returns, so both parser branches
    agree and downstream code does not have to ask which one ran."""
    doc = PDFParser().parse(spec_pdf)

    assert isinstance(doc.tables, list)
    for table in doc.tables:
        assert isinstance(table, list) and table
        for row in table:
            assert isinstance(row, list)
            assert all(isinstance(cell, str) for cell in row), \
                f"non-string cell in {row!r} — None must be normalised to ''"


def test_text_still_comes_through_alongside_the_tables(spec_pdf):
    """Adding table extraction must not disturb the text path."""
    doc = PDFParser().parse(spec_pdf)

    assert doc.pages, "no pages extracted"
    assert "GRADE" in doc.text


def test_a_pdf_with_no_tables_reports_none_without_raising(tmp_path):
    path = tmp_path / "prose.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 72), "All workmanship shall comply with the Specification.")
    doc.save(str(path))
    doc.close()

    parsed = PDFParser().parse(str(path))
    assert parsed.tables == []
    assert "workmanship" in parsed.text


def test_the_dead_helper_is_gone():
    """`_blocks_to_tables` returned [] unconditionally and had exactly one
    caller. Leaving it in place would let a future caller reintroduce the
    silent-empty behaviour this change removed."""
    assert not hasattr(PDFParser, "_blocks_to_tables")
