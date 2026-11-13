"""PDF tables keep every body row through ingest.

A table ruled only at its top, under its header and at its bottom has no
horizontal edge between body rows; pdfplumber then folds the whole body into
one row of newline-joined cells. These tests build their own PDFs and check
that every body row comes out as its own pipe row, aligned with its values,
on both sides of a page break, and that a one-row table is left as it was.
"""
from __future__ import annotations

import fitz  # PyMuPDF
import pdfplumber
import pytest

from app.core.doc_index import _pdf_tables_markdown, _unfold_table_row

_COLS = (72.0, 252.0, 372.0, 492.0)  # column edges, x
_ROW_H = 18.0


def _draw_open_table(page, header, body, top=100.0):
    """Draw a table with vertical rules and only three horizontal rules."""
    bottom = top + _ROW_H * (1 + len(body))
    for y in (top, top + _ROW_H, bottom):
        page.draw_line((_COLS[0], y), (_COLS[-1], y))
    for x in _COLS:
        page.draw_line((x, top), (x, bottom))
    for r, row in enumerate([header, *body]):
        y = top + _ROW_H * r + 13
        for c, text in enumerate(row):
            page.insert_text((_COLS[c] + 4, y), text, fontsize=10)


def _build_pdf(path, pages):
    doc = fitz.open()
    for header, body in pages:
        page = doc.new_page(width=612, height=792)
        _draw_open_table(page, header, body)
    doc.save(str(path))
    doc.close()


def _page_tables(path):
    with pdfplumber.open(str(path)) as pdf:
        out = []
        for page in pdf.pages:
            out.append(_pdf_tables_markdown(page))
            page.close()
        return out


_HEADER = ("Activity", "Average", "Minimum")
_AREAS = [(f"Area {n}", str(37 * n), str(11 * n)) for n in range(1, 9)]


def _pipe(rows):
    return [" | ".join(r) for r in rows]


@pytest.mark.parametrize(
    ("pages", "expected"),
    [
        # Open 8-row table split 4/4 across a page break; the continuation
        # page has no header. Every row survives, aligned, on its own page.
        (
            [(_HEADER, _AREAS[:4]), (("", "", ""), _AREAS[4:])],
            [_pipe([_HEADER, *_AREAS[:4]]), _pipe(_AREAS[4:])],
        ),
        # A single-row table: header + its one row, as before.
        (
            [(("Item", "Qty", "Unit"), [("Concrete C40", "945", "m3")])],
            [["Item | Qty | Unit", "Concrete C40 | 945 | m3"]],
        ),
    ],
    ids=["open_table_across_page_break", "single_row_table"],
)
def test_pdf_table_rows_survive_extraction(tmp_path, pages, expected):
    pdf = tmp_path / "table.pdf"
    _build_pdf(pdf, pages)

    got = [md.splitlines() for md in _page_tables(pdf)]

    assert got == expected


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        # Folded body: equal line counts in every filled cell -> k rows.
        (
            ["A\nB\nC", "1\n2\n3", "x\ny\nz"],
            [["A", "1", "x"], ["B", "2", "y"], ["C", "3", "z"]],
        ),
        # Empty cell in a folded row stays empty in every unfolded row.
        (["A\nB", "", "1\n2"], [["A", "", "1"], ["B", "", "2"]]),
        # One wrapped description cell: a real multi-line cell, untouched.
        (
            ["Long wrapped\ndescription", "945", "m3"],
            [["Long wrapped\ndescription", "945", "m3"]],
        ),
        # Differing line counts: not line-aligned, untouched.
        (["A\nB\nC", "1\n2", "x"], [["A\nB\nC", "1\n2", "x"]]),
        # Only one filled cell: nothing to align against, untouched.
        (["A\nB", "", ""], [["A\nB", "", ""]]),
        # Plain single-line row.
        (["A", "1", "x"], [["A", "1", "x"]]),
    ],
)
def test_unfold_table_row_shapes(cells, expected):
    assert _unfold_table_row(cells) == expected
