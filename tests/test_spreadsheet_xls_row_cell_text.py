"""A legacy .xls row chunk states each cell as the document text does.

BIFF stores every number as a float and every date as a day serial. The row
chunk reader printed the raw value, so a quantity of 40 read "40.0" and a date
read "45678.0", while the document-text reader of the same sheet
(`text_extractors._xls_cell`) prints "40" and an ISO date. The row chunk and
the flat text then disagree on the same figure.

No BIFF writer is installed in CI, so each test builds its workbook from
xlrd's own `Cell` objects and hands it to both readers through
`xlrd.open_workbook`. Two invented inputs per decision.
"""
from __future__ import annotations

import pytest
import xlrd
from xlrd.sheet import Cell

from app.core import doc_index, text_extractors


class _Sheet:
    def __init__(self, name, rows):
        self.name = name
        self._rows = rows
        self.nrows = len(rows)
        self.ncols = max((len(r) for r in rows), default=0)

    def row(self, r):
        cells = list(self._rows[r])
        return cells + [Cell(xlrd.XL_CELL_EMPTY, "")] * (self.ncols - len(cells))

    def cell_value(self, r, c):
        return self.row(r)[c].value


class _Book:
    datemode = 0

    def __init__(self, sheets):
        self._sheets = sheets

    def sheets(self):
        return self._sheets


def _text(v):
    return Cell(xlrd.XL_CELL_TEXT, v)


def _num(v):
    return Cell(xlrd.XL_CELL_NUMBER, float(v))


def _date(serial):
    return Cell(xlrd.XL_CELL_DATE, float(serial))


def _write(monkeypatch, tmp_path, sheet_name, rows):
    book = _Book([_Sheet(sheet_name, rows)])
    monkeypatch.setattr(xlrd, "open_workbook", lambda **_kw: book)
    path = tmp_path / "legacy.xls"
    path.write_bytes(b"biff placeholder; the reader is handed the book above")
    return str(path)


def _header():
    return [_text(h) for h in ("Description", "Qty", "Unit", "Rate", "Amount")]


@pytest.mark.parametrize(
    "desc,qty,unit,rate,amount",
    [
        ("Kerb laying", 40, "m", 12, 480),
        ("Excavation to reduced level", 1200, "m3", 25, 30000),
    ],
)
def test_a_whole_number_figure_is_stated_without_a_float_tail(
    monkeypatch, tmp_path, desc, qty, unit, rate, amount,
):
    path = _write(monkeypatch, tmp_path, "Bill 2", [
        _header(),
        [_text(desc), _num(qty), _text(unit), _num(rate), _num(amount)],
    ])
    chunks = doc_index._spreadsheet_row_chunks(path, "legacy.xls", ".xls")
    assert len(chunks) == 1, chunks
    chunk = chunks[0]
    assert f"Quantity: {qty};" in chunk, chunk
    assert f"Rate: {rate};" in chunk, chunk
    assert chunk.endswith(f"Amount: {amount}"), chunk
    assert ".0;" not in chunk and not chunk.endswith(".0"), chunk


@pytest.mark.parametrize(
    "qty,rate,amount",
    [(2.5, 12.35, 30.875), (0.75, 410.2, 307.65)],
)
def test_a_fractional_figure_is_kept_exact(
    monkeypatch, tmp_path, qty, rate, amount,
):
    path = _write(monkeypatch, tmp_path, "Bill 3", [
        _header(),
        [_text("Blinding"), _num(qty), _text("m3"), _num(rate), _num(amount)],
    ])
    chunks = doc_index._spreadsheet_row_chunks(path, "legacy.xls", ".xls")
    assert len(chunks) == 1, chunks
    assert f"Quantity: {qty};" in chunks[0], chunks
    assert f"Rate: {rate};" in chunks[0], chunks
    assert chunks[0].endswith(f"Amount: {amount}"), chunks


@pytest.mark.parametrize(
    "rows",
    [
        [
            [_text("Ref"), _text("Description"), _text("Qty"), _text("Date")],
            [_text("A.1"), _text("Formwork to slab soffit"), _num(310), _date(45678)],
        ],
        [
            [_text("Description"), _text("Qty"), _text("Rate"), _text("Issued")],
            [_text("Rebar T16"), _num(18.4), _num(2950), _date(45300)],
            [_text("Rebar T20"), _num(7), _num(2950), _date(45301)],
        ],
    ],
)
def test_the_row_reader_and_the_text_reader_agree_cell_for_cell(
    monkeypatch, tmp_path, rows,
):
    path = _write(monkeypatch, tmp_path, "S1", rows)
    sheets = doc_index._spreadsheet_sheets(path, ".xls")
    flat, _meta = text_extractors._xls(path, "legacy.xls", False)
    row_reader = [" | ".join(c for c in row if c) for row in sheets[0][1]]
    assert row_reader == flat.split("\n")
    # The date cell is a date, not a day serial.
    assert all(not row.endswith(".0") for row in row_reader), row_reader
