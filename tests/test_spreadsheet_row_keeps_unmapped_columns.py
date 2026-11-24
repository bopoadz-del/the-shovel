"""A spreadsheet row chunk keeps every headed cell the row states.

The BOQ column resolver names only the description and figure columns. A
sheet also prints an item reference, a bill column or a remark beside them;
before this, the row chunk dropped those cells, so an ask that names the
row by the reference the document gives it could not reach it. Each extra
cell is now carried under the sheet's own header text. Every workbook is
built inside the test; two invented inputs per decision.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core import doc_index


def _chunks(tmp_path, rows: list[list]) -> list[str]:
    wb = Workbook()
    ws = wb.active
    ws.title = "Priced"
    for row in rows:
        ws.append(list(row))
    path = str(tmp_path / "sheet.xlsx")
    wb.save(path)
    return doc_index._spreadsheet_row_chunks(path, "sheet.xlsx", ".xlsx")


@pytest.mark.parametrize(
    "header, row, carried",
    [
        (
            ["Item", "Description", "Qty", "Unit", "Rate", "Amount", "Remarks"],
            ["3.1.4", "Kerb laying on bed", 40, "m", 12, 480, "provisional"],
            ["Item: 3.1.4", "Remarks: provisional"],
        ),
        (
            ["Ref No.", "Spec Clause", "Description", "Unit", "Quantity", "Rate", "Total"],
            ["E/12", "S-207", "Gully grating", "nr", 6, 90, 540],
            ["Ref No.: E/12", "Spec Clause: S-207"],
        ),
    ],
)
def test_an_unmapped_headed_cell_is_carried_under_its_header(
    tmp_path, header, row, carried,
):
    chunks = _chunks(tmp_path, [["BILL NO. 4 - EXTERNALS"], header, row])

    assert len(chunks) == 1, chunks
    for text in carried:
        assert text in chunks[0], chunks[0]


@pytest.mark.parametrize(
    "header, row, description, qty_head",
    [
        (
            ["", "Description", "Qty", "Unit", "Rate", "Amount"],
            ["loose note", "Topsoil strip 150mm", 900, "m2", 3, 2700],
            "Topsoil strip 150mm",
            "Qty",
        ),
        (
            ["Description", "QTY", "Unit", "Rate", "Amount", ""],
            ["Geotextile membrane", 310, "m2", 7, 2170, "stray"],
            "Geotextile membrane",
            "QTY",
        ),
    ],
)
def test_headerless_and_fixed_label_columns_are_not_carried_again(
    tmp_path, header, row, description, qty_head,
):
    chunks = _chunks(tmp_path, [["BILL NO. 9 - LANDSCAPE"], header, row])

    assert len(chunks) == 1, chunks
    text = chunks[0]
    # A cell under a blank header has no name to carry it by.
    assert "loose note" not in text and "stray" not in text, text
    # Columns already printed under a fixed label are not repeated.
    assert text.count(description) == 1, text
    assert f"{qty_head}:" not in text, text
    assert "Description:" not in text, text
