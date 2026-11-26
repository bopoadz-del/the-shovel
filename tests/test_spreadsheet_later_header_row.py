"""A later column-header row in a sheet is structure, and it re-heads the rows under it.

A bill sheet re-prints its column header for each bill. The row chunker skipped
a verbatim reprint only. A reprint in other case or with a unit suffix
(``RATE (AED)``) was emitted as a measured row (``Item: DESCRIPTION; Quantity:
QTY ...``), and a later table with its columns in another order was read
through the first header's columns, so its figures landed under the wrong
labels. The header test that finds the first header now reads every later
label row too.

Every workbook is built inside the test, and every decision has two inputs.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core import doc_index

_FIRST_HEADER = ["Item", "Description", "Qty", "Unit", "Rate", "Amount"]


def _chunks(tmp_path, rows: list[list]) -> list[str]:
    wb = Workbook()
    ws = wb.active
    ws.title = "Bills"
    for row in rows:
        ws.append(list(row))
    path = tmp_path / "bill.xlsx"
    wb.save(str(path))
    return doc_index._spreadsheet_row_chunks(str(path), "bill.xlsx", ".xlsx")


@pytest.mark.parametrize(
    "reheader",
    [
        ["ITEM", "DESCRIPTION", "QTY", "UNIT", "RATE (AED)", "AMOUNT (AED)"],
        ["Ref", "Description of Work", "Quantity", "Unit", "Unit Price", "Total"],
    ],
)
def test_reworded_header_reprint_is_not_a_measured_row(tmp_path, reheader):
    chunks = _chunks(tmp_path, [
        ["BILL 1 - EARTHWORKS"],
        _FIRST_HEADER,
        ["1.1", "Topsoil strip", 500, "m2", 3, 1500],
        ["BILL 2 - PAVING"],
        reheader,
        ["2.1", "Block paving", 80, "m2", 45, 3600],
    ])
    assert len(chunks) == 2, chunks
    # No header cell is printed as a field value.
    values = {f.split(": ", 1)[1] for c in chunks for f in c.split("; ")}
    assert not values & set(reheader), chunks
    paving = chunks[1]
    assert "Item: Block paving" in paving
    assert "Quantity: 80" in paving and "Amount: 3600" in paving


@pytest.mark.parametrize(
    "header,row,expect",
    [
        (
            ["Description", "Unit", "Qty", "Amount", "Rate"],
            ["Kerb laying", "m", 40, 480, 12],
            ["Item: Kerb laying", "Quantity: 40", "Unit: m", "Rate: 12", "Amount: 480"],
        ),
        (
            ["Ref", "Item Description", "Rate (USD)", "Quantity", "Unit", "Amount (USD)"],
            ["7.2", "Gully pot", 150, 6, "nr", 900],
            ["Item: Gully pot", "Quantity: 6", "Unit: nr", "Rate: 150", "Amount: 900"],
        ),
    ],
)
def test_rows_under_a_reordered_header_read_through_its_columns(
    tmp_path, header, row, expect,
):
    chunks = _chunks(tmp_path, [
        ["BILL 1 - EARTHWORKS"],
        _FIRST_HEADER,
        ["1.1", "Topsoil strip", 500, "m2", 3, 1500],
        ["BILL 2 - DRAINAGE"],
        header,
        row,
    ])
    assert len(chunks) == 2, chunks
    for part in expect:
        assert part in chunks[1], chunks[1]
    # The first table is still read through its own header.
    assert "Item: Topsoil strip" in chunks[0] and "Quantity: 500" in chunks[0]


@pytest.mark.parametrize(
    "row,desc,qty",
    [
        (["A.4", "Item description plate", 4, "No", 12, 48], "Item description plate", 4),
        (["B.2", "Name plate", 2, "Number", 5, 10], "Name plate", 2),
    ],
    ids=["description-and-no", "name-and-number"],
)
def test_measured_row_whose_words_read_as_column_names_stays_measured(
    tmp_path, row, desc, qty,
):
    # Its cells resolve as column names, but it states figures: a header row
    # never does, so it is a measured row and the first header still holds.
    chunks = _chunks(tmp_path, [
        ["BILL 1 - SIGNAGE"],
        _FIRST_HEADER,
        row,
        ["A.5", "Post footing", 3, "nr", 20, 60],
    ])
    assert len(chunks) == 2, chunks
    assert f"Item: {desc}" in chunks[0] and f"Quantity: {qty}" in chunks[0]
    assert "Item: Post footing" in chunks[1] and "Quantity: 3" in chunks[1]
