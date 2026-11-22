"""A spreadsheet row chunk keeps the bill that scopes it under an item preamble.

``_spreadsheet_row_chunks`` turns a description-only row into inherited
context for the measured rows below it. Before this, every such row REPLACED
the inherited bill heading, so a row under an item preamble read
``Bill: <preamble>`` and lost its bill: a "quantity of X in bill N" ask could
not reach it. Now the band is a path: the bill heading, then the latest
description-only label under it.

Every workbook here is built by the test; the wording is invented. Each
decision runs on two inputs.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core import doc_index

_HEADER = ["Item", "Description", "Qty", "Unit", "Rate", "Amount"]


def _chunks(tmp_path, rows: list[list]) -> list[str]:
    wb = Workbook()
    ws = wb.active
    ws.title = "Priced"
    for row in rows:
        ws.append(list(row))
    path = str(tmp_path / "priced.xlsx")
    wb.save(path)
    return doc_index._spreadsheet_row_chunks(path, "priced.xlsx", ".xlsx")


def _bill_of(chunk: str) -> str:
    return chunk.split("; Bill: ", 1)[1].split("; Item: ", 1)[0]


@pytest.mark.parametrize("bill, preamble, item", [
    ("BILL NO. 5 - DRAINAGE",
     "Precast concrete chambers complete with cover slab",
     "Two coats bituminous coating to external faces"),
    ("Bill No 12: External Works",
     "Kerbs; precast concrete, bedded and backed in concrete",
     "Straight, 125 x 255 mm"),
])
def test_item_preamble_is_filed_under_the_bill_heading(tmp_path, bill, preamble, item):
    chunks = _chunks(tmp_path, [
        [bill], _HEADER,
        ["1", "Excavate by hand", 10, "m3", 20, 200],
        ["", preamble, "", "", "", ""],
        ["2", item, 40, "m", 15, 600],
    ])
    assert len(chunks) == 2, chunks
    first, second = chunks
    assert _bill_of(first) == bill
    assert _bill_of(second) == f"{bill} > {preamble}"


@pytest.mark.parametrize("first_bill, second_bill", [
    ("BILL NO. 3 - SUBSTRUCTURE", "BILL NO. 4 - FRAME"),
    ("Bill No 8 - Roads", "Bill Number 9 - Lighting"),
])
def test_a_bill_heading_label_opens_a_new_bill(tmp_path, first_bill, second_bill):
    """A label in the bills' heading grammar replaces the path, preamble and all."""
    chunks = _chunks(tmp_path, [
        [first_bill], _HEADER,
        ["", "Concrete grade C40 in", "", "", "", ""],
        ["1", "Columns", 12, "m3", 400, 4800],
        [second_bill],
        ["1", "Beams", 9, "m3", 420, 3780],
    ])
    assert [_bill_of(c) for c in chunks] == [
        f"{first_bill} > Concrete grade C40 in", second_bill,
    ]


@pytest.mark.parametrize("first_bill, second_bill", [
    ("SITE CLEARANCE", "EARTHWORKS"),
    ("Section A - Preliminaries", "Section B - Demolition"),
])
def test_the_label_above_a_repeated_header_opens_a_new_bill(tmp_path, first_bill, second_bill):
    """A later bill re-prints the header under its own heading: that label is the bill,
    whatever its wording, not a preamble under the previous bill."""
    chunks = _chunks(tmp_path, [
        [first_bill], _HEADER,
        ["1", "Clear vegetation", 500, "m2", 2, 1000],
        [second_bill], _HEADER,
        ["1", "Excavate topsoil", 150, "m3", 9, 1350],
    ])
    assert [_bill_of(c) for c in chunks] == [first_bill, second_bill]


@pytest.mark.parametrize("bill", ["BILL NO. 2 - PIPEWORK", "Bill No 6 - Ducts"])
def test_a_repeated_header_after_a_measured_row_keeps_the_bill(tmp_path, bill):
    """A header re-printed with no label directly above it (a page break) opens
    nothing: the preamble before the measured row is not promoted to a bill."""
    chunks = _chunks(tmp_path, [
        [bill], _HEADER,
        ["", "Pipes in trenches, depth not exceeding 2 m", "", "", "", ""],
        ["1", "150 mm diameter", 80, "m", 30, 2400],
        _HEADER,
        ["2", "225 mm diameter", 60, "m", 45, 2700],
    ])
    path = f"{bill} > Pipes in trenches, depth not exceeding 2 m"
    assert [_bill_of(c) for c in chunks] == [path, path]


@pytest.mark.parametrize("first, second", [
    ("Pipes in trenches, depth not exceeding 2 m", "Pipes in trenches, depth 2 to 4 m"),
    ("Blockwork, 100 mm thick", "Blockwork, 200 mm thick"),
])
def test_a_later_preamble_replaces_the_earlier_one(tmp_path, first, second):
    """Only the latest preamble scopes a row; preambles never pile up."""
    bill = "BILL NO. 1 - GENERAL"
    chunks = _chunks(tmp_path, [
        [bill], _HEADER,
        ["", first, "", "", "", ""],
        ["1", "Item one", 5, "m", 10, 50],
        ["", second, "", "", "", ""],
        ["2", "Item two", 6, "m", 11, 66],
    ])
    assert [_bill_of(c) for c in chunks] == [f"{bill} > {first}", f"{bill} > {second}"]


@pytest.mark.parametrize("label, item", [
    ("Substructure works", "Pile caps"),
    ("Mechanical installation", "Chilled water pipework"),
])
def test_the_first_label_on_a_sheet_with_no_heading_is_the_bill(tmp_path, label, item):
    chunks = _chunks(tmp_path, [
        _HEADER,
        ["", label, "", "", "", ""],
        ["1", item, 3, "nr", 100, 300],
    ])
    assert [_bill_of(c) for c in chunks] == [label]
