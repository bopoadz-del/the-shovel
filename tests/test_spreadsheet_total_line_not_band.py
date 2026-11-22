"""A spreadsheet total line is not the band of the rows that follow it.

``_spreadsheet_row_chunks`` turns a row with no measured item into inherited
context (the band) for the measured rows below it. Before this, a row that
carries an amount but no item description (a carried-forward, brought-forward
or to-collection line at a page break) was taken as that band. It replaced the
bill heading with its caption and put the running total into every following
row's locator: ``Bill: Brought forward | 1800``.

Every workbook here is built by the test; the wording and figures are invented.
Each decision runs on two inputs.
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


@pytest.mark.parametrize("bill, carried, brought, running", [
    ("BILL NO. 5 - DRAINAGE", "Carried forward", "Brought forward", 1800),
    ("Section 3: Roadworks", "To next page", "From previous page", 7350),
])
def test_a_total_line_keeps_the_bill_heading(tmp_path, bill, carried, brought, running):
    chunks = _chunks(tmp_path, [
        [bill], _HEADER,
        ["1", "Excavate by hand", 10, "m3", running / 10, running],
        [carried, "", "", "", "", running],
        [brought, "", "", "", "", running],
        ["2", "Backfill with selected material", 6, "m3", 30, 180],
    ])
    assert [_bill_of(c) for c in chunks] == [bill, bill]


@pytest.mark.parametrize("caption, column, figure", [
    ("Total carried to collection", 5, 98765),
    ("Subtotal", 2, 4321),
])
def test_a_total_line_is_not_a_chunk_and_leaves_no_figure_in_a_locator(
    tmp_path, caption, column, figure,
):
    """The total line is neither emitted as a row nor carried in a band, whether
    its figure sits in the amount or the quantity column."""
    total = ["", "", "", "", "", ""]
    total[0] = caption
    total[column] = figure
    chunks = _chunks(tmp_path, [
        ["Bill No 2 - Pipework"], _HEADER,
        ["1", "Pipe 150 dia", 40, "m", 25, 1000],
        total,
        ["2", "Pipe 225 dia", 12, "m", 40, 480],
    ])
    assert len(chunks) == 2, chunks
    assert not any(caption in c for c in chunks)
    assert not any(str(figure) in _bill_of(c) for c in chunks)
