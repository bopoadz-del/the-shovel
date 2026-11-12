"""Spreadsheet ingestion must emit one retrievable chunk per measured row.

A bill of quantities (or any measured spreadsheet) keeps its figures in rows
whose context — the sheet they live on and the bill/section heading above them
— the raw-text extractor drops. The prose chunker then splits a bill into
fragments where the item description, its quantity/rate and the bill number
that scopes them land in different chunks, so a "quantity of X in bill Y" or
"unit rate of X" ask misses the row that states the figure.

``_spreadsheet_row_chunks`` is the general fix: each data row becomes one
chunk carrying the sheet name, the most recent bill/section heading inherited
from above, and the row's own cells labelled with their column names. These
tests parametrise over two distinct synthetic layouts each (no live names) to
pin the mechanism.
"""
from __future__ import annotations

import pytest
from openpyxl import Workbook

from app.core import doc_index


def _write_xlsx(path: str, sheets: list[tuple[str, list[list]]]) -> str:
    wb = Workbook()
    first = True
    for name, rows in sheets:
        if first:
            ws = wb.active
            ws.title = name
            first = False
        else:
            ws = wb.create_sheet(name)
        for row in rows:
            ws.append(list(row))
    wb.save(path)
    return path


@pytest.mark.parametrize(
    "second_bill",
    [
        # (heading, item description, qty, unit, rate, amount)
        ("BILL NO.2 - SUBSTRUCTURE", "Reinforced concrete in raft", 950, "m3", 380, 361000),
        ("BILL NO.2 - DRAINAGE", "Reinforced concrete in culvert", 620, "m3", 410, 254200),
    ],
)
def test_single_sheet_multiple_bills_inherit_heading(tmp_path, second_bill):
    """Layout 1: one sheet, two bills, the second re-prints its header row.

    Every measured row must carry the sheet name and the bill heading that
    scopes it — not the earlier bill's heading and not the repeated header.
    """
    heading, item, qty, unit, rate, amount = second_bill
    path = _write_xlsx(str(tmp_path / "priced_bill.xlsx"), [(
        "Bill 1 - Site Works",
        [
            ["BILL NO.1 - SITE WORKS"],
            ["Item", "Description", "Qty", "Unit", "Rate", "Amount"],
            ["A.1", "Excavation to reduced level", 1200, "m3", 25, 30000],
            ["A.2", "Concrete blinding 75mm", 300, "m2", 45, 13500],
            [heading],
            ["Item", "Description", "Qty", "Unit", "Rate", "Amount"],
            ["B.1", item, qty, unit, rate, amount],
        ],
    )])

    chunks = doc_index._spreadsheet_row_chunks(path, "priced_bill.xlsx", ".xlsx")

    assert len(chunks) == 3, chunks

    excavation = next(c for c in chunks if "Excavation" in c)
    assert "Sheet: Bill 1 - Site Works" in excavation
    assert "Bill: BILL NO.1 - SITE WORKS" in excavation
    assert "Quantity: 1200" in excavation
    assert "Rate: 25" in excavation

    second = next(c for c in chunks if item in c)
    # The second bill's heading is inherited, not the first bill's.
    assert f"Bill: {heading}" in second
    assert "BILL NO.1" not in second
    assert f"Quantity: {qty}" in second
    assert f"Rate: {rate}" in second

    # No chunk re-prints the repeated header row as if it were an item.
    assert not any("Item: Description" in c for c in chunks)


@pytest.mark.parametrize(
    "second_sheet",
    [
        # (bill heading, item description, qty, rate)
        ("BILL NO.5 - FINISHES", "Wall tiling", 1200, 55),
        ("BILL NO.6 - PAINTING", "Emulsion to walls", 540, 12),
    ],
)
def test_title_block_and_multiple_sheets_inherit_heading(tmp_path, second_sheet):
    """Layout 2: a title-block banner above the header, and a second sheet.

    The banner rows above the header are skipped; the bill heading still
    scopes the rows beneath it, and each sheet keeps its own name.
    """
    sheet2_bill, sheet2_item, sheet2_qty, sheet2_rate = second_sheet
    path = _write_xlsx(str(tmp_path / "schedule_of_works.xlsx"), [
        (
            "Sheet1",
            [
                ["PROJECT: EXAMPLE ISLANDS"],
                ["For M/s EMPLOYER"],
                [],
                ["BILL NO.4 - APARTMENTS"],
                ["Item", "Description", "Qty", "Unit", "Rate", "Amount"],
                ["A", "Shear wall", 3533, "m3", 380, 1342540],
            ],
        ),
        (
            "Sheet2",
            [
                [sheet2_bill],
                ["Item", "Description", "Qty", "Unit", "Rate", "Amount"],
                ["F1", sheet2_item, sheet2_qty, "m2", sheet2_rate, 66000],
            ],
        ),
    ])

    chunks = doc_index._spreadsheet_row_chunks(
        path, "schedule_of_works.xlsx", ".xlsx"
    )

    assert len(chunks) == 2, chunks

    shear = next(c for c in chunks if "Shear wall" in c)
    assert "Sheet: Sheet1" in shear
    assert "Bill: BILL NO.4 - APARTMENTS" in shear
    assert "Quantity: 3533" in shear

    second = next(c for c in chunks if sheet2_item in c)
    assert "Sheet: Sheet2" in second
    assert f"Bill: {sheet2_bill}" in second
    assert f"Quantity: {sheet2_qty}" in second
    assert f"Rate: {sheet2_rate}" in second


def test_wired_into_structured_chunks(tmp_path):
    """The row chunks are part of the structured (table) stage, not the prose
    text path, so a spreadsheet indexes both the raw text and these chunks."""
    path = _write_xlsx(str(tmp_path / "any_spreadsheet.xlsx"), [(
        "Rates",
        [
            ["Description", "Quantity", "Unit", "Rate", "Amount"],
            ["Supply and lay 100mm ductile iron pipe", 240, "lm", 180, 43200],
        ],
    )])

    structured = doc_index._structured_chunks(
        path, "any_spreadsheet.xlsx", ".xlsx", "project_id_ignored",
    )

    assert any(
        "Item: Supply and lay 100mm ductile iron pipe" in c
        and "Rate: 180" in c
        for c in structured
    ), structured


@pytest.mark.parametrize(
    "payload",
    [
        b"this is not a zip file",
        b"<html>not a spreadsheet</html>",
    ],
)
def test_parse_failure_yields_empty_not_exception(tmp_path, payload):
    """A non-spreadsheet payload (or a broken one) degrades to [], never raises."""
    not_a_sheet = tmp_path / "not_really.xlsx"
    not_a_sheet.write_bytes(payload)

    assert doc_index._spreadsheet_row_chunks(
        str(not_a_sheet), "not_really.xlsx", ".xlsx"
    ) == []


@pytest.mark.parametrize("ext", [".pdf", ".png", ".txt"])
def test_non_spreadsheet_extensions_are_ignored(ext):
    filename = f"x{ext}"
    assert doc_index._spreadsheet_row_chunks(filename, filename, ext) == []


@pytest.mark.parametrize(
    "csv_lines, first_item, first_qty, first_rate",
    [
        (
            (
                "Supply ductile iron pipe,240,lm,180,43200\n"
                "Manhole type M1,12,no,1500,18000\n"
            ),
            "ductile iron pipe", 240, 180,
        ),
        (
            (
                "Handrail,88,lm,95,8360\n"
                "Kerb edging,300,lm,22,6600\n"
            ),
            "Handrail", 88, 95,
        ),
    ],
)
def test_csv_rows_are_chunked_without_sheet_name(
    tmp_path, csv_lines, first_item, first_qty, first_rate
):
    """CSV has no sheet name; the row chunks still label each measured column."""
    path = tmp_path / "rates.csv"
    path.write_text(
        "Description,Quantity,Unit,Rate,Amount\n" + csv_lines,
        encoding="utf-8",
    )

    chunks = doc_index._spreadsheet_row_chunks(str(path), "rates.csv", ".csv")

    assert len(chunks) == 2, chunks
    first = next(c for c in chunks if first_item in c)
    assert "Sheet:" not in first
    assert f"Quantity: {first_qty}" in first
    assert f"Rate: {first_rate}" in first
