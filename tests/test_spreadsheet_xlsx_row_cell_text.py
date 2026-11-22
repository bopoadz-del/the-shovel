"""An .xlsx row chunk states each cell at the precision the sheet holds.

Excel computes and displays a number to 15 significant digits, but it writes
a cached formula result to the package as the nearest binary double, so
0.1 + 0.2 is stored as ``0.30000000000000004``. The row chunk reader printed
``str(value)``, which put that tail into the chunk as a figure the sheet never
shows. A datetime with no time of day printed ``2025-01-21 00:00:00``, where
the .xls row reader prints the date alone.

openpyxl's writer rounds floats when it saves, so each test builds a workbook
with a placeholder figure and then writes the stored value into the sheet XML
itself, the way Excel stores it. Two invented inputs per decision.
"""
from __future__ import annotations

import datetime
import zipfile

import openpyxl
import pytest

from app.core import doc_index

_PLACEHOLDER = 7777.25


def _book(tmp_path, stored: str | None = None, when=None) -> str:
    src = tmp_path / "src.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Bill 4"
    ws.append(["Description", "Unit", "Qty", "Rate", "Amount", "Date"])
    ws.append(["Gully grating", "nr", 6, 0.5, _PLACEHOLDER, when])
    wb.save(src)
    if stored is None:
        return str(src)
    dst = tmp_path / "book.xlsx"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.startswith("xl/worksheets/"):
                text = data.decode("utf-8")
                assert f"<v>{_PLACEHOLDER}</v>" in text
                data = text.replace(
                    f"<v>{_PLACEHOLDER}</v>", f"<v>{stored}</v>"
                ).encode("utf-8")
            zout.writestr(item, data)
    return str(dst)


def _amount(path: str) -> str:
    chunks = doc_index._spreadsheet_row_chunks(path, "book.xlsx", ".xlsx")
    assert len(chunks) == 1, chunks
    return chunks[0].split("Amount: ", 1)[1].split(";", 1)[0]


@pytest.mark.parametrize(
    "stored, stated",
    [("0.30000000000000004", "0.3"), ("1234.5600000000002", "1234.56")],
)
def test_binary_tail_of_a_cached_float_is_not_a_figure(tmp_path, stored, stated):
    assert _amount(_book(tmp_path, stored)) == stated


@pytest.mark.parametrize(
    "stored, stated",
    [("123456.789012345", "123456.789012345"), ("0.000123456", "0.000123456")],
)
def test_a_figure_within_fifteen_digits_is_kept_whole(tmp_path, stored, stated):
    assert _amount(_book(tmp_path, stored)) == stated


@pytest.mark.parametrize(
    "stored, stated",
    [("494.00000000000006", "494"), ("1.9999999999999998", "2")],
)
def test_a_whole_number_prints_without_a_decimal_tail(tmp_path, stored, stated):
    assert _amount(_book(tmp_path, stored)) == stated


@pytest.mark.parametrize(
    "when, stated",
    [
        (datetime.datetime.fromisoformat("2025-01-21"), "2025-01-21"),
        (datetime.datetime.fromisoformat("2024-12-03"), "2024-12-03"),
        (datetime.datetime.fromisoformat("2025-01-21T14:30"), "2025-01-21 14:30:00"),
        (datetime.datetime.fromisoformat("2024-12-03T08:05"), "2024-12-03 08:05:00"),
    ],
)
def test_a_date_cell_is_its_date_unless_it_states_a_time(tmp_path, when, stated):
    sheets = doc_index._spreadsheet_sheets(_book(tmp_path, when=when), ".xlsx")
    assert sheets[0][1][1][5] == stated
