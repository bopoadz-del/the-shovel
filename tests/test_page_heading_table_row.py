"""A table row is not the heading a PDF page is filed under.

PDF pages carry their tables as `` | ``-separated rows after the text
layer. A header row in capitals or a numbered item row looked heading-like,
so a page with no heading of its own was filed under a row of its table.
Every input is built here; nothing is read from a corpus.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import heading_line_for_page, page_heading_lines

PROSE = (
    "The contractor shall keep the works clean and remove debris daily "
    "from the site and the access roads used by delivery vehicles. "
) * 2


@pytest.mark.parametrize("row", [
    "ITEM | DESCRIPTION | UNIT | QTY",
    "NO. | ACTIVITY | START | FINISH",
])
def test_capital_header_row_is_not_a_heading(row):
    assert heading_line_for_page(f"{PROSE}\n{row}\n7 | Formwork | m2 | 90") == ""


@pytest.mark.parametrize("row", [
    "1 | Excavation in soft ground | m3 | 450",
    "B | Earthworks | sum | 1",
])
def test_numbered_item_row_is_not_a_heading(row):
    assert heading_line_for_page(f"{PROSE}\n{row}") == ""


@pytest.mark.parametrize("heading", ["4.1 Concrete Works", "## Earthworks"])
def test_heading_above_a_table_still_names_the_page(heading):
    page = f"{PROSE}\nITEM | DESCRIPTION | UNIT\n{heading}\n{PROSE}"
    assert page_heading_lines(page, [[3, 0]]) == {3: heading}


@pytest.mark.parametrize("heading", ["SCOPE; MATERIALS", "2.3 Testing; Acceptance"])
def test_semicolon_heading_is_still_a_heading(heading):
    # Only the pipe marks a table cell: a ';' inside a heading is not a row.
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{PROSE}") == heading


@pytest.mark.parametrize("line", ["SECTION 5 |", "| GENERAL NOTES"])
def test_a_single_cell_is_not_a_table_row(line):
    assert heading_line_for_page(f"{PROSE}\n{line}") == line
