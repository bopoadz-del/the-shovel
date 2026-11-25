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


@pytest.mark.parametrize("rows", [
    "1 | Excavation in soft ground | m3 | 450\n2 | Backfill | m3 | 120",
    "B | Earthworks | sum | 1\nC | Drainage | sum | 1",
])
def test_numbered_item_rows_are_not_a_heading(rows):
    assert heading_line_for_page(f"{PROSE}\n{rows}") == ""


@pytest.mark.parametrize("rows", [
    "ITEM | DESCRIPTION\nA1 | EARTHWORKS",
    "NO. | ACTIVITY\n1 | Setting out",
])
def test_two_cell_rows_are_a_table(rows):
    # The boundary: two cells is already a row.
    assert heading_line_for_page(f"{PROSE}\n{rows}") == ""


@pytest.mark.parametrize("rows", [
    "ITEM | DESCRIPTION | QTY\n\nA1 | EARTHWORKS | 1",
    "NO. | ACTIVITY\n  \n1 | Setting out",
])
def test_rows_apart_by_a_blank_line_are_one_table(rows):
    assert heading_line_for_page(f"{PROSE}\n{rows}") == ""


@pytest.mark.parametrize("row", ["| ITEM | DESCRIPTION |", "| NO. | ACTIVITY | QTY |"])
def test_a_markdown_row_opening_on_the_separator_is_a_row(row):
    assert heading_line_for_page(f"{PROSE}\n{row}\n{PROSE}") == ""


@pytest.mark.parametrize("heading", [
    "## Scope | Phase 2",
    "PART 2 | TECHNICAL SPECIFICATION",
])
def test_a_single_title_holding_the_separator_is_a_heading(heading):
    # Between prose, with no row before or after it, the line is a title.
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{PROSE}") == heading


@pytest.mark.parametrize("heading", ["## Scope | Phase 2", "# BILL 3 | SUBSTRUCTURE"])
def test_a_markdown_heading_above_a_table_is_not_a_row(heading):
    page = f"{PROSE}\n{heading}\nITEM | DESCRIPTION | QTY\nA1 | Excavation | 450"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize("heading", ["SCOPE; MATERIALS", "2.3 Testing; Acceptance"])
def test_a_semicolon_heading_above_a_table_is_not_a_row(heading):
    # Next to a row, a ';' heading still holds no cells: only the pipe splits.
    page = f"{PROSE}\n{heading}\nITEM | DESCRIPTION | QTY\nA1 | Excavation | 450"
    assert heading_line_for_page(page) == heading


def test_an_empty_page_has_no_heading():
    assert heading_line_for_page("") == ""
    assert heading_line_for_page("\n  \n") == ""


@pytest.mark.parametrize("heading", ["4.1 Concrete Works", "## Earthworks"])
def test_heading_above_a_table_still_names_the_page(heading):
    page = f"{PROSE}\nITEM | DESCRIPTION | UNIT\n1 | Blinding | m3\n{heading}\n{PROSE}"
    assert page_heading_lines(page, [[3, 0]]) == {3: heading}


@pytest.mark.parametrize("heading", ["SCOPE; MATERIALS", "2.3 Testing; Acceptance"])
def test_semicolon_heading_is_still_a_heading(heading):
    # Only the pipe marks a table cell: a ';' inside a heading is not a row.
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{PROSE}") == heading


@pytest.mark.parametrize("line", ["SECTION 5 |", "| GENERAL NOTES"])
def test_a_single_cell_is_not_a_table_row(line):
    assert heading_line_for_page(f"{PROSE}\n{line}") == line
