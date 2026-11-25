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
    "ITEM | DESCRIPTION\nA1 | EARTHWORKS\nA2 | DRAINAGE",
    "NO. | ACTIVITY\n1 | Setting out\n2 | Clearing",
])
def test_three_rows_of_two_cells_are_a_table(rows):
    # The row boundary: two columns need three rows to be a table.
    assert heading_line_for_page(f"{PROSE}\n{rows}") == ""


@pytest.mark.parametrize("pair", [
    "PROJECT ALPHA | VOLUME 2\nSECTION 03300 | CAST-IN-PLACE CONCRETE",
    "ITEM | DESCRIPTION\nA1 | EARTHWORKS",
])
def test_two_lines_of_two_cells_are_not_taken_as_a_table(pair):
    # Two titles holding the separator look exactly like a 2 x 2 table, so
    # below both bounds the lines are read as they always were: the first
    # is the page's heading.
    assert heading_line_for_page(f"{PROSE}\n{pair}\n{PROSE}") == pair.split("\n")[0]


@pytest.mark.parametrize("rows", [
    "ITEM | DESCRIPTION | QTY\nA1 | EXCAVATION | 450",
    "NO. | ACTIVITY | DAYS\n1 | Setting out | 3",
])
def test_two_rows_of_three_cells_are_a_table(rows):
    # The column boundary: three columns make two rows a table.
    assert heading_line_for_page(f"{PROSE}\n{rows}") == ""


@pytest.mark.parametrize("rows", [
    "ITEM | DESCRIPTION | QTY\n\nA1 | EARTHWORKS | 1",
    "NO. | ACTIVITY\n  \n1 | Setting out\n\n2 | Clearing",
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


@pytest.mark.parametrize("page", ["", "\n  \n"])
def test_an_empty_page_has_no_heading(page):
    assert heading_line_for_page(page) == ""


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


# The extractor writes every cell, blank ones too (" | ".join of the cells),
# so a section row holding one filled cell has as many separators as its
# header. These pages are built the way that writer builds them.
def _written(table):
    return "\n".join(" | ".join(c or "" for c in row) for row in table)


@pytest.mark.parametrize("section", ["SUBSTRUCTURE", "EXTERNAL WORKS"])
def test_a_one_cell_section_row_is_a_row_of_its_table(section):
    table = [["ITEM", "DESCRIPTION", "UNIT", "QTY"], [None, section, None, None],
             ["1", "Excavation", "m3", "450"], ["2", "Backfill", "m3", "120"]]
    assert heading_line_for_page(f"{PROSE}\n{_written(table)}") == ""


@pytest.mark.parametrize("section", ["SUBSTRUCTURE", "EXTERNAL WORKS"])
def test_a_table_opening_on_a_section_row_gives_no_heading(section):
    # A table alone on its page, its first line a section row.
    table = [[None, section, None, None], ["1", "Excavation", "m3", "450"],
             ["2", "Backfill", "m3", "120"]]
    assert heading_line_for_page(_written(table)) == ""


@pytest.mark.parametrize("rows", [
    "ITEM | DESCRIPTION | QTY\n1 | Excavation | 450\n2 | Backfill | 120",
    "NO. | ACTIVITY\n1 | Setting out\n2 | Clearing",
])
def test_a_table_at_the_top_of_a_page_gives_no_heading(rows):
    assert heading_line_for_page(rows) == ""
    assert heading_line_for_page(f"{rows}\n4.2 Drainage\n{PROSE}") == "4.2 Drainage"


@pytest.mark.parametrize("rows", [
    "ITEM|DESCRIPTION|QTY\nA1|EXCAVATION|450",
    "NO.\t|\tACTIVITY\t|\tDAYS\n1  |  Setting out  |  3",
])
def test_rows_however_the_separator_is_spaced_are_a_table(rows):
    assert heading_line_for_page(f"{PROSE}\n{rows}\n{PROSE}") == ""


@pytest.mark.parametrize("row", ["|ITEM|DESCRIPTION|", "|NO.|ACTIVITY|QTY|"])
def test_an_unspaced_markdown_row_is_a_row(row):
    assert heading_line_for_page(f"{PROSE}\n{row}\n{PROSE}") == ""


@pytest.mark.parametrize("title", [
    "PART 2 | TECHNICAL SPECIFICATION",
    "VOLUME 3 | BILL OF QUANTITIES",
])
def test_a_pipe_title_above_a_wider_table_is_the_heading(title):
    page = f"{PROSE}\n{title}\nITEM | DESCRIPTION | QTY\nA1 | Excavation | 450"
    assert heading_line_for_page(page) == title


@pytest.mark.parametrize("heading", ["4.1 Concrete Works", "SECTION 7 DRAINAGE"])
def test_a_repeated_table_header_is_furniture_even_where_it_stands_alone(heading):
    # Page 1 holds the header alone (a header-only table) and a heading
    # below it; pages 2-4 repeat the header above an item row. The repeated
    # header is a running line, so page 1 is filed under its own heading.
    header = "ITEM | DESCRIPTION | QTY"
    pages = [f"{PROSE}\n{header}\n{PROSE}\n{heading}\n"] + [
        f"{header}\n{n} | Item {n} | 10\n5.{n} Drainage {n}\n{PROSE}\n"
        for n in (2, 3, 4)
    ]
    text, starts, at = "", [], 0
    for page, body in enumerate(pages, start=1):
        starts.append([page, at])
        text += body
        at += len(body)
    found = page_heading_lines(text, starts)
    assert found[1] == heading
    assert [found[p] for p in (2, 3, 4)] == [f"5.{n} Drainage {n}" for n in (2, 3, 4)]


@pytest.mark.parametrize("heading", [
    "# PART 2 | CONCRETE | VOLUME 3",
    "## 4.1 Formwork | Falsework | Props",
])
def test_a_hash_heading_as_wide_as_the_table_below_is_still_the_heading(heading):
    # Same separator count as the rows under it, so only the # keeps it
    # out of the table's run.
    page = f"{PROSE}\n{heading}\nITEM | DESCRIPTION | QTY\nA1 | Excavation | 450"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize("title", [
    "SECTION 5 | PART 2 | VOLUME 3",
    "BLOCK B | LEVEL 4 | ZONE 2 | GRID C",
])
def test_one_line_of_three_or_more_cells_alone_is_a_title(title):
    # One line is no run: however many cells it holds, alone between prose
    # it is read as it always was.
    assert heading_line_for_page(f"{PROSE}\n{title}\n{PROSE}") == title
