"""A table row is not the heading a PDF page is filed under.

PDF pages carry their tables as `` | ``-separated rows after the text
layer. A header row in capitals or a numbered item row looked heading-like,
so a page with no heading of its own was filed under a row of its table.
Every input is built here; nothing is read from a corpus.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import (
    _table_cell_copies,
    _table_row_indexes,
    heading_line_for_page,
    page_heading_lines,
)

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


# --- A row whose cell wrapped is split over lines that add up to its width.
@pytest.mark.parametrize("table", [
    "ITEM | DESCRIPTION | UNIT | QTY\n1 | Excavation in rock\n"
    "not exceeding 2 m deep | m3 | 450\n2 | Backfill with selected\n"
    "material in layers | m3 | 120",
    "1.1 | Supply and install reinforced\nconcrete grade C40 | m3 | 20\n"
    "1.2 | Formwork to soffits | m2 | 90",
])
def test_rows_with_a_wrapped_cell_are_not_a_heading(table):
    assert heading_line_for_page(table) == ""
    assert heading_line_for_page(f"{PROSE}\n{table}") == ""


@pytest.mark.parametrize("heading", ["4.1 Earthworks", "SECTION 2 DRAINAGE"])
def test_heading_above_a_wrapped_first_row_is_kept(heading):
    table = (
        "1 | Excavation in rock\nnot exceeding 2 m deep | m3 | 450\n"
        "2 | Backfill | m3 | 120"
    )
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{table}") == heading


@pytest.mark.parametrize("pieces", [
    # Pieces adding up to more or less than the width are not one row.
    "1 | Excavation in rock\nnot | exceeding | 2 m | deep",
    "1 | Excavation in rock\nNOT EXCEEDING 2 M DEEP",
])
def test_pieces_that_do_not_add_up_to_the_width_are_not_a_row(pieces):
    table = "ITEM | DESCRIPTION | UNIT | QTY\n7 | Formwork | m2 | 90"
    text = f"{PROSE}\n{table}\n{pieces}"
    # The first piece is no row of the table: it keeps its own reading (a
    # numbered line), where pieces adding up to the width are a row (above).
    assert heading_line_for_page(text) == pieces.splitlines()[0]


# --- A markdown row of exactly two separators, indented or not.
@pytest.mark.parametrize("row", [" | SUBSTRUCTURE | ", "|SUBSTRUCTURE|", "   |ITEM|DESCRIPTION|"])
def test_lone_markdown_row_is_not_a_heading(row):
    assert heading_line_for_page(f"{PROSE}\n{row}\n{PROSE}") == ""


@pytest.mark.parametrize("line", ["SUBSTRUCTURE | WORKS", "PART 2 | SCOPE"])
def test_lone_line_holding_the_separator_not_at_its_start_is_kept(line):
    assert heading_line_for_page(f"{PROSE}\n{line}\n{PROSE}") == line


@pytest.mark.parametrize("heading", ["  # PART 2 | SCOPE | REV A", "    # VOLUME 3 | SECTION | 04"])
def test_indented_markdown_heading_above_a_table_of_its_width_is_kept(heading):
    table = "A | B | C | D\n1 | Earthworks | m3 | 10\n2 | Drainage | m | 40"
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{table}") == heading.strip()
    # As wide as the table below it: only the indented "#" keeps it out.
    narrow = "A | B | C\n1 | Earthworks | m3\n2 | Drainage | m"
    assert heading_line_for_page(f"{PROSE}\n{heading}\n{narrow}") == heading.strip()


# --- The text layer repeats the table's cells before the rows.
@pytest.mark.parametrize("cells,rows", [
    ("ITEM\nDESCRIPTION\nUNIT\nQTY\n1\nExcavation\nm3\n450",
     "ITEM | DESCRIPTION | UNIT | QTY\n1 | Excavation | m3 | 450"),
    ("NO.\nACTIVITY\nSTART\nFINISH\n1\nSetting out\nMAY\nJUNE",
     "NO. | ACTIVITY | START | FINISH\n1 | Setting out | MAY | JUNE"),
])
def test_text_layer_copy_of_a_cell_is_not_a_heading(cells, rows):
    assert heading_line_for_page(f"{PROSE}\n{cells}\n{rows}") == ""


@pytest.mark.parametrize("heading", ["EARTHWORKS", "BILL NO. 2 DRAINAGE"])
def test_heading_the_table_repeats_as_a_cell_is_kept(heading):
    cells = f"ITEM\nDESCRIPTION\nUNIT\nQTY\n{heading}\n1\nExcavation\nm3\n450"
    rows = (
        f"ITEM | DESCRIPTION | UNIT | QTY\n | {heading} |  | \n"
        "1 | Excavation | m3 | 450"
    )
    page = f"{PROSE}\n{heading}\n{cells}\n{rows}"
    assert heading_line_for_page(page) == heading
    # Without its own line the cell is no heading.
    assert heading_line_for_page(f"{PROSE}\n{cells}\n{rows}") == ""


# --- The page as the PDF extractor composes it: a pipe title in the text
# layer is followed by the table's cells, then by the rows.
@pytest.mark.parametrize("title,cells,rows", [
    ("SECTION 03300 | CAST-IN-PLACE CONCRETE",
     "Property\nValue\nSlump\n75 mm\nStrength\n40 MPa",
     "Property | Value\nSlump | 75 mm\nStrength | 40 MPa"),
    ("PART 2 | PRODUCTS",
     "Item\nGrade\nCement\nCEM I\nSand\nZone 2",
     "Item | Grade\nCement | CEM I\nSand | Zone 2"),
])
def test_pipe_title_in_the_text_layer_is_kept(title, cells, rows):
    assert heading_line_for_page(f"{PROSE}\n{title}\n{cells}\n{rows}") == title


# --- One real PDF per shape through the PDF extractor and the page headings.
def _ruled_table_pdf(path, title, widths, rows):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("pdfplumber")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 60), PROSE[:70], fontsize=9)
    y = 80
    if title:
        page.insert_text((50, y + 10), title, fontsize=11)
        y += 30
    xs = [50]
    for w in widths:
        xs.append(xs[-1] + w)
    ys = [y]
    for row in rows:
        height = 16 * max(len(c.split("\n")) for c in row)
        for j, cell in enumerate(row):
            for k, piece in enumerate(cell.split("\n")):
                page.insert_text((xs[j] + 3, y + 12 + 16 * k), piece, fontsize=9)
        y += height
        ys.append(y)
    for yy in ys:
        page.draw_line((xs[0], yy), (xs[-1], yy))
    for xx in xs:
        page.draw_line((xx, ys[0]), (xx, ys[-1]))
    doc.save(str(path))


@pytest.mark.parametrize("title,widths,rows,expected", [
    (None, [40, 220, 50, 50],
     [["ITEM", "DESCRIPTION", "UNIT", "QTY"],
      ["1", "Excavation in rock\nnot exceeding 2 m deep", "m3", "450"],
      ["2", "Backfill with selected\nmaterial in layers", "m3", "120"]], None),
    (None, [40, 220, 50, 50],
     [["1.1", "Supply and install reinforced\nconcrete grade C40", "m3", "20"],
      ["1.2", "Formwork to soffits", "m2", "90"],
      ["1.3", "Rebar to slabs", "t", "4"]], None),
    ("SECTION 03300 | CAST-IN-PLACE CONCRETE", [120, 120],
     [["Property", "Value"], ["Slump", "75 mm"], ["Strength", "40 MPa"]],
     "SECTION 03300 | CAST-IN-PLACE CONCRETE"),
    ("PART 2 | PRODUCTS", [120, 120],
     [["Item", "Grade"], ["Cement", "CEM I"], ["Sand", "Zone 2"]],
     "PART 2 | PRODUCTS"),
])
def test_real_pdf_page_heading(tmp_path, title, widths, rows, expected):
    from app.core.doc_index import _extract_pdf

    path = tmp_path / "page.pdf"
    _ruled_table_pdf(path, title, widths, rows)
    text, meta = _extract_pdf(str(path), "page.pdf")
    assert " | " in text  # the rows were written after the text layer
    headings = page_heading_lines(text, meta.get("page_starts"))
    assert headings.get(1) == expected


# --- A run of pieces only: no line holds the width, so it is no table.
@pytest.mark.parametrize("first,last", [
    ("SECTION 4 | DRAINAGE", "SECTION 5 | ROADS"),
    ("PART 2 | PIPES", "PART 3 | CHAMBERS"),
])
def test_lines_that_only_add_up_to_a_later_width_are_not_a_table(first, last):
    # 1 + 2 and 2 + 1 separators add up to the 3 of the table after the
    # prose, but none of the four lines holds 3: the first is a title.
    page = (
        f"{PROSE}\n{first}\nPipes | 150 mm | uPVC\nManholes | precast | 1200\n"
        f"{last}\n{PROSE}\nA | B | C | D\n1 | Kerb | m | 40\n2 | Gully | nr | 6"
    )
    assert heading_line_for_page(page) == first


# --- Two wrapped cells in one row: three short pieces and a full row.
@pytest.mark.parametrize("table", [
    [["1.1", "Excavation in\nrock", "Disposal off\nsite", "450"],
     ["1.2", "Backfill", "Stockpile", "120"]],
    [["A1", "Blockwork to\nwalls", "Render both\nfaces", "90"],
     ["A2", "Lintels", "Precast", "14"]],
])
def test_the_widest_table_the_lines_make_is_taken(table):
    # The three pieces alone look like a 2-column table of three rows; read
    # that way the full row after them would stand alone, a numbered title.
    page = f"{PROSE}\n{_written(table)}\n{PROSE}"
    assert heading_line_for_page(page) == ""



def _wrapped_table(cell: str, unit: str) -> list[str]:
    return [
        "Item | Description | Unit | Qty", "1 | Concrete | m3 | 5",
        "2 | Formwork", cell, f"| {unit} | 9", "3 | Rebar | t | 2",
    ]


# A wrapped row whose cell runs onto a line with no separator: that line is
# a row, and the text layer's one loose copy of the cell is no heading.
@pytest.mark.parametrize("cell,unit", [("SLABS", "m2"), ("PILE CAPS", "m3")])
def test_a_text_layer_copy_of_a_wrapped_cell_is_no_heading(cell, unit):
    page = "\n".join(_wrapped_table(cell, unit) + [cell])
    assert heading_line_for_page(page) == ""


# The same text twice off the rows is the page's own line, not a copy.
@pytest.mark.parametrize("cell,unit", [("SLABS", "m2"), ("PILE CAPS", "m3")])
def test_a_wrapped_cell_also_written_twice_off_the_table_is_a_heading(cell, unit):
    page = "\n".join([cell] + _wrapped_table(cell, unit) + [cell])
    assert heading_line_for_page(page) == cell


# Copies are lines off the rows only: the continuation line that is itself a
# row is never one, and the loose line after the table is.
@pytest.mark.parametrize("cell,unit", [("SLABS", "m2"), ("PILE CAPS", "m3")])
def test_cell_copies_are_off_the_rows(cell, unit):
    lines = _wrapped_table(cell, unit) + [cell]
    rows = _table_row_indexes(lines)
    assert lines.index(cell) in rows
    assert _table_cell_copies(lines, rows) == {len(lines) - 1}
