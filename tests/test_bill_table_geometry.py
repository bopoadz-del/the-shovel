"""P-B3: scanned bill page -> table rows from OCR word geometry.

Every input is a synthetic page built here from word boxes: no document, no
bill name, no fixture file. Each decision is exercised with at least two
inputs.
"""

from __future__ import annotations

import random

import pytest

from app.core import bill_table_geometry as g

H = 20.0       # word height in px
CW = 9.0       # px per character
LINE = 34.0    # baseline pitch


def _words_for_cell(text: str, x: float, y: float, page: int) -> list[g.Word]:
    out, cur = [], x
    for tok in text.split():
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW  # one-character space inside a cell
    return out


def _page(columns: list[tuple[str, float]], lines: list[list[str] | str], page: int,
          header: bool = True, top: float = 100.0) -> list[g.Word]:
    """Lay out a bill page. ``columns`` = [(header label, x)], each line is a
    list of cell texts (one per column, "" for empty) or a str for a label
    line printed at the left margin."""
    words: list[g.Word] = []
    y = top
    if header:
        for label, x in columns:
            words += _words_for_cell(label, x, y, page)
        y += LINE
    for line in lines:
        if isinstance(line, str):
            words += _words_for_cell(line, columns[0][1], y, page)
        else:
            for (label, x), text in zip(columns, line):
                if text:
                    words += _words_for_cell(text, x, y, page)
        y += LINE
    return words


COLS_A = [("Item", 40), ("Description", 120), ("Unit", 600), ("Qty", 700),
          ("Rate", 820), ("Amount", 960)]
LINES_A = [
    "Bill 3 Pipework",
    ["A", "Supply pipe 150 dia", "m", "120", "45.50", "5,460.00"],
    ["B", "Gate valve 150 dia", "Nr", "4", "1,200.00", "4,800.00"],
    ["C", "Thrust block", "Nr", "6", "300.00", "1,800.00"],
]


def _fields(rows):
    return [(r.section, r.fields.get("description"), r.fields.get("unit"),
             r.fields.get("quantity"), r.fields.get("rate"), r.fields.get("total"))
            for r in rows]


@pytest.mark.parametrize("seed", [1, 7])
def test_word_order_does_not_change_rows(seed):
    """Row-major and column-major OCR order (and a shuffle) give the same rows."""
    words = _page(COLS_A, LINES_A, page=1)
    row_major = sorted(words, key=lambda w: (w.top, w.left))
    column_major = sorted(words, key=lambda w: (w.left, w.top))
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    a = _fields(g.extract_bill_rows(row_major))
    b = _fields(g.extract_bill_rows(column_major))
    c = _fields(g.extract_bill_rows(shuffled))
    assert a == b == c
    assert len(a) == 3
    assert a[1] == ("Bill 3 Pipework", "Gate valve 150 dia", "Nr", "4", "1,200.00", "4,800.00")


def test_two_column_orders_both_parse():
    """Two bills that print their columns in different orders and with
    different header words both resolve to the same fields."""
    cols_b = [("Ref", 40), ("Amount SAR", 160), ("Unit Rate SAR", 330),
              ("Quantity", 520), ("Unit", 640), ("Description", 740)]
    lines_b = [
        "Section 7 Earthworks",
        ["E1", "2,000.00", "10.00", "200", "m3", "Excavate trench"],
        ["E2", "750.00", "2.50", "300", "m2", "Trim and compact"],
    ]
    rows_a = g.extract_bill_rows(_page(COLS_A, LINES_A, page=1))
    rows_b = g.extract_bill_rows(_page(cols_b, lines_b, page=1))
    # The margin label prints a number that lands under a figure column; it
    # is a section label, not a measured row.
    assert {r.section for r in rows_b} == {"Section 7 Earthworks"}
    assert [r.fields["description"] for r in rows_b] == ["Excavate trench", "Trim and compact"]
    assert [r.fields["rate"] for r in rows_b] == ["10.00", "2.50"]
    assert [r.extra.get("Ref") for r in rows_b] == ["E1", "E2"]
    assert [r.extra.get("Item") for r in rows_a] == ["A", "B", "C"]
    sum_a = sum(g.parse_number(r.fields["total"]) for r in rows_a)
    sum_b = sum(g.parse_number(r.fields["total"]) for r in rows_b)
    assert sum_a == pytest.approx(12060.0)
    assert sum_b == pytest.approx(2750.0)


def test_section_split_across_pages_keeps_columns_and_section():
    p1 = _page(COLS_A, LINES_A, page=4)
    p2 = _page(COLS_A, [["D", "Air valve", "Nr", "2", "900.00", "1,800.00"],
                        "Bill 4 Roads",
                        ["A", "Kerb", "m", "50", "20.00", "1,000.00"]],
               page=5, header=False)
    rows = g.extract_bill_rows(p2 + p1)  # pages arrive out of order
    assert [(r.source_page, r.section, r.fields["description"]) for r in rows] == [
        (4, "Bill 3 Pipework", "Supply pipe 150 dia"),
        (4, "Bill 3 Pipework", "Gate valve 150 dia"),
        (4, "Bill 3 Pipework", "Thrust block"),
        (5, "Bill 3 Pipework", "Air valve"),
        (5, "Bill 4 Roads", "Kerb"),
    ]


def test_page_without_any_header_gives_no_rows():
    prose = _page(COLS_A, ["General conditions apply", "Rates include overheads"],
                  page=1, header=False)
    figures_only = _page(COLS_A, [["", "", "", "12", "3.00", "36.00"]], page=1, header=False)
    assert g.extract_bill_rows(prose) == []
    assert g.extract_bill_rows(figures_only) == []


@pytest.mark.parametrize("cells,expected", [
    ({"quantity": "4", "rate": "1,200.00", "total": "4,800.00"}, ""),
    ({"quantity": "3", "rate": "10", "total": "30.10"}, ""),          # within tolerance
    ({"quantity": "4", "rate": "1,200.00", "total": "4,900.00"}, "qty x rate != amount"),
    ({"quantity": "4", "rate": "", "total": "4,900.00"}, ""),         # cannot check
    ({"quantity": "(2)", "rate": "5", "total": "-10"}, ""),           # negative forms
])
def test_check_column(cells, expected):
    assert g.check_row(cells) == expected


def test_mismatched_row_is_kept_and_flagged():
    lines = ["Bill 9", ["A", "Pipe", "m", "10", "5.00", "50.00"],
             ["B", "Valve", "Nr", "2", "5.00", "99.00"]]
    rows = g.extract_bill_rows(_page(COLS_A, lines, page=1))
    assert [r.check for r in rows] == ["", "qty x rate != amount"]


def test_header_requires_description_and_a_figure_column():
    y = 10.0
    no_figure = _words_for_cell("Item", 40, y, 1) + _words_for_cell("Description", 120, y, 1) \
        + _words_for_cell("Remarks", 600, y, 1)
    no_desc = _words_for_cell("Qty", 40, y, 1) + _words_for_cell("Rate", 120, y, 1) \
        + _words_for_cell("Amount", 600, y, 1)
    ok = _words_for_cell("Description", 40, y, 1) + _words_for_cell("Qty", 400, y, 1) \
        + _words_for_cell("Rate", 600, y, 1)
    assert g.detect_header(g.group_lines(no_figure)[0]) is None
    assert g.detect_header(g.group_lines(no_desc)[0]) is None
    cols = g.detect_header(g.group_lines(ok)[0])
    assert [c.field for c in cols] == ["description", "quantity", "rate"]


def test_tesseract_dict_drops_blank_and_negative_confidence():
    data = {"text": ["", "Qty", " ", "Rate", "x"], "conf": ["-1", "91", "-1", "88", "-1"],
            "left": [0, 10, 0, 50, 90], "top": [0, 5, 0, 5, 5],
            "width": [0, 30, 0, 40, 9], "height": [0, 20, 0, 20, 20]}
    words = g.words_from_tesseract_data(data, page=3)
    assert [(w.text, w.page, w.left) for w in words] == [("Qty", 3, 10.0), ("Rate", 3, 50.0)]


def test_workbook_reads_back_through_the_row_chunker(tmp_path):
    """The derived sheet is consumed by the existing spreadsheet row path:
    one chunk per measured row, the section inherited as the bill heading."""
    from app.core import doc_index

    p1 = _page(COLS_A, LINES_A, page=1)
    p2 = _page(COLS_A, ["Bill 4 Roads", ["A", "Kerb", "m", "50", "20.00", "1,000.00"]],
               page=2, header=False)
    rows = g.extract_bill_rows(p1 + p2)
    path = tmp_path / "derived.xlsx"
    counts = g.write_bill_workbook(rows, str(path))
    assert list(counts.values()) == [3, 1]
    chunks = doc_index._spreadsheet_row_chunks(str(path), "derived.xlsx", ".xlsx")
    assert len(chunks) == 4
    assert "Bill: Bill 3 Pipework" in chunks[1] and "Item: Gate valve 150 dia" in chunks[1]
    assert "Rate: 1,200.00" in chunks[1] and "Amount: 4,800.00" in chunks[1]
    assert "Bill: Bill 4 Roads" in chunks[3] and "Quantity: 50" in chunks[3]


def test_workbook_keeps_source_page_and_unmapped_columns(tmp_path):
    import openpyxl

    rows = g.extract_bill_rows(_page(COLS_A, LINES_A, page=12))
    path = tmp_path / "d.xlsx"
    g.write_bill_workbook(rows, str(path))
    ws = openpyxl.load_workbook(path).worksheets[0]
    values = [list(r) for r in ws.iter_rows(values_only=True)]
    assert values[0][0] == "Bill 3 Pipework" and not any(values[0][1:])
    assert values[1] == ["Description", "Unit", "Quantity", "Rate", "Amount", "Item",
                         "Check", "Source Page"]
    assert values[2][-1] == 12 and values[2][5] == "A"
