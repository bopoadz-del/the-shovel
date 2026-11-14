"""P-B3: scanned bill page -> table rows from OCR word geometry.

Every input is a synthetic page built here from word boxes: no document, no
bill name, no fixture file. Each decision is exercised with at least two
inputs.
"""

from __future__ import annotations

import random

import pytest

from app.core import bill_table_geometry as g

H = 20.0  # word height in px
CW = 9.0  # px per character
LINE = 34.0  # baseline pitch


def _words_for_cell(text: str, x: float, y: float, page: int) -> list[g.Word]:
    out, cur = [], x
    for tok in text.split():
        w = CW * len(tok)
        out.append(g.Word(tok, page, cur, y, w, H))
        cur += w + CW  # one-character space inside a cell
    return out


def _page(
    columns: list[tuple[str, float]],
    lines: list[list[str] | str],
    page: int,
    header: bool = True,
    top: float = 100.0,
) -> list[g.Word]:
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
            for (_label, x), text in zip(columns, line):
                if text:
                    words += _words_for_cell(text, x, y, page)
        y += LINE
    return words


COLS_A = [("Item", 40), ("Description", 120), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]
LINES_A = [
    "Bill 3 Pipework",
    ["A", "Supply pipe 150 dia", "m", "120", "45.50", "5,460.00"],
    ["B", "Gate valve 150 dia", "Nr", "4", "1,200.00", "4,800.00"],
    ["C", "Thrust block", "Nr", "6", "300.00", "1,800.00"],
]
COLS_B = [
    ("Ref", 40),
    ("Amount SAR", 160),
    ("Unit Rate SAR", 330),
    ("Quantity", 520),
    ("Unit", 640),
    ("Description", 740),
]
LINES_B = [
    "Section 7 Earthworks",
    ["E1", "2,000.00", "10.00", "200", "m3", "Excavate trench"],
    ["E2", "750.00", "2.50", "300", "m2", "Trim and compact"],
]


def _fields(rows):
    return [
        (
            r.section,
            r.fields.get("description"),
            r.fields.get("unit"),
            r.fields.get("quantity"),
            r.fields.get("rate"),
            r.fields.get("total"),
        )
        for r in rows
    ]


@pytest.mark.parametrize(
    "cols,lines,seed",
    [(COLS_A, LINES_A, 1), (COLS_B, LINES_B, 7)],
)
def test_word_order_does_not_change_rows(cols, lines, seed):
    """Row-major and column-major OCR order (and a shuffle) give the same rows."""
    words = _page(cols, lines, page=1)
    row_major = sorted(words, key=lambda w: (w.top, w.left))
    column_major = sorted(words, key=lambda w: (w.left, w.top))
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    a = _fields(g.extract_bill_rows(row_major))
    b = _fields(g.extract_bill_rows(column_major))
    c = _fields(g.extract_bill_rows(shuffled))
    assert a == b == c
    assert len(a) == len(lines) - 1


@pytest.mark.parametrize(
    "cols,lines,section,descs,rates,extra_label,extra,total",
    [
        (
            COLS_A,
            LINES_A,
            "Bill 3 Pipework",
            ["Supply pipe 150 dia", "Gate valve 150 dia", "Thrust block"],
            ["45.50", "1,200.00", "300.00"],
            "Item",
            ["A", "B", "C"],
            12060.0,
        ),
        # Different column order and header words; the margin label prints a
        # number that lands under a figure column and is still a section label.
        (
            COLS_B,
            LINES_B,
            "Section 7 Earthworks",
            ["Excavate trench", "Trim and compact"],
            ["10.00", "2.50"],
            "Ref",
            ["E1", "E2"],
            2750.0,
        ),
    ],
)
def test_column_orders_parse(cols, lines, section, descs, rates, extra_label, extra, total):
    rows = g.extract_bill_rows(_page(cols, lines, page=1))
    assert [r.fields["description"] for r in rows] == descs
    assert [r.fields["rate"] for r in rows] == rates
    assert [r.extra.get(extra_label) for r in rows] == extra
    assert {r.section for r in rows} == {section}
    assert sum(g.parse_number(r.fields["total"]) for r in rows) == pytest.approx(total)


_CONT = [
    ["D", "Air valve", "Nr", "2", "900.00", "1,800.00"],
    "Bill 4 Roads",
    ["A", "Kerb", "m", "50", "20.00", "1,000.00"],
]


@pytest.mark.parametrize(
    "pages,expected",
    [
        # header on the first page only; the next page continues its columns
        (
            [(4, LINES_A, True), (5, _CONT, False)],
            [
                (4, "Bill 3 Pipework", "Supply pipe 150 dia"),
                (4, "Bill 3 Pipework", "Gate valve 150 dia"),
                (4, "Bill 3 Pipework", "Thrust block"),
                (5, "Bill 3 Pipework", "Air valve"),
                (5, "Bill 4 Roads", "Kerb"),
            ],
        ),
        # header printed again on the next page: section kept across the break
        (
            [(1, LINES_A[:2], True), (2, LINES_A[2:], True)],
            [
                (1, "Bill 3 Pipework", "Supply pipe 150 dia"),
                (2, "Bill 3 Pipework", "Gate valve 150 dia"),
                (2, "Bill 3 Pipework", "Thrust block"),
            ],
        ),
    ],
)
def test_section_split_across_pages(pages, expected):
    words: list[g.Word] = []
    for page, lines, header in reversed(pages):  # pages arrive out of order
        words += _page(COLS_A, lines, page=page, header=header)
    rows = g.extract_bill_rows(words)
    assert [(r.source_page, r.section, r.fields["description"]) for r in rows] == expected


@pytest.mark.parametrize(
    "lines",
    [
        ["General conditions apply", "Rates include overheads"],
        [["", "", "", "12", "3.00", "36.00"]],
    ],
)
def test_page_without_any_header_gives_no_rows(lines):
    assert g.extract_bill_rows(_page(COLS_A, lines, page=1, header=False)) == []


@pytest.mark.parametrize(
    "cells,expected",
    [
        ({"quantity": "4", "rate": "1,200.00", "total": "4,800.00"}, ""),
        ({"quantity": "3", "rate": "10", "total": "30.10"}, ""),  # within tolerance
        ({"quantity": "4", "rate": "1,200.00", "total": "4,900.00"}, "qty x rate != amount"),
        ({"quantity": "4", "rate": "", "total": "4,900.00"}, ""),  # cannot check
        ({"quantity": "(2)", "rate": "5", "total": "-10"}, ""),  # negative forms
    ],
)
def test_check_column(cells, expected):
    assert g.check_row(cells) == expected


@pytest.mark.parametrize(
    "text,expected",
    [("1,200.50", 1200.5), ("(3)", -3.0), ("-7", -7.0), ("12a", None), ("", None)],
)
def test_parse_number(text, expected):
    assert g.parse_number(text) == expected


@pytest.mark.parametrize(
    "lines,checks",
    [
        (
            ["Bill 9", ["A", "Pipe", "m", "10", "5.00", "50.00"], ["B", "Valve", "Nr", "2", "5.00", "99.00"]],
            ["", "qty x rate != amount"],
        ),
        (
            ["Bill 9", ["A", "Pipe", "m", "10", "5.00", "51.00"], ["B", "Valve", "Nr", "2", "5.00", "10.00"]],
            ["qty x rate != amount", ""],
        ),
    ],
)
def test_mismatched_row_is_kept_and_flagged(lines, checks):
    rows = g.extract_bill_rows(_page(COLS_A, lines, page=1))
    assert [r.check for r in rows] == checks


@pytest.mark.parametrize(
    "cells,expected",
    [
        ([("Item", 40), ("Description", 120), ("Remarks", 600)], None),
        ([("Qty", 40), ("Rate", 120), ("Amount", 600)], None),
        ([("Description", 40), ("Qty", 400)], None),  # fewer than three cells
        ([("Description", 40), ("Qty", 400), ("Rate", 600)], ["description", "quantity", "rate"]),
        ([("Description", 40), ("Unit", 400), ("Amount SAR", 600)], ["description", "unit", "total"]),
    ],
)
def test_header_requires_description_and_a_figure_column(cells, expected):
    words: list[g.Word] = []
    for label, x in cells:
        words += _words_for_cell(label, x, 10.0, 1)
    cols = g.detect_header(g.group_lines(words)[0])
    assert (None if cols is None else [c.field for c in cols]) == expected


@pytest.mark.parametrize(
    "data,expected",
    [
        (
            {
                "text": ["", "Qty", " ", "Rate", "x"],
                "conf": ["-1", "91", "-1", "88", "-1"],
                "left": [0, 10, 0, 50, 90],
                "top": [0, 5, 0, 5, 5],
                "width": [0, 30, 0, 40, 9],
                "height": [0, 20, 0, 20, 20],
            },
            [("Qty", 10.0), ("Rate", 50.0)],
        ),
        (
            {
                "text": ["Amount", "", "12.00"],
                "conf": ["70", "95", "0"],
                "left": [300, 0, 400],
                "top": [9, 0, 9],
                "width": [60, 0, 40],
                "height": [18, 0, 18],
            },
            [("Amount", 300.0), ("12.00", 400.0)],
        ),
    ],
)
def test_tesseract_dict_drops_blank_and_negative_confidence(data, expected):
    words = g.words_from_tesseract_data(data, page=3)
    assert [(w.text, w.left) for w in words] == expected
    assert {w.page for w in words} == {3}


@pytest.mark.parametrize(
    "words_in,expected_lines",
    [
        # two lines, words given bottom-up and right-to-left
        (
            [("b2", 200, 52), ("a2", 10, 50), ("b1", 200, 12), ("a1", 10, 10)],
            [["a1", "b1"], ["a2", "b2"]],
        ),
        # a small vertical jitter inside a line stays on that line
        (
            [("x", 10, 10), ("y", 100, 14), ("z", 10, 60)],
            [["x", "y"], ["z"]],
        ),
    ],
)
def test_group_lines(words_in, expected_lines):
    words = [g.Word(t, 1, x, y, 30, H) for t, x, y in words_in]
    assert [[w.text for w in line] for line in g.group_lines(words)] == expected_lines


@pytest.mark.parametrize(
    "pages,counts,probe,needles",
    [
        (
            [(1, LINES_A, True), (2, ["Bill 4 Roads", ["A", "Kerb", "m", "50", "20.00", "1,000.00"]], False)],
            [3, 1],
            1,
            ["Bill: Bill 3 Pipework", "Item: Gate valve 150 dia", "Rate: 1,200.00", "Amount: 4,800.00"],
        ),
        (
            [(1, LINES_A, True)],
            [3],
            2,
            ["Bill: Bill 3 Pipework", "Item: Thrust block", "Quantity: 6", "Unit: Nr"],
        ),
    ],
)
def test_workbook_reads_back_through_the_row_chunker(tmp_path, pages, counts, probe, needles):
    """The derived sheet is consumed by the existing spreadsheet row path:
    one chunk per measured row, the section inherited as the bill heading."""
    from app.core import doc_index

    words: list[g.Word] = []
    for page, lines, header in pages:
        words += _page(COLS_A, lines, page=page, header=header)
    path = tmp_path / "derived.xlsx"
    assert list(g.write_bill_workbook(g.extract_bill_rows(words), str(path)).values()) == counts
    chunks = doc_index._spreadsheet_row_chunks(str(path), "derived.xlsx", ".xlsx")
    assert len(chunks) == sum(counts)
    for needle in needles:
        assert needle in chunks[probe]


@pytest.mark.parametrize(
    "cols,lines,page,extra_label,first_extra",
    [
        (COLS_A, LINES_A, 12, "Item", "A"),
        (COLS_B, LINES_B, 3, "Ref", "E1"),
    ],
)
def test_workbook_keeps_source_page_and_unmapped_columns(tmp_path, cols, lines, page, extra_label, first_extra):
    import openpyxl

    path = tmp_path / "d.xlsx"
    g.write_bill_workbook(g.extract_bill_rows(_page(cols, lines, page=page)), str(path))
    ws = openpyxl.load_workbook(path).worksheets[0]
    values = [list(r) for r in ws.iter_rows(values_only=True)]
    assert values[0][0] == lines[0] and not any(values[0][1:])
    assert values[1] == ["Description", "Unit", "Quantity", "Rate", "Amount", extra_label, "Check", "Source Page"]
    assert values[2][-1] == page and values[2][5] == first_extra


@pytest.mark.parametrize(
    "names,expected",
    [
        (["Bill 1", "Bill 1"], ["Bill 1", "Bill 1 2"]),
        (["a/b:c", ""], ["a b c", "Bill"]),
    ],
)
def test_sheet_titles_are_valid_and_unique(names, expected):
    used: set[str] = set()
    assert [g._sheet_title(n, used) for n in names] == expected


@pytest.mark.parametrize(
    "word,cx,cy,right",
    [
        (g.Word("a", 1, 10, 20, 30, 40), 25.0, 40.0, 40.0),
        (g.Word("b", 2, 0, 0, 8, 6), 4.0, 3.0, 8.0),
    ],
)
def test_word_geometry(word, cx, cy, right):
    assert (word.cx, word.cy, word.right) == (cx, cy, right)


@pytest.mark.parametrize(
    "line,expected",
    [
        ([], []),
        (_words_for_cell("Unit Rate", 10, 0, 1) + _words_for_cell("Qty", 300, 0, 1), [["Unit", "Rate"], ["Qty"]]),
    ],
)
def test_cells_by_gap(line, expected):
    assert [[w.text for w in c] for c in g._cells_by_gap(line)] == expected


@pytest.mark.parametrize(
    "lines,sheets",
    [
        ([], [("Bill", ["Description", "Unit", "Quantity", "Rate", "Amount"])]),
        (LINES_A[:2], [("Bill 3 Pipework", ["Bill 3 Pipework"])]),
    ],
)
def test_workbook_always_has_a_sheet(tmp_path, lines, sheets):
    import openpyxl

    rows = g.extract_bill_rows(_page(COLS_A, lines, page=1)) if lines else []
    path = tmp_path / "w.xlsx"
    g.write_bill_workbook(rows, str(path))
    wb = openpyxl.load_workbook(path)
    got = [(ws.title, [c for c in next(ws.iter_rows(values_only=True)) if c is not None]) for ws in wb.worksheets]
    assert got == sheets


@pytest.mark.parametrize(
    "data,page,expected",
    [
        (
            {"text": ["Qty", ""], "conf": ["90", "-1"], "left": [5, 0], "top": [6, 0], "width": [27, 0], "height": [20, 0]},
            2,
            [("Qty", 2, 5.0)],
        ),
        (
            {"text": ["Rate", "9.50"], "conf": ["80", "77"], "left": [1, 60], "top": [2, 2], "width": [36, 36], "height": [20, 20]},
            7,
            [("Rate", 7, 1.0), ("9.50", 7, 60.0)],
        ),
    ],
)
def test_ocr_page_words_reads_tesseract_boxes(monkeypatch, data, page, expected):
    import pytesseract

    seen = {}

    def _image_to_data(image, lang, output_type):
        seen["args"] = (image, lang, output_type)
        return data

    monkeypatch.setattr(pytesseract, "image_to_data", _image_to_data)
    words = g.ocr_page_words("IMG", page, lang="eng")
    assert [(w.text, w.page, w.left) for w in words] == expected
    assert seen["args"] == ("IMG", "eng", pytesseract.Output.DICT)
