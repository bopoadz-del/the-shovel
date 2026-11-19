"""The converter's section pass (bill_table_geometry.section_paths).

Two invented scanned-bill layouts that differ in running header, page-label
form and where the section title is printed, plus the negatives (no running
header; too few pages to see one). Every input is built here from word boxes.
"""

from __future__ import annotations

import pytest

from app.core import bill_table_geometry as g

H = 20.0
HEAVY = 28.0
CW = 9.0
LINE = 34.0


def _text(words, text, x, y, page, h=H):
    cur = x
    for tok in text.split():
        w = CW * len(tok)
        words.append(g.Word(tok, page, cur, y, w, h))
        cur += w + CW


def _page(page, columns, above, body):
    """``above``: (text, height) lines printed above the column header;
    ``body``: (text, height) label lines or a list of cell texts (a row)."""
    words: list[g.Word] = []
    y = 60.0
    for text, h in above:
        _text(words, text, columns[0][1], y, page, h)
        y += LINE
    for label, x in columns:
        _text(words, label, x, y, page)
    y += LINE
    for line in body:
        if isinstance(line, tuple):
            _text(words, line[0], columns[0][1], y, page, line[1])
        else:
            for (_lbl, x), cell in zip(columns, line):
                if cell:
                    _text(words, cell, x, y, page)
        y += LINE
    return words


COLS_A = [("Description", 40), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]
COLS_B = [("Description", 80), ("Qty", 560), ("Unit", 660), ("Rate", 780), ("Amount", 920)]


def _layout_a():
    """Running header + 'Bill N Page N-P' label; the title printed above the
    header on the FIRST page of each section only; page 4's label misread."""
    run = ("Northfield Depot Extension", H)
    pages = [
        (1, [run, ("Bill 1 Page 1-1", H), ("EARTHWORKS", HEAVY)],
         [("Excavation", HEAVY), ("Excavate in any material including", H),
          ("disposal off site", H), ["Depth not exceeding 2 m", "m3", "10", "5", "50"]]),
        (2, [run, ("Bill 1 Page 1-2", H)],
         [("Excavation (cont'd)", HEAVY), ["Depth 2 to 4 m", "m3", "4", "8", "32"]]),
        (3, [run, ("Bill 2 Page 2-3", H), ("DRAINAGE", HEAVY)],
         [("Pipes", HEAVY), ["Pipe 150 mm", "m", "20", "3", "60"]]),
        (4, [run, ("Bill 2 Page 2-S", H)],
         [("Pipes (cont'd)", HEAVY), ["Pipe 225 mm", "m", "6", "5", "30"]]),
        (5, [run, ("Bill 2 Page 2-5", H)],
         [("Manholes", HEAVY), ["Manhole type A", "nr", "2", "100", "200"]]),
    ]
    return [w for p, above, body in pages for w in _page(p, COLS_A, above, body)]


def _layout_b():
    """Two running lines + a dotted label 'S.s.P'; the title (two lines)
    printed above the header on the SECOND page of a section; the
    continuation mark is '-contd' with an OCR variant '-contd.'."""
    run1, run2 = ("Harbour Road Upgrade", H), ("Tender Volume Four", H)
    pages = [
        (1, [run1, run2, ("4.2.01", H)],
         [("Kerbs", HEAVY), ["Precast kerb straight", "40", "m", "2", "80"]]),
        (2, [run1, run2, ("4.2.02", H), ("ROADWORKS", HEAVY), ("Surfacing and kerbs", HEAVY)],
         [("Kerbs -contd", HEAVY), ["Precast kerb curved", "10", "m", "3", "30"]]),
        (3, [run1, run2, ("4.3.03", H)],
         [("Paving", HEAVY), ["Block paving", "50", "m2", "4", "200"]]),
        (4, [run1, run2, ("4.3.04", H), ("FOOTWAYS", HEAVY)],
         [("Paving -contd.", HEAVY), ["Flag paving", "5", "m2", "6", "30"]]),
    ]
    return [w for p, above, body in pages for w in _page(p, COLS_B, above, body)]


def _paths(rows):
    return [(r.fields["description"], r.path) for r in rows]


@pytest.mark.parametrize("words, expected", [
    (_layout_a(), [
        ("Depth not exceeding 2 m", ("EARTHWORKS", "Excavation", "Excavate in any material including disposal off site")),
        ("Depth 2 to 4 m", ("EARTHWORKS", "Excavation")),
        ("Pipe 150 mm", ("DRAINAGE", "Pipes")),
        ("Pipe 225 mm", ("DRAINAGE", "Pipes")),
        ("Manhole type A", ("DRAINAGE", "Manholes")),
    ]),
    (_layout_b(), [
        ("Precast kerb straight", ("ROADWORKS / Surfacing and kerbs", "Kerbs")),
        ("Precast kerb curved", ("ROADWORKS / Surfacing and kerbs", "Kerbs")),
        ("Block paving", ("FOOTWAYS", "Paving")),
        ("Flag paving", ("FOOTWAYS", "Paving")),
    ]),
], ids=["layout-a", "layout-b"])
def test_path_is_title_subheading_preamble(words, expected):
    assert _paths(g.extract_bill_rows(words)) == expected


@pytest.mark.parametrize("words, running, label", [
    (_layout_a(), {"northfield depot extension", "bill # page #-#"}, "bill # page #-#"),
    (_layout_b(), {"harbour road upgrade", "tender volume four", "#.#.#"}, "#.#.#"),
], ids=["layout-a", "layout-b"])
def test_running_lines_and_page_label(words, running, label):
    pages = len({w.page for w in words})
    found = g.running_lines(_printed(words), pages)
    assert running <= found.masks
    assert found.page_label == label


def _printed(words):
    """Every label the converter printed, through its own label observer path."""
    seen: list = []
    orig = g.section_paths

    def spy(rows, printed, pages):
        seen.extend(printed)
        return rows
    g.section_paths = spy
    try:
        g.extract_bill_rows(words)
    finally:
        g.section_paths = orig
    return seen


@pytest.mark.parametrize("words, expected", [
    (_layout_a(), {1: ("1", "1"), 2: ("1", "1"), 3: ("2", "2"), 4: ("2", "2"), 5: ("2", "2")}),
    (_layout_b(), {1: ("4", "2"), 2: ("4", "2"), 3: ("4", "3"), 4: ("4", "3")}),
], ids=["misread-label-inherits", "dotted-label"])
def test_page_sections(words, expected):
    printed = _printed(words)
    running = g.running_lines(printed, len({w.page for w in words}))
    assert g.page_sections(printed, running) == expected


def _lab(text, page=1, top=0.0):
    return g._Label(text, 40.0, H, page=page, top=top, bottom=top + H)


@pytest.mark.parametrize("texts, expected", [
    (["Kerbs", "Paving", "Kerbs (cont'd)", "Paving (contd)"],
     {"kerbs (cont'd)": "Kerbs", "paving (contd)": "Paving"}),
    (["Kerbs", "Kerbs (b)"], {}),  # a suffix after one heading only is not a mark
    (["Kerbs", "Kerbs straight", "Paving", "Paving straight"], {}),  # unpunctuated word
    (["Pipes", "Pipes (150)", "Ducts", "Ducts (150)"], {}),  # a figure is not a mark
], ids=["mark-after-two-headings", "one-heading-only", "plain-word", "figure"])
def test_continuation_map(texts, expected):
    assert g.continuation_map([_lab(t) for t in texts]) == expected


@pytest.mark.parametrize("a, b, wrapped", [
    (_lab("one", top=0.0), _lab("two", top=34.0), True),       # next line, same style
    (_lab("one", top=0.0), _lab("two", top=80.0), False),      # a blank line between
    (_lab("one", top=0.0), _lab("two", page=2, top=34.0), False),  # another page
    (_lab("one", top=0.0), g._Label("two", 40.0, HEAVY, page=1, top=34.0, bottom=62.0), False),  # heavier
], ids=["adjacent", "gap", "other-page", "heavier"])
def test_wrapped(a, b, wrapped):
    assert g._wrapped(a, b) is wrapped


def _no_running(n_pages):
    words: list[g.Word] = []
    for p in range(1, n_pages + 1):
        words += _page(p, COLS_A, [(f"Heading of sheet {chr(64 + p) * p}", H)],
                       [[f"Item {p}", "m", "1", "2", "2"]])
    return words


def _repeating_two_pages():
    words: list[g.Word] = []
    for p in (1, 2):
        words += _page(p, COLS_A, [("Same top line", H)], [[f"Item {p}", "m", "1", "2", "2"]])
    return words


@pytest.mark.parametrize("words", [_no_running(3), _repeating_two_pages()],
                         ids=["no-running-header", "too-few-pages"])
def test_document_without_running_header_is_left_alone(words):
    pages = len({w.page for w in words})
    assert not g.running_lines(_printed(words), pages).masks
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [tuple(lab.text for lab in r.labels) for r in rows]
