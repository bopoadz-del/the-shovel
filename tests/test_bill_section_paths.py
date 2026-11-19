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


def _unchanged(words):
    rows = g.extract_bill_rows(words)
    return [r.path for r in rows] == [tuple(lab.text for lab in r.labels) for r in rows]


def _no_section_info(labels):
    """Four pages with a constant running line, a per-page label line from
    ``labels`` (None = no label line) and a different title on each page."""
    titles = ["CONCRETE", "FORMWORK", "MASONRY", "METALWORK"]
    words: list[g.Word] = []
    for p in range(1, 5):
        above = [("Eastgate Pumping Station", H)]
        if labels[p - 1] is not None:
            above.append((labels[p - 1], H))
        above.append((titles[p - 1], HEAVY))
        words += _page(p, COLS_A, above, [("Walls", HEAVY), [f"Item {p}", "m", "1", "2", "2"]])
    return words


@pytest.mark.parametrize("labels", [
    [None] * 4,                                            # a constant header, no label
    ["Contract 12 Volume 3"] * 4,                          # constant figures, nothing varies
    [f"Page {p}" for p in range(1, 5)],                    # a page number only
    [f"Sheet {p} of 9" for p in range(1, 5)],              # page number plus a constant count
], ids=["no-label", "constant-figures", "page-number-only", "page-of-count"])
def test_label_without_section_information_leaves_rows_unchanged(labels):
    words = _no_section_info(labels)
    printed = _printed(words)
    running = g.running_lines(printed, 4)
    assert running.masks  # the running header IS found ...
    assert g.page_sections(printed, running) == {}  # ... but gives no sections
    assert _unchanged(words)


@pytest.mark.parametrize("labels, expected", [
    ([f"Bill {b} Sheet {p} of 9" for b, p in ((1, 1), (1, 2), (2, 3), (2, 4))],
     {1: ("1", "9"), 2: ("1", "9"), 3: ("2", "9"), 4: ("2", "9")}),
    # every page its own section: the fields tie, the LAST is the page number
    ([f"Part {s} Leaf {p}" for s, p in ((3, 7), (4, 8), (5, 9), (6, 10))],
     {1: ("3",), 2: ("4",), 3: ("5",), 4: ("6",)}),
], ids=["section-beside-page-count", "tie-last-is-page"])
def test_section_field_beside_page_number_is_section_information(labels, expected):
    words = _no_section_info(labels)
    printed = _printed(words)
    assert g.page_sections(printed, g.running_lines(printed, 4)) == expected


def _labelled_pages(labels):
    """One page per entry of ``labels``: a constant running line, the label
    line and a different title on each page."""
    words: list[g.Word] = []
    for p, label in enumerate(labels, start=1):
        above = [("Eastgate Pumping Station", H), (label, H),
                 (f"Heading {chr(64 + p) * 3}", HEAVY)]
        words += _page(p, COLS_A, above, [("Walls", HEAVY), [f"Item {p}", "m", "1", "2", "2"]])
    return words


def _many_sections(fmt, sections, per_section):
    """More sections than pages per section; the page number restarts at 1
    in every section, so the SECTION field has the most distinct values."""
    return [fmt.format(b=b, p=p) for b in range(1, sections + 1)
            for p in range(1, per_section + 1)]


@pytest.mark.parametrize("fmt, sections, per_section", [
    ("Bill {b} Page {p}", 6, 3),
    ("Page {p} Bill {b}", 7, 2),   # the page field first: no tie-break helps
], ids=["bill-then-page", "page-then-bill"])
def test_page_number_is_the_fastest_moving_field(fmt, sections, per_section):
    labels = _many_sections(fmt, sections, per_section)
    words = _labelled_pages(labels)
    printed = _printed(words)
    found = g.page_sections(printed, g.running_lines(printed, len(labels)))
    # one section key per bill, every page of a bill in it
    assert found == {i + 1: (str(i // per_section + 1),) for i in range(len(labels))}
    assert len(set(found.values())) == sections


@pytest.mark.parametrize("fmt", ["Volume 2 Page {p}", "Page {p} Volume 2"],
                         ids=["constant-then-page", "page-then-constant"])
def test_restarting_page_number_alone_is_not_section_information(fmt):
    # the page number restarts (as if per section) but nothing else in the
    # label varies: there is no section field to key the rows by
    labels = [fmt.format(p=p) for _ in range(6) for p in range(1, 4)]
    words = _labelled_pages(labels)
    printed = _printed(words)
    assert g.page_sections(printed, g.running_lines(printed, len(labels))) == {}
    assert _unchanged(words)


def _repeating_title(title, label_misread_on=None):
    """A real section title printed in the title band on every page of its
    section; the first section spans most of the document's pages, so its
    title repeats like a running line. Optionally one page's label is
    misread into a near-identical line."""
    words: list[g.Word] = []
    pages = [(1, "1", title[0], "Basins"), (2, "1", title[0], "Sinks"), (3, "1", title[0], "Baths"),
             (4, "2", title[1], "Doors"), (5, "2", title[1], "Frames")]
    for p, b, t, sub in pages:
        label = f"Part {b} Leaf {b}-{p}"
        if p == label_misread_on:
            label = f"Part {b} Leaf {b}-I"
        words += _page(p, COLS_B, [("Riverside Clinic Phase Two", H), (label, H), (t, HEAVY)],
                       [(sub, HEAVY), [f"Item {p}", "1", "nr", "2", "2"]])
    return words


@pytest.mark.parametrize("title, misread", [
    (("SANITARY INSTALLATIONS", "ELECTRICAL INSTALLATIONS"), None),
    (("JOINERY", "IRONMONGERY"), 2),  # near-identical misread label stays furniture
], ids=["repeating-title", "repeating-title-misread-label"])
def test_repeating_title_band_line_is_a_title_not_furniture(title, misread):
    words = _repeating_title(title, misread)
    printed = _printed(words)
    running = g.running_lines(printed, 5)
    assert g._masked(title[0]) in running.masks  # it repeats on most pages ...
    assert running.page_label == "part # leaf #-#"
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [  # ... and is still the section title
        (title[0], "Basins"), (title[0], "Sinks"), (title[0], "Baths"),
        (title[1], "Doors"), (title[1], "Frames"),
    ]


def _page_number_only_bill(sections):
    """A page label that is only a page number; the section title prints in the
    title band on the FIRST page of each section and nowhere else. ``sections``
    lists ``(title, page_count)`` pairs, so a section spans several pages and
    its title must be carried onto its continuation pages."""
    run = ("Bridgeport Viaduct", H)
    words: list[g.Word] = []
    page = 1
    for title, count in sections:
        for i in range(count):
            above = [run, (f"Page {page}", H)]
            if i == 0:
                above.append((title, HEAVY))
            words += _page(page, COLS_A, above,
                           [("Bulk", HEAVY), [f"Item {page}", "m3", "1", "2", "2"]])
            page += 1
    return words


@pytest.mark.parametrize("sections", [
    [("EARTHWORKS", 3), ("CONCRETE", 2)],
    [("DEMOLITION", 2), ("DRAINAGE", 3), ("PAVING", 1)],
], ids=["two-sections", "three-sections"])
def test_page_number_only_label_keys_sections_by_title_band(sections):
    words = _page_number_only_bill(sections)
    printed = _printed(words)
    running = g.running_lines(printed, len({w.page for w in words}))
    assert running.page_label == "page #"
    assert g.page_sections(printed, running) == {}  # the label carries no section field
    rows = g.extract_bill_rows(words)
    want = [(title, "Bulk") for title, count in sections for _ in range(count)]
    assert [r.path for r in rows] == want


def _page_number_only_no_title(pages):
    """A page label that is only a page number and NO title band: the heading
    above the header is the running line and the label, nothing else."""
    run = ("Bridgeport Viaduct", H)
    words: list[g.Word] = []
    for p in range(1, pages + 1):
        words += _page(p, COLS_A, [run, (f"Page {p}", H)],
                       [("Bulk", HEAVY), [f"Item {p}", "m3", "1", "2", "2"]])
    return words


@pytest.mark.parametrize("pages", [4, 5], ids=["four-pages", "five-pages"])
def test_page_number_only_label_without_title_band_is_left_alone(pages):
    words = _page_number_only_no_title(pages)
    printed = _printed(words)
    running = g.running_lines(printed, pages)
    assert running.page_label == "page #"
    assert g.page_sections(printed, running) == {}
    assert _unchanged(words)  # no title band: no section information at all
