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
def test_label_without_section_information_uses_title_band(labels):
    words = _no_section_info(labels)
    printed = _printed(words)
    running = g.running_lines(printed, 4)
    assert running.masks  # the running header IS found ...
    assert g.page_sections(printed, running) == {}  # ... but gives no sections
    # ... so the title band becomes the section: each page's title is its
    # section, and the constant "Walls" line on every page is furniture.
    assert [r.path for r in g.extract_bill_rows(words)] == [
        ("CONCRETE",), ("FORMWORK",), ("MASONRY",), ("METALWORK",),
    ]


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
def test_restarting_page_number_alone_falls_back_to_title_band(fmt):
    # the page number restarts (as if per section) but nothing else in the
    # label varies: there is no section field to key the rows by, so the
    # title band carries the sections.
    labels = [fmt.format(p=p) for _ in range(6) for p in range(1, 4)]
    words = _labelled_pages(labels)
    printed = _printed(words)
    assert g.page_sections(printed, g.running_lines(printed, len(labels))) == {}
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [  # one section per page, from the title
        (f"Heading {chr(64 + p) * 3}",) for p in range(1, len(labels) + 1)
    ]


def _titled_first_pages(pages):
    """Pages given as (page, label, title-or-None): a constant running line,
    a page-number-only label, and a title band printed only where a title is
    given (no body heading, so each row's path is its section title)."""
    words: list[g.Word] = []
    for p, label, title in pages:
        above = [("Riverside Depot", H), (label, H)]
        if title:
            above.append((title, HEAVY))
        words += _page(p, COLS_A, above, [[f"Item {p}", "m", "1", "2", "2"]])
    return words


@pytest.mark.parametrize("pages, expected", [
    ([
        (1, "Page 1", "SUBSTRUCTURE"), (2, "Page 2", None), (3, "Page 3", None),
        (4, "Page 4", "SUPERSTRUCTURE"), (5, "Page 5", None),
    ], [
        ("SUBSTRUCTURE",), ("SUBSTRUCTURE",), ("SUBSTRUCTURE",),
        ("SUPERSTRUCTURE",), ("SUPERSTRUCTURE",),
    ]),
    ([
        (1, "Page 1", "SUBSTRUCTURE"), (2, "Page 2", "SUBSTRUCTURE"), (3, "Page 3", None),
        (4, "Page 4", "SUPERSTRUCTURE"), (5, "Page 5", None),
    ], [
        ("SUBSTRUCTURE",), ("SUBSTRUCTURE",), ("SUBSTRUCTURE",),
        ("SUPERSTRUCTURE",), ("SUPERSTRUCTURE",),
    ]),
], ids=["title-first-page-only", "title-repeats-on-a-continuation-page"])
def test_title_band_sections_inherit_and_repeat_without_new_section(pages, expected):
    words = _titled_first_pages(pages)
    printed = _printed(words)
    running = g.running_lines(printed, 5)
    assert running.page_label == "page #"
    assert g.page_sections(printed, running) == {}  # the label is a page number only
    assert [r.path for r in g.extract_bill_rows(words)] == expected


@pytest.mark.parametrize("labels", [
    [f"Page {p}" for p in range(1, 5)],       # a page number only
    ["Contract 12 Volume 3"] * 4,             # constant figures, nothing varies
], ids=["page-number-only", "constant-figures"])
def test_running_header_without_title_band_is_left_alone(labels):
    # a running header and a page label, but no title printed above the
    # header: there is no section to read, so the paths stay as built
    words: list[g.Word] = []
    for p in range(1, 5):
        above = [("Eastgate Pumping Station", H), (labels[p - 1], H)]
        words += _page(p, COLS_A, above, [[f"Item {p}", "m", "1", "2", "2"]])
    printed = _printed(words)
    running = g.running_lines(printed, 4)
    assert running.masks
    assert g.page_sections(printed, running) == {}
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [tuple(lab.text for lab in r.labels) for r in rows]


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


def _section_heading_words(prefix, pages=6):
    """A constant running header, a page-number-only label, a two-state
    title band, and body headings ``{prefix} N`` whose masked form appears
    on most pages while no one heading repeats (a section heading per bill,
    not a running header)."""
    words: list[g.Word] = []
    layout = [
        (1, "GENERAL ITEMS", 1, "Item 1"),
        (2, "GENERAL ITEMS", 1, "Item 2"),
        (3, "GENERAL ITEMS", 2, "Item 3"),
        (4, "SUPPLEMENTARY", 2, "Item 4"),
        (5, "SUPPLEMENTARY", 3, "Item 5"),
        (6, "SUPPLEMENTARY", 3, "Item 6"),
    ]
    for p, band, n, item in layout:
        above = [("Contractor Limited", H), (f"Page {p}", H), (band, HEAVY)]
        body = [(f"{prefix} {n}", HEAVY), [item, "m", "1", "2", "2"]]
        words += _page(p, COLS_A, above, body)
    return words


@pytest.mark.parametrize("prefix", ["BILL NO.", "SECTION"], ids=["bill-no", "section"])
def test_section_heading_is_not_a_running_line(prefix):
    printed: list[g._Label] = []
    for p, n in ((1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (6, 3)):
        printed += [
            _lab("Contractor Limited", page=p),
            _lab(f"Page {p}", page=p),
            _lab(f"{prefix} {n}", page=p),
        ]
    running = g.running_lines(printed, 6)
    assert g._masked(f"{prefix} 1") not in running.masks  # a section heading, not furniture
    assert "page #" in running.masks                      # the page label stays furniture
    assert running.page_label == "page #"


@pytest.mark.parametrize("prefix, expected", [
    ("BILL NO.", [
        ("GENERAL ITEMS / BILL NO. 1",),
        ("GENERAL ITEMS / BILL NO. 1",),
        ("GENERAL ITEMS / BILL NO. 2",),
        ("SUPPLEMENTARY / BILL NO. 2",),
        ("SUPPLEMENTARY / BILL NO. 3",),
        ("SUPPLEMENTARY / BILL NO. 3",),
    ]),
    ("SECTION", [
        ("GENERAL ITEMS", "SECTION 1"),
        ("GENERAL ITEMS", "SECTION 1"),
        ("GENERAL ITEMS", "SECTION 2"),
        ("SUPPLEMENTARY", "SECTION 2"),
        ("SUPPLEMENTARY", "SECTION 3"),
        ("SUPPLEMENTARY", "SECTION 3"),
    ]),
], ids=["bill-no", "section"])
def test_section_heading_survives_in_the_path(prefix, expected):
    """A repeated per-bill heading is not a running line and survives in the
    path: a bill heading (the ``_HEADING_RE`` grammar) joins the title band
    as the section title wherever it is printed; a generic section heading is
    a body sub-level."""
    words = _section_heading_words(prefix)
    assert [r.path for r in g.extract_bill_rows(words)] == expected

def _heading_is_page_label_words(heading_fn, title, pages=6):
    """A constant running header, a constant title line, and a per-bill
    heading whose number is the only varying digit on the page, so
    ``running_lines`` reads the heading as the page label (which then
    carries no section field of its own)."""
    words: list[g.Word] = []
    layout = [(1, 1, "Item 1"), (2, 1, "Item 2"), (3, 15, "Item 3"),
              (4, 15, "Item 4"), (5, 16, "Item 5"), (6, 16, "Item 6")]
    for p, n, item in layout:
        above = [("Contractor Limited", H), (heading_fn(n), HEAVY), (title, HEAVY)]
        words += _page(p, COLS_A, above, [[item, "m", "1", "2", "2"]])
    return words


@pytest.mark.parametrize("heading_fn, title", [
    (lambda n: f"BILL NO. {n}", "GENERAL SUMMARY"),
    (lambda n: f"Bill Number {n}", "BILL OF QUANTITIES"),
], ids=["bill-no", "bill-number"])
def test_bill_heading_survives_as_a_path_level_when_it_is_the_page_label(heading_fn, title):
    """A per-bill heading read as the page label must still be a path level.
    It is a section heading wherever it is printed, even when its masked form
    is the page's only varying digit and it would otherwise be furniture."""
    words = _heading_is_page_label_words(heading_fn, title)
    printed = _printed(words)
    running = g.running_lines(printed, 6)
    assert g.page_sections(printed, running) == {}
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
    ]


def _footer_page_label_words(heading_fn, title, pages=6):
    """A constant running header, a bill heading with its constant title
    printed above the column header, and the page label printed below the
    header (a footer). The heading's title line repeats on every page, so the
    footer label's vertical position must not make it page furniture."""
    words: list[g.Word] = []
    layout = [(1, 1, "Item 1"), (2, 1, "Item 2"), (3, 15, "Item 3"),
              (4, 15, "Item 4"), (5, 16, "Item 5"), (6, 16, "Item 6")]
    for p, n, item in layout:
        above = [("Contractor Limited", H), (heading_fn(n), HEAVY), (title, HEAVY)]
        body = [[item, "m", "1", "2", "2"], (f"Page {p}", H)]
        words += _page(p, COLS_A, above, body)
    return words


@pytest.mark.parametrize("heading_fn, title", [
    (lambda n: f"BILL NO. {n}", "GENERAL SUMMARY"),
    (lambda n: f"Bill Number {n}", "BILL OF QUANTITIES"),
], ids=["bill-no", "bill-number"])
def test_heading_title_line_survives_a_footer_page_label(heading_fn, title):
    """A bill heading's title line (the line directly below the number) is
    part of the heading, so it stays a path level even when the page label is
    a footer and the title line repeats like a running line."""
    words = _footer_page_label_words(heading_fn, title)
    printed = _printed(words)
    running = g.running_lines(printed, 6)
    assert running.page_label == "page #"
    assert g.page_sections(printed, running) == {}
    assert g._masked(title) in running.masks  # it repeats on most pages ...
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [  # ... yet it is still the heading title
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
    ]


def _below_header_heading_words(heading_fn, title):
    """A constant running header, the column header, then a bill heading and
    its constant title printed BELOW the column header (above_header=False),
    and the page label as a footer. The heading is the section wherever it
    is printed."""
    words: list[g.Word] = []
    layout = [(1, 1, "Item 1"), (2, 1, "Item 2"), (3, 15, "Item 3"),
              (4, 15, "Item 4"), (5, 16, "Item 5"), (6, 16, "Item 6")]
    for p, n, item in layout:
        above = [("Contractor Limited", H)]
        body = [(heading_fn(n), HEAVY), (title, HEAVY),
                [item, "m", "1", "2", "2"], (f"Page {p}", H)]
        words += _page(p, COLS_A, above, body)
    return words


@pytest.mark.parametrize("heading_fn, title", [
    (lambda n: f"BILL NO. {n}", "GENERAL SUMMARY"),
    (lambda n: f"Bill Number {n}", "BILL OF QUANTITIES"),
], ids=["bill-no", "bill-number"])
def test_heading_below_the_header_is_still_a_path_level(heading_fn, title):
    """A bill heading printed below the column header (above_header=False) is
    still a section title: the title-band fallback collects it regardless of
    above/below-header placement, so it gives the same path as the same
    heading printed above the header."""
    words = _below_header_heading_words(heading_fn, title)
    printed = _printed(words)
    running = g.running_lines(printed, 6)
    assert g.page_sections(printed, running) == {}
    heading = next(lab for lab in printed
                   if g._HEADING_RE.match(lab.text) and not g._CARRIED_RE.search(lab.text))
    assert heading.above_header is False  # printed below the column header
    rows = g.extract_bill_rows(words)
    assert [r.path for r in rows] == [
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(1)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(15)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
        (f"{heading_fn(16)} / {title}",),
    ]


@pytest.mark.parametrize("heading_fn", [
    lambda n: f"BILL NO. {n}",
    lambda n: f"Bill Number {n}",
], ids=["bill-no", "bill-number"])
def test_geometry_diagnostic_counts_sides_and_label_position(heading_fn):
    """The read-only diagnostic reports, per masked form, how many labels sit
    above vs below the column header, and the page label's vertical position."""
    words = _footer_page_label_words(heading_fn, "GENERAL SUMMARY")
    printed = _printed(words)
    diag = g.geometry_diagnostic(printed, 6)
    assert diag["page_label"][1]["above_header"] is False  # a footer label
    title_mask = g._masked("GENERAL SUMMARY")
    assert diag["masks"][title_mask]["above_header"] == 6
    assert diag["masks"][title_mask]["below_header"] == 0


def _note_pages(note_lines, first_line_case="upper"):
    """Pages sharing a running header and a page label whose section digit
    steps once; page 1 prints a heading, then a note the page wraps, then one
    measured row. Without the repetition there is no running header and the
    section pass leaves the read alone."""
    run = ("Harbour Works", H)
    note = [
        "The tenderer is to price every item at the rate stated and",
        "no separate claim for any additional quantity or rate will be",
        "entertained under this section of the works, and any item left",
        "blank is deemed included in the rates entered elsewhere in",
        "this bill of quantities for the whole of the works.",
    ][:note_lines]
    if first_line_case == "lower" and note:
        note = [note[0][0].lower() + note[0][1:]] + note[1:]
    pages = []
    for page, section in ((1, 7), (2, 7), (3, 8)):
        above = [run, ("Bill %d Page %d-%d" % (section, section, page), H)]
        if page == 1:
            body = [("PRICING NOTES", HEAVY)] + [(line, H) for line in note] + [
                ["Reinforced concrete in raft", "m3", "95", "380", "36,100"],
            ]
        else:
            body = [["Reinforced concrete in wall", "m3", "12", "380", "4,560"]]
        pages.append(_page(page, COLS_A, above, body))
    return [w for page_words in pages for w in page_words]


@pytest.mark.parametrize("n_lines", [4, 5, 9])
def test_a_wrapped_note_does_not_open_a_sheet(n_lines):
    """A note the page wraps over many lines joins into one long path level.
    It names no bill, section or sub-heading, so the measured rows stay under
    the heading that owns the table instead of opening a sheet named after a
    fragment of prose."""
    rows = g.extract_bill_rows(_note_pages(n_lines))
    assert rows, "the layout must still measure its rows"
    raft = next(r for r in rows if r.fields.get("description") == "Reinforced concrete in raft")
    assert raft.path == ("PRICING NOTES",), raft.path


@pytest.mark.parametrize("first_line", ["upper", "lower"])
def test_short_wrap_stays_and_a_lower_case_continuation_does_not(first_line):
    """A two-line item preamble is part of the path; the same two lines read
    as a continuation of the sentence above (lower-case opening) are not: a
    heading opens a sentence, a notice that wraps does not."""
    rows = g.extract_bill_rows(_note_pages(2, first_line))
    raft = next(r for r in rows if r.fields.get("description") == "Reinforced concrete in raft")
    if first_line == "upper":
        assert len(raft.path) == 2 and raft.path[0] == "PRICING NOTES", raft.path
        assert raft.path[1].startswith("The tenderer is to price"), raft.path
    else:
        assert raft.path == ("PRICING NOTES",), raft.path


@pytest.mark.parametrize("heading", ["PRICING NOTES", "2.9.3 Demolition and Site Clearance", "Kerbs"])
def test_a_heading_is_never_dropped_by_the_prose_rule(heading):
    """The rule reads shape, not words: an upper-case heading, a numbered
    sub-heading and a plain sub-heading all stay the level that names the
    rows' sheet."""
    pages = []
    for page, section in ((1, 7), (2, 7), (3, 8)):
        above = [("Harbour Works", H), ("Bill %d Page %d-%d" % (section, section, page), H)]
        if page == 1:
            body = [(heading, HEAVY), ["Reinforced concrete in raft", "m3", "95", "380", "36,100"]]
        else:
            body = [["Reinforced concrete in wall", "m3", "12", "380", "4,560"]]
        pages.append(_page(page, COLS_A, above, body))
    rows = g.extract_bill_rows([w for p in pages for w in p])
    raft = next(r for r in rows if r.fields.get("description") == "Reinforced concrete in raft")
    assert raft.path == (heading,), raft.path
