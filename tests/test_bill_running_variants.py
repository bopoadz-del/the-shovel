"""A running line OCR reads several ways is still one running line.

``running_lines`` decides a running line by repetition: a masked line on
most pages. A scan misreads a letter of the running header or the page label
on some pages, so the one printed line arrives as several masked forms, and
each form alone can fall short of most pages. These tests build invented
scanned bills from word boxes; every decision is tested on two inputs, one
each side.
"""

from __future__ import annotations

import pytest

from app.core import bill_table_geometry as g

H = 20.0
HEAVY = 28.0
CW = 9.0
LINE = 34.0
COLS = [("Description", 40), ("Unit", 600), ("Qty", 700), ("Rate", 820), ("Amount", 960)]


def _text(words, text, x, y, page, h=H):
    cur = x
    for tok in text.split():
        w = CW * len(tok)
        words.append(g.Word(tok, page, cur, y, w, h))
        cur += w + CW


def _page(page, above, body):
    words: list[g.Word] = []
    y = 60.0
    for text, h in above:
        _text(words, text, COLS[0][1], y, page, h)
        y += LINE
    for label, x in COLS:
        _text(words, label, x, y, page)
    y += LINE
    for line in body:
        if isinstance(line, tuple):
            _text(words, line[0], COLS[0][1], y, page, line[1])
        else:
            for (_lbl, x), cell in zip(COLS, line):
                if cell:
                    _text(words, cell, x, y, page)
        y += LINE
    return words


def _printed(words):
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


# Six pages, two bills of three pages; the title prints on each bill's first
# page. ``runs`` and ``labels`` give each page's running line and page label.
def _bill(runs, labels):
    titles = {1: "EARTHWORKS", 4: "DRAINAGE"}
    heads = {1: "Excavation", 2: "Excavation (cont'd)", 3: "Filling",
             4: "Pipes", 5: "Pipes (cont'd)", 6: "Manholes"}
    words: list[g.Word] = []
    for p in range(1, 7):
        above = [(runs[p - 1], H), (labels[p - 1], H)]
        if p in titles:
            above.append((titles[p], HEAVY))
        words += _page(p, above, [(heads[p], HEAVY), [f"Item {p}", "m3", "10", "5", "50"]])
    return words


STEADY_LABELS = ["Bill 1 Page 1-1", "Bill 1 Page 1-2", "Bill 1 Page 1-3",
                 "Bill 2 Page 2-4", "Bill 2 Page 2-5", "Bill 2 Page 2-6"]
# One running line read three ways (a letter misread), two pages each:
# no single form is on most of the six pages.
MISREAD_RUN = ["Westbrook Terminal Works", "Westbrook Terminal Works",
               "Wcstbrook Terminal Works", "Wcstbrook Terminal Works",
               "Westbrook Terrninal Works", "Westbrook Terrninal Works"]
# Three different lines, two pages each: the same page spread, not one line.
THREE_LINES = ["Westbrook Terminal Works", "Westbrook Terminal Works",
               "Harbour Quay Extension", "Harbour Quay Extension",
               "Northfield Rail Sidings", "Northfield Rail Sidings"]
# The page label read two ways, each on three of six pages, every digit kept.
MISREAD_LABELS = ["Bill 1 Page 1-1", "BiIl 1 Page 1-2", "Bill 1 Page 1-3",
                  "BiIl 2 Page 2-4", "Bill 2 Page 2-5", "BiIl 2 Page 2-6"]


@pytest.mark.parametrize("runs, is_running", [
    (MISREAD_RUN, True),
    (THREE_LINES, False),
], ids=["one-line-misread", "three-different-lines"])
def test_a_misread_running_line_is_counted_once(runs, is_running):
    running = g.running_lines(_printed(_bill(runs, STEADY_LABELS)), 6)
    assert all(running.holds(r) for r in runs) is is_running
    assert ("westbrook terminal works" in running.masks) is is_running


@pytest.mark.parametrize("labels, label_on_every_page", [
    (MISREAD_LABELS, True),
    (STEADY_LABELS, True),
], ids=["label-read-two-ways", "label-read-one-way"])
def test_a_misread_page_label_is_still_the_label(labels, label_on_every_page):
    printed = _printed(_bill(MISREAD_RUN, labels))
    running = g.running_lines(printed, 6)
    assert running.page_label == "bill # page #-#"
    assert all(running.is_label(t) for t in labels) is label_on_every_page
    sections = g.page_sections(printed, running)
    assert sections == {1: ("1", "1"), 2: ("1", "1"), 3: ("1", "1"),
                        4: ("2", "2"), 5: ("2", "2"), 6: ("2", "2")}


@pytest.mark.parametrize("page4, read_on_page4", [
    ("BiIl 2 Page 2-4", True),   # misread letter, every digit kept: read
    ("BiIl 2 Page 2-A", False),  # a digit read as a letter: not read
], ids=["digits-kept", "digit-lost"])
def test_a_misread_label_is_read_for_its_fields_only_with_every_digit(page4, read_on_page4):
    labels = list(MISREAD_LABELS)
    labels[3] = page4
    printed = _printed(_bill(MISREAD_RUN, labels))
    running = g.running_lines(printed, 6)
    assert running.is_label(page4) is read_on_page4
    # Either way page 4 belongs to bill 2: read from its own label, or
    # inherited from the page before it when that label lost a digit.
    sections = g.page_sections(printed, running)
    assert sections[4] == (("2", "2") if read_on_page4 else ("1", "1"))


@pytest.mark.parametrize("runs, labels", [
    (MISREAD_RUN, MISREAD_LABELS),
    (["Westbrook Terminal Works"] * 6, STEADY_LABELS),
], ids=["misread-furniture", "steady-furniture"])
def test_misread_furniture_is_never_a_path_level(runs, labels):
    rows = g.extract_bill_rows(_bill(runs, labels))
    paths = {r.fields["description"]: r.path for r in rows}
    furniture = {g._masked(t) for t in runs + labels}
    for path in paths.values():
        assert not furniture & {g._masked(level) for level in path}
    assert paths["Item 1"][0] == "EARTHWORKS"
    assert paths["Item 5"][0] == "DRAINAGE"


@pytest.mark.parametrize("runs, stays", [
    (THREE_LINES, True),
    (MISREAD_RUN, False),
], ids=["different-lines-stay", "misread-line-goes"])
def test_lines_that_only_share_a_page_spread_are_not_merged(runs, stays):
    running = g.running_lines(_printed(_bill(runs, STEADY_LABELS)), 6)
    assert (not running.holds(runs[2])) is stays
