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
# One running line read three ways (a letter misread), two pages each and
# scattered over the bill: no single form is on most of the six pages.
MISREAD_RUN = ["Westbrook Terminal Works", "Wcstbrook Terminal Works",
               "Westbrook Terrninal Works", "Westbrook Terminal Works",
               "Wcstbrook Terminal Works", "Westbrook Terrninal Works"]
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


# Two different headings alike in their letters, each reprinted on pages of
# its own section: two lines, never one misread line, however the pages fall.
# A misreading is one glyph confusion; it is one line whether it is scattered
# among the line's pages or holds a run of pages of its own.
BLOCKS = [0, 0, 0, 1, 1, 1]
SCATTERED = [0, 1, 0, 1, 1, 0]
REPEATED = [0, 0, 1, 1, 0, 1]


@pytest.mark.parametrize("first, second, layout, one_line", [
    ("Internal Finishes", "External Finishes", SCATTERED, False),
    ("Internal Walls", "External Walls", SCATTERED, False),
    ("Internal Doors", "Internal Floors", BLOCKS, False),
    ("External Piling", "External Tiling", BLOCKS, False),
    ("Internal Doors", "Internal Floors", SCATTERED, False),
    ("Internal Doors", "Internal Floors", REPEATED, False),
    ("External Piling", "External Tiling", REPEATED, False),
    ("Westbrook Terminal Works", "Westbrook Terrninal Works", BLOCKS, True),
    ("Westbrook Terminal Works", "Wcstbrook Terminal Works", BLOCKS, True),
    ("Westbrook Terminal Works", "Westbrook Terrninal Works", SCATTERED, True),
    ("Westbrook Terminal Works", "Wcstbrook Terminal Works", SCATTERED, True),
    ("Westbrook Terminal", "Westbrook TerminaI", REPEATED, True),
], ids=["finishes-two-headings", "walls-two-headings", "doors-floors-two-sections",
        "piling-tiling-two-sections", "doors-floors-scattered", "doors-floors-repeated",
        "piling-tiling-repeated", "slip-in-a-run-of-its-own", "e-for-c-in-a-run-of-its-own",
        "glyph-read-as-two", "glyph-read-as-one", "slip-at-the-end-of-a-word"])
def test_alike_headings_that_each_recur_are_pooled_only_when_one_is_misread(
        first, second, layout, one_line):
    runs = [(first, second)[i] for i in layout]
    words = _bill(runs, STEADY_LABELS)
    running = g.running_lines(_printed(words), 6)
    assert bool({g._masked(first), g._masked(second)} & running.masks) is one_line
    rows = g.extract_bill_rows(words)
    paths = {r.fields["description"]: r.path for r in rows}
    top = paths["Item 1"][0].split(" / ")[0]
    assert top == ("EARTHWORKS" if one_line else first)


@pytest.mark.parametrize("a, b, misread", [
    ("bill", "biil", True),          # l read as i
    ("terminal", "terrninal", True),  # m read as rn
    ("terminal", "terminai", True),   # l read as i, the last glyph
    ("works", "wcrks", True),         # o read as c
    ("terminal", "termnal", False),   # a glyph dropped: not a confusion
    ("pit", "pits", False),           # a glyph added: not a confusion
    ("doors", "floors", False),       # d for fl: another word
    ("piling", "tiling", False),      # p for t: another word
    ("works", "worms", False),        # k for m: another word
    ("internal", "external", False),  # two glyphs for two: another word
    ("a", "b", False),                # the whole word changed
    ("roads", "roofs", False),        # two separate changes
    ("terminal", "terxyzinal", False),  # one glyph read as three
    ("ce", "ee", False),              # a confusion, but half the word: another word
    ("cee", "eee", True),             # the same confusion under half the word
    ("pit", "pots", False),           # a glyph changed and one added
], ids=["l-as-i", "m-as-rn", "last-glyph", "o-as-c", "dropped", "added-at-end",
        "d-for-fl", "p-for-t", "k-for-m", "two-for-two", "whole-word", "two-changes",
        "one-for-three", "half-the-word", "under-half", "changed-and-added"])
def test_a_word_is_misread_by_one_glyph_only(a, b, misread):
    assert g._slipped_word(a, b) is misread
    assert g._slipped_word(b, a) is misread


@pytest.mark.parametrize("a, b, alike", [
    ("a" * 17 + "bcd", "a" * 17 + "xyz", True),    # 34/40 = 0.85, on the bound
    ("a" * 17 + "bcd", "a" * 16 + "wxyz", False),  # 32/40 = 0.80, under it
    # Different lengths: the length bound itself is 34/40 = 0.85, on it.
    ("x" * 17, "x" * 17 + "y" * 6, True),
    ("x" * 17, "x" * 17 + "y" * 7, False),          # 34/41, under it
], ids=["on-the-bound", "under-the-bound", "lengths-on-the-bound", "lengths-under-the-bound"])
def test_likeness_holds_at_its_bound(a, b, alike):
    assert g._alike(a, b) is alike
    assert g._alike(b, a) is alike


@pytest.mark.parametrize("anchor, other", [
    ("westbrook terminal works", "westbrook terrninal works"),  # the misreading is longer
    ("westbrook terrninal works", "westbrook terminal works"),  # the misreading is shorter
    ("westbrook terminal", "westbrook terminai"),  # the slip is the word's last glyph
    ("westbrook terminai", "westbrook terminal"),  # either way round
], ids=["longer-reading", "shorter-reading", "last-glyph", "last-glyph-reversed"])
def test_a_misreading_is_found_whichever_form_is_longer(anchor, other):
    found = g._misread_forms([anchor], [anchor, other, "harbour quay extension"])
    assert sorted(found[anchor]) == sorted([anchor, other])


PARTS = ["Part 1 Works"] * 3 + ["Part 2 Works"] * 3
CONTRACT = ["Contract 4 Westbrook Works", "Contract 4 Wcstbrook Works"] * 3


@pytest.mark.parametrize("runs, labels, form, kept", [
    # A reading of the label is the label: it stays a running line.
    (MISREAD_RUN, MISREAD_LABELS, "biil # page #-#", True),
    # A line whose digits change with the section is a heading, not furniture.
    (PARTS, STEADY_LABELS, "part # works", False),
    # A misread line whose digits never change does not vary: still furniture.
    (CONTRACT, STEADY_LABELS, "contract # westbrook works", True),
], ids=["label-reading", "section-number", "constant-number-misread"])
def test_which_digit_lines_stay_running_lines(runs, labels, form, kept):
    running = g.running_lines(_printed(_bill(runs, labels)), 6)
    assert running.page_label == "bill # page #-#"
    assert (form in running.masks) is kept


def _lowered_label_bill(page4):
    # Page 4 prints one extra line above its label, so its label sits lower.
    words: list[g.Word] = []
    for p in range(1, 7):
        label = page4 if p == 4 else STEADY_LABELS[p - 1]
        above = [(MISREAD_RUN[p - 1], H)] + ([("Addendum", H)] if p == 4 else []) + [(label, H)]
        words += _page(p, above, [("Excavation", HEAVY), [f"Item {p}", "m3", "10", "5", "50"]])
    return words


@pytest.mark.parametrize("page4, read", [
    ("BiIl 2 Page 2-4", True),   # a misread label keeping its digits is read
    ("BiIl 2 Page 2-A", False),  # one that lost a digit is not
], ids=["digits-kept", "digit-lost"])
def test_the_furniture_band_and_diagnostic_read_a_misread_label(page4, read):
    printed = _printed(_lowered_label_bill(page4))
    running = g.running_lines(printed, 6)
    band = g.furniture_band(printed, running)
    assert (band[4] > band[1]) is read
    diag = g.geometry_diagnostic(printed, 6)
    assert (4 in diag["page_label"]) is read


@pytest.mark.parametrize("headings, reprints", [(400, 1), (400, 2)],
                         ids=["one-page-headings", "recurring-headings"])
def test_headings_are_not_compared_pairwise(headings, reprints, monkeypatch):
    # Each page prints its label and distinct headings, alike in length and
    # word count, each on one page or reprinted on two: the work grows with
    # the headings, not with their pairs.
    pages = 100
    printed = [g._Label(f"Bill 1 Page 1-{p}", 40, H, page=p) for p in range(1, pages + 1)]
    printed += [g._Label(f"Heading {chr(97 + i % 26)}{chr(97 + i // 26 % 26)} item work",
                         40, H, page=1 + (i * reprints + r) % pages)
                for i in range(headings) for r in range(reprints)]
    calls = []
    real = g._misread
    monkeypatch.setattr(g, "_misread", lambda a, b: calls.append(1) or real(a, b))
    running = g.running_lines(printed, pages)
    assert running.page_label == "bill # page #-#"
    assert len(calls) <= 2 * headings


@pytest.mark.parametrize("spellings, label", [
    # A line read three ways on every page has more texts than the page
    # label, but its digits take three values to the label's six: the label
    # is the line whose digits vary most, not the one read most ways.
    (["Works", "Wcrks", "Werks"], "page #"),
    (["Works"], "page #"),
], ids=["misspelled-line", "one-spelling"])
def test_the_page_label_is_the_line_whose_digits_vary_most(spellings, label):
    printed = [g._Label(f"Page {p}", 40, H, page=p) for p in range(1, 7)]
    printed += [g._Label(f"Part {1 + (p - 1) // 2} {word}", 40, H, page=p)
                for p in range(1, 7) for word in spellings]
    assert g.running_lines(printed, 6).page_label == label


# The running line read right on three of six pages; on each other page a
# different one-off misreading. One-off readings are not pooled, so the line
# falls short of most pages; the same misreadings each seen twice are pooled.
ONE_OFF_READINGS = ["Westbrook Terminal Works"] * 3 + [
    "Wcstbrook Terminal Works", "Westbrook Terrninal Works", "Westbrook Terminal Wcrks"]


@pytest.mark.parametrize("runs, is_running", [
    (ONE_OFF_READINGS, False),
    (MISREAD_RUN, True),
], ids=["one-off-readings", "recurring-readings"])
def test_only_recurring_readings_add_their_pages(runs, is_running):
    running = g.running_lines(_printed(_bill(runs, STEADY_LABELS)), 6)
    assert ("westbrook terminal works" in running.masks) is is_running


@pytest.mark.parametrize("b, misread", [
    ("wcstbrook terminal works", True),    # one word slipped
    ("wcstbrook terrninal works", False),  # two words slipped: counts alone
    ("westbrook terminal wcrks", True),    # the last word slipped
    ("westbrook terminal wcrks x", False),  # and a word added: another line
], ids=["one-word", "two-words", "last-word", "word-added"])
def test_a_misreading_differs_from_its_line_in_one_word(b, misread):
    line = "westbrook terminal works"
    assert g._misread(line, b) is misread
    assert g._misread(b, line) is misread


@pytest.mark.parametrize("a, b, misread", [
    ("bill # page #", "biil # page #", True),  # 22/24 alike: the line misread
    ("bill #", "biil #", False),               # 10/12 alike: under the likeness
], ids=["long-enough", "too-short"])
def test_a_one_glyph_slip_still_needs_the_line_likeness(a, b, misread):
    assert g._misread(a, b) is misread


# A bill that repeats its trade sections per building: a steady running line
# and page label, a building title in heavy type on its first page, and a
# trade heading as a body heading. Two trade headings one letter apart each
# hold the same pages in both buildings; they are sections, never one line.
def _buildings_bill(trades):
    words: list[g.Word] = []
    for p in range(1, 9):
        building = "BUILDING A" if p <= 4 else "BUILDING B"
        trade = trades[((p - 1) // 2) % 2]
        above = [("Harbour Quay Extension", H), (f"Bill 1 Page 1-{p}", H)]
        if p in (1, 5):
            above.append((building, HEAVY))
        words += _page(p, above, [(trade, HEAVY), [f"Item {p}", "m3", "10", "5", "50"]])
    return words


@pytest.mark.parametrize("trades", [
    ("Internal Doors", "Internal Floors"),   # one letter apart
    ("Concrete Walls", "Concrete Slabs"),    # not alike at all
], ids=["one-letter-apart", "unlike"])
def test_repeated_trade_sections_keep_their_headings(trades):
    rows = g.extract_bill_rows(_buildings_bill(trades))
    paths = {r.fields["description"]: " / ".join(r.path) for r in rows}
    assert trades[0] in paths["Item 1"] and trades[1] in paths["Item 3"]
    assert trades[0] in paths["Item 5"] and trades[1] in paths["Item 7"]
    running = g.running_lines(_printed(_buildings_bill(trades)), 8)
    assert not {g._masked(t) for t in trades} & running.masks
