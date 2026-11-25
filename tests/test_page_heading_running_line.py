"""A page's heading line is never the document's page furniture (RQ-2).

``page_heading_lines`` files each split piece under the first heading-like line
of its page. A PDF that prints a running header (the document's title, a page
label) on top of every page then filed every piece of every page under that
same line, so the line told no page from another. The running line is found
by repetition only -- the bill reader's rule (``running_lines``): a line that
repeats, digits masked, on more than half of at least three pages.

Every text here is invented. Each decision is made on two inputs.
"""
from __future__ import annotations

import pytest

from app.core import chunk_bounds as cb


def _doc(pages: list[list[str]]) -> tuple[str, list[list[int]]]:
    """Join page line lists into one text with its ``[[page, offset], ...]``."""
    text = ""
    starts: list[list[int]] = []
    for number, lines in enumerate(pages, 1):
        starts.append([number, len(text)])
        text += "\n".join(lines) + "\n\n"
    return text, starts


_BODY = "The contractor shall keep the area dry and clean at all times."

_HEADER_A = "HARBOUR DEPOT EXTENSION - VOLUME 4 - SPECIFICATION"
_SECTIONS_A = ["2.1 EARTHWORKS", "2.2 DRAINAGE", "2.3 PAVING", "3.1 FENCING"]
_HEADER_B = "RIVERSIDE SCHOOL WORKS - PART B - TECHNICAL NOTES"
_SECTIONS_B = ["7 Ventilation", "8 Lighting", "9 Fire Alarm", "10 Lifts", "11 Signage"]


@pytest.mark.parametrize(
    "header, sections",
    [(_HEADER_A, _SECTIONS_A), (_HEADER_B, _SECTIONS_B)],
)
def test_running_header_does_not_name_every_page(header, sections):
    text, starts = _doc([[header, s, _BODY] for s in sections])
    got = cb.page_heading_lines(text, starts)
    assert got == {i: s for i, s in enumerate(sections, 1)}


@pytest.mark.parametrize(
    "label",
    ["PAGE {n} OF 9", "SHEET {n} - ISSUE C"],
)
def test_page_label_with_changing_number_is_furniture(label):
    sections = ["4.1 MASONRY", "4.2 RENDER", "4.3 TILING"]
    text, starts = _doc(
        [[label.format(n=n), s, _BODY] for n, s in enumerate(sections, 3)]
    )
    assert cb.page_heading_lines(text, starts) == {
        i: s for i, s in enumerate(sections, 1)
    }


@pytest.mark.parametrize("header", [_HEADER_A, _HEADER_B])
def test_page_with_only_furniture_keeps_it(header):
    # Page 3 has no heading of its own: it keeps the running line rather
    # than losing its heading altogether.
    pages = [
        [header, "5.1 STEELWORK", _BODY],
        [header, "5.2 CLADDING", _BODY],
        [header, _BODY],
    ]
    text, starts = _doc(pages)
    got = cb.page_heading_lines(text, starts)
    assert got == {1: "5.1 STEELWORK", 2: "5.2 CLADDING", 3: header}


@pytest.mark.parametrize("header", [_HEADER_A, _HEADER_B])
def test_two_pages_are_too_few_to_call_a_line_running(header):
    # Below three pages repetition proves nothing: the first heading-like
    # line names the page, as before.
    text, starts = _doc([[header, "1.1 GENERAL", _BODY], [header, "1.2 SITE", _BODY]])
    assert cb.page_heading_lines(text, starts) == {1: header, 2: header}


@pytest.mark.parametrize(
    "repeats_on, running",
    [(2, False), (3, True)],
)
def test_a_line_must_repeat_on_more_than_half_the_pages(repeats_on, running):
    # Five pages: a line on two of them is a heading that recurs, a line on
    # three of them is furniture.
    sections = ["6.1 DOORS", "6.2 WINDOWS", "6.3 HARDWARE", "6.4 GLAZING", "6.5 SEALANTS"]
    pages = [
        ([_HEADER_A] if i < repeats_on else []) + [s, _BODY]
        for i, s in enumerate(sections)
    ]
    text, starts = _doc(pages)
    got = cb.page_heading_lines(text, starts)
    expected_first = sections[0] if running else _HEADER_A
    assert got[1] == expected_first
    assert got[5] == sections[4]


@pytest.mark.parametrize("header, sections", [(_HEADER_A, _SECTIONS_A), (_HEADER_B, _SECTIONS_B)])
def test_split_pieces_open_on_their_own_section(monkeypatch, header, sections):
    # End to end through the piece heading: each split piece of a page opens
    # on that page's section line, not on the document's running header.
    monkeypatch.setenv("CHUNK_MAX_CHARS", "400")
    monkeypatch.setenv("CHUNK_MIN_CHARS", "0")
    long_body = " ".join(
        f"Clause {k} sets the tolerance at {k + 2} mm for this element." for k in range(12)
    )
    text, starts = _doc([[header, s, long_body] for s in sections])
    page_texts = [text[o:(starts[i + 1][1] if i + 1 < len(starts) else len(text))]
                  for i, (_, o) in enumerate(starts)]
    chunks: list[str] = []
    pages: list[int | None] = []
    parents: list[int | None] = []
    for number, body in enumerate(page_texts, 1):
        got = cb.bound_chunks([body], split_parents=parents)
        chunks.extend(got)
        pages.extend([number] * len(got))
    headings = cb.page_heading_lines(text, starts)
    out, _ = cb.attach_page_headings(chunks, pages, parents, headings)
    pieces = [(c, p) for c, p, par in zip(out, pages, parents) if par is not None]
    assert len(pieces) > len(sections)
    for chunk, page in pieces:
        assert chunk.split("\n", 1)[0] == sections[page - 1]
