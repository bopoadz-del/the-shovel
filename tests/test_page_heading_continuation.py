"""A page that only continues a section is filed under that section.

``page_heading_lines`` names each PDF page by its first heading-like line, and
``attach_page_headings`` opens every split piece of the page with it. A page
that carries on the section begun on the page before it has no heading of its
own, so its pieces were filed under nothing. Such a page is now filed under the
section still open at the end of the nearest earlier page: the last numbered or
markdown heading that page holds, else the line it was filed under. Every input
below is invented; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import attach_page_headings, page_heading_lines


def _doc(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, starts = "", []
    for number, page in enumerate(pages, 1):
        starts.append((number, len(text)))
        text += page + "\n"
    return text, starts


@pytest.mark.parametrize(
    "first,second,open_section",
    [
        (
            "5.1 Blinding\nBlinding is laid on the formation.\n5.2 Ground Slab\nThe slab is cast in bays",
            "not longer than six metres\nwith sealed joints between them.",
            "5.2 Ground Slab",
        ),
        (
            "## Roofing\nThe deck is primed.\n## Gutters\nGutters fall towards the outlets",
            "at not less than one in two hundred\nand are fixed with stainless brackets.",
            "## Gutters",
        ),
    ],
)
def test_continuation_page_takes_the_last_section_of_the_page_before(first, second, open_section):
    text, starts = _doc([first, second])
    headings = page_heading_lines(text, starts)
    assert headings[2] == open_section


@pytest.mark.parametrize(
    "pages,expected",
    [
        (
            ["7 Drainage\nPipes are bedded on gravel", "laid to the falls shown", "and tested before backfilling"],
            "7 Drainage",
        ),
        (
            ["B Painting\nSurfaces are rubbed down", "between every coat applied", "until an even finish shows"],
            "B Painting",
        ),
    ],
)
def test_section_carries_over_several_continuation_pages(pages, expected):
    text, starts = _doc(pages)
    headings = page_heading_lines(text, starts)
    assert headings[2] == expected
    assert headings[3] == expected


@pytest.mark.parametrize(
    "first,filed",
    [
        ("GENERAL REQUIREMENTS\nThe contractor keeps the site tidy", "GENERAL REQUIREMENTS"),
        ("SITE SECURITY\nHoardings are kept closed at night", "SITE SECURITY"),
    ],
)
def test_page_with_no_numbered_heading_carries_the_line_it_is_filed_under(first, filed):
    text, starts = _doc([first, "and the gates are locked by the guard"])
    assert page_heading_lines(text, starts)[2] == filed


@pytest.mark.parametrize(
    "first,section",
    [
        ("2.3 Mortar\nMortar is mixed by machine\nfor the full batch\nDRAFT ISSUE", "2.3 Mortar"),
        ("9 Glazing\nPanes are set in beads\nwith a bedding tape\nNOT FOR CONSTRUCTION", "9 Glazing"),
    ],
)
def test_capital_line_at_the_page_foot_does_not_replace_the_open_section(first, section):
    text, starts = _doc([first, "and sealed on the outer face"])
    assert page_heading_lines(text, starts)[2] == section


@pytest.mark.parametrize(
    "number,title",
    [("A-101", "Ground Floor Plan"), ("S-204", "Roof Framing")],
)
def test_drawing_sheet_name_is_not_carried_to_the_next_page(number, title):
    def title_block(page: str) -> dict:
        return {"drawing_number": number, "drawing_title": title} if number in page else {}

    text, starts = _doc([f"{number}\nscale 1 to 100", "notes continue here in plain words"])
    headings = page_heading_lines(text, starts, title_block)
    assert headings[1] == f"{number} - {title}"
    assert 2 not in headings


@pytest.mark.parametrize(
    "pages",
    [
        ["laid on a bed of sand", "3 Paving\nSlabs are cut neatly"],
        ["fixed at both ends", "8 Handrails\nRails are welded"],
    ],
)
def test_page_with_no_section_before_it_stays_unfiled(pages):
    text, starts = _doc(pages)
    headings = page_heading_lines(text, starts)
    assert 1 not in headings
    assert headings[2].split()[0].isdigit()


@pytest.mark.parametrize(
    "first,piece,section",
    [
        ("6.4 Screeds\nScreeds are laid level", "and cured under polythene sheets", "6.4 Screeds"),
        ("12 Doors\nDoors are hung on three hinges", "and fitted with a closer", "12 Doors"),
    ],
)
def test_split_piece_on_a_continuation_page_opens_on_the_open_section(first, piece, section):
    text, starts = _doc([first, piece])
    headings = page_heading_lines(text, starts)
    chunks, pages = attach_page_headings([piece], [2], [0], headings)
    assert chunks[0].split("\n", 1)[0] == section
    assert pages == [2]


@pytest.mark.parametrize(
    "first,section",
    [
        (
            "3.1 Mixing\nConcrete is mixed on site.\n2 bags of cement are used per batch.",
            "3.1 Mixing",
        ),
        (
            "7.4 Base Plates\nPlates are set on grout.\n12 bolts are fixed to each plate, as shown.",
            "7.4 Base Plates",
        ),
    ],
)
def test_sentence_opening_on_a_figure_does_not_become_the_open_section(first, section):
    text, starts = _doc([first, "and the work is checked by the engineer"])
    assert page_heading_lines(text, starts)[2] == section


@pytest.mark.parametrize(
    "section,number,title",
    [("4.0 Doors", "A-301", "Door Schedule"), ("6.2 Roof Drainage", "M-110", "Roof Plan")],
)
def test_drawing_sheet_closes_the_section_open_before_it(section, number, title):
    def title_block(page: str) -> dict:
        return {"drawing_number": number, "drawing_title": title} if number in page else {}

    pages = [
        f"{section}\nLeaves are hung on three hinges.",
        f"{number}\nscale 1 to 50",
        "notes continue here in plain words",
    ]
    text, starts = _doc(pages)
    headings = page_heading_lines(text, starts, title_block)
    assert headings[1] == section
    assert headings[2] == f"{number} - {title}"
    assert 3 not in headings
