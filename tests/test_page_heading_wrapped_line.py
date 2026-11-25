"""A wrapped line of prose is not the heading a page is filed under.

A PDF page wraps its sentences at a fixed width. A wrapped line that opens on a
number ("300 mm thick ...") or a single capital ("A minimum cover ...") and
stops mid-sentence passes the heading-like shape test, so it named the page
ahead of the page's own section heading. The line after it opens on a
lowercase word: that is the sign of a sentence carried on, and such a line is
skipped. Every input below is invented; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import heading_line_for_page, page_heading_lines


def _page(opener: str, carry: str, heading: str) -> str:
    return f"{opener}\n{carry}\n{heading}\nThe works are measured net as fixed.\n"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("450 mm deep ground beams are cast against a blinding of", "lean mix concrete and sand.", "3.4 GROUND BEAMS"),
        ("75 kg of fixings are allowed for each tonne of steel", "erected on the upper floors.", "11 Steel Fixings"),
    ],
)
def test_wrapped_line_opening_on_a_number_does_not_name_the_page(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A sealant joint is formed between each frame and the", "surrounding masonry opening.", "2.6 Sealants"),
        ("C class bricks are laid below the damp proof course in", "every external wall.", "14 BRICKWORK"),
    ],
)
def test_wrapped_line_opening_on_one_capital_does_not_name_the_page(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize("marker", ["a) all joints are struck flush", "(iv) the samples are labelled"])
def test_heading_followed_by_a_lowercase_list_item_still_names_the_page(marker):
    page = f"7.1 Pointing Works\n{marker}\nb) the face is brushed clean\n"
    assert heading_line_for_page(page) == "7.1 Pointing Works"


@pytest.mark.parametrize(
    "heading,body",
    [
        ("9 ROOF COVERINGS", "Tiles are fixed with two nails each."),
        ("B Drainage Schedule", "Pipes are laid to the falls shown."),
    ],
)
def test_heading_followed_by_a_capitalised_line_still_names_the_page(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize("heading", ["## Levels of the site", "# 5 Notes on the works"])
def test_markdown_heading_followed_by_a_lowercase_line_still_names_the_page(heading):
    page = f"{heading}\nrecorded by the surveyor before stripping.\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize(
    "first_head,second_head",
    [("4.1 Excavation", "4.2 FILLING"), ("6 PAINTING", "7 Glazing")],
)
def test_every_page_of_a_document_is_named_by_its_own_section(first_head, second_head):
    page_one = _page("200 mm of topsoil is stripped and stockpiled for", "reuse in the landscaped areas.", first_head)
    page_two = _page("A mist coat is applied to new plaster before the", "two finishing coats.", second_head)
    text = page_one + page_two
    headings = page_heading_lines(text, [(1, 0), (2, len(page_one))])
    assert headings == {1: first_head, 2: second_head}


# --- grader faults on 38e67f0 (REPORT_OPUSGRADE_S616): each decision, two inputs ---


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A minimum cover of 25 mm is kept to the outer", "ties.", "5.3 COVER TO TIES"),
        ("Joints are raked out and left open for later", "use) as directed.", "8 Pointing"),
    ],
)
def test_short_closing_word_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "heading,body",
    [
        ("5.2 MIXING WATER", "pH of the water is tested weekly."),
        ("12 Brick Sizes", "mm figures below are work sizes."),
    ],
)
def test_capitalised_heading_over_a_lowercase_unit_line_still_names_the_page(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize(
    "heading,body",
    [
        ("8 Fire stopping", "mm gaps are sealed with mastic."),
        ("2 Floor screeds", "kg of cement per bag is printed."),
    ],
)
def test_heading_with_as_many_capitals_as_lowercase_words_is_not_prose(heading, body):
    assert heading_line_for_page(f"{heading}\n{body}\n") == heading


@pytest.mark.parametrize("marker", ["iv. the samples are labelled", "xii) the face is brushed clean", "(b) the edges are sealed"])
def test_sentence_case_heading_over_a_list_item_still_names_the_page(marker):
    page = f"7.1 Pointing of joints\n{marker}\n"
    assert heading_line_for_page(page) == "7.1 Pointing of joints"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("600 mm wide strip footings are poured on the", "same day as excavated.", "2.2 FOOTINGS"),
        ("A damp proof membrane is lapped 150 mm at all", "joints and taped.", "6 Membranes"),
    ],
)
def test_blank_line_inside_a_wrapped_sentence_is_skipped(opener, carry, heading):
    page = f"{opener}\n\n{carry}\n{heading}\n"
    assert heading_line_for_page(page) == heading


@pytest.mark.parametrize("heading", ["3 Cover to the bars", "9 Making good of finishes"])
def test_prose_shaped_heading_on_the_last_line_still_names_the_page(heading):
    assert heading_line_for_page(f"Notes are given below.\n{heading}\n") == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A lintel bearing is as set out in appendix", "iv.", "10 LINTELS"),
        ("150 mm kerbs are laid to the profile of detail", "c.", "7.4 Kerbs"),
    ],
)
def test_bare_letter_or_numeral_that_ends_a_sentence_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("C Grade is used", "throughout the frame.", "4 STEEL"),
        ("A Mesh is laid", "over the membrane.", "6 SLABS"),
    ],
)
def test_the_opener_is_not_counted_as_a_capitalised_word(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize("marker", ["viii) the joints are raked out", "xiii. the bays are cast in turn"])
def test_four_letter_roman_numeral_is_a_list_marker(marker):
    page = f"7.1 Pointing of joints\n{marker}\n"
    assert heading_line_for_page(page) == "7.1 Pointing of joints"


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("A sealant bead is run where the frames are made", "up. Joints are then tooled.", "8 SEALANTS"),
        ("300 mm wide bays are laid and the forms are struck", "in. Each bay is then cured.", "5.2 Slab Bays"),
    ],
)
def test_two_letter_word_that_ends_a_sentence_is_not_a_list_marker(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading


@pytest.mark.parametrize(
    "opener,carry,heading",
    [
        ("2 Bars at 150 200 250 centres to", "each face of the wall.", "2.1 REINFORCEMENT"),
        ("7 Ties of 6 8 10 mm at", "spacings shown on the drawings.", "7.3 Links"),
    ],
)
def test_figures_in_a_wrapped_line_are_not_counted_as_capitalised_words(opener, carry, heading):
    assert heading_line_for_page(_page(opener, carry, heading)) == heading
