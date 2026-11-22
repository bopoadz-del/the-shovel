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


@pytest.mark.parametrize("heading", ["## Site levels", "# 5 External works"])
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
