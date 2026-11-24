"""A continuation page is filed under its open section, not the page furniture.

Two rules meet on a page that only continues a section: the running-line rule
(a line the document repeats on most pages names no page) and the carry rule
(a page with no heading of its own takes the section still open at the end of
the page before). A numbered running footer is heading-like, so without both
rules agreeing it was taken as the open section, or filed the continuation
page under itself. Every text is invented; two inputs per decision.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import page_heading_lines


def _doc(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, starts = "", []
    for number, page in enumerate(pages, 1):
        starts.append((number, len(text)))
        text += page + "\n"
    return text, starts


@pytest.mark.parametrize(
    "opening,footer,open_section",
    [
        (
            "3.1 Formwork\nForms are oiled before use.\n3.2 Striking\nSoffit forms stay in place",
            "9 ISSUED FOR TENDER",
            "3.2 Striking",
        ),
        (
            "## Glazing\nPanes are bedded in putty.\n## Sealants\nJoints are primed and filled",
            "12 MARINA OFFICES PACKAGE C",
            "## Sealants",
        ),
    ],
)
def test_running_footer_neither_opens_nor_names_a_continuation_page(opening, footer, open_section):
    pages = [
        f"{opening}\n{footer}",
        f"until the cubes reach strength\nand the engineer agrees.\n{footer}",
        f"after which the props are eased\nin the order shown.\n{footer}",
        f"and the surface is made good\nbefore any finish.\n{footer}",
    ]
    text, starts = _doc(pages)
    headings = page_heading_lines(text, starts)
    assert headings[2] == open_section
    assert headings[4] == open_section


@pytest.mark.parametrize(
    "footer",
    ["9 ISSUED FOR TENDER", "12 MARINA OFFICES PACKAGE C"],
)
def test_furniture_still_names_a_page_with_no_open_section(footer):
    pages = [
        f"until the cubes reach strength\nand the engineer agrees.\n{footer}",
        f"after which the props are eased\nin the order shown.\n{footer}",
        f"and the surface is made good\nbefore any finish.\n{footer}",
    ]
    text, starts = _doc(pages)
    headings = page_heading_lines(text, starts)
    assert headings[1] == footer
