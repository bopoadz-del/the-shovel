"""Edge cases of the continuation-page carry in ``page_heading_lines``.

Every input is invented here; nothing is read from a stored document.
"""

from __future__ import annotations

import pytest

from app.core.chunk_bounds import page_heading_lines


def _paged(pages: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text, starts = "", []
    for number, body in enumerate(pages, start=1):
        starts.append((number, len(text)))
        text += body + "\n"
    return text, starts


@pytest.mark.parametrize("page_starts", [None, []])
def test_no_text_and_no_pages_files_nothing(page_starts):
    assert page_heading_lines("", page_starts) == {}


@pytest.mark.parametrize(
    "first, blank, last, section",
    [
        ("3.1 Formwork\nShutters are struck after curing.", "", "the props stay in place.", "3.1 Formwork"),
        ("## 7 Drainage\nPipes are laid to falls.", "   ", "and the outfall is tested.", "## 7 Drainage"),
    ],
)
def test_blank_page_does_not_break_the_carry(first, blank, last, section):
    text, starts = _paged([first, blank, last])
    assert page_heading_lines(text, starts) == {1: section, 2: section, 3: section}


@pytest.mark.parametrize(
    "lone",
    [
        "only a sentence that carries on from nowhere.",
        "and the joints are sealed after the pour.",
    ],
)
def test_single_page_with_no_heading_is_unfiled(lone):
    text, starts = _paged([lone])
    assert page_heading_lines(text, starts) == {}
