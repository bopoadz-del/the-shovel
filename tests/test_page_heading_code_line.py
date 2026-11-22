"""A page is not filed under a line that only names a code.

Every input is invented here; nothing is read from a corpus.
"""
from __future__ import annotations

import pytest

from app.core.chunk_bounds import _is_heading_line, heading_line_for_page


@pytest.mark.parametrize("line", ["Ref: KX/07", "Our Ref: QQ/554/ZT"])
def test_label_then_code_is_not_a_heading(line):
    assert _is_heading_line(line) is False


@pytest.mark.parametrize("line", ["REF-MNO-0031", "TRX-88/B2"])
def test_bare_code_is_not_a_heading(line):
    assert _is_heading_line(line) is False


@pytest.mark.parametrize(
    "line", ["TO: NORTHGATE BUILDERS LLC", "SUBJECT: PUMP STATION HANDOVER"]
)
def test_label_then_words_is_still_a_heading(line):
    assert _is_heading_line(line) is True


@pytest.mark.parametrize("line", ["SECTION 7:", "PART 4:"])
def test_label_with_nothing_after_it_is_still_a_heading(line):
    assert _is_heading_line(line) is True


@pytest.mark.parametrize("line", ["# QXR-220", "## RTW-14/C"])
def test_markdown_heading_stays_a_heading_even_when_a_code(line):
    assert _is_heading_line(line) is True


@pytest.mark.parametrize("line", ["5 Retaining Walls", "SK-311 REV C"])
def test_title_with_a_word_beside_a_code_is_still_a_heading(line):
    assert _is_heading_line(line) is True


@pytest.mark.parametrize(
    "code,subject",
    [
        ("Ref: KX/07", "HOARDING REMOVAL PLAN"),
        ("REF-MNO-0031", "6 Fire Main Testing"),
    ],
)
def test_page_is_filed_under_the_subject_not_the_code(code, subject):
    page = f"{code}\n{subject}\nThe contractor shall submit the method first."
    assert heading_line_for_page(page) == subject
