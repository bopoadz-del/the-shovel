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


@pytest.mark.parametrize("line", ["DRAWING NO: ()", "REVISION: [ ]"])
def test_label_with_only_empty_marks_after_it_is_still_a_heading(line):
    # The value is only brackets: no token is left, so it names no code.
    assert _is_heading_line(line) is True


@pytest.mark.parametrize("label", ["SUBCONTRACT REFERENCE NO", "SUPPLIER DELIVERY REF NO"])
def test_a_label_at_the_length_bound_is_read_as_a_label(label):
    assert len(label) == 24
    assert _is_heading_line(f"{label}: KX/07") is False


@pytest.mark.parametrize("label", ["SUBCONTRACTS REFERENCE NO", "SUPPLIER DELIVERY REFS NO"])
def test_words_past_the_label_bound_are_a_title(label):
    # 25 characters before the colon: the words are part of the line, not a label.
    assert len(label) == 25
    assert _is_heading_line(f"{label}: KX/07") is True


@pytest.mark.parametrize("line", ["SK-12 GA", "C/04 WC LAYOUT"])
def test_a_two_letter_word_beside_a_code_is_a_title(line):
    assert _is_heading_line(line) is True


@pytest.mark.parametrize("line", ["REF-MNO-0031 B", "Ref: KX/07 C"])
def test_a_single_letter_beside_a_code_is_still_a_code(line):
    # A revision letter is not a word.
    assert _is_heading_line(line) is False


@pytest.mark.parametrize("line", ["REF: C0451", "OUR REF: 22B07"])
def test_a_code_of_letters_and_digits_is_not_a_word(line):
    assert _is_heading_line(line) is False
