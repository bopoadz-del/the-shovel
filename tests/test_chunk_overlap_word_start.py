"""The carried overlap of ``chunk_text_with_overlap`` starts on a boundary.

A character offset opened the next chunk mid-token: ``C40`` became a bare
``0`` and a BOQ chunk opened on ``oundations`` instead of its row code.
Every input is invented here; two inputs per decision.
"""
from __future__ import annotations

import re

import pytest

from app.core.doc_index import _overlap_start, chunk_text_with_overlap

_ROW_CODE = re.compile(r"^[A-Z] \d+\.\d+ ")


def _prose(sentence: str, times: int) -> str:
    return " ".join([sentence] * times)


@pytest.mark.parametrize("sentence", [
    "Reinforced concrete grade C40 placed in foundations and compacted thoroughly.",
    "Blockwork walls 200 mm thick laid in mortar class M4 with ties every 450 mm.",
])
def test_prose_chunk_never_opens_on_a_word_fragment(sentence):
    text = _prose(sentence, 20)
    words = set(text.split())
    chunks = chunk_text_with_overlap(text, target_chars=300, overlap=50)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk.split()[0] in words, chunk[:40]


def _boq(code_letter: str, unit: str, rows: int) -> str:
    return "\n".join(
        f"{code_letter} {500 + k}.{k % 9 + 1} Excavation for foundations depth "
        f"not exceeding 2 m  {unit}  {120 + k}  {45.5 + k}"
        for k in range(rows)
    )


@pytest.mark.parametrize("letter,unit", [("D", "m3"), ("E", "m2")])
def test_boq_chunk_after_a_row_break_opens_on_its_row_code(letter, unit):
    chunks = chunk_text_with_overlap(_boq(letter, unit, 30), target_chars=300, overlap=50)
    assert len(chunks) >= 3
    for chunk in chunks:
        assert _ROW_CODE.match(chunk), chunk[:50]


@pytest.mark.parametrize("text,start,end,expect", [
    ("alpha beta gamma delta", 8, 16, 11),   # mid "beta" -> after its space
    ("one two three four five", 9, 18, 14),  # mid "three" -> start of "four"
])
def test_word_window_snaps_past_the_fragment(text, start, end, expect):
    assert _overlap_start(text, start, end, whole_lines=False) == expect


@pytest.mark.parametrize("text,start,end", [
    ("alpha beta gamma delta", 6, 16),   # already the start of "beta"
    ("xxxxxxxxxxxxxxxxxxxxxx", 5, 15),   # no whitespace: keep the offset
])
def test_word_window_keeps_an_offset_with_nothing_to_protect(text, start, end):
    assert _overlap_start(text, start, end, whole_lines=False) == start


@pytest.mark.parametrize("text,start,end,expect", [
    ("row one 12\nrow two 34\nrow three", 4, 22, 11),
    ("A 1.1 x 5\nB 2.2 y 6\nC 3.3 z", 2, 20, 10),
])
def test_line_window_carries_whole_lines_only(text, start, end, expect):
    assert _overlap_start(text, start, end, whole_lines=True) == expect


@pytest.mark.parametrize("text,start,end", [
    ("a long row with no newline in reach", 3, 20),
    ("another row ending without a break", 0, 15),
])
def test_line_window_without_a_line_start_carries_nothing(text, start, end):
    assert _overlap_start(text, start, end, whole_lines=True) == end


@pytest.mark.parametrize("start,end", [(10, 10), (12, 9)])
def test_empty_window_returns_its_end(start, end):
    assert _overlap_start("some words here and there", start, end, whole_lines=False) == end
