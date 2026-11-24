"""A computed answer prints a bounded row list, and counts the rows it leaves out.

The figure and its count state the whole row set; the inline row list only
shows what was matched. Every test builds its own rows (generated labels, no
document, bill, item or project name) and every decision is checked on two
independently built inputs.
"""
from __future__ import annotations

import re

import pytest

from app.agents.answer_counted_rows import name_dropped_rows
from app.agents.answer_row_totals import reconcile
from app.agents.core import computed_answer as ca
from app.agents.core.tool_registry import Statement

EN = ca.labels_for("en")
AR = ca.labels_for("ar")


def _rows(n: int, stem: str, width: int = 0) -> tuple[str, ...]:
    """``n`` generated row labels; ``width`` pads each to that many chars."""
    out = []
    for i in range(1, n + 1):
        label = f"{stem} {i}"
        out.append(label.ljust(width, "x") if width else label)
    return tuple(out)


def _parts(text: str, lab: dict) -> list[str]:
    return text.split(lab["list_sep"])


# -- a short list is printed whole -----------------------------------------

@pytest.mark.parametrize("rows", [_rows(3, "pipe run"), _rows(ca.ROWS_SHOWN, "valve set")])
def test_a_list_within_the_bounds_is_printed_whole(rows):
    text = ca.shown_rows(rows, EN)
    assert _parts(text, EN) == list(rows)
    assert "further rows" not in text


# -- a list longer than the row bound keeps the leading rows and counts the rest

@pytest.mark.parametrize("n", [ca.ROWS_SHOWN + 3, ca.ROWS_SHOWN * 30 + 7])
def test_a_list_over_the_row_bound_counts_what_it_leaves_out(n):
    rows = _rows(n, "line")
    parts = _parts(ca.shown_rows(rows, EN), EN)
    assert parts[:-1] == list(rows[: ca.ROWS_SHOWN])
    assert parts[-1] == EN["more_rows"].format(n=n - ca.ROWS_SHOWN)


@pytest.mark.parametrize("stem", ["bay", "grid line"])
def test_the_row_bound_is_exact_at_one_over(stem):
    rows = _rows(ca.ROWS_SHOWN + 1, stem)
    parts = _parts(ca.shown_rows(rows, EN), EN)
    assert len(parts) == ca.ROWS_SHOWN + 1
    assert parts[-1] == EN["more_rows"].format(n=1)


# -- long labels stop the list at the character bound ------------------------

@pytest.mark.parametrize("n, width", [(5, ca.ROWS_SHOWN_CHARS // 3), (8, ca.ROWS_SHOWN_CHARS // 2 + 1)])
def test_long_labels_stop_at_the_character_bound(n, width):
    rows = _rows(n, "described item", width=width)
    parts = _parts(ca.shown_rows(rows, EN), EN)
    shown = parts[:-1]
    assert 1 <= len(shown) < n
    assert len(EN["list_sep"].join(shown)) <= ca.ROWS_SHOWN_CHARS
    # the next row would have crossed the bound
    assert len(EN["list_sep"].join(rows[: len(shown) + 1])) > ca.ROWS_SHOWN_CHARS
    assert parts[-1] == EN["more_rows"].format(n=n - len(shown))


@pytest.mark.parametrize("width", [ca.ROWS_SHOWN_CHARS + 1, ca.ROWS_SHOWN_CHARS * 3])
def test_one_row_longer_than_the_bound_is_still_shown(width):
    rows = _rows(1, "long row", width=width)
    assert ca.shown_rows(rows, EN) == rows[0]


@pytest.mark.parametrize("width", [ca.ROWS_SHOWN_CHARS + 1, ca.ROWS_SHOWN_CHARS * 2])
def test_a_first_row_over_the_bound_leads_and_the_rest_are_counted(width):
    rows = _rows(1, "long row", width=width) + _rows(4, "short")
    parts = _parts(ca.shown_rows(rows, EN), EN)
    assert parts == [rows[0], EN["more_rows"].format(n=4)]


@pytest.mark.parametrize("rows, lab", [((), EN), ([], AR)])
def test_an_empty_row_set_prints_nothing(rows, lab):
    assert ca.shown_rows(rows, lab) == ""


# -- the remainder is worded in the question's language -----------------------

@pytest.mark.parametrize("n", [ca.ROWS_SHOWN + 2, ca.ROWS_SHOWN + 40])
def test_the_remainder_is_worded_in_the_question_language(n):
    rows = _rows(n, "بند")
    text = ca.shown_rows(rows, AR)
    assert text.endswith(AR["more_rows"].format(n=n - ca.ROWS_SHOWN))
    assert AR["list_sep"] in text
    assert not re.search(r"[A-Za-z]", text)


# -- through the renderer the turn uses ---------------------------------------

def _sum_statement(n: int, stem: str, figure: float) -> Statement:
    return Statement(figure=figure, what="quantity", unit="m", basis="sum",
                     terms=(stem,), rows=_rows(n, stem))


@pytest.mark.parametrize("n, stem, figure", [(329, "bore", 4410.0), (53, "fitting", 212.5)])
def test_a_turn_answer_over_a_large_row_set_is_bounded(n, stem, figure):
    st = _sum_statement(n, stem, figure)
    call = ca.Call(tool="t", heading="Bill row count and total", statements=(st,),
                   asked={}, error="", declared=())
    text = ca.answer_turn([call], "en", question=f"total {stem}")
    rows_line = next(line for line in text.splitlines() if line.strip().startswith(EN["rows"]))
    assert EN["more_rows"].format(n=n - ca.ROWS_SHOWN) in rows_line
    assert f"{stem} {ca.ROWS_SHOWN + 1};" not in text
    assert f"{stem} {n}" not in text
    # the figure the tool stated is printed unchanged
    assert ca._figure_text(figure) in text
    # bounded by the row bounds, not by the set's size
    assert len(text) < ca.ROWS_SHOWN_CHARS + 400


@pytest.mark.parametrize("n, stem, figure", [(4, "door", 9.0), (ca.ROWS_SHOWN, "window", 31.0)])
def test_a_turn_answer_over_a_small_row_set_prints_every_row(n, stem, figure):
    st = _sum_statement(n, stem, figure)
    text = ca.render_call("Computed result", [st], asked={}, error="", language="en")
    for row in st.rows:
        assert row in text
    assert "further rows" not in text


@pytest.mark.parametrize("n, stem, figure", [(40, "slab", 880.0), (ca.ROWS_SHOWN + 1, "beam", 77.0)])
def test_the_stated_total_survives_the_row_total_check(n, stem, figure):
    """The bounded list leaves the tool's own total alone on the exit the
    turn runs (runtime hands both checks the turn's computed statements)."""
    st = _sum_statement(n, stem, figure)
    text = ca.render_call("Computed result", [st], asked={}, error="", language="en")
    assert reconcile(text, [st]) == text
    assert name_dropped_rows(text, [st]) == text


def test_every_language_words_the_remainder():
    for language, lab in ca.LABELS.items():
        assert "{n}" in lab["more_rows"], language
        assert lab["more_rows"].format(n=7)
