"""Answer style, length: a computed answer states what it found once.

* Two calls of one tool that found the same figures over the same rows are
  one result, whatever words each was asked with (``_found``).
* A row list already listed in the answer is named as the same rows, not
  listed again; a different row list is still listed.

Every input is built here (invented tools, rows, figures and words); each
decision is checked with two inputs, one side that changes and one that must
not.
"""
from __future__ import annotations

import pytest

from app.agents.core.computed_answer import LABELS, Call, answer_turn
from app.agents.core.tool_registry import Statement

EN, AR = LABELS["en"], LABELS["ar"]


def _rows(stem: str, n: int) -> tuple:
    return tuple(f"{stem} {i} (p.{i})" for i in range(1, n + 1))


def _count(rows, terms, *, src="ledger.xlsx"):
    return Statement(figure=float(len(rows)), what="rows", count=len(rows), basis="count",
                     rows=rows, terms=terms, source=src)


def _sum(rows, figure, unit, terms, *, what="quantity", src="ledger.xlsx", working=""):
    return Statement(figure=figure, what=what, unit=unit, basis="sum", rows=rows,
                     terms=terms, source=src, working=working)


def _call(tool, terms, statements):
    return Call(tool=tool, heading="Row tally", statements=tuple(statements),
                asked={"match": " ".join(terms)})


# Two different worlds: (row stem, row count, figure, unit, first words, wider words)
WORLDS = [
    ("Gravel bed type", 14, 5120.0, "m3", ("gravel",), ("gravel", "bed", "type")),
    ("Copper cable core", 23, 870.5, "m", ("copper",), ("copper", "cable")),
]


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_a_second_call_that_found_the_same_is_not_shown_again(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    first = _call("tally", words, [_count(rows, words), _sum(rows, fig, unit, words)])
    again = _call("tally", wider, [_count(rows, wider), _sum(rows, fig, unit, wider)])
    text = answer_turn([first, again], "en")
    assert text.count("Row tally") == 1
    assert text.count(f"{rows[0]};") == 1
    assert EN["several"].split("{n}")[1].strip() not in text


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_a_second_call_that_found_a_different_figure_is_still_shown(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    first = _call("tally", words, [_count(rows, words), _sum(rows, fig, unit, words)])
    other = _call("tally", wider, [_count(rows[:-1], wider), _sum(rows[:-1], fig - 1, unit, wider)])
    text = answer_turn([first, other], "en")
    assert text.count("Row tally") == 2
    assert EN["several"].format(n=2) in text


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_same_figures_from_another_tool_are_both_shown(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    first = _call("tally", words, [_sum(rows, fig, unit, words)])
    second = Call(tool="tally_other", heading="Other tally",
                  statements=(_sum(rows, fig, unit, words),), asked={"match": words[0]})
    text = answer_turn([first, second], "en")
    assert "Row tally" in text and "Other tally" in text


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_same_figures_with_a_different_missing_part_are_both_shown(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    gone = Statement(figure=None, what="amount", basis="sum", terms=wider, failure="")
    first = _call("tally", words, [_sum(rows, fig, unit, words)])
    second = _call("tally", wider, [_sum(rows, fig, unit, wider), gone])
    text = answer_turn([first, second], "en")
    assert text.count("Row tally") == 2


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_a_row_list_listed_once_is_named_after(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    call = _call("tally", words, [_count(rows, words), _sum(rows, fig, unit, words)])
    text = answer_turn([call], "en")
    assert text.count(f"{rows[0]};") == 1
    assert f"{EN['rows']}: {EN['same_rows']}" in text
    assert EN["more_rows"].format(n=n - 10) in text


@pytest.mark.parametrize("stem, n, fig, unit, words, wider", WORLDS)
def test_a_different_row_list_is_still_listed(stem, n, fig, unit, words, wider):
    rows = _rows(stem, n)
    priced = rows[1:]
    call = _call("tally", words, [_count(rows, words), _sum(priced, fig, unit, words)])
    text = answer_turn([call], "en")
    # Holds on main too (the unchanged side): the label is new, so read it softly.
    assert f"{EN['rows']}: {EN.get('same_rows', '(none)')}" not in text
    assert text.count(f"{rows[2]};") == 2


@pytest.mark.parametrize("stem, n", [("Duct run", 3), ("Anchor bolt", 12)])
def test_the_same_rows_are_named_in_the_question_language(stem, n):
    rows = _rows(stem, n)
    call = _call("tally", ("x",), [_count(rows, ("x",)), _sum(rows, 9.0, "m", ("x",))])
    text = answer_turn([call], "ar")
    assert f"{AR['rows']}: {AR['same_rows']}" in text
    assert EN["same_rows"] not in text


def test_every_language_words_the_same_rows():
    assert all(lab.get("same_rows") for lab in LABELS.values())
