"""The clarifying question offers each distinct value once.

The module collapses cases by value ("one value = one case"); the options it
lists must follow the same rule, so a value the evidence repeats across
passages or sources does not appear as several options. Every input here is
built by the test itself, two inputs per decision.
"""

from __future__ import annotations

import pytest

from app.agents import clarify_cases as cc
from app.agents.clarify_cases import Case


@pytest.fixture(autouse=True)
def _on(monkeypatch):
    monkeypatch.setenv(cc.FLAG_ENV, "1")


def _options(q: str) -> list[str]:
    body = q.split("supports more than one answer: ", 1)[1]
    body = body.rsplit(". Which one is meant?", 1)[0]
    return body.split("; ")


QUESTION = "What is the figure?"


def _draft(value: str) -> str:
    return f"The figure is {value}."


def _measures(rows: list[tuple[str, str, str]]) -> list[Case]:
    return [Case(subject=s, value=v, source=src, kind="measure") for s, v, src in rows]


@pytest.mark.parametrize(
    "rows,repeated,other,first",
    [
        (
            [
                ("concrete cover", "40 mm", "Doc A p1"),
                ("concrete cover", "40 mm", "Doc B p3"),
                ("backfill depth", "300 mm", "Doc C p2"),
                ("pipe bedding", "40 mm", "Doc D p5"),
            ],
            "40 mm", "300 mm", "40 mm in Doc A p1 (concrete cover)",
        ),
        (
            [
                ("retention", "5%", "Sheet K"),
                ("advance payment", "15%", "Sheet L"),
                ("retention release", "5%", "Sheet M"),
                ("retention", "5%", "Sheet N"),
                ("retention", "5%", "Sheet K"),
            ],
            "5%", "15%", "5% in Sheet K (retention)",
        ),
    ],
)
def test_repeated_value_is_listed_once(rows, repeated, other, first):
    q = cc.clarifying_question(QUESTION, _draft(repeated), _measures(rows))
    assert q is not None
    opts = _options(q)
    assert len(opts) == 2
    assert sum(o.startswith(f"{repeated} in ") for o in opts) == 1
    assert sum(o.startswith(f"{other} in ") for o in opts) == 1
    # The repeated value is named by its first appearance.
    assert opts[0] == first


@pytest.mark.parametrize(
    "rows,values",
    [
        (
            [
                ("concrete cover", "50 mm", "Doc E p1"),
                ("backfill depth", "300 mm", "Doc F p2"),
                ("pipe bedding", "150 mm", "Doc G p4"),
            ],
            ["50 mm", "300 mm", "150 mm"],
        ),
        (
            [
                ("slab thickness", "250 mm", "Doc P p1"),
                ("wall thickness", "200 mm", "Doc Q p6"),
            ],
            ["250 mm", "200 mm"],
        ),
    ],
)
def test_distinct_values_each_keep_their_option(rows, values):
    q = cc.clarifying_question(QUESTION, _draft(values[0]), _measures(rows))
    assert q is not None
    opts = _options(q)
    assert [o.split(" in ", 1)[0] for o in opts] == values
    for opt, (_subject, _value, source) in zip(opts, rows):
        assert source in opt


@pytest.mark.parametrize(
    "rows",
    [
        [("concrete cover", "40 mm", "Doc H p1"), ("backfill depth", "40 m", "Doc I p1")],
        [("pipe diameter", "600 mm", "Doc J p2"), ("trench length", "600 m", "Doc R p3")],
    ],
)
def test_same_number_in_another_unit_stays_its_own_option(rows):
    # The value key is (number, unit): the same number in another unit is a
    # different answer.
    q = cc.clarifying_question(QUESTION, _draft(rows[0][1]), _measures(rows))
    assert q is not None
    assert len(_options(q)) == 2
