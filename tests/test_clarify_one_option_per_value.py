"""The clarifying question offers each distinct value once.

The module collapses cases by value ("one value = one case"); the options it
lists must follow the same rule, so a value the evidence repeats across
passages or sources does not appear as several options. Every input here is
built by the test itself.
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


def test_repeated_value_is_listed_once():
    cases = [
        Case(subject="concrete cover", value="40 mm", source="Doc A p1", kind="measure"),
        Case(subject="concrete cover", value="40 mm", source="Doc B p3", kind="measure"),
        Case(subject="backfill depth", value="300 mm", source="Doc C p2", kind="measure"),
        Case(subject="pipe bedding", value="40 mm", source="Doc D p5", kind="measure"),
    ]
    q = cc.clarifying_question(QUESTION, _draft("40 mm"), cases)
    assert q is not None
    opts = _options(q)
    assert len(opts) == 2
    assert sum(o.startswith("40 mm in ") for o in opts) == 1
    assert sum(o.startswith("300 mm in ") for o in opts) == 1
    # The repeated value is named by its first appearance.
    assert opts[0] == "40 mm in Doc A p1 (concrete cover)"


def test_distinct_values_each_keep_their_option():
    cases = [
        Case(subject="concrete cover", value="50 mm", source="Doc E p1", kind="measure"),
        Case(subject="backfill depth", value="300 mm", source="Doc F p2", kind="measure"),
        Case(subject="pipe bedding", value="150 mm", source="Doc G p4", kind="measure"),
    ]
    q = cc.clarifying_question(QUESTION, _draft("50 mm"), cases)
    assert q is not None
    opts = _options(q)
    assert [o.split(" in ", 1)[0] for o in opts] == ["50 mm", "300 mm", "150 mm"]
    assert "Doc F p2" in opts[1] and "Doc G p4" in opts[2]


def test_same_number_in_another_unit_stays_its_own_option():
    # The value key is (number, unit): 40 mm and 40 m are different answers.
    cases = [
        Case(subject="concrete cover", value="40 mm", source="Doc H p1", kind="measure"),
        Case(subject="backfill depth", value="40 m", source="Doc I p1", kind="measure"),
    ]
    q = cc.clarifying_question(QUESTION, _draft("40 mm"), cases)
    assert q is not None
    assert len(_options(q)) == 2
