"""A bare clause number is an identifier the sweep can match.

A document heads and cites its own clauses without a label word ("4.2.1
Minimum cover ..."). The sweep used to read only labelled designations, so
a draft citing "Clause 4.2.1" never shared an identifier with the heading
"4.2.1" that states another figure. Every input below is invented here; two
inputs per decision, each with a negative.
"""

from __future__ import annotations

import pytest

from app.agents.discrepancy_sweep import (
    _designations,
    _same_identifier,
    sweep_differences,
    sweep_targets,
)


# --- the reader: a bare clause number is read, by shape ----------------------

@pytest.mark.parametrize("text, expected", [
    ("6.3.2 Lap length shall be 600 mm.", ["6.3.2"]),
    ("as required by 11.4.7.2, the gap is 25 mm", ["11.4.7.2"]),
])
def test_bare_clause_number_is_read(text, expected):
    assert _designations(text) == expected


@pytest.mark.parametrize("text", [
    "the strip is 3.75 m wide",            # a decimal has one dot
    "issued 07.09.2031, the gap is 25 mm",  # a dotted date has a year part
])
def test_decimal_or_dotted_date_is_not_a_clause(text):
    assert _designations(text) == []


@pytest.mark.parametrize("text, expected", [
    ("Section 9.1.4 sets the gap at 25 mm.", ["section 9.1.4"]),
    ("See Cl.2.8.6 for the lap.", ["cl.2.8.6"]),
])
def test_labelled_clause_is_read_once_not_twice(text, expected):
    # The number inside a labelled designation is not read again bare.
    assert _designations(text) == expected


# --- matching: bare matches labelled; two labels never match -----------------

@pytest.mark.parametrize("a, b", [
    ("6.3.2", "section 6.3.2"),
    ("cl.11.4.7", "11.4.7"),
])
def test_bare_number_matches_labelled_same_number(a, b):
    assert _same_identifier(a, b)


@pytest.mark.parametrize("a, b", [
    ("table 6.3.2", "section 6.3.2"),   # different structures, same number
    ("6.3.2", "section 6.3.21"),        # a longer number is another clause
    ("6.3.2", "6.3.2.1"),               # a sub-clause is another clause
])
def test_different_structure_or_number_never_matches(a, b):
    assert not _same_identifier(a, b)


# --- the sweep end to end ----------------------------------------------------

@pytest.mark.parametrize("stated, heading, other", [
    ("Under Section 6.3.2 the lap length is 600 mm.",
     "6.3.2 Lap length shall be 750 mm.", "750 mm"),
    ("Per Cl.9.1.4 the joint gap is 25 mm.",
     "9.1.4 Joint gap: 20 mm minimum.", "20 mm"),
])
def test_labelled_draft_recalls_bare_heading_with_other_figure(
    stated, heading, other,
):
    hits = sweep_differences(stated, [("spec", heading)])
    assert len(hits) == 1
    assert hits[0].differing_value == other


@pytest.mark.parametrize("stated, heading", [
    ("Under Section 6.3.2 the lap length is 600 mm.",
     "6.3.2 Lap length shall be 600 mm."),          # same figure
    ("Per Cl.9.1.4 the joint gap is 25 mm.",
     "9.1.5 Joint gap: 20 mm minimum."),            # another clause
])
def test_same_figure_or_other_clause_is_not_recalled(stated, heading):
    assert sweep_differences(stated, [("spec", heading)]) == []


@pytest.mark.parametrize("stated, candidate", [
    ("Table 6.3.2 gives a lap length of 600 mm.",
     "Section 6.3.2 sets the lap length at 750 mm."),
    ("Figure 4.4.1 shows a gap of 25 mm.",
     "Clause 4.4.1 sets the gap at 20 mm."),
])
def test_two_labels_on_one_number_are_not_recalled(stated, candidate):
    assert sweep_differences(stated, [("spec", candidate)]) == []


@pytest.mark.parametrize("draft, query", [
    ("see 6.3.2: the lap length is 600 mm.", "6.3.2"),
    ("the joint gap is 25 mm as 9.1.4 requires.", "9.1.4"),
])
def test_unlabelled_citation_gives_the_sweep_a_target(draft, query):
    stated, q = sweep_targets(draft)
    assert stated and q == query
