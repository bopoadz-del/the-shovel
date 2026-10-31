"""#45 follow-up (G2 post-hoc FAIL on bfa0250): normalise_rate's annually/yearly
rule and its word-boundary rule were untested, and value_and_unit's docstring
had been made dead by an import above it. These pin all three.
"""
from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas_planning import normalise_rate
from app.lib.formula_registry import value_and_unit


@pytest.mark.parametrize("word", ["annually", "yearly", "Annually", "YEARLY"])
def test_annual_rate_words_normalise_to_per_year(word):
    """Kills a mutant that drops annually/yearly from _RATE_WORDS: without the
    rule the word passes through unchanged instead of becoming /year."""
    assert normalise_rate(word) == "/year"


@pytest.mark.parametrize("word", ["biweekly", "semimonthly", "dailyrate"])
def test_non_rate_words_are_left_unchanged(word):
    """The word-boundary rule. "daily" inside "dailyrate" must NOT match (drop
    the \\b and it would be rewritten); biweekly/semimonthly are not rate
    words and stay as written."""
    assert normalise_rate(word) == word


@pytest.mark.parametrize("word", ["perch", "percent"])
def test_per_run_into_a_nonunit_is_left_alone(word):
    """"per" joined to something that is not a unit the table knows stays as
    written -- only "per<unit>" is split."""
    assert normalise_rate(word) == word


@pytest.mark.parametrize("text,expected", [
    ("SAR 5,000 annually", (5000.0, "SAR/year")),
    ("AED 1,200 yearly", (1200.0, "AED/year")),
])
def test_value_and_unit_carries_the_annual_rate(text, expected):
    assert value_and_unit(text) == expected


@pytest.mark.parametrize("value,expected", [
    ({"value": 20, "unit": "ft"}, (20.0, "ft")),
    ({"value": 1200.0, "unit": "AED"}, (1200.0, "AED")),
])
def test_value_and_unit_reads_a_value_unit_dict(value, expected):
    """Pins the dict branch (the function's first `if`): drop it and a
    ``{"value":.., "unit":..}`` mapping falls through to ``None``."""
    assert value_and_unit(value) == expected


def test_value_and_unit_has_a_real_docstring():
    """The import used to sit above the string literal, making it a dead
    statement rather than the function's __doc__. Restored above the import."""
    assert value_and_unit.__doc__ is not None and "20 ft" in value_and_unit.__doc__
