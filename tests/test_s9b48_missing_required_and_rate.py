"""#48b (close-and-split of #48): the two G6 survivors on df94661.

O9 -- the describe_calculation_params fallback in _missing_required (the
groupless path) -- and W3 -- the \\bper\\b word boundary in normalise_rate --
were untested. Also folds in the #45 rate-word / value_and_unit-docstring
coverage (#51), so #48b subsumes it.
"""
from __future__ import annotations

import pytest

import app.lib.construction_formulas as cf
from app.agents.base.formulas.construction_formulas_planning import normalise_rate
from app.lib.formula_registry import value_and_unit


# ── O9: the describe_calculation_params else-branch (groupless calc) ─────────
@pytest.mark.parametrize("bound,expected", [
    ({"y": 2}, ["x"]),          # x has no default -> required -> omitted -> asked
    ({"x": 5, "y": 2}, []),     # x given -> nothing missing
])
def test_missing_required_else_branch_reports_signature_required(bound, expected):
    """A calc with no _REQUIRED_GROUPS entry takes the describe_calculation_
    params else-branch. A signature-required param (no default) that is NOT in
    the registry's declared set is caught ONLY there, not by the declared-
    required tail -- so dropping the else-branch (O9) stops reporting it."""
    def _fn(x, y=1):
        return x + y

    assert cf._missing_required(_fn, bound, name=None) == expected


# ── W3: the \bper\b word boundary in normalise_rate ──────────────────────────
@pytest.mark.parametrize("text,expected", [
    (" per day", "/day"),        # a standalone "per" is the rate separator
    ("paper day", "paper day"),  # "per" inside "paper" must NOT match (\b)
])
def test_per_word_boundary(text, expected):
    assert normalise_rate(text) == expected


# ── #45 rate words + value_and_unit docstring (folds in #51) ─────────────────
@pytest.mark.parametrize("word", ["annually", "yearly", "Annually", "YEARLY"])
def test_annual_rate_words_normalise_to_per_year(word):
    assert normalise_rate(word) == "/year"


@pytest.mark.parametrize("word", ["biweekly", "semimonthly", "dailyrate", "perch", "percent"])
def test_non_rate_words_are_left_unchanged(word):
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
    assert value_and_unit(value) == expected


def test_value_and_unit_has_a_real_docstring():
    assert value_and_unit.__doc__ is not None and "20 ft" in value_and_unit.__doc__
