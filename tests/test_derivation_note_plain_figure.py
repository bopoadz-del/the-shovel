"""The working-and-result note writes the computed figure as working does.

The note appended by ``_annotate_derivation_mismatches`` told the client
"`5% x 1,280,000` is **6.4e+04**". A figure in an answer is grouped digits,
never an exponent. Every text below is invented; each decision has two
inputs.
"""

from __future__ import annotations

import pytest

from app.agents import runtime as rt


@pytest.fixture(autouse=True)
def _check_on(monkeypatch):
    monkeypatch.setenv("DERIVATION_CHECK", "1")


@pytest.mark.parametrize(
    ("text", "figure"),
    [
        ("Area: 12 × 9,500 = 100,000 m2 for the deck.", "114,000"),
        ("Hire: 3 × 450,000 = 1,000,000 for the quarter.", "1,350,000"),
    ],
)
def test_a_large_computed_figure_is_grouped_digits(text, figure):
    note = rt._annotate_derivation_mismatches(text)
    assert f"**{figure}**" in note
    assert "e+" not in note


@pytest.mark.parametrize(
    ("text", "figure"),
    [
        ("Factor: 0.3 × 0.25 = 0.5 on the load.", "0.075"),
        ("Ratio: 0.4 × 0.15 = 0.9 of the span.", "0.06"),
    ],
)
def test_a_small_computed_figure_keeps_its_significant_digits(text, figure):
    note = rt._annotate_derivation_mismatches(text)
    assert f"**{figure}**" in note
    assert "e-" not in note


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (1088000.0, "1,088,000"),
        (23436.0, "23,436"),
        (1508.22, "1,508"),
        (27.4, "27.4"),
        (0.000123, "0.000123"),
        (-6.0, "-6"),
        (-64000.0, "-64,000"),
        (0.0, "0"),
        (1e-15, "0"),
        (float("inf"), "inf"),
        (float("nan"), "nan"),
    ],
)
def test_figure_written_without_exponent(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    "text",
    [
        "Area: 12 × 9,500 = 114,000 m2 for the deck.",
        "Factor: 0.3 × 0.25 = 0.075 on the load.",
    ],
)
def test_working_that_agrees_gets_no_note(text):
    assert rt._annotate_derivation_mismatches(text) == text
