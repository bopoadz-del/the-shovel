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


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (12.3449, "12.34"),
        (1.23456, "1.235"),
        (0.0123449, "0.01234"),
        (9999.5, "10,000"),
        (0.99996, "1"),
    ],
)
def test_figure_keeps_four_significant_figures(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (1e-13, "0.0000000000001"),
        (1.234e-10, "0.0000000001234"),
        (-2.5e-12, "-0.0000000000025"),
        (1e-15, "0.000000000000001"),
    ],
)
def test_a_tiny_figure_is_never_written_as_zero(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (999999999999999.0, "999,999,999,999,999"),
        (123456789012345.0, "123,456,789,012,345"),
    ],
)
def test_the_largest_plain_figure_keeps_every_digit(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (1e15, "1 × 10^15"),
        (2.0**60, "1.153 × 10^18"),
        (1e301, "1 × 10^301"),
        (-2.5e17, "-2.5 × 10^17"),
        (9.9996e20, "1 × 10^21"),
        (1e-16, "1 × 10^-16"),
        (-1.2346e-20, "-1.235 × 10^-20"),
    ],
)
def test_a_figure_beyond_float_digits_is_mantissa_and_power(value, written):
    figure = rt._derivation_figure(value)
    assert figure == written
    assert len(figure) < 20


@pytest.mark.parametrize(
    ("text", "figure"),
    [
        ("Area: 0.0000001 x 0.000001 = 0.6 m2.", "0.0000000000001"),
        ("Strain: 0.000002 x 0.00003 = 0.4 of the bar.", "0.00000000006"),
    ],
)
def test_the_note_never_calls_a_tiny_product_zero(text, figure):
    note = rt._annotate_derivation_mismatches(text)
    assert f"**{figure}**" in note
    assert "**0**" not in note


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (12.345, "12.35"),
        (2.0345, "2.035"),
        (0.0012345, "0.001235"),
        (0.0020345, "0.002035"),
    ],
)
def test_a_plain_figure_rounds_half_up_from_its_shortest_form(value, written):
    # Each value sits on a half at the fourth figure as written. Half-even
    # rounds each one down, and so does reading the float's binary value for
    # every one but the first (whose binary value lies just over the half).
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (1.2345e20, "1.235 × 10^20"),
        (-6.0345e-18, "-6.035 × 10^-18"),
    ],
)
def test_a_mantissa_rounds_half_up_from_its_shortest_form(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (999999999999999.9, "1 × 10^15"),
        (999999999999999.5, "1 × 10^15"),
        (-999999999999999.9, "-1 × 10^15"),
        (9.99996e-16, "0.000000000000001"),
        (9.9996e-16, "0.000000000000001"),
    ],
)
def test_a_figure_that_rounds_across_a_bound_takes_the_form_of_what_it_shows(
    value, written
):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (999999999999999.4, "999,999,999,999,999"),
        (999999999999998.9, "999,999,999,999,999"),
        (9.994e-16, "9.994 × 10^-16"),
        (9.9994e-16, "9.999 × 10^-16"),
    ],
)
def test_a_figure_that_stays_inside_its_bound_keeps_its_form(value, written):
    assert rt._derivation_figure(value) == written


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (999999999999999.9, 1e15),
        (9.99996e-16, 1e-15),
    ],
)
def test_one_displayed_quantity_has_one_form(a, b):
    assert rt._derivation_figure(a) == rt._derivation_figure(b)
