"""Edge cases for the two bind rules in ``extract_calculation_params_from_text``:
a union type read member by member, a type name that only starts like a
container, two inputs measuring the same dimension, an adjective that is not
written straight after the figure, and a figure taken by its adjective that
must not come back as a bare number.

Every calculator is built here; no registered formula or product answer
supplies an expected value. Every decision has two inputs.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import (
    _takes_a_figure,
    extract_calculation_params_from_text as read,
)


# --- type: a written union is read member by member -----------------------

def _unions(a: str | None, b: float | None, c: str | float, d: list | None) -> dict:
    return {}


@pytest.mark.parametrize("name", ["a", "d"])
def test_optional_text_or_list_is_closed_to_a_figure(name):
    assert _takes_a_figure(_unions, name) is False


@pytest.mark.parametrize("name", ["b", "c"])
def test_a_union_with_a_number_member_is_open_to_a_figure(name):
    assert _takes_a_figure(_unions, name) is True


# --- type: a name that only begins like a container is not one -----------

class Settlement(float):
    pass


class Strand(float):
    pass


def _custom(s: Settlement, t: Strand) -> dict:
    return {}


@pytest.mark.parametrize("name", ["s", "t"])
def test_a_type_that_only_starts_like_a_container_is_open(name):
    assert _takes_a_figure(_custom, name) is True


# --- dimension: two inputs measuring it is no single fit ------------------

def _two_thick(top_thickness_mm: float, base_thickness_mm: float) -> dict:
    return {}


@pytest.mark.parametrize("figure", [250, 180])
def test_thick_figure_with_two_thickness_inputs_binds_neither(figure):
    assert read(_two_thick, f"Wall {figure} mm thick, check it.") == {}


# --- dimension: the adjective must follow the figure at once --------------

def _span(span_mm: float) -> dict:
    return {"span_mm": span_mm}


@pytest.mark.parametrize("figure", [250, 400])
def test_adjective_after_a_comma_does_not_name_the_figure(figure):
    assert read(_span, f"Opening {figure} mm, thick walls each side.") == {"span_mm": float(figure)}


# --- dimension: a figure it took is not read again as a bare number -------

def _span_and_count(span_mm: float, count: float) -> dict:
    return {}


@pytest.mark.parametrize("figure", [250, 300])
def test_thick_figure_is_not_reread_as_a_bare_count(figure):
    out = read(_span_and_count, f"Slab {figure} mm thick, how many?")
    assert "count" not in out and "span_mm" not in out


@pytest.mark.parametrize("ask", [
    "Slab 0.25 m thick, how many?",
    "Wall 1.5 m high, how many?",
])
def test_decimal_figure_taken_by_its_adjective_is_not_reread_in_part(ask):
    # The whole figure is spent by its adjective; no part of it ("0", "1")
    # may come back as a bare number for another input.
    assert read(_span_and_count, ask) == {}


# --- type: a callable with no readable signature --------------------------

@pytest.mark.parametrize("fn", [max, min])
def test_a_callable_without_a_signature_is_read_as_open_not_a_crash(fn):
    assert _takes_a_figure(fn, "x") is True
