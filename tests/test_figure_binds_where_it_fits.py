"""A figure in the ask binds only to an input that can hold it.

Two rules, each read from the calculator's own signature, never from a
case name:

  type      - an input declared as text, a flag, a list or a mapping never
              takes a lone figure by its unit or as a bare number;
  dimension - a figure written with its dimension adjective ("250 mm
              thick") is that dimension, so it binds by unit only to an
              input that measures it, never to a span or a diameter.

Each calculator here is built in the test, so no registered formula and no
product answer supplies an expected value. Every decision has two inputs.
"""
from __future__ import annotations

from typing import List

import pytest

from app.lib.construction_formulas import extract_calculation_params_from_text as read


# --- type: a bare number --------------------------------------------------

def _text_input(note: str) -> dict:
    return {"note": note}


def _number_input(count: float) -> dict:
    return {"count": count}


@pytest.mark.parametrize("figure", [310, 280])
def test_bare_figure_never_binds_to_a_text_input(figure):
    ask = f"Concrete at {figure} per m3. Cost?"
    assert read(_text_input, ask) == {}


@pytest.mark.parametrize("figure", [310, 280])
def test_bare_figure_still_binds_to_a_number_input(figure):
    ask = f"Concrete at {figure} per m3. Cost?"
    assert read(_number_input, ask) == {"count": float(figure)}


# --- type: a figure written with the input's unit -------------------------

def _bore_list(bores_mm: List[float]) -> dict:
    return {"bores_mm": bores_mm}


def _bore(bore_mm: float) -> dict:
    return {"bore_mm": bore_mm}


@pytest.mark.parametrize("figure", [100, 80])
def test_unit_figure_never_binds_to_a_list_input(figure):
    assert read(_bore_list, f"Ducts of {figure} mm, grout needed?") == {}


@pytest.mark.parametrize("figure", [100, 80])
def test_unit_figure_still_binds_to_a_number_input(figure):
    assert read(_bore, f"Ducts of {figure} mm, grout needed?") == {"bore_mm": float(figure)}


# --- type: a figure in another unit of the same kind ----------------------

def _size_list(sizes_mm: List[float]) -> dict:
    return {"sizes_mm": sizes_mm}


def _size(size_mm: float) -> dict:
    return {"size_mm": size_mm}


@pytest.mark.parametrize("feet", [15.75, 10.0])
def test_converted_figure_never_binds_to_a_list_input(feet):
    assert read(_size_list, f"A member {feet} ft, check it.") == {}


@pytest.mark.parametrize("feet, mm", [
    (15.75, 4800.6),  # 15.75 ft x 304.8 mm/ft
    (10.0, 3048.0),   # 10 ft x 304.8 mm/ft
])
def test_converted_figure_still_binds_to_a_number_input(feet, mm):
    out = read(_size, f"A member {feet} ft, check it.")
    assert out["size_mm"] == pytest.approx(mm, rel=1e-6)


# --- dimension: the adjective after the figure ----------------------------

def _span(span_mm: float) -> dict:
    return {"span_mm": span_mm}


def _raft(raft_thickness_mm: float) -> dict:
    return {"raft_thickness_mm": raft_thickness_mm}


@pytest.mark.parametrize("figure", [250, 200])
def test_thick_figure_never_binds_to_a_span(figure):
    assert read(_span, f"Slab {figure} mm thick, is it enough?") == {}


@pytest.mark.parametrize("figure", [250, 200])
def test_thick_figure_binds_to_the_input_that_measures_thickness(figure):
    out = read(_raft, f"Raft {figure} mm thick, check uplift.")
    assert out == {"raft_thickness_mm": float(figure)}


@pytest.mark.parametrize("figure", [250, 200])
def test_figure_without_an_adjective_still_binds_by_its_unit(figure):
    assert read(_span, f"Opening {figure} mm, is it enough?") == {"span_mm": float(figure)}


# --- type: an input the signature does not declare a type for -------------

from app.lib.construction_formulas import _takes_a_figure  # noqa: E402


def _open_inputs(a, b):
    return {"a": a, "b": b}


def _typed_inputs(note: str, rows: List[float]) -> dict:
    return {"note": note, "rows": rows}


@pytest.mark.parametrize("fn, name", [(_open_inputs, "a"), (_open_inputs, "b")])
def test_an_undeclared_type_is_open_to_a_figure(fn, name):
    assert _takes_a_figure(fn, name) is True


@pytest.mark.parametrize("fn, name", [(_typed_inputs, "depth"), (_open_inputs, "span")])
def test_a_name_the_signature_lacks_is_read_as_open_not_a_crash(fn, name):
    assert _takes_a_figure(fn, name) is True


@pytest.mark.parametrize("name", ["note", "rows"])
def test_a_declared_text_or_list_input_is_closed_to_a_figure(name):
    assert _takes_a_figure(_typed_inputs, name) is False
