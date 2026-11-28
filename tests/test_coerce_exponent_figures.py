"""A figure written with an exponent is read whole at the calculator's front
door: "3e1" is 30, not 3 followed by a unit "e". Every case builds its own
input; each decision is checked with at least two different figures."""
from __future__ import annotations

import math

import pytest

from app.lib import construction_formulas as cf


def _result(name, params):
    env = cf.run_calculation(name, params)
    inner = env.get("result", env)
    assert isinstance(inner, dict) and not inner.get("error"), env
    return inner


@pytest.mark.parametrize("length, expected", [("3e1", 6.0), ("1.2E+2", 24.0)])
def test_concrete_volume_reads_an_exponent_length_whole(length, expected):
    out = _result("concrete_volume", {"length_m": length, "width_m": "2",
                                      "thickness_m": "0.1", "waste_factor": 0})
    assert math.isclose(out["net_volume_m3"], expected)


@pytest.mark.parametrize("depth, expected_bank", [("1.5e0", 120.0), ("2.5E0", 200.0)])
def test_excavation_volume_reads_an_exponent_depth_whole(depth, expected_bank):
    out = _result("excavation_volume", {"length_m": "4e1", "width_m": "2", "depth_m": depth})
    assert math.isclose(out["bank_volume_m3"], expected_bank)


@pytest.mark.parametrize("thickness, expected", [("1e-1", 6.0), ("2.5E-1", 15.0)])
def test_a_negative_exponent_makes_the_figure_smaller(thickness, expected):
    out = _result("concrete_volume", {"length_m": "30", "width_m": "2",
                                      "thickness_m": thickness, "waste_factor": 0})
    assert math.isclose(out["net_volume_m3"], expected)


@pytest.mark.parametrize("text, expected", [
    ("1.25e+06", 1250000.0),
    ("3e1", 30.0),
    ("-2.5E-3", -0.0025),
    ("4E2 m", 400.0),
    ("7e-2 m3", 0.07),
])
def test_coerce_scalar_keeps_the_exponent(text, expected):
    got = cf._coerce_scalar(text)
    assert isinstance(got, float)
    assert math.isclose(got, expected)


@pytest.mark.parametrize("text, expected", [("12", 12), ("- 5", -5), ("1,250", 1250), ("3 each", 3)])
def test_a_figure_without_point_or_exponent_stays_an_int(text, expected):
    got = cf._coerce_scalar(text)
    assert got == expected and isinstance(got, int)


@pytest.mark.parametrize("text", ["abc", "e5", "--5", "+-3", "m 12"])
def test_text_that_is_not_a_figure_is_left_as_written(text):
    assert cf._coerce_scalar(text) == text


@pytest.mark.parametrize("text, expected", [("4e2", 400.0), ("6.5E-1", 0.65)])
def test_a_bare_exponent_figure_is_a_bare_figure(text, expected):
    assert math.isclose(cf._bare_figure(text), expected)


def test_a_bare_figure_with_a_unit_or_words_is_not_bare():
    assert cf._bare_figure("4e2 m") is None
    assert cf._bare_figure("about 3e1") is None


@pytest.mark.parametrize("text, expected", [("1e2, 2e2", [100.0, 200.0]), ("3e0 4e-1", [3.0, 0.4])])
def test_a_list_of_exponent_figures_is_read_item_by_item(text, expected):
    got = cf._coerce_list(text)
    assert len(got) == len(expected)
    assert all(math.isclose(a, b) for a, b in zip(got, expected))


def test_a_refused_text_input_still_fails_the_calculation():
    env = cf.run_calculation("concrete_volume", {"length_m": "thirty", "width_m": "2",
                                                 "thickness_m": "0.1"})
    inner = env.get("result", env)
    assert env.get("status") != "success" or (isinstance(inner, dict) and inner.get("error")), env
