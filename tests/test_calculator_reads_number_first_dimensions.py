"""The calculator reads dimensions written the way a site engineer writes
them -- "12 m long, 8 m wide and 200 mm thick" -- converts each to the
unit its parameter declares, infers the shape from the dimensions given,
and asks rather than returning 0 when none was given."""
from __future__ import annotations

import math

import pytest

from app.lib import construction_formulas as cf


def _bind(name, text):
    return cf.extract_calculation_params_from_text(cf.CALCULATORS[name], text)


@pytest.mark.parametrize("text", [
    "How much concrete do I need for a ground slab 12 m long, 8 m wide and 200 mm thick?",
    "slab length 12 m, width 8 m, thickness 200 mm",
    "a slab 1200 cm long, 8 m wide, 0.2 m thick",
])
def test_slab_dimensions_bind_in_metres(text):
    p = _bind("concrete_volume", text)
    assert p == pytest.approx({"length_m": 12.0, "width_m": 8.0, "thickness_m": 0.2})
    out = cf.run_calculation("concrete_volume", p)["result"]
    assert out["net_volume_m3"] == pytest.approx(19.2)


def test_a_column_given_by_diameter_and_height_is_a_cylinder():
    p = _bind("concrete_volume", "a circular column 0.6 m in diameter and 3.5 m high")
    assert p == pytest.approx({"diameter_m": 0.6, "height_m": 3.5})
    out = cf.run_calculation("concrete_volume", p)["result"]
    assert out["net_volume_m3"] == pytest.approx(math.pi * 0.3 ** 2 * 3.5, rel=1e-3)


def test_a_millimetre_parameter_takes_metres_converted():
    assert cf._to_param_unit(0.2, "m", "thickness_mm") == pytest.approx(200.0)
    assert cf._to_param_unit(200, "mm", "thickness_m") == pytest.approx(0.2)
    assert cf._to_param_unit(5, None, "thickness_m") == 5


def test_a_volume_with_no_dimensions_asks_instead_of_returning_zero():
    out = cf.run_calculation("concrete_volume", {})
    assert out["status"] == "error" and "missing required" in out["error"]


def test_a_deflection_question_binds_every_input():
    """'E is 30,000 MPa and I is 5.4 x 10^9 mm4', 'a uniform load of 20 kN/m'."""
    p = _bind("beam_deflection_ss_udl",
              "A simply supported beam spans 6 m with a uniform load of 20 kN/m. "
              "E is 30,000 MPa and I is 5.4 x 10^9 mm4. How much will it deflect?")
    assert p == pytest.approx({"span_m": 6.0, "w_kn_m": 20.0, "ec_mpa": 30000.0, "i_mm4": 5.4e9})
    assert cf.run_calculation("beam_deflection_ss_udl", p)["result"]["value"] == pytest.approx(2.08, abs=0.01)


@pytest.mark.parametrize("raw,value", [("5.4 x 10^9", 5.4e9), ("5.4×10^9", 5.4e9), ("2.1e5", 2.1e5),
                                       ("1,250", 1250.0), ("0.2", 0.2)])
def test_figures_in_engineering_notation(raw, value):
    assert cf._parse_text_number(raw) == pytest.approx(value)


def test_a_figures_unit_is_matched_against_the_formula_declaration():
    assert cf._declared_units(cf.CALCULATORS["beam_deflection_ss_udl"])["w_kn_m"] == "kN/m"
    p = _bind("beam_deflection_ss_udl", "6 m span, 20 kN/m, E 30000 MPa, I 5.4e9 mm4")
    assert p.get("w_kn_m") == pytest.approx(20.0)


@pytest.mark.parametrize("num,unit,dest,expected", [
    (2.0e-4, "m4", "i_mm4", 2.0e8), (200, "GPa", "ec_mpa", 200000.0), (30, "N/mm2", "fck_mpa", 30.0),
    (200, "mm", "thickness_m", 0.2),
])
def test_figures_convert_within_their_unit_family(num, unit, dest, expected):
    assert cf._to_param_unit(num, unit, dest) == pytest.approx(expected)


def test_a_figure_of_another_kind_keeps_its_unit_for_the_input_check():
    assert cf._to_param_unit(5, "kN", "thickness_m") == "5 kN"
