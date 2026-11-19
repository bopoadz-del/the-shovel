"""KS-2: on every turn the model is offered each calculator WITH its labelled
inputs, units and ranges, built from the same registry the driver schema uses.

The default (non-driver) chat path used to offer ``construction_calc`` as a
bare enum of names; only driver mode carried the labelled schema. Both paths
now read one builder, ``driver.tools.calculator_catalogue``.

Each test builds its own registry (two synthetic formulas: one in the base
offer, one only in another hat), so nothing here names a real calculator.
"""
from types import SimpleNamespace

import pytest

from app.agents.driver import tools as driver_tools
from app.lib import formula_registry


def _param(label, unit, lo, hi, required):
    return SimpleNamespace(label=label, unit=unit, lo=lo, hi=hi, required=required)


def _spec(name, display, owner, desc, params):
    return SimpleNamespace(name=name, display_name=display, owner=owner, description=desc,
                           fn=lambda **_: None, _params=params)


ALPHA = _spec("zz_alpha_calc", "Alpha check", "base", "Checks alpha.", {
    "span_m": _param("clear span", "m", 0.5, 40.0, True),
    "factor": _param("", "-", None, None, False),
})
BETA = _spec("zz_beta_calc", "Beta load", "otherhat", "Loads beta.", {
    "pressure_kpa": _param("design pressure", "kPa", 1.0, 900.0, True),
    "count": _param("number of bays", "", None, None, False),
})


@pytest.fixture
def two_formula_registry(monkeypatch):
    monkeypatch.setattr(formula_registry, "specs_for", lambda hat: [ALPHA])
    monkeypatch.setattr(formula_registry, "all_specs", lambda: [ALPHA, BETA])
    monkeypatch.setattr(formula_registry, "parameters", lambda spec: spec._params)
    driver_tools._SCHEMA_CACHE.clear()
    yield
    driver_tools._SCHEMA_CACHE.clear()


def _default_path_description():
    from app.agents.runtime import _construction_calc_tool_schema
    return _construction_calc_tool_schema()["function"]["description"]


@pytest.mark.parametrize("spec,key,expected", [
    # base-offer formula: label, unit and range
    (ALPHA, "span_m", "span_m = clear span (m, 0.5 to 40)"),
    # formula offered only in another hat: label, unit and range too
    (BETA, "pressure_kpa", "pressure_kpa = design pressure (kPa, 1 to 900)"),
])
def test_default_path_offers_label_unit_and_range(two_formula_registry, spec, key, expected):
    desc = _default_path_description()
    assert f"- {spec.name}: {spec.display_name}" in desc
    assert expected in desc


@pytest.mark.parametrize("key,expected", [
    # unitless optional input with no label: key-derived label, marked optional
    ("factor", "factor = factor (optional)"),
    # unitless optional input with a label
    ("count", "count = number of bays (optional)"),
])
def test_optional_inputs_are_marked_on_the_default_path(two_formula_registry, key, expected):
    assert expected in _default_path_description()


def test_both_paths_offer_the_same_catalogue(two_formula_registry):
    driver_desc = driver_tools.calculator_schema(None)["function"]["description"]
    default_desc = _default_path_description()
    _specs, lines = driver_tools.calculator_catalogue(None)
    assert [s.name for s in _specs] == ["zz_alpha_calc", "zz_beta_calc"]
    for line in lines:
        assert line in driver_desc
        assert line in default_desc


def test_a_formula_added_to_the_registry_is_offered_without_code(monkeypatch, two_formula_registry):
    gamma = _spec("zz_gamma_calc", "Gamma rate", "thirdhat", "Rates gamma.", {
        "rate_per_day": _param("output rate", "m2/day", 1.0, 500.0, True),
    })
    monkeypatch.setattr(formula_registry, "all_specs", lambda: [ALPHA, BETA, gamma])
    desc = _default_path_description()
    assert "rate_per_day = output rate (m2/day, 1 to 500)" in desc
    # and one that is not registered is not offered
    assert "zz_delta_calc" not in desc
