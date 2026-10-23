"""``count`` is an element quantity. Dropping it prices one element.

Live construction_calc call (figures only)::

    calculation=concrete_volume
    length=2.5, width=2.5, depth=1.2, thickness=1.2, count=24, waste_factor=0.07

One element is 2.5 x 2.5 x 1.2 = 7.5 m3, x 1.07 = 8.025 m3.
Twenty-four elements are 180.0 m3 net, 192.6 m3 with 7% waste.
192.6 x 420 = 80892.00. The calculator does not price; the cost
assertion is at this test only.

``quantity=24`` already returns the twenty-four-element totals.
``count`` must do the same on every calculator that takes ``quantity``.
"""
from __future__ import annotations

import inspect

import pytest

from app.lib.construction_formulas import (
    CALCULATORS,
    bind_calculation_params,
    run_calculation,
)
from tests.test_construction_calc_tool import _agent, _call

_L = 2.5
_W = 2.5
_T = 1.2
_N = 24
_WASTE = 0.07
_RATE = 420
_NET = _L * _W * _T * _N  # 180
_WITH_WASTE = _NET * (1.0 + _WASTE)  # 192.6
_COST = _WITH_WASTE * _RATE  # 80892
_ONE = _L * _W * _T * (1.0 + _WASTE)  # 8.025


def _volume(params: dict):
    env = run_calculation("concrete_volume", params)
    assert env.get("status") == "success", env
    return env["result"]


def _assert_twenty_four(inner: dict) -> None:
    assert inner["net_volume_m3"] == pytest.approx(_NET)
    assert inner["volume_m3"] == pytest.approx(_WITH_WASTE)
    assert inner["volume_m3"] != pytest.approx(_ONE)
    assert inner["quantity"] == pytest.approx(_N)
    priced = round(float(inner["volume_m3"]) * _RATE, 2)
    assert priced == pytest.approx(_COST)
    assert f"{priced:,.2f}" == "80,892.00"


def test_count_24_with_depth_and_thickness_is_twenty_four_elements():
    inner = _volume({
        "length": _L,
        "width": _W,
        "depth": _T,
        "thickness": _T,
        "count": _N,
        "waste_factor": _WASTE,
    })
    _assert_twenty_four(inner)


def test_count_24_without_the_redundant_depth_is_the_same_total():
    inner = _volume({
        "length": _L,
        "width": _W,
        "thickness": _T,
        "count": _N,
        "waste_factor": _WASTE,
    })
    _assert_twenty_four(inner)
    depth_only = _volume({
        "length": _L,
        "width": _W,
        "depth": _T,
        "count": _N,
        "waste_factor": _WASTE,
    })
    _assert_twenty_four(depth_only)


def test_count_as_numeric_string_matches_the_integer_count():
    inner = _volume({
        "length": _L,
        "width": _W,
        "thickness": _T,
        "count": "24",
        "waste_factor": _WASTE,
    })
    _assert_twenty_four(inner)


def test_count_and_quantity_when_equal_are_not_doubled():
    inner = _volume({
        "length": _L,
        "width": _W,
        "depth": _T,
        "thickness": _T,
        "count": _N,
        "quantity": _N,
        "waste_factor": _WASTE,
    })
    _assert_twenty_four(inner)


def test_count_and_quantity_when_they_differ_is_a_tool_error():
    params = {
        "length": _L,
        "width": _W,
        "thickness": _T,
        "count": 24,
        "quantity": 12,
        "waste_factor": _WASTE,
    }
    env = run_calculation("concrete_volume", params)
    assert env.get("status") == "error", env
    err = str(env.get("error") or "")
    assert "count" in err
    assert "quantity" in err
    assert "24" in err
    assert "12" in err
    tool = _call(_agent(["construction"]), "concrete_volume", params)
    assert tool.get("ok") is False, tool
    assert tool["result"]["status"] == "error"
    tool_err = str(tool["result"].get("error") or "")
    assert "24" in tool_err and "12" in tool_err
    assert "count" in tool_err and "quantity" in tool_err


def test_rebar_weight_count_matches_quantity():
    common = {"bar_diameter_mm": 16, "total_length_m": 12.5}
    by_quantity = run_calculation("rebar_weight", {**common, "quantity": 8})
    by_count = run_calculation("rebar_weight", {**common, "count": 8})
    by_string = run_calculation("rebar_weight", {**common, "count": "8"})
    one = run_calculation("rebar_weight", {**common, "quantity": 1})
    assert by_quantity["status"] == "success", by_quantity
    assert by_count["status"] == "success", by_count
    assert by_string["status"] == "success", by_string
    mass = by_quantity["result"]["total_mass_kg"]
    assert by_count["result"]["total_mass_kg"] == pytest.approx(mass)
    assert by_string["result"]["total_mass_kg"] == pytest.approx(mass)
    assert mass != pytest.approx(one["result"]["total_mass_kg"])


def test_unknown_argument_is_a_retryable_tool_error():
    params = {
        "length": _L,
        "width": _W,
        "thickness": _T,
        "quantity": 1,
        "waste_factor": _WASTE,
        "bogus_param": 1,
    }
    env = run_calculation("concrete_volume", params)
    assert env.get("status") == "error", env
    assert "bogus_param" in str(env.get("error") or "")
    assert "signature" in env
    assert isinstance(env.get("expected_params"), list)
    tool = _call(_agent(["construction"]), "concrete_volume", params)
    assert tool.get("ok") is False, tool
    assert tool["result"]["status"] == "error"
    assert "bogus_param" in str(tool["result"].get("error") or "")


def test_count_aliases_quantity_on_every_calculator_that_takes_it():
    found: list[str] = []
    for name, fn in sorted(CALCULATORS.items()):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        if "quantity" not in sig.parameters:
            continue
        found.append(name)
        bound = bind_calculation_params(fn, {"count": 6})
        assert bound.get("quantity") == 6, (name, bound)
    assert "concrete_volume" in found
    assert "rebar_weight" in found
    assert "unit_cost_total" in found
    assert "resource_line_cost" in found
    assert len(found) >= 4


def test_platform_context_keys_are_not_unknown_arguments():
    """text / prior_text / formula / name / calculation are platform keys."""
    inner = _volume({
        "length": _L,
        "width": _W,
        "thickness": _T,
        "count": _N,
        "waste_factor": _WASTE,
        "text": "2.5 x 2.5 x 1.2",
        "prior_text": "no element count here",
        "formula": "L*W*T",
        "name": "concrete_volume",
        "calculation": "concrete_volume",
        "message": "volume",
        "query": "volume",
    })
    _assert_twenty_four(inner)


def test_var_keyword_calculators_are_the_only_unknown_arg_swallowers():
    """Audit: only a ``**kwargs`` signature can absorb an unbound key.

    ``concrete_mix_slip_form`` documents that extra kwargs are ignored so a
    live probe that sends slump / w/c next to the name still returns the
    locked mix. Every other calculator must surface an unknown key as a
    tool error instead of dropping it.
    """
    swallowers: list[str] = []
    for name, fn in sorted(CALCULATORS.items()):
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue
        params = sig.parameters.values()
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
            swallowers.append(name)
    assert swallowers == ["concrete_mix_slip_form"]
