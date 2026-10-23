"""Calculator outputs print the unit and label the registry declares.

A unit token that names another field (``duration_units``) is not a unit.
A per-day rate is a duration in days, so it is not labelled as man-hours.
"""
from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas_planning import parse_unit
from app.agents.driver import render
from app.lib.construction_formulas import run_calculation
from app.lib.formula_registry import all_specs

_BARE = {"", "-", "currency", "ratio", "count"}


def test_lane_output_units_are_real_units():
    """Planning and procurement declare a converter unit, or no unit."""
    offenders = []
    for spec in all_specs():
        if spec.owner not in ("planning", "procurement"):
            continue
        for name, unit in spec.outputs.items():
            token = str(unit or "")
            if token in _BARE or token.startswith("currency"):
                continue
            if parse_unit(token) is None:
                offenders.append(f"{spec.name}.{name}={token}")
    assert offenders == []


@pytest.mark.parametrize("rate_unit", ["per day", "m2 per gang-day"])
@pytest.mark.parametrize(
    "params",
    [
        {"quantity": 80, "productivity": 2, "productivity_rate": 2},
        {"quantity": 80, "productivity": 2},
    ],
)
def test_a_per_day_rate_prints_days_not_a_placeholder_or_manhours(params, rate_unit):
    params = {**params, "rate_unit": rate_unit}
    env = run_calculation("productivity_manpower_duration", params)
    assert env["status"] == "success", env
    text = render.render_calculation(env, params)
    assert "Result: duration 40 days." in text
    assert "manhours required" not in text.lower()
    assert "40 h" not in text
    assert "duration_units" not in text
    assert "duration units" not in text.lower()


def test_manhour_productivity_keeps_the_declared_hour_unit():
    params = {"quantity": 80, "productivity": 2}
    env = run_calculation("productivity_manpower_duration", params)
    text = render.render_calculation(env, params)
    assert "manhours required 40 h" in text


def _assert_days(env, inputs):
    assert env.get("status") == "success", env
    text = render.render_calculation(env, inputs)
    assert "Result: duration 40 days." in text
    assert "manhours required" not in text.lower()
    assert "40 h" not in text


@pytest.mark.parametrize("written", ["2 m2 per gang-day", "2 m2/gang-day"])
def test_construction_calc_keeps_a_gang_day_rate_out_of_manhours(written):
    """The live tool passes the rate with its unit on the value."""
    import asyncio

    from app.agents.base.tools import handle_construction_calc
    from app.agents.core.tool_registry import ToolCall

    args = {
        "calculation": "productivity_manpower_duration",
        "params": {"quantity": 80, "productivity": written},
    }
    call = ToolCall(
        agent=None, name="construction_calc", args=args,
        user_message="How long does 80 m2 take at a gang-day rate?",
    )
    out = asyncio.run(handle_construction_calc(call))
    inputs = args["params"]
    _assert_days(out["result"], inputs)


@pytest.mark.parametrize("written", ["2 m2 per gang-day", "2 m2/gang-day"])
def test_workflow_envelope_keeps_a_gang_day_rate_out_of_manhours(written):
    """A workflow envelope reaches the same calculator through the container."""
    import asyncio

    from app.containers.construction import ConstructionContainer

    envelope = {
        "action": "construction_calc",
        "message": "How long does 80 m2 take at a gang-day rate?",
        "calculation": "productivity_manpower_duration",
        "quantity": 80,
        "productivity": written,
    }
    out = asyncio.run(ConstructionContainer().construction_calc(envelope, {}))
    _assert_days(out, {"quantity": 80, "productivity": written})


def test_an_hour_unit_on_the_rate_stays_manhours():
    env = run_calculation("productivity_manpower_duration", {
        "quantity": 80,
        "productivity_rate": "2 per hour",
    })
    text = render.render_calculation(env, {"quantity": 80, "productivity_rate": "2 per hour"})
    assert env["status"] == "success", env
    assert "Result: manhours required 40 h." in text
    assert "40 days" not in text
