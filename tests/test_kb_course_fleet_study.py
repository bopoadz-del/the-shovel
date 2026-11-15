"""Course program, owner Ruling 2: the fleet study calculator, the course's way.

The two fleet studies of the owner-supplied course material (roads and
earthworks section 2, p26-30; blocks W7 and W8, CONFIRMED in verify file C)
reach their hauling-machine counts by one route: the output of one machine
at each of the three haul distances, (60 / cycle) x load, is combined into
the hourly rate per machine, and the count is the required hourly rate over
that combined rate, rounded up. The course's printed counts are the expected
answers. Each required rate is derived from the study's own inputs through
``equipment.required_hourly_rate``; no printed intermediate figure is used.
Invented cases state their hand calculation.
"""
from __future__ import annotations

import pint
import pytest
import sympy

from app.blocks import _knowledge as kb

RATE = "equipment.fleet_study_hourly_rate"
COUNT = "equipment.fleet_study_machines"
NEW_IDS = [RATE, COUNT]

_UREG = pint.UnitRegistry()


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


def cycles(short, middle, long_):
    return {"cycle_time_short": short, "cycle_time_middle": middle, "cycle_time_long": long_}


@pytest.mark.parametrize("rule_id", NEW_IDS)
def test_entry_carries_course_source_and_declared_units(rule_id):
    e = kb.get_rule(rule_id)
    assert e is not None and e["type"] == "formula"
    assert e["provenance"]["course_reference"].startswith("owner-supplied course material, ")
    assert e["provenance"]["project"] is None
    assert e["applicability"]["project_specific"] is False
    assert "not a code or standard requirement" in e["statement"]
    symbols = {str(s) for s in sympy.sympify(e["expression"]).free_symbols}
    assert symbols == set(e["variables"]), rule_id
    for v in e["variables"].values():
        _UREG(v["unit"])
    if e["thresholds"]["unit"] != "machines":  # a count, as equipment.machines_required
        _UREG(e["thresholds"]["unit"])


@pytest.mark.parametrize(
    "cycle_var, cycle, load, per_hour",
    [
        ("cycle_time_short", 1, 1, 60.0),  # hand: 60 / 1 x 1 = 60 m3/hr
        ("cycle_time_long", 4, 2, 30.0),   # hand: 60 / 4 x 2 = 30 m3/hr
    ],
)
def test_rate_unit_is_volume_per_hour_from_declared_variables(cycle_var, cycle, load, per_hour):
    e = kb.get_rule(RATE)
    # the 60 is minutes per hour: (60 min/hr / cycle min) x load m3 -> m3/hr
    q = (60 * _UREG("minute/hour") / (cycle * _UREG(e["variables"][cycle_var]["unit"]))) * (
        load * _UREG(e["variables"]["unit_capacity"]["unit"])
    )
    assert q.to(e["thresholds"]["unit"]).magnitude == pytest.approx(per_hour)


@pytest.mark.parametrize(
    "load, c, combined",
    [
        (20, (10, 12, 15), 300.0),  # course W7 scrapers: 120 + 100 + 80 = 300 m3/hr
        (14, (20, 30, 40), 91.0),   # course W8 trucks: 42 + 28 + 21 = 91 m3/hr
        (10, (5, 10, 20), 210.0),   # invented: 120 + 60 + 30 = 210 m3/hr
    ],
)
def test_hourly_rate_per_machine_combines_the_three_distances(load, c, combined):
    out = ev(RATE, unit_capacity=load, **cycles(*c))
    assert out["result"] == pytest.approx(combined)
    assert out["unit"] == "meter**3/hour"


def _required_rate(quantity, days, hours):
    return ev("equipment.required_hourly_rate",
              total_quantity=quantity, working_days=days, hours_per_day=hours)["result"]


@pytest.mark.parametrize(
    "quantity, load, c, unrounded, machines",
    [
        # course W7: 9.3e6 / (210 x 20) = 2,214.3 m3/hr; / 300 = 7.38, say 8 scrapers
        (9.3e6, 20, (10, 12, 15), 7.381, 8),
        # course W8: 10e6 / (210 x 20) = 2,381 m3/hr; / 91 = 26.2 -> 27 trucks
        (10e6, 14, (20, 30, 40), 26.164, 27),
    ],
)
def test_course_fleet_studies_give_the_printed_counts(quantity, load, c, unrounded, machines):
    required = _required_rate(quantity, 210, 20)
    out = ev(COUNT, required_rate=required, unit_capacity=load, **cycles(*c))
    assert out["result"] == machines
    assert out["unrounded"] == pytest.approx(unrounded, abs=1e-3)
    assert out["unit"] == "machines"


@pytest.mark.parametrize(
    "required, machines",
    [
        (420.0, 2),   # invented: 420 / 210 = 2.0 exactly -> 2, not rounded past an exact count
        (421.0, 3),   # invented: 421 / 210 = 2.005 -> 3, a fraction of a machine is a machine
        (1000.0, 5),  # invented: 1000 / 210 = 4.76 -> 5
    ],
)
def test_count_rounds_up_at_the_boundary(required, machines):
    out = ev(COUNT, required_rate=required, unit_capacity=10, **cycles(5, 10, 20))
    assert out["result"] == machines


def test_each_distance_counts_in_the_combined_rate():
    base = ev(RATE, unit_capacity=10, **cycles(5, 10, 20))["result"]
    for key in ("cycle_time_short", "cycle_time_middle", "cycle_time_long"):
        c = cycles(5, 10, 20)
        c[key] = c[key] * 2  # halves that distance's output only
        assert ev(RATE, unit_capacity=10, **c)["result"] < base, key


@pytest.mark.parametrize(
    "rule_id, partial",
    [
        (RATE, {"unit_capacity": 20, "cycle_time_short": 10, "cycle_time_middle": 12}),
        (COUNT, {"unit_capacity": 20, **cycles(10, 12, 15)}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


@pytest.mark.parametrize(
    "rule_id, expression",
    [
        ("equipment.cycle_production", "60 / cycle_time * unit_capacity"),
        ("equipment.machines_required", "ceiling(required_rate / production_per_machine)"),
    ],
)
def test_single_haul_calculators_are_left_as_they_are(rule_id, expression):
    # Ruling 2: the general cycle formulas stay separate calculators.
    assert kb.get_rule(rule_id)["expression"] == expression
