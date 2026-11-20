"""Course program, file C, second formula batch: required output rates with
the shift split, and the sand-cone sand calibration.

Every entry is a ``type=formula`` knowledge entry evaluated by
``app.blocks._knowledge.evaluate``. Values come from the owner-supplied
course material (roads and earthworks section 2) as checked in its verify
file, or are invented; each case states its hand calculation.

The two fleet studies' target-rate steps are NOT used as test oracles
(advisor ruling 09:21Z: their printed answers stay out of the tests); the
rate formulas are tested on invented cases.
"""
from __future__ import annotations

import pytest
import sympy

from app.blocks import _knowledge as kb

NEW_IDS = [
    "equipment.required_daily_rate",
    "equipment.shift_quantity",
    "equipment.required_hourly_rate",
    "earthworks.sand_bulk_density",
]


def ev(rule_id, **values):
    return kb.evaluate(rule_id, **values)


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
    assert all(v.get("unit") for v in e["variables"].values())
    assert e["thresholds"]["unit"]


@pytest.mark.parametrize(
    "rule_id, partial",
    [
        ("equipment.required_hourly_rate", {"total_quantity": 100000, "working_days": 100}),
        ("equipment.shift_quantity", {"daily_quantity": 1000, "shift_hours": 11}),
        ("earthworks.sand_bulk_density", {"sand_mass": 1512}),
    ],
)
def test_missing_required_input_raises_not_defaulted(rule_id, partial):
    with pytest.raises(ValueError):
        ev(rule_id, **partial)


# ---------------------------------------------------------------------------
# Required output: daily, per shift, hourly (invented cases, by hand)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quantity, days, daily",
    [
        (600000, 120, 5000.0),    # 600,000 / 120 = 5,000 m3/day
        (250000, 80, 3125.0),     # 250,000 / 80 = 3,125 m3/day
    ],
)
def test_required_daily_rate(quantity, days, daily):
    out = ev("equipment.required_daily_rate", total_quantity=quantity, working_days=days)
    assert out["result"] == pytest.approx(daily)


@pytest.mark.parametrize(
    "daily, shift_h, day_h, shift_q",
    [
        (5000, 11, 20, 2750.0),   # 5,000 x 11 / 20 = 2,750 m3 on the day shift
        (5000, 9, 20, 2250.0),    # 5,000 x 9 / 20 = 2,250 m3 on the night shift
        (3125, 8, 16, 1562.5),    # 3,125 x 8 / 16 = 1,562.5 m3
    ],
)
def test_shift_quantity(daily, shift_h, day_h, shift_q):
    out = ev("equipment.shift_quantity", daily_quantity=daily, shift_hours=shift_h, hours_per_day=day_h)
    assert out["result"] == pytest.approx(shift_q)


@pytest.mark.parametrize(
    "quantity, days, day_h, shifts",
    [
        # 600,000 / (120 x 20) = 250 m3/hr; day 2,750/11 = 250, night 2,250/9 = 250
        (600000, 120, 20, (11, 9)),
        # 250,000 / (80 x 16) = 195.3125 m3/hr; each 8 h shift 1,562.5/8 = 195.3125
        (250000, 80, 16, (8, 8)),
    ],
)
def test_hourly_rate_equals_every_shifts_rate(quantity, days, day_h, shifts):
    hourly = ev("equipment.required_hourly_rate", total_quantity=quantity,
                working_days=days, hours_per_day=day_h)["result"]
    assert hourly == pytest.approx(quantity / (days * day_h))
    daily = ev("equipment.required_daily_rate", total_quantity=quantity, working_days=days)["result"]
    for h in shifts:
        q = ev("equipment.shift_quantity", daily_quantity=daily, shift_hours=h, hours_per_day=day_h)["result"]
        assert q / h == pytest.approx(hourly)


@pytest.mark.parametrize(
    "quantity, days, day_h, unit_rate, machines",
    [
        # 250 m3/hr / 120 m3/hr per dozer = 2.083 -> 3 dozers
        (600000, 120, 20, 120, 3),
        # 195.3125 / 100 = 1.953 -> 2
        (250000, 80, 16, 100, 2),
    ],
)
def test_hourly_rate_feeds_machine_count(quantity, days, day_h, unit_rate, machines):
    hourly = ev("equipment.required_hourly_rate", total_quantity=quantity,
                working_days=days, hours_per_day=day_h)["result"]
    n = ev("equipment.machines_required", required_rate=hourly, production_per_machine=unit_rate)
    assert n["result"] == machines


# ---------------------------------------------------------------------------
# Sand calibration (invented, by hand); chain to hole volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sand, volume, density",
    [
        (1512, 1000, 1.512),   # 1,512 g / 1,000 cm3 = 1.512 g/cm3
        (4410, 3000, 1.47),    # 4,410 / 3,000 = 1.47 g/cm3
    ],
)
def test_sand_bulk_density_feeds_hole_volume(sand, volume, density):
    rho = ev("earthworks.sand_bulk_density", sand_mass=sand, container_volume=volume)["result"]
    assert rho == pytest.approx(density)
    # hole: (8000 - 2514 - 1564) = 3922 g of sand in the hole -> 3922 / rho
    V = ev("earthworks.sand_cone_hole_volume", sand_before=8000, sand_after=2514,
           sand_in_cone=1564, sand_density=rho)["result"]
    assert V == pytest.approx(3922 / density)
