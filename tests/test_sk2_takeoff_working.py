"""S-K2 item 5: take-off formulas print working that reproduces their figures.

The audit's S-K2 entry asks for "takeoff formulas with printed working (areas,
opening deductions, volumes, rebar by cited density)". The figures were already
right; these tests hold the WORKING to the same standard: every figure a
take-off reports appears in its own working at the precision it is reported,
a rate the working states is the rate the arithmetic used, and a quantity
floored at zero says so instead of printing ``a - b = 0``.

Every input is built here from a seeded generator; nothing is read from a
document and no expected value is a literal copied from the product.
"""
from __future__ import annotations

import math
import random
import re

import pytest

from app.agents.base.formulas.construction_formulas_earthwork import (
    backfill_volume,
    excavation_volume,
)
from app.agents.base.formulas.construction_formulas_quantities import concrete_volume
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    interior_finishes_takeoff,
    rebar_by_area,
)

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_SEEDS = (3, 17)


def _working(result: dict) -> str:
    """The printed working: the note, or the basis lines joined."""
    if "note" in result:
        return str(result["note"])
    return " ".join(str(line) for line in result.get("basis", ()))


def _numbers(text: str) -> list[float]:
    return [float(m) for m in _NUMBER.findall(text)]


def _shown(value: float, text: str, places: int) -> bool:
    """True when ``value`` is printed in ``text`` to within its reported rounding."""
    tol = 0.5 * 10 ** (-places) + 1e-9
    return any(abs(n - value) <= tol for n in _numbers(text))


def _waste_rates(rng: random.Random, n: int) -> list[float]:
    # Half-percent steps: the rates a whole-percent format misprints.
    return [rng.randrange(1, 40) / 200.0 for _ in range(n)]


@pytest.mark.parametrize("seed", _SEEDS)
def test_concrete_working_states_the_waste_rate_it_applied(seed):
    rng = random.Random(seed)
    for waste in _waste_rates(rng, 12):
        r = concrete_volume(
            length_m=round(rng.uniform(1, 40), 2),
            width_m=round(rng.uniform(1, 30), 2),
            thickness_m=round(rng.uniform(0.1, 2), 2),
            waste_factor=waste,
        )
        text = _working(r)
        assert _shown(waste * 100.0, text, 6), (waste, text)
        assert _shown(r["net_volume_m3"], text, 3), text
        assert _shown(r["volume_with_waste_m3"], text, 3), text


@pytest.mark.parametrize("seed", _SEEDS)
def test_excavation_working_reproduces_bank_and_loose(seed):
    rng = random.Random(seed)
    for _ in range(12):
        r = excavation_volume(
            length_m=round(rng.uniform(0.5, 60), 3),
            width_m=round(rng.uniform(0.5, 20), 3),
            depth_m=round(rng.uniform(0.2, 6), 3),
            bulking_factor=rng.randrange(5, 40) / 100.0,
        )
        text = _working(r)
        assert _shown(r["bank_volume_m3"], text, 3), text
        assert _shown(r["loose_volume_m3"], text, 3), text


@pytest.mark.parametrize("seed", _SEEDS)
def test_backfill_working_reproduces_void_and_import(seed):
    rng = random.Random(seed)
    for _ in range(12):
        bank = round(rng.uniform(50, 2000), 2)
        r = backfill_volume(
            excavation_bank_m3=bank,
            structure_volume_m3=round(bank * rng.uniform(0.05, 0.95), 2),
            swell_factor=rng.randrange(5, 30) / 100.0,
        )
        text = _working(r)
        assert "below zero" not in text
        assert _shown(r["void_volume_m3"], text, 3), text
        assert _shown(r["loose_backfill_needed_m3"], text, 3), text


@pytest.mark.parametrize("bank, structure", [(120.0, 150.0), (40.5, 41.0)])
def test_backfill_working_names_the_floor_it_applied(bank, structure):
    r = backfill_volume(excavation_bank_m3=bank, structure_volume_m3=structure)
    text = _working(r)
    assert r["void_volume_m3"] == 0
    assert "below zero" in text
    assert _shown(bank - structure, text, 3), text


@pytest.mark.parametrize("seed", _SEEDS)
def test_rebar_by_area_working_cites_the_density(seed):
    rng = random.Random(seed)
    for _ in range(12):
        density = float(rng.choice((7850, 7800, 7860, 7900)))
        r = rebar_by_area(
            area_m2=round(rng.uniform(5, 500), 1),
            spacing_mm=float(rng.choice((100, 125, 150, 200, 250))),
            bar_diameter_mm=float(rng.choice((8, 10, 12, 16, 20, 25))),
            both_ways=rng.random() < 0.5,
            density_kg_m3=density,
        )
        text = _working(r)
        assert _shown(density, text, 0), text
        assert _shown(r["bars_per_m"], text, 3), text
        assert _shown(r["total_bar_length_m"], text, 2), text
        assert _shown(r["total_mass_kg"], text, 2), text


def _finishes(rng: random.Random, window: float) -> dict:
    return interior_finishes_takeoff(
        floor_area_m2=round(rng.uniform(20, 400), 1),
        perimeter_m=round(rng.uniform(15, 120), 1),
        room_count=rng.randrange(1, 8),
        floor_to_ceiling_m=round(rng.uniform(2.4, 4.2), 2),
        door_width_m=0.9,
        door_height_m=2.1,
        doors_per_room=1,
        shared_wall_fraction=0.5,
        window_deduction_m2=window,
    )


@pytest.mark.parametrize("seed", _SEEDS)
def test_finishes_working_reproduces_each_opening_deducted_area(seed):
    rng = random.Random(seed)
    for _ in range(12):
        r = _finishes(rng, round(rng.uniform(0, 10), 1))
        text = _working(r)
        for key in ("wall_paint_m2", "blockwork_m2", "skirting_m", "door_deduction_m2"):
            assert _shown(r[key], text, 2), (key, text)


@pytest.mark.parametrize("excess", [50.0, 900.0])
def test_finishes_working_names_the_floor_it_applied(excess):
    rng = random.Random(int(excess))
    probe = _finishes(random.Random(int(excess)), 0.0)
    gross = probe["gross_wall_face_m2"]
    r = _finishes(rng, gross + excess)
    text = _working(r)
    assert r["wall_paint_m2"] == 0
    assert "below zero" in text
    assert _shown(-(excess + r["door_deduction_m2"]), text, 2), text
    assert not math.isnan(r["blockwork_m2"])


# A negative dimension is refused, never turned into a figure with working.
# Two inputs per take-off: the guard is the first rule of each.
@pytest.mark.parametrize("fn, kwargs", [
    (excavation_volume, dict(length_m=-4.0, width_m=3.0, depth_m=1.5)),
    (excavation_volume, dict(length_m=10.0, width_m=6.0, depth_m=-0.2)),
    (backfill_volume, dict(excavation_bank_m3=-80.0, structure_volume_m3=20.0)),
    (backfill_volume, dict(excavation_bank_m3=150.0, structure_volume_m3=-5.0)),
    (concrete_volume, dict(length_m=-8.0, width_m=4.0, thickness_m=0.25)),
    (concrete_volume, dict(shape="cylinder", diameter_m=0.6, height_m=-3.0)),
])
def test_a_negative_dimension_is_refused_never_computed(fn, kwargs):
    r = fn(**kwargs)
    assert "error" in r, r
    assert ">= 0" in r["error"]
    assert not any(k.endswith(("_m3", "_m2")) for k in r), r
