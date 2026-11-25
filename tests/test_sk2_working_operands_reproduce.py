"""Take-off working: every printed step reproduces from the operands it prints.

Answers REPORT_OPUSGRADE_S674_66b9e3f_FAIL. The working multiplied figures
already rounded for display but printed the unrounded product
("6.667 x 100 x 2 = 1333.33"), and a waste rate printed with ``:g`` was not
the rate applied ("33.3333%", "1e-05%").

The checker here is general: it splits the working into ``expression =
figure`` steps, evaluates each expression exactly from the text as printed
(units dropped, ``x`` read as multiply, ``pi`` as pi) and requires the
printed figure to be that value rounded half-up at the precision printed.
Every input is built here from a seeded generator or a construction chosen
for the boundary under test; no expected figure is copied from the product.
"""
from __future__ import annotations

import random
import re
from decimal import ROUND_HALF_UP, Decimal, getcontext

import pytest

from app.agents.base.formulas.construction_formulas_earthwork import backfill_volume
from app.agents.base.formulas.construction_formulas_quantities import (
    _concrete_quantity,
    concrete_volume,
    fixed,
    plain,
    rate_percent,
    shown,
)
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    interior_finishes_takeoff,
    rebar_by_area,
)
from app.lib.construction_formulas_quantities import format_concrete_volume_line

getcontext().prec = 50
_PI = Decimal("3.14159265358979323846264338327950288419716939937510")
_UNIT = re.compile(r"\b(?!pi\b)[A-Za-z][\w/()]*")
_FIGURE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_SEEDS = (5, 23)


def _steps(result: dict) -> list[str]:
    if "note" in result:
        return [s for s in str(result["note"]).split(";")]
    return [str(line) for line in result.get("basis", ())]


def _evaluate(expression: str) -> Decimal | None:
    """The exact value of a printed arithmetic expression, or None when the
    text is not one (a prose line of the basis)."""
    text = expression.replace(" x ", " * ").replace("^", "**")
    text = re.sub(r"\bpi\b", "_PI", _UNIT.sub("", text))
    if not re.fullmatch(r"[\d.\s*/+\-()_PI]+", text) or not _NUMBER.search(text):
        return None
    text = _NUMBER.sub(lambda m: f"Decimal('{m.group(0)}')", text)
    try:
        return eval(text, {"Decimal": Decimal, "_PI": _PI})  # noqa: S307 -- test-built text
    except (SyntaxError, ZeroDivisionError, TypeError):
        return None


def _check_steps(result: dict) -> list[tuple[str, Decimal]]:
    """Every ``expression = figure`` step; asserts each reproduces. Returns
    (step, printed figure) for the steps checked."""
    checked = []
    for step in _steps(result):
        parts = step.split(" = ")
        if len(parts) < 2:
            continue
        figure = _FIGURE.match(parts[-1])
        value = _evaluate(parts[-2])
        if figure is None or value is None:
            continue
        printed = Decimal(figure.group(1))
        places = -printed.as_tuple().exponent
        expected = value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
        assert printed == expected, f"{step!r}: operands give {value}, printed {printed}"
        checked.append((step.strip(), printed))
    return checked


@pytest.mark.parametrize("seed", _SEEDS)
def test_rebar_by_area_every_step_reproduces_from_its_operands(seed):
    rng = random.Random(seed)
    for _ in range(300):
        spacing = rng.choice([rng.randint(10, 2000), round(rng.uniform(10, 2000), 1)])
        result = rebar_by_area(
            area_m2=round(rng.uniform(0.5, 5000), rng.choice([0, 1, 2])),
            spacing_mm=spacing,
            bar_diameter_mm=rng.choice([8, 10, 12, 16, 20, 25, 32]),
            both_ways=rng.random() < 0.5,
        )
        checked = _check_steps(result)
        assert len(checked) == 4, checked
        figures = [f for _, f in checked]
        assert float(figures[1]) == result["total_bar_length_m"]
        assert float(figures[3]) == result["total_mass_kg"]
        assert checked[0][0].startswith(f"bars/m = 1000/{spacing:g}")


@pytest.mark.parametrize("seed", _SEEDS)
def test_finishes_every_step_reproduces_from_its_operands(seed):
    rng = random.Random(seed)
    for _ in range(300):
        result = interior_finishes_takeoff(
            floor_area_m2=round(rng.uniform(5, 500), 2),
            perimeter_m=round(rng.uniform(5, 200), 2),
            room_count=rng.randint(1, 12),
            floor_to_ceiling_m=round(rng.uniform(2.2, 4.5), 2),
            door_width_m=round(rng.uniform(0.6, 1.2), 2),
            door_height_m=round(rng.uniform(2.0, 2.4), 2),
            doors_per_room=rng.randint(0, 3),
            shared_wall_fraction=round(rng.uniform(0, 1), 2),
            window_deduction_m2=round(rng.uniform(0, 40), 2),
        )
        checked = dict((s.split(" = ")[0], f) for s, f in _check_steps(result))
        assert {"wall paint", "blockwork", "skirting"} <= set(checked), checked
        if result["skirting_m"] > 0:
            assert float(checked["skirting"]) == result["skirting_m"]
        if result["wall_paint_m2"] > 0:
            assert float(checked["wall paint"]) == result["wall_paint_m2"]
        assert float(checked["blockwork"]) == result["blockwork_m2"]


@pytest.mark.parametrize("doors_per_room, width", [(3, 1.0), (4, 0.9)])
def test_finishes_skirting_names_the_floor_it_applied(doors_per_room, width):
    # perimeter shorter than the door widths: skirting is floored at zero.
    perimeter = doors_per_room * width - 0.5
    result = interior_finishes_takeoff(
        floor_area_m2=20.0, perimeter_m=perimeter, room_count=1,
        floor_to_ceiling_m=3.0, door_width_m=width, door_height_m=0.1,
        doors_per_room=doors_per_room,
    )
    skirting = next(s for s in result["basis"] if s.startswith("skirting = "))
    assert "below zero" in skirting
    assert result["skirting_m"] == 0.0
    _check_steps(result)


@pytest.mark.parametrize("doors_per_room, width", [(2, 1.0), (3, 0.5)])
def test_finishes_an_exact_zero_is_not_called_below_zero(doors_per_room, width):
    # perimeter equal to the door widths: the difference is exactly zero.
    result = interior_finishes_takeoff(
        floor_area_m2=20.0, perimeter_m=doors_per_room * width, room_count=1,
        floor_to_ceiling_m=3.0, door_width_m=width, door_height_m=0.1,
        doors_per_room=doors_per_room,
    )
    skirting = next(s for s in result["basis"] if s.startswith("skirting = "))
    assert "below zero" not in skirting
    assert skirting.endswith("= 0.00 m")


@pytest.mark.parametrize("volume", [75.0, 312.5])
def test_backfill_equal_bank_and_structure_is_not_called_below_zero(volume):
    result = backfill_volume(excavation_bank_m3=volume, structure_volume_m3=volume)
    assert "below zero" not in result["note"]
    assert result["void_volume_m3"] == 0.0


@pytest.mark.parametrize("waste", [1 / 3, 1e-7, 1 / 7])
def test_the_stated_waste_rate_is_exactly_the_rate_applied(waste):
    result = concrete_volume(length_m=3.0, width_m=2.0, thickness_m=0.5, waste_factor=waste)
    applied = Decimal(repr(result["waste_factor"]))
    note_pct = re.search(r"\+(\S+)% waste", result["note"]).group(1)
    assert Decimal(note_pct) / 100 == applied
    line = format_concrete_volume_line(result)
    line_pct = re.search(r"documented (\S+)% waste", line).group(1)
    factor = re.search(r"× (\S+)\)", line).group(1)
    assert Decimal(line_pct) / 100 == applied
    assert Decimal(factor) == 1 + applied
    assert "e-" not in note_pct + line_pct + factor


@pytest.mark.parametrize("value, places, expected", [
    ("107.405", 2, "107.41"), ("2.0005", 3, "2.001"), ("-1.255", 2, "-1.26"),
])
def test_fixed_rounds_the_exact_decimal_half_up(value, places, expected):
    assert fixed(Decimal(value), places) == Decimal(expected)
    assert str(fixed(Decimal(value), places)) == expected


@pytest.mark.parametrize("value, text", [(0.1, "0.1"), (1e-07, "0.0000001"), (150, "150")])
def test_shown_and_plain_are_the_figure_as_written(value, text):
    assert shown(value) == Decimal(text)
    assert plain(value) == text


@pytest.mark.parametrize("fraction, percent", [(0.025, "2.5"), (0.07, "7")])
def test_rate_percent_is_the_exact_shift_of_the_fraction(fraction, percent):
    assert rate_percent(fraction) == percent


@pytest.mark.parametrize("missing", [None, ""])
def test_a_missing_element_count_is_one_element(missing):
    assert _concrete_quantity(missing) == 1.0
    result = concrete_volume(length_m=2.0, width_m=3.0, thickness_m=0.5, quantity=missing)
    assert result["net_volume_m3"] == 3.0


@pytest.mark.parametrize("zero", [-0.0, Decimal("-0.000")])
def test_plain_never_prints_a_signed_zero(zero):
    # A floored or cancelled figure can be a negative zero; the working line
    # states it as 0, never "-0".
    assert plain(zero) == "0"


@pytest.mark.parametrize("diameter_mm, unit_mass", [(12, "0.8878"), (20, "2.4662")])
def test_rebar_unit_mass_is_stated_to_four_places(diameter_mm, unit_mass):
    note = rebar_by_area(area_m2=10, spacing_mm=200, bar_diameter_mm=diameter_mm)["note"]
    stated = re.search(r"unit mass = \S+ = (\S+) kg/m;", note).group(1)
    assert stated == unit_mass
