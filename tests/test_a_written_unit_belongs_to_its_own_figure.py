"""A unit the ask writes is read only from the figure's own writing.

Chains M03 / M23 / M27 (TESTER 1857Z, live 867dff41..868d88aa): the calculator
was right in CI, the turn failed at the input read. The server gives a bare
figure the unit words the ask writes after it, so the unit check can refuse a
rate passed as a mass ("10 lifts/h" as a 10 t crane). It took those words
from ANY place the same number was written:

- "30 kN/m ultimate ... f'c 30": the strength took the load's "kN/m" and was
  refused as a force per length;
- "live 4.0 kN/m2 ... (KLL 4)": the element factor took "kN/m2", cut to
  "kN/m", and was refused;
- "one 0.9 x 2.1 door per room": the door height took "door per room".

The rules now: a figure written twice, once bare and once with words, has no
written unit of its own; a figure that is one side of a product has none
either (the words after a product name the thing); a rate's last unit is read
whole ("kN/m2", not "kN/m"). Every ask below is invented; every expected
value is worked by hand in its comment. Each runs through the chat door
(``construction_calc`` handler) with the turn's words set as the loop sets them.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.agents.base.tools import handle_construction_calc
from app.agents.core.tool_registry import ToolCall
from app.lib import construction_formulas as cf


def _chat(ask: str, calculation: str, params: dict[str, Any]) -> dict[str, Any]:
    token = cf.set_current_ask(ask)
    try:
        call = ToolCall(agent=None, name="construction_calc",
                        args={"calculation": calculation, "params": dict(params)},
                        user_message=ask, history=[])
        out = asyncio.run(handle_construction_calc(call))
    finally:
        cf._CURRENT_ASK.reset(token)
    assert isinstance(out, dict) and isinstance(out.get("result"), dict), out
    return out["result"]


def _ok(ask, calculation, params):
    env = _chat(ask, calculation, params)
    assert env.get("status") == "success", env
    return env["result"]


# ---- 1. a figure written bare for itself and with words for another quantity

@pytest.mark.parametrize("ask, params, capacity", [
    # a = 603.19*500/(0.85*25*250) = 56.77 mm; 0.9*603.19*500*(450-28.385) = 114.44 kN.m
    ("Beam 250 wide on a 5 m span, 25 kN/m factored. 3 bars of 16 mm, d 450, "
     "fy 500, f'c 25, ACI. Does it carry it?",
     {"steel_area_mm2": 603.19, "fy_mpa": 500, "width_mm": 250, "eff_depth_mm": 450,
      "fc_mpa": 25, "code": "ACI"}, 114.44),
    # a = 1256.64*460/(0.85*40*350) = 48.58 mm; 0.9*1256.64*460*(600-24.29) = 299.51 kN.m
    ("Girder 350 x 700, 40 kN/m ultimate over 7 m, 4 bars of 20 mm, d 600, fy 460, "
     "f'c 40. Check the moment to ACI.",
     {"steel_area_mm2": 1256.64, "fy_mpa": 460, "width_mm": 350, "eff_depth_mm": 600,
      "fc_mpa": 40, "code": "ACI"}, 299.51),
])
def test_a_strength_does_not_take_the_load_units_of_the_same_number(ask, params, capacity):
    out = _ok(ask, "rc_beam_moment_capacity", params)
    assert out["moment_capacity_kn_m"] == pytest.approx(capacity, abs=0.02)


@pytest.mark.parametrize("ask, live, area, kll, reduced", [
    # 0.25 + 4.57/sqrt(2*50) = 0.707; 2.0*0.707 = 1.414 kN/m2
    ("8 floors, 50 m2 tributary each, live 2.0 kN/m2, dead 6 kN/m2. "
     "Reduce the live load to ASCE 7 (KLL 2).", 2.0, 50, 2, 1.414),
    # 0.25 + 4.57/sqrt(3*90) = 0.25 + 4.57/16.432 = 0.5281; 3.0*0.5281 = 1.584 kN/m2
    ("Interior column, 90 m2 tributary, office live 3.0 kN/m2 on 5 levels, "
     "ASCE 7 with KLL 3. Reduced live load?", 3.0, 90, 3, 1.584),
])
def test_an_element_factor_does_not_take_the_load_units_of_the_same_number(
        ask, live, area, kll, reduced):
    out = _ok(ask, "live_load_reduction", {"base_live_load_kn_m2": live,
                                           "tributary_area_m2": area,
                                           "code": "ASCE", "kll": kll})
    assert out["reduced_live_load_kn_m2"] == pytest.approx(reduced, abs=0.002)


# ---- 2. a side of a product has no unit words of its own

@pytest.mark.parametrize("ask, params, paint", [
    # 36*2.7 - 3*0.8*2.0 = 97.2 - 4.8 = 92.4 m2
    ("Flat of 80 m2, 36 m of walls, 3 rooms, 2.7 m high, one 0.8 x 2.0 door per room. "
     "Paint area?",
     {"floor_area_m2": 80, "perimeter_m": 36, "room_count": 3, "floor_to_ceiling_m": 2.7,
      "door_width_m": 0.8, "door_height_m": 2.0, "doors_per_room": 1}, 92.4),
    # 50*3.2 - 5*1.0*2.4 = 160 - 12 = 148.0 m2
    ("Villa wing 150 m2, 50 m of wall, 5 bedrooms at 3.2 m ceiling, a 1.0 x 2.4 door "
     "per bedroom. Wall paint?",
     {"floor_area_m2": 150, "perimeter_m": 50, "room_count": 5, "floor_to_ceiling_m": 3.2,
      "door_width_m": 1.0, "door_height_m": 2.4, "doors_per_room": 1}, 148.0),
])
def test_a_door_side_is_not_read_as_door_per_room(ask, params, paint):
    out = _ok(ask, "interior_finishes_takeoff", params)
    assert out["wall_paint_m2"] == pytest.approx(paint, abs=0.01)


@pytest.mark.parametrize("figure, text", [
    (2.0, "one 0.8 x 2.0 door per room"),        # the product's last side
    (0.8, "one 0.8 x 2.0 door per room"),        # its first side
    (1.2, "a 1.2 × 3 window per bay"),           # written with the times sign
    (30, "30 kN/m ultimate, f'c 30, ACI"),       # once with words, once bare
    (4, "live 4.0 kN/m2, dead 5.0 (KLL 4)"),
])
def test_a_figure_without_words_of_its_own_gets_none(figure, text):
    assert cf._words_written_after(figure, text) == ""


# ---- negatives: what still holds

@pytest.mark.parametrize("ask", [
    "Each crane does 10 lifts/h over an 8 h day; the spare crane also does 10 lifts/h. "
    "We have 240 lifts to make. How many cranes?",
    "Each crane manages 12 lifts per h on a 10 h shift. 360 lifts in total. Cranes?",
])
def test_a_rate_written_for_the_figure_is_still_refused_as_a_mass(ask):
    figure = 10 if "10 lifts" in ask else 12
    env = _chat(ask, "crane_planning", {"total_lift_demand_tons": 240,
                                        "crane_capacity_tons": figure})
    assert env["status"] == "error" and "lifts" in env["error"], env


@pytest.mark.parametrize("figure, text, words", [
    (1500, "at 1,500 m2 per gang-day. How long?", "m2 per gang-day"),
    (12, "12 m2 per crew-hour, then 12 m2 per crew-hour", "m2 per crew-hour"),
    (3, "3 m3/h from the pump, 6 m long", "m3/h from the pump"),
])
def test_words_written_only_after_the_figure_are_still_its_own(figure, text, words):
    assert cf._words_written_after(figure, text) == words


@pytest.mark.parametrize("words, unit", [
    ("kN/m2 allowable", "kN/m2"),
    ("N/mm2 at 28 days", "N/mm2"),
])
def test_a_rate_unit_is_read_whole(words, unit):
    assert cf._RATE_WORDS.match(words).group(0) == unit


@pytest.mark.parametrize("ask, load", [
    ("Column load 250 kN/m2 on a pad. Bearing pressure?", 250),
    ("The column brings 180 N/mm2 onto the footing; pressure under it?", 180),
])
def test_a_pressure_given_for_a_force_is_refused_as_a_pressure(ask, load):
    env = _chat(ask, "foundation_bearing_pressure", {
        "foundation_width_m": 2, "foundation_length_m": 2, "column_load_kn": load,
        "soil_bearing_capacity_kn_m2": 300})
    assert env["status"] == "error", env
    assert "pressure" in env["error"] and "force per length" not in env["error"], env


@pytest.mark.parametrize("ask, load", [
    # q = 250 / (2*2) = 62.5 kN/m2
    ("Footing 2 x 2 m, bearing 250 kN/m2. Column load 250?", 250),
    # q = 180 / (2*2) = 45.0 kN/m2
    ("Pad 2 x 2 m on soil good for 180 kN/m2; load is 180, check it.", 180),
])
def test_a_figure_also_written_bare_runs_on_the_callers_value(ask, load):
    out = _ok(ask, "foundation_bearing_pressure", {
        "foundation_width_m": 2, "foundation_length_m": 2, "column_load_kn": load,
        "soil_bearing_capacity_kn_m2": 300})
    assert out["bearing_pressure_kn_m2"] == pytest.approx(load / 4, abs=0.01)
