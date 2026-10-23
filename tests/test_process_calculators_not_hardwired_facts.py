"""Interior estimating is a PROCESS with per-project variables, never facts.

Operator directive 2026-08-15: floor-to-ceiling heights, door schedules,
productivity norms and rates "are variables, change in every project and
even floor -- we want process learned, not hardwired facts across the app."

These calculators encode only the geometry and build-up PROCESS. Every
project-specific value is an input: a missing required variable is a
refusal that names what to supply, never a silent default. The fence values
are hand-computed from the live campaign take-off (192 rooms, 1,947.87 m2,
2,342.20 m) with a declared test assumption pack.
"""
from __future__ import annotations

import pytest

from app.agents import formulas as F


def takeoff(**kw):
    return F.run_formula("interior_finishes_takeoff", kw)


def line(**kw):
    return F.run_formula("resource_line_cost", kw)


# hand-computed: gross wall 2342.20*2.70=6323.94; doors 192*0.9*2.1=362.88;
# paint 5961.06; block 5961.06*0.5=2980.53; skirting 2342.20-172.80=2169.40
LIVE = dict(floor_area_m2=1947.87, perimeter_m=2342.20, room_count=192,
            floor_to_ceiling_m=2.70, door_width_m=0.9, door_height_m=2.1,
            doors_per_room=1)


def test_the_live_takeoff_computes_the_hand_checked_quantities():
    r = takeoff(**LIVE)
    assert r["gross_wall_face_m2"] == pytest.approx(6323.94)
    assert r["door_deduction_m2"] == pytest.approx(362.88)
    assert r["wall_paint_m2"] == pytest.approx(5961.06)
    assert r["blockwork_m2"] == pytest.approx(2980.53)
    assert r["skirting_m"] == pytest.approx(2169.40)
    assert r["floor_tiling_m2"] == pytest.approx(1947.87)


def test_missing_height_is_a_refusal_not_a_default():
    r = takeoff(floor_area_m2=100, perimeter_m=80, room_count=5,
                floor_to_ceiling_m=0)
    assert r.get("status") == "error"
    assert "floor_to_ceiling_m" in r["error"]
    assert "never assumed" in r["error"]


def test_every_variable_is_echoed_in_the_basis():
    """The process is auditable: the basis restates each input so a reviewer
    sees exactly which project variables produced the quantities."""
    r = takeoff(**LIVE)
    basis = " ".join(r["basis"])
    for token in ("2.7", "0.9", "2.1", "192", "shared wall fraction 0.5"):
        assert token in basis, token


def test_changing_the_height_changes_only_wall_derived_quantities():
    """Per-floor variability is the point: the same rooms at a different
    height move the wall quantities and nothing else."""
    a = takeoff(**LIVE)
    b = takeoff(**{**LIVE, "floor_to_ceiling_m": 3.20})
    assert b["floor_tiling_m2"] == a["floor_tiling_m2"]
    assert b["skirting_m"] == a["skirting_m"]
    assert b["wall_paint_m2"] == pytest.approx(2342.20 * 3.20 - 362.88)
    assert b["wall_paint_m2"] > a["wall_paint_m2"]


def test_resource_line_hand_checked():
    r = line(quantity=1947.87, daily_output=12, day_rate=275,
             material_rate_per_unit=225, plant_fraction_of_labour=0.03)
    assert r["labour_cost"] == pytest.approx(44638.69, abs=0.01)
    assert r["plant_cost"] == pytest.approx(1339.16, abs=0.01)
    assert r["material_cost"] == pytest.approx(438270.75, abs=0.01)
    assert r["total_cost"] == pytest.approx(484248.60, abs=0.01)


def test_missing_productivity_or_rate_refuses():
    r = line(quantity=100, daily_output=0, day_rate=275)
    assert r.get("status") == "error"
    assert "never" in r["error"]


def test_missing_material_rate_is_no_sourced_rate_not_zero():
    r = line(quantity=100, daily_output=10, day_rate=280)
    assert r["material_cost"] == "NO SOURCED RATE"
    assert r["total_is_partial"] is True
    # the total that IS computed is labour+plant only
    assert r["total_cost"] == pytest.approx(100 / 10 * 280)
