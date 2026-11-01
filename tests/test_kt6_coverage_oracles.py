"""K-T6 PR-C: oracle + boundary tests for the single-coverage formulas.

A gap scan (scripts, see the PR) found these six formulas were invoked only by
the generic smoke table -- executed, but with no test asserting a known-correct
value. Each test below asserts a hand-computed figure for >=2 distinct inputs,
plus the boundary/refusal the formula defines. Full-dotted imports so the
mutation gate selects these against the formula modules.
"""
from __future__ import annotations

import pytest

from app.agents.hats.design.formulas.construction_formulas import (
    beam_deflection_cantilever_point_load,
    beam_deflection_ss_point_load_midspan,
    thermal_shrinkage_equivalence,
    unit_weight_concrete,
)
from app.agents.hats.planning.formulas.construction_formulas import (
    plumbing_flow_programme,
)
from app.agents.hats.quantities.formulas.construction_formulas_quantities import (
    interior_finishes_takeoff,
)


# --- beam deflection: delta = P*L^3 / (c*E*I); c = 3 cantilever, 48 ss-midspan ---
@pytest.mark.parametrize("p,L,E,I,expected", [
    (30, 4, 30000, 1e9, 21.33),
    (10, 3, 25000, 5e8, 7.2),
])
def test_cantilever_point_load_deflection_mm(p, L, E, I, expected):
    assert beam_deflection_cantilever_point_load(p, L, E, I) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("p,L,E,I,expected", [
    (30, 4, 30000, 1e9, 1.33),
    (48, 5, 30000, 1e9, 4.17),
])
def test_ss_midspan_point_load_deflection_mm(p, L, E, I, expected):
    assert beam_deflection_ss_point_load_midspan(p, L, E, I) == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize("formula", [
    beam_deflection_cantilever_point_load,
    beam_deflection_ss_point_load_midspan,
])
def test_beam_deflection_rejects_negative_load(formula):
    with pytest.raises(ValueError):
        formula(-5, 4, 30000, 1e9)


# --- thermal shrinkage as a temperature drop: delta_t = strain / alpha ---
@pytest.mark.parametrize("strain,alpha,expected", [
    (0.0002, 10e-6, 20.0),
    (0.0003, 12e-6, 25.0),
])
def test_thermal_shrinkage_equivalence_c(strain, alpha, expected):
    r = thermal_shrinkage_equivalence(strain, alpha)
    assert r["equivalent_temp_drop_c"] == pytest.approx(expected, abs=0.1)


# --- unit weight of concrete: reinforced 2500, plain 2400 kg/m3 ---
@pytest.mark.parametrize("reinforced,expected", [(True, 2500), (False, 2400)])
def test_unit_weight_concrete_kg_m3(reinforced, expected):
    assert unit_weight_concrete(reinforced=reinforced)["unit_weight_kg_m3"] == expected


# --- plumbing flow programme: 8 fixed stages + area-scaled durations ---
@pytest.mark.parametrize("area,floors,total_days", [(400, 1, 68), (1200, 1, 127)])
def test_plumbing_flow_programme_total_days(area, floors, total_days):
    r = plumbing_flow_programme(area, floors)
    assert len(r["stages"]) == 8
    assert r["total_days"] == total_days


# --- interior finishes take-off: geometry invariants ---
@pytest.mark.parametrize(
    "floor_area, perimeter, room_count, height, door_width, door_height, "
    "doors_each, shared_fraction, window_deduction, "
    "expected_screed, expected_paint, expected_skirting, expected_block",
    [
        (100, 40, 4, 3.0, 0.9, 2.1, 1, 0.5, 0.0, 100.0, 112.44, 36.4, 56.22),
        (50, 30, 2, 2.7, 0.8, 2.0, 1, 0.5, 5.0, 50.0, 72.8, 28.4, 36.4),
    ],
)
def test_interior_finishes_takeoff_quantities(
    floor_area, perimeter, room_count, height, door_width, door_height,
    doors_each, shared_fraction, window_deduction,
    expected_screed, expected_paint, expected_skirting, expected_block,
):
    r = interior_finishes_takeoff(
        floor_area, perimeter, room_count, height,
        door_width_m=door_width, door_height_m=door_height,
        doors_per_room=doors_each, shared_wall_fraction=shared_fraction,
        window_deduction_m2=window_deduction,
    )
    assert r["floor_screed_m2"] == pytest.approx(expected_screed, abs=0.01)
    assert r["wall_paint_m2"] == pytest.approx(expected_paint, abs=0.01)
    assert r["skirting_m"] == pytest.approx(expected_skirting, abs=0.01)
    assert r["blockwork_m2"] == pytest.approx(expected_block, abs=0.01)


@pytest.mark.parametrize("h", [0.0, -1.0])
def test_interior_finishes_requires_floor_to_ceiling(h):
    r = interior_finishes_takeoff(100, 40, 4, h)
    assert r.get("status") == "error"
