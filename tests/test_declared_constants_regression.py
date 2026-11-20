"""D1 rule C, part 2 — regression for the real formulas that now declare their
constants, plus a direct exam of the ``Constants`` machinery.

Two inputs per decision, own inputs, nothing real hard-coded beyond the
computed regression values (each expected number was produced by the same
formula on this branch before the constants rewrite; the tests pin the number
so a dropped rule or a lost body is seen).
"""

from __future__ import annotations

import pytest

from app.agents.base.formulas.construction_formulas import diaphragm_wall_panel_volume
from app.agents.hats.design.formulas.construction_formulas import (
    foundation_bearing_pressure,
    post_tensioning_force,
)
from app.agents.hats.design.formulas.construction_formulas_structural_steel import weld_capacity
from app.agents.hats.quantities.formulas.construction_formulas_quantities import rebar_by_area
from app.agents.hats.safety.formulas.construction_formulas import wind_load_on_formwork
from app.lib.formula_registry import Constant, Constants, recording_constants


# --- real formulas: the declared-constant rewrite still computes -----------------

@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(panel_length=10, wall_thickness=1.2, excavation_depth=30, panel_count=5), "total_volume_m3", 1800.0),
        (dict(panel_length=6, wall_thickness=0.8, excavation_depth=20, panel_count=2), "total_volume_m3", 192.0),
    ],
)
def test_diaphragm_wall_panel_volume_computes(args, key, expected):
    out = diaphragm_wall_panel_volume(**args)
    assert out[key] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("args", "error_part"),
    [
        (dict(panel_length=0, wall_thickness=1.2, excavation_depth=30, panel_count=5), "> 0"),
        (dict(panel_length=10, wall_thickness=0, excavation_depth=30, panel_count=5), "> 0"),
    ],
)
def test_diaphragm_wall_panel_volume_rejects_non_positive(args, error_part):
    out = diaphragm_wall_panel_volume(**args)
    assert "error" in out and error_part in out["error"]


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(foundation_width_m=2, foundation_length_m=3, column_load_kn=500,
              soil_bearing_capacity_kn_m2=200, foundation_depth_m=1.5), "bearing_pressure_kn_m2", 83.33),
        (dict(foundation_width_m=2.5, foundation_length_m=4, column_load_kn=800,
              soil_bearing_capacity_kn_m2=150, foundation_depth_m=1.0), "bearing_pressure_kn_m2", 80.0),
    ],
)
def test_foundation_bearing_pressure_computes(args, key, expected):
    out = foundation_bearing_pressure(**args)
    assert out[key] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("args",),
    [
        (dict(foundation_width_m=0, foundation_length_m=3, column_load_kn=500,
              soil_bearing_capacity_kn_m2=200, foundation_depth_m=1.5),),
        (dict(foundation_width_m=2, foundation_length_m=0, column_load_kn=500,
              soil_bearing_capacity_kn_m2=200, foundation_depth_m=1.5),),
    ],
)
def test_foundation_bearing_pressure_rejects_non_positive(args):
    out = foundation_bearing_pressure(**args)
    assert "error" in out


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(span_m=8, slab_thickness_m=0.25, live_load_kn_m2=5, dead_load_kn_m2=7,
              tendon_stress_n_mm2=1100, tendon_diameter_mm=15, eccentricity_ratio=0.5), "tendon_force_kn", 336.0),
        (dict(span_m=6, slab_thickness_m=0.2, live_load_kn_m2=3, dead_load_kn_m2=5,
              tendon_stress_n_mm2=1200, tendon_diameter_mm=12.7, eccentricity_ratio=0.3), "tendon_force_kn", 281.2),
    ],
)
def test_post_tensioning_force_computes(args, key, expected):
    out = post_tensioning_force(**args)
    assert out[key] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("args",),
    [
        (dict(span_m=8, slab_thickness_m=0.25, live_load_kn_m2=5, dead_load_kn_m2=None,
              tendon_stress_n_mm2=1100, tendon_diameter_mm=15, eccentricity_ratio=0.5),),
        (dict(span_m=6, slab_thickness_m=0.2, live_load_kn_m2=3, dead_load_kn_m2=None,
              tendon_stress_n_mm2=1200, tendon_diameter_mm=12.7, eccentricity_ratio=0.3),),
    ],
)
def test_post_tensioning_force_defaults_dead_load(args):
    out = post_tensioning_force(**args)
    assert isinstance(out["tendon_force_kn"], (int, float))


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(leg_size_mm=6, length_mm=200, electrode_strength_mpa=480, code="EC3", beta_w=0.85, steel_grade="S275"),
         "capacity_kn", 221.29),
        (dict(leg_size_mm=8, length_mm=100, electrode_strength_mpa=430, code="EC3", beta_w=0.8, steel_grade="S235"),
         "capacity_kn", 140.42),
    ],
)
def test_weld_capacity_computes(args, key, expected):
    out = weld_capacity(**args)
    assert out[key] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(area_m2=100, spacing_mm=200, bar_diameter_mm=12, both_ways=True, density_kg_m3=7850),
         "total_mass_kg", 887.81),
        (dict(area_m2=50, spacing_mm=150, bar_diameter_mm=10, both_ways=False, density_kg_m3=7850),
         "total_mass_kg", 205.51),
    ],
)
def test_rebar_by_area_computes(args, key, expected):
    out = rebar_by_area(**args)
    assert out[key] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("args", "key", "expected"),
    [
        (dict(wind_velocity_m_s=20, formwork_area_m2=50, formwork_height_m=4, formwork_width_m=6, shape_factor=1.3),
         "wind_force_kn", 15.938),
        (dict(wind_velocity_m_s=10, formwork_area_m2=20, formwork_height_m=3, formwork_width_m=5, shape_factor=1.0),
         "wind_force_kn", 1.226),
    ],
)
def test_wind_load_on_formwork_computes(args, key, expected):
    out = wind_load_on_formwork(**args)
    assert out[key] == pytest.approx(expected, abs=0.01)


# --- Constants machinery, named directly so the gate selects the dunder methods ---

def _direct_constant() -> Constants:
    return Constants(
        plate=Constant(2.5, "ratio", "invented plate", source="Invented Cl. 1"),
        edge=Constant(0.5, "ratio", "invented edge"),
    )


@pytest.mark.parametrize("bad", [0.0, "2.5", None])
def test_constants_init_rejects_a_non_constant(bad):
    with pytest.raises(TypeError):
        Constants.__init__(Constants.__new__(Constants), x=bad)


@pytest.mark.parametrize("good", [1.0, 2.0])
def test_constants_init_stores_declared(good):
    c = Constants.__new__(Constants)
    Constants.__init__(c, plate=Constant(good, "ratio", "invented"))
    assert c.plate == good


@pytest.mark.parametrize("name", ["missing", "other"])
def test_constants_getattr_raises_for_unknown(name):
    c = _direct_constant()
    with pytest.raises(AttributeError):
        Constants.__getattr__(c, name)


@pytest.mark.parametrize("name", ["plate", "edge"])
def test_constants_getattr_reads_value(name):
    c = _direct_constant()
    assert isinstance(Constants.__getattr__(c, name), float)


@pytest.mark.parametrize("name", ["plate", "edge"])
def test_constants_setattr_is_read_only(name):
    c = _direct_constant()
    with pytest.raises(AttributeError):
        Constants.__setattr__(c, name, 9.0)


@pytest.mark.parametrize("name", ["plate", "edge"])
def test_recording_constants_enter_exit_records(name):
    rc = recording_constants()
    recording_constants.__enter__(rc)
    try:
        c = _direct_constant()
        _ = Constants.__getattr__(c, name)
    finally:
        recording_constants.__exit__(rc, None, None, None)
    assert [r["name"] for r in rc.rows] == [name]


@pytest.mark.parametrize("name", ["plate", "edge"])
def test_recording_constants_dedups_repeated_reads(name):
    c = _direct_constant()
    rc = recording_constants()
    recording_constants.__enter__(rc)
    try:
        Constants.__getattr__(c, name)
        Constants.__getattr__(c, name)
    finally:
        recording_constants.__exit__(rc, None, None, None)
    assert [r["name"] for r in rc.rows] == [name]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (dict(leg_size_mm=6, length_mm=200, code="EC3", steel_grade="S275"), 198.23),
        (dict(leg_size_mm=6, length_mm=200, code="EC3", steel_grade="S355"), 222.05),
    ],
)
def test_weld_capacity_reads_tabulated_parent_metal(args, expected):
    out = weld_capacity(**args)
    assert out["capacity_kn"] == pytest.approx(expected, abs=0.01)


@pytest.mark.parametrize(
    ("velocity", "area", "width", "height"),
    [
        (20, None, 0.0, 0.0),
        (10, 0.0, 0.0, 0.0),
    ],
)
def test_wind_load_on_formwork_needs_area_or_dims(velocity, area, width, height):
    out = wind_load_on_formwork(
        wind_velocity_m_s=velocity, formwork_area_m2=area,
        formwork_width_m=width, formwork_height_m=height,
    )
    assert "error" in out
