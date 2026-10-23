"""Tests for the reference-table calculators. These check the cited factor/
threshold lookups + the honest ``kind=reference_table`` labelling (not physics)."""
import pytest

from app.lib.construction_formulas import CALCULATORS


def test_carbon_footprint_concrete_c30():
    # 100 m3 * 320 kgCO2e/m3 = 32,000 kgCO2e = 32 tCO2e.
    r = CALCULATORS["carbon_footprint_concrete"](volume_m3=100, grade="c30")
    assert r["kind"] == "reference_table"
    assert r["total_kgco2e"] == pytest.approx(32_000, abs=1)
    assert r["total_tco2e"] == pytest.approx(32.0, abs=0.01)


def test_carbon_footprint_custom_epd_factor():
    r = CALCULATORS["carbon_footprint_concrete"](volume_m3=50, embodied_kgco2e_m3=200)
    assert r["total_kgco2e"] == pytest.approx(10_000, abs=1)


def test_leed_levels():
    assert CALCULATORS["leed_points_estimate"](points=62)["certification_level"] == "Gold"
    assert CALCULATORS["leed_points_estimate"](points=85)["certification_level"] == "Platinum"
    assert CALCULATORS["leed_points_estimate"](points=45)["certification_level"] == "Certified"
    assert CALCULATORS["leed_points_estimate"](points=30)["certification_level"] == "Not certified"


def test_bim_clash_tolerance_lod350():
    r = CALCULATORS["bim_clash_tolerance"](lod=350)
    assert r["kind"] == "reference_table"
    assert r["clash_tolerance_mm"] == 12.0


def test_laser_scan_accuracy_scales_with_range():
    # 2 mm at 10 m -> 8 mm at 40 m (linear).
    r = CALCULATORS["laser_scan_accuracy"](range_m=40, accuracy_at_10m_mm=2.0)
    assert r["estimated_accuracy_mm"] == pytest.approx(8.0, abs=0.01)
