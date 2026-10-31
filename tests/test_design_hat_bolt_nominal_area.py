"""A nominal metric bolt size is its tensile stress area. A property class sets fub."""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


def _bolt(params: dict) -> dict:
    env = run_calculation("bolt_shear_capacity", params)
    assert env["status"] == "success", env
    return env["result"]


def test_nominal_size_and_class_resolve():
    # EN 1993-1-8 Table 3.4: Fv,Rd = αv * fub * As / γM2.
    # M20 As = 245 mm², class 8.8 fub = 800, αv = 0.6 → 94.08 kN.
    m20 = _bolt({
        "bolt_area_mm2": "M20",
        "grade": "8.8",
        "code": "eurocode",
        "n_shear_planes": 1,
    })
    assert m20["capacity_kn"] == pytest.approx(0.6 * 800 * 245 / 1.25 / 1000.0, abs=0.05)

    # M16 As = 157 mm², class 10.9 fub = 1000, αv = 0.5.
    m16 = _bolt({
        "bolt_area_mm2": "M16",
        "grade": 10.9,
        "code": "eurocode",
    })
    assert m16["capacity_kn"] == pytest.approx(0.5 * 1000 * 157 / 1.25 / 1000.0, abs=0.05)


def test_numeric_area_and_explicit_strength_stay():
    # M22 As passed as a number, fub passed explicitly. Not the size table.
    env = run_calculation("bolt_shear_capacity", {
        "bolt_area_mm2": 303,
        "shear_strength_mpa": 800,
        "n_shear_planes": 1,
        "code": "eurocode",
        "alpha_v": 0.6,
    })
    assert env["status"] == "success", env
    assert env["result"]["capacity_kn"] == pytest.approx(116.35, abs=0.05)


def test_missing_area_is_still_reported():
    env = run_calculation("bolt_shear_capacity", {})
    assert env["status"] == "needs_input"
    assert "bolt_area_mm2" in env["error"]
    assert "(mm2)" in env["error"]


def test_zero_area_is_rejected():
    env = run_calculation("bolt_shear_capacity", {
        "bolt_area_mm2": 0, "shear_strength_mpa": 372,
    })
    assert env["status"] == "error"
