"""Axial capacity does not require a slenderness check the caller did not ask for."""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


def _column(params: dict) -> dict:
    env = run_calculation("column_axial_capacity", params)
    assert env["status"] == "success", env
    return env["result"]


def test_tied_column_capacity_without_length_or_radius():
    # phi.Pn = 0.65 * 0.80 * (0.85*fc*(Ag-Ast) + fy*Ast).
    tied = _column({
        "gross_area_mm2": 160_000,
        "steel_area_mm2": 2_400,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "code": "aci",
        "tie_type": "tied",
    })
    assert tied["axial_capacity_kn"] == pytest.approx(2613.9, abs=0.5)
    assert "slenderness_ratio" not in tied

    other = _column({
        "gross_area_mm2": 250_000,
        "steel_area_mm2": 3_000,
        "fc_mpa": 40,
        "fy_mpa": 500,
        "code": "aci",
    })
    p0 = 0.85 * 40 * (250_000 - 3_000) + 500 * 3_000
    assert other["axial_capacity_kn"] == pytest.approx(0.65 * 0.80 * p0 / 1000.0, abs=0.5)
    assert "slenderness_ratio" not in other


def test_eurocode_capacity_without_slenderness_inputs():
    euro = _column({
        "gross_area_mm2": 160_000,
        "steel_area_mm2": 2_400,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "code": "eurocode",
    })
    fcd, fyd = 30 / 1.5, 420 / 1.15
    nrd = fcd * (160_000 - 2_400) + fyd * 2_400
    assert euro["axial_capacity_kn"] == pytest.approx(nrd / 1000.0, abs=0.5)
    assert "slenderness_ratio" not in euro


def test_supplied_length_and_radius_still_assess_slenderness():
    checked = _column({
        "gross_area_mm2": 160_000,
        "steel_area_mm2": 4_000,
        "fc_mpa": 30,
        "fy_mpa": 420,
        "unsupported_length_mm": 3000,
        "radius_gyration_mm": 115.47,
        "code": "aci",
    })
    assert checked["slenderness_ratio"] == pytest.approx(25.98, abs=0.05)
    assert checked["axial_capacity_kn"] == pytest.approx(2942.2, abs=0.5)
