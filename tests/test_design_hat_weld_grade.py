"""Fillet-weld strength follows the stated steel grade. An explicit fu or βw wins."""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation


def _weld(params: dict) -> dict:
    env = run_calculation("weld_capacity", params)
    assert env["status"] == "success", env
    return env["result"]


def _eurocode_kn(fu: float, beta_w: float, leg_mm: float, length_mm: float) -> float:
    throat = 0.707 * leg_mm
    fvw_d = fu / (math.sqrt(3.0) * beta_w * 1.25)
    return fvw_d * throat * length_mm / 1000.0


def test_stated_grade_sets_parent_metal_strength():
    # EN 1993-1-1 Table 3.1 fu (t ≤ 40 mm) and EN 1993-1-8 Table 4.1 βw.
    s275 = _weld({
        "leg_size_mm": 6,
        "length_mm": 100,
        "code": "eurocode",
        "text": "6 mm fillet weld, 100 mm long, S275 steel to Eurocode.",
    })
    assert s275["capacity_kn"] == pytest.approx(
        _eurocode_kn(430, 0.85, 6, 100), abs=0.05)
    assert s275["capacity_kn"] == pytest.approx(99.12, abs=0.05)

    s235 = _weld({
        "leg_size_mm": 8, "length_mm": 200, "code": "eurocode", "steel_grade": "S235",
    })
    assert s235["capacity_kn"] == pytest.approx(
        _eurocode_kn(360, 0.80, 8, 200), abs=0.05)


def test_explicit_electrode_and_beta_are_not_replaced():
    # The historic electrode default, passed on purpose, stays that strength.
    explicit = _weld({
        "leg_size_mm": 6,
        "length_mm": 100,
        "electrode_strength_mpa": 490,
        "code": "eurocode",
        "beta_w": 0.9,
        "steel_grade": "S275",
    })
    assert explicit["capacity_kn"] == pytest.approx(
        _eurocode_kn(490, 0.9, 6, 100), abs=0.05)


def test_no_grade_keeps_the_electrode_default():
    plain = _weld({"leg_size_mm": 6, "length_mm": 100, "code": "aci"})
    throat = 0.707 * 6
    expected = 0.75 * 0.60 * 490 * throat * 100 / 1000.0
    assert plain["capacity_kn"] == pytest.approx(expected, abs=0.05)
