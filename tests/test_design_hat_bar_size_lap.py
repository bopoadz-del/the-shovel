"""Lap length uses the cited code's bar-size factor.

ACI 318-19 §25.4.2.2: ψs = 0.8 for No. 19 and smaller, 1.0 for No. 22 and larger.
EN 1992-1-1 §8.4.2: η2 = 1.0 for db ≤ 32 mm, (132 − db)/100 above that.
A 20 mm bar stays on the existing oracle (ψs = 1, η2 = 1).
"""
from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas.construction_formulas_structural_rc import (
    rebar_lap_length,
)


def _aci_lap(db: float, fy: float = 420.0, fc: float = 30.0, conf: float = 1.5) -> float:
    psi_s = 0.8 if db <= 19.0 else 1.0
    ld = (fy * psi_s / (1.1 * math.sqrt(fc) * conf)) * db
    return 1.3 * ld


def _ec_lap(db: float, fy: float = 420.0, fc: float = 30.0) -> float:
    eta2 = 1.0 if db <= 32.0 else (132.0 - db) / 100.0
    fyd = fy / 1.15
    fctm = 0.3 * fc ** (2.0 / 3.0)
    fctd = 0.7 * fctm / 1.5
    fbd = 2.25 * eta2 * fctd
    return 1.5 * (db / 4.0) * (fyd / fbd)


def test_a_small_bar_lap_includes_the_aci_size_factor():
    r = rebar_lap_length(bar_diameter_mm=16, fy_mpa=420, fc_mpa=30, code="aci")
    assert r["lap_length_mm"] == pytest.approx(_aci_lap(16), abs=0.2)


def test_a_large_bar_keeps_psi_s_at_one():
    r = rebar_lap_length(bar_diameter_mm=25, fy_mpa=420, fc_mpa=30, code="aci",
                         confinement_ratio=1.5)
    assert r["lap_length_mm"] == pytest.approx(_aci_lap(25), abs=0.2)


def test_a_bar_above_32_mm_reduces_eurocode_bond_stress():
    r = rebar_lap_length(bar_diameter_mm=40, fy_mpa=420, fc_mpa=30, code="eurocode")
    assert r["lap_length_mm"] == pytest.approx(_ec_lap(40), abs=0.5)


def test_fy_above_420_mpa_includes_the_grade_factor():
    """ACI 318-19 Table 25.4.2.5: ψg steps with fy (1, 1.15, then 1.3)."""
    db, fc, conf, psi_s = 16.0, 30.0, 1.5, 0.8
    for fy, psi_g in ((420.0, 1.0), (500.0, 1.15), (550.0, 1.15), (690.0, 1.3)):
        ld = (fy * psi_s * psi_g / (1.1 * math.sqrt(fc) * conf)) * db
        r = rebar_lap_length(bar_diameter_mm=db, fy_mpa=fy, fc_mpa=fc, code="aci")
        assert r["lap_length_mm"] == pytest.approx(1.3 * ld, abs=0.2)
        assert f"ψg={psi_g}" in r["note"]
