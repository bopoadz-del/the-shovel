"""Composite-column resistance is the plastic section of EN 1994-1-1 §6.7.3.2.

Npl,Rd = Aa*fyd + 0.85*Ac*fcd + As*fsd, with fcd = fck/1.5.
A flat 10 percent is not a steel contribution.
"""
from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas.construction_formulas import (
    composite_column_design,
)


def _npl(diameter_mm, fck, steel_area=0.0, steel_fy=355.0,
         rebar_area=0.0, rebar_fy=500.0) -> float:
    ag = math.pi * (diameter_mm / 2.0) ** 2
    ac = ag - steel_area - rebar_area
    return (
        steel_area * steel_fy
        + 0.85 * ac * (fck / 1.5)
        + rebar_area * (rebar_fy / 1.15)
    ) / 1000.0


def test_a_plain_column_uses_the_concrete_factors():
    r = composite_column_design(
        axial_load_kn=1000, column_diameter_mm=500, concrete_grade_n_mm2=40,
    )
    expect = _npl(500, 40)
    assert r["option_a"]["capacity_kn"] == pytest.approx(expect, abs=1)
    assert r["option_a"]["capacity_kn"] == pytest.approx(4451, abs=1)


def test_steel_area_contributes_its_yield_not_a_flat_ten_percent():
    plain = composite_column_design(
        axial_load_kn=1000, column_diameter_mm=500, concrete_grade_n_mm2=40,
        i_beam_weight_t=2.0,
    )
    with_steel = composite_column_design(
        axial_load_kn=1000, column_diameter_mm=500, concrete_grade_n_mm2=40,
        steel_area_mm2=10000, steel_fy_mpa=355,
    )
    assert plain["option_a"]["capacity_kn"] == pytest.approx(_npl(500, 40), abs=1)
    assert with_steel["option_a"]["capacity_kn"] == pytest.approx(
        _npl(500, 40, steel_area=10000, steel_fy=355), abs=1,
    )
    assert with_steel["option_a"]["capacity_kn"] != pytest.approx(
        plain["option_a"]["capacity_kn"] * 1.10, abs=1,
    )
