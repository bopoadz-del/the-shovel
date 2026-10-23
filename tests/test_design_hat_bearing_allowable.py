"""Allowable bearing already includes the factor of safety.

The foundation is adequate when the net pressure does not exceed that allowable.
Requiring a second factor of 2 marks a pressure inside the allowable as unsafe.
"""
from __future__ import annotations

import pytest

from app.agents.hats.design.formulas.construction_formulas import (
    foundation_bearing_pressure,
)


def test_pressure_inside_the_allowable_is_adequate():
    # 2000 kN on 3×3 m is 222 kPa. Allowable 300 kPa. FOS = 1.35.
    r = foundation_bearing_pressure(
        foundation_width_m=3, foundation_length_m=3,
        column_load_kn=2000, soil_bearing_capacity_kn_m2=300,
    )
    assert r["fos_against_bearing"] == pytest.approx(1.35, abs=0.01)
    assert r["bearing_ok"] is True


def test_pressure_above_the_allowable_is_not():
    r = foundation_bearing_pressure(
        foundation_width_m=3, foundation_length_m=3,
        column_load_kn=4000, soil_bearing_capacity_kn_m2=300,
    )
    assert r["bearing_ok"] is False
