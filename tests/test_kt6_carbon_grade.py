"""K-T6 PR-A: carbon_footprint_concrete must not silently cost an unknown
grade at the c30 figure. A recognised grade (or an explicit EPD factor)
computes; an unrecognised grade is refused with the known grades named and the
ICE/EPD source cited. Full-dotted import so the mutation gate selects this.
"""
from __future__ import annotations

import pytest

from app.agents.hats.design.formulas.construction_formulas_reference_tables import (
    _CONCRETE_ECO2_KGM3,
    carbon_footprint_concrete,
)


@pytest.mark.parametrize("volume", [-1.0, -100.0])
def test_negative_volume_is_refused(volume):
    # Kills the `if volume_m3 < 0` drop_rule: without it a negative volume
    # would compute a (nonsense) negative total instead of refusing.
    r = carbon_footprint_concrete(volume, grade="c30")
    assert "error" in r and "volume_m3" in r["error"], r
    assert "total_kgco2e" not in r


@pytest.mark.parametrize("grade, factor", [("c25", 290.0), ("c40", 390.0)])
def test_known_grade_computes_from_the_table(grade, factor):
    r = carbon_footprint_concrete(10.0, grade=grade)
    assert "error" not in r, r
    assert r["factor_kgco2e_m3"] == factor
    assert r["total_kgco2e"] == pytest.approx(10.0 * factor, abs=0.1)


@pytest.mark.parametrize("grade", ["c99", "super-eco"])
def test_unknown_grade_is_refused_not_defaulted(grade):
    r = carbon_footprint_concrete(10.0, grade=grade)
    assert "error" in r, r
    assert "total_kgco2e" not in r
    # Names the known grades and cites the source; never the silent 320 figure.
    for known in _CONCRETE_ECO2_KGM3:
        assert known in r["error"], r
    assert "ICE" in r.get("standard", "")


@pytest.mark.parametrize("grade", ["c99", "whatever"])
def test_explicit_epd_factor_overrides_unknown_grade(grade):
    # An explicit supplier EPD factor computes regardless of grade label.
    r = carbon_footprint_concrete(10.0, grade=grade, embodied_kgco2e_m3=410.0)
    assert "error" not in r, r
    assert r["factor_kgco2e_m3"] == 410.0
    assert r["total_kgco2e"] == pytest.approx(4100.0, abs=0.1)
