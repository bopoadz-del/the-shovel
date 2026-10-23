"""Eurocode span/depth limits are effective depths. Thickness adds the cover.

EN 1992-1-1 §7.4.2 gives l/d. §4.4.1 nominal cover for XC1, class S4, is
15 mm plus the 10 mm deviation allowance.
"""
from __future__ import annotations

import pytest

from app.agents.hats.design.formulas.construction_formulas_structural_rc import (
    slab_thickness_min,
)


def test_eurocode_thickness_adds_nominal_cover_to_the_effective_depth():
    r = slab_thickness_min(span_mm=6000, support_condition="simply_supported", code="eurocode")
    assert r["min_thickness_mm"] == pytest.approx(6000 / 20 + 25, abs=0.1)


def test_a_stated_bar_adds_half_its_diameter():
    r = slab_thickness_min(
        span_mm=6000, support_condition="simply_supported", code="eurocode",
        cover_mm=25, bar_diameter_mm=12,
    )
    assert r["min_thickness_mm"] == pytest.approx(6000 / 20 + 25 + 6, abs=0.1)


def test_aci_table_is_already_a_thickness():
    r = slab_thickness_min(
        span_mm=6000, support_condition="simply_supported", code="aci", fy_mpa=420,
    )
    assert r["min_thickness_mm"] == pytest.approx(300.0, abs=0.1)
