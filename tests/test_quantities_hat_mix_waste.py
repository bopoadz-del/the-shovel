"""A waste allowance of 0 to 1 is a fraction added to the dry volume.

10 m³ of 1:2:4 at 5 percent is 10 × 1.54 × 1.05 = 16.17 m³, not 5 percent of the dry volume.
"""
from __future__ import annotations

import pytest

from app.agents.hats.quantities.formulas.construction_formulas_planning import (
    concrete_mix_proportions,
)


def test_five_percent_waste_increases_the_dry_volume():
    r = concrete_mix_proportions(
        wet_volume=10, cement_parts=1, sand_parts=2, aggregate_parts=4,
        waste_factor=0.05,
    )
    assert r["dry_volume"] == pytest.approx(16.17, abs=0.001)
    assert r["cement_volume"] == pytest.approx(16.17 / 7, abs=0.001)


def test_no_waste_keeps_the_dry_factor_alone():
    r = concrete_mix_proportions(
        wet_volume=10, cement_parts=1, sand_parts=2, aggregate_parts=4,
    )
    assert r["dry_volume"] == pytest.approx(15.4, abs=1e-6)
