"""The metric modulus converts N/mm² to kg/cm² by 10.197, not by 10.

Ec = 15000 √(f'c) with f'c in kg/cm². 1 N/mm² = 10.197 kg/cm².
"""
from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas.construction_formulas import (
    modulus_of_elasticity_concrete,
)

_NMM2_TO_KGCM2 = 10.197


def test_c40_uses_the_technical_conversion():
    r = modulus_of_elasticity_concrete(40)
    expect = round(15000 * math.sqrt(40 * _NMM2_TO_KGCM2))
    assert r["value"] == pytest.approx(expect, abs=0.5)
    assert r["value"] == pytest.approx(302941, abs=0.5)
    assert r["unit"] == "kg/cm2"
