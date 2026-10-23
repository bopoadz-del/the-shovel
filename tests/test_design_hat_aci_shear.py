"""Concrete shear with no stirrups follows ACI 318-19 Table 22.5.5.1.

The 0.17√fc equation is only for members that already have Av ≥ Av,min.
Without stirrups, Vc uses the size-effect row.
"""
from __future__ import annotations

import math

import pytest

from app.agents.hats.design.formulas.construction_formulas_structural_rc import (
    rc_beam_shear_capacity,
)


def _phi_vc_no_stirrups(bw: float, d: float, fc: float, rho: float) -> float:
    sqrt_fc = min(math.sqrt(fc), 8.3)
    lam_s = min(1.0, math.sqrt(2.0 / (1.0 + 0.004 * d)))
    vc = 0.66 * lam_s * (rho ** (1.0 / 3.0)) * sqrt_fc
    vc = min(vc, 0.42 * sqrt_fc)
    return 0.75 * vc * bw * d / 1000.0


def test_a_beam_with_no_stirrups_uses_the_size_effect():
    r = rc_beam_shear_capacity(
        width_mm=300, eff_depth_mm=500, fc_mpa=30, code="aci", rho_l=0.01,
    )
    assert r["shear_capacity_kn"] == pytest.approx(71.5, abs=0.1)
    assert r["shear_capacity_kn"] == pytest.approx(
        _phi_vc_no_stirrups(300, 500, 30, 0.01), abs=0.05,
    )


def test_the_size_effect_follows_depth():
    r = rc_beam_shear_capacity(
        width_mm=300, eff_depth_mm=550, fc_mpa=30, code="aci", rho_l=0.01,
    )
    assert r["shear_capacity_kn"] == pytest.approx(
        _phi_vc_no_stirrups(300, 550, 30, 0.01), abs=0.05,
    )
