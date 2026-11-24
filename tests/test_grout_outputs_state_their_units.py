"""A PT grout result states every figure it computes, each in the unit it is in.

``calculation_statement`` states only outputs whose DECLARED unit is not
``''`` / ``'-'`` and labels each figure with that declared unit. The grout
checks computed a pressure in psi and in kgf/cm2 and echoed the mixing time in
minutes, but declared those outputs unitless, so none of them was ever stated.
They also declared the grout take per metre of duct as metres, so that figure
was stated under the wrong unit.

Each case below builds its own inputs and checks the stated figure against a
hand computation in the declared unit, independent of the formula's code:
  * litres per metre of duct = pi * (d/2)^2 mm2 * 1000 mm / 1e6 (1 L = 1e6 mm3)
  * 1 N/mm2 = 145.0377 psi = 10.19716 kgf/cm2
  * the mixing time is the input time, in minutes
"""
from __future__ import annotations

import math

import pytest

from app.agents.base.tools import calculation_statement
from app.lib.construction_formulas import run_calculation

PSI_PER_N_MM2 = 145.0377
KGF_CM2_PER_N_MM2 = 10.19716


def _stated(diameter_mm: float, pressure_n_mm2: float, mixing_min: float) -> list[tuple[float, str]]:
    """(figure, unit) for every figure the statement builder states."""
    result = run_calculation("grout_pressure_calc", {
        "tendon_duct_diameter_mm": diameter_mm,
        "required_pressure_n_mm2": pressure_n_mm2,
        "mixing_time_minutes": mixing_min,
    })
    assert result.get("status") == "success", result
    envelope = {"name": "construction_calc", "ok": True, "result": result}
    return [(st.figure, st.unit) for st in calculation_statement(envelope)]


def _figures_in(stated: list[tuple[float, str]], unit: str) -> list[float]:
    return [figure for figure, u in stated if u == unit]


@pytest.mark.parametrize("diameter_mm", [40.0, 90.0])
def test_the_grout_take_is_stated_in_litres_per_metre(diameter_mm):
    stated = _stated(diameter_mm, 0.5, 3.0)
    expected = math.pi * (diameter_mm / 2) ** 2 * 1000 / 1e6
    assert _figures_in(stated, "L/m") == [pytest.approx(expected, abs=1e-3)]
    # The litres-per-metre figure is never stated as a length.
    assert not _figures_in(stated, "m")


@pytest.mark.parametrize("pressure_n_mm2", [0.4, 0.9])
def test_the_grout_pressure_is_stated_in_psi_and_kgf_cm2(pressure_n_mm2):
    stated = _stated(60.0, pressure_n_mm2, 3.0)
    assert _figures_in(stated, "psi") == [pytest.approx(pressure_n_mm2 * PSI_PER_N_MM2, abs=0.06)]
    assert _figures_in(stated, "kgf/cm2") == [pytest.approx(pressure_n_mm2 * KGF_CM2_PER_N_MM2, abs=0.06)]


@pytest.mark.parametrize("mixing_min", [2.5, 7.0])
def test_the_mixing_time_is_stated_in_minutes(mixing_min):
    stated = _stated(60.0, 0.5, mixing_min)
    assert _figures_in(stated, "min") == [pytest.approx(mixing_min)]
