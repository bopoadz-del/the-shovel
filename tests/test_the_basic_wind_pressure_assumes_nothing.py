"""A factor nobody supplied is an invented input.

Live 589e637, SET4 T14: "What is the basic wind velocity pressure for a wind
speed of 40 m/s?" returned 833.7 Pa in all 20 runs. The basic velocity
pressure is 0.613 V^2 = 980.8 Pa; the answer was 15% lower because
``wind_pressure`` defaulted the directionality factor to Kd = 0.85 -- a site
assumption the operator never made and the answer's own working then printed
as if it had been given.

Kz, Kzt and Kd default to unity: the caller supplies a factor or there is no
factor. Passing one explicitly is unchanged.
"""
import pytest

from app.lib.construction_formulas_loads import wind_pressure


def test_the_basic_velocity_pressure_applies_no_unrequested_factor():
    r = wind_pressure(40)
    assert r["velocity_pressure_pa"] == pytest.approx(980.8, abs=0.2)
    assert r["velocity_pressure_kn_m2"] == pytest.approx(0.9808, abs=0.001)


def test_the_note_shows_unity_factors_not_a_hidden_0_85():
    note = wind_pressure(40)["note"]
    assert "0.85" not in note, f"a factor nobody supplied is in the working: {note}"
    assert "0.613" in note


@pytest.mark.parametrize("kd,expected", [(0.85, 833.7), (1.0, 980.8), (0.9, 882.7)])
def test_an_explicitly_supplied_factor_is_honoured(kd, expected):
    assert wind_pressure(40, kd=kd)["velocity_pressure_pa"] == pytest.approx(expected, abs=0.2)


def test_exposure_and_topography_also_default_to_unity():
    assert wind_pressure(40, kz=1.0, kzt=1.0, kd=1.0)["velocity_pressure_pa"] == pytest.approx(
        wind_pressure(40)["velocity_pressure_pa"], abs=0.01)


def test_the_eurocode_basic_pressure_is_unchanged():
    # qb = 0.5 x 1.25 x 40^2 = 1000 Pa -- a different reference quantity, and
    # not what this fix touches.
    assert wind_pressure(40, code="eurocode")["velocity_pressure_pa"] == pytest.approx(1000.0, abs=0.2)
