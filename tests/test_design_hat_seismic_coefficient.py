"""A stated seismic coefficient is Cs. The code formulas apply only when it is absent."""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation


def _shear(params: dict) -> dict:
    env = run_calculation("seismic_base_shear", params)
    assert env["status"] == "success", env
    return env["result"]


def test_a_supplied_coefficient_is_cs():
    # V = Cs * W. Two different coefficients, so this is the input, not one product.
    stated = _shear({
        "text": "Seismic coefficient 0.1 and building weight 5000 kN, what's the base shear?",
    })
    assert stated["base_shear_kn"] == pytest.approx(500.0, abs=0.05)
    assert stated["seismic_response_coefficient_cs"] == pytest.approx(0.1, abs=1e-6)

    other = _shear({"seismic_weight_kn": 8000, "seismic_coefficient": 0.2})
    assert other["base_shear_kn"] == pytest.approx(1600.0, abs=0.05)
    assert other["seismic_response_coefficient_cs"] == pytest.approx(0.2, abs=1e-6)


def test_omitted_coefficient_keeps_the_code_formula():
    # ASCE 7-16: Cs = 1.0 * 1.0 / 8 = 0.125; V = 1250 kN. Not the supplied-coefficient path.
    asce = _shear({
        "seismic_weight_kn": 10000, "code": "aci", "sds": 1.0, "r": 8.0, "ie": 1.0,
    })
    assert asce["base_shear_kn"] == pytest.approx(1250.0, abs=0.5)
    assert asce["seismic_response_coefficient_cs"] == pytest.approx(0.125, abs=1e-4)
