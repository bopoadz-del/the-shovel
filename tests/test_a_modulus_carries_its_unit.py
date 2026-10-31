"""A bare number in kg/cm2 gets converted by guesswork.

Live SET4.1 T12, 7/10. The answer stated the ACI formula and its substitution
correctly -- "Ec = 4700 x sqrt(35) = 4700 x 5.9161" -- and then printed
**28,062 MPa**. That arithmetic gives 27,806. 28,062 is the tool's output,
280,624, divided by 10; the tool returns
``15000 x sqrt(f'c)`` in **kg/cm2** as a bare float, and 280,624 kg/cm2 is
27,520 MPa (x0.0980665), not 28,062.

So three different numbers were in play for one quantity, and nothing in the
payload said which unit it was in. The value now carries its unit and its own
MPa conversion, and the ACI 318-19 SI form is available on request -- that is
what an ACI question asks for.
"""
import math

import pytest

from app.lib import construction_formulas as cf

CALC = "modulus_of_elasticity_concrete"


def _result(params):
    out = cf.run_calculation(CALC, params)
    assert out["status"] == "success", out
    return out["result"]


# ── the default form is unchanged, but now says what it is ─────────────────

def test_the_default_value_is_unchanged():
    # 1 N/mm² = 10.197 kg/cm², then Ec = 15000 √(f'c).
    assert _result({"fck_n_mm2": 40})["value"] == pytest.approx(
        round(15_000 * math.sqrt(40 * 10.197)), abs=0.5)
    assert _result({"fck_n_mm2": 25})["value"] == pytest.approx(
        round(15_000 * math.sqrt(25 * 10.197)), abs=0.5)


def test_the_default_value_names_its_unit():
    r = _result({"fck_n_mm2": 35})
    assert r["unit"] == "kg/cm2"
    assert r["value"] == pytest.approx(round(15_000 * math.sqrt(35 * 10.197)), abs=0.5)


def test_the_mpa_conversion_is_done_here_not_guessed():
    r = _result({"fck_n_mm2": 35})
    kg = round(15_000 * math.sqrt(35 * 10.197))
    assert r["value_mpa"] == pytest.approx(round(kg * 0.0980665), abs=1.0)
    assert r["value_mpa"] != pytest.approx(28_062.0, abs=1.0)


# ── the ACI form, which is what an ACI question asks for ───────────────────

@pytest.mark.parametrize("code", ["aci", "ACI", "aci318", "ACI-318", "aci_318_19"])
def test_the_aci_form_is_4700_root_fc_in_mpa(code):
    r = _result({"fck_n_mm2": 35, "code": code})
    assert r["value"] == pytest.approx(round(4700 * math.sqrt(35)), abs=0.5)
    assert r["value"] == pytest.approx(27_806.0, abs=1.0)
    assert r["unit"] == "MPa"


def test_the_aci_form_names_the_clause():
    assert "19.2.2.1b" in _result({"fck_n_mm2": 35, "code": "aci"})["standard"]


def test_an_unknown_code_keeps_the_default_form():
    # Not _norm_code: that helper maps every unknown string to ACI, which
    # would silently change this function's default.
    r = _result({"fck_n_mm2": 35, "code": "something else"})
    assert r["unit"] == "kg/cm2"
    assert r["value"] == pytest.approx(round(15_000 * math.sqrt(35 * 10.197)), abs=0.5)


def test_the_two_forms_are_different_numbers():
    metric = _result({"fck_n_mm2": 35})
    aci = _result({"fck_n_mm2": 35, "code": "aci"})
    assert metric["value_mpa"] != pytest.approx(aci["value"], abs=1.0), (
        "if these were the same number the unit confusion would not matter")


def test_zero_strength_is_asked_for_in_both_forms():
    for params in ({"fck_n_mm2": 0}, {"fck_n_mm2": 0, "code": "aci"}):
        out = cf.run_calculation(CALC, params)
        assert out["status"] == "needs_input" and out["needs_input"] is True, out
        assert "characteristic concrete strength" in out["question"]


# ── the same defect, left behind in the sibling function ───────────────────
#
# Live SET5 E12, asked as a follow-up to E11: "to the same code and the same
# concrete, what is its modulus of rupture?" -- f'c = 30 MPa, ACI 318-19.
# One run answered 4.16 MPa, which is this tool's output: 2.4*sqrt(300)/10 =
# 4.157, the metric-technical form in kg/cm2 roots. ACI 318-19 Eq. 19.2.3.1
# is 0.62*sqrt(f'c) = 3.40 MPa. The other run happened to reason it by hand
# and passed, which is what made it look flaky rather than deterministic.
#
# E11 was fixed in #707 and E12 was not, because they are two functions.

RUPTURE = "modulus_of_rupture"


def _rupture(params):
    out = cf.run_calculation(RUPTURE, params)
    assert out["status"] == "success", out
    return out["result"]


def test_the_rupture_default_form_is_unchanged():
    # The F-W audit suite pins these three keys and these three numbers.
    r = _rupture({"fck_n_mm2": 40})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(4.8, abs=0.005)
    assert r["split_cylinder_aci_n_mm2"] == pytest.approx(3.56, abs=0.01)
    assert r["tensile_pct_of_compressive"] == pytest.approx(8.9, abs=0.05)


@pytest.mark.parametrize("code", ["aci", "ACI", "aci318", "ACI-318", "aci_318_19"])
def test_the_rupture_aci_form_is_062_root_fc(code):
    r = _rupture({"fck_n_mm2": 30, "code": code})
    assert r["value"] == pytest.approx(0.62 * math.sqrt(30), abs=0.005)
    assert r["value"] == pytest.approx(3.396, abs=0.005)
    assert r["unit"] == "MPa"


def test_the_rupture_aci_form_is_not_the_live_wrong_answer():
    assert _rupture({"fck_n_mm2": 30, "code": "aci"})["value"] != pytest.approx(
        4.157, abs=0.01), "4.16 MPa was the figure the operator was given"


def test_the_rupture_aci_form_names_the_clause():
    assert "19.2.3.1" in _rupture({"fck_n_mm2": 30, "code": "aci"})["standard"]


def test_an_unknown_code_keeps_the_default_rupture_form():
    r = _rupture({"fck_n_mm2": 30, "code": "something else"})
    assert r["modulus_of_rupture_n_mm2"] == pytest.approx(4.157, abs=0.005)


def test_both_moduli_answer_the_same_code_consistently():
    # E11 and E12 are one conversation: naming ACI once must govern both.
    ec = _result({"fck_n_mm2": 30, "code": "aci"})
    fr = _rupture({"fck_n_mm2": 30, "code": "aci"})
    assert ec["unit"] == fr["unit"] == "MPa"
    assert ec["value"] == pytest.approx(25_742.0, abs=1.0)
    assert fr["value"] == pytest.approx(3.396, abs=0.005)
