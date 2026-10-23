"""The bolt shear coefficient comes from the standard's class table.

Lane 4 on 12e9321: an M16 bolt with fub 1000 MPa gave 75.36 kN (alpha_v
0.6) where EN 1993-1-8 Table 3.4 gives 62.8 kN (class 10.9, alpha_v 0.5).
Two causes: "EC3" / "Eurocode 3" / "EN 1993-1-8" were not recognised as the
Eurocode (only "ec", "eurocode", "en", "en1993" were), so the AISC path ran;
and with no class given the default 0.6 was used even when fub names the
class. The answer now states the class and the coefficient it used.
"""
import pytest

from app.lib import construction_formulas as cf
from app.agents.hats.design.formulas.construction_formulas_structural_steel import _norm_code


@pytest.mark.parametrize("code", ["EC3", "Eurocode 3", "EN 1993-1-8", "BS EN 1993-1-8", "ec", "eurocode"])
def test_every_eurocode_spelling_is_the_eurocode(code):
    assert _norm_code(code) == "eurocode"


@pytest.mark.parametrize("code", ["AISC 360", "aci", "ACI 318", "", None, "english"])
def test_other_codes_stay_on_the_aisc_path(code):
    assert _norm_code(code) == "aci"


def _run(**p):
    return cf.run_calculation("bolt_shear_capacity", p)["result"]


def test_an_fub_that_names_one_class_takes_that_class_coefficient():
    r = _run(bolt_area_mm2=157, shear_strength_mpa=1000, code="EC3")
    assert r["capacity_kn"] == 62.8 and r["bolt_class"] == "10.9" and r["alpha_v"] == 0.5
    assert "10.9" in r["note"] and "0.5" in r["note"]


def test_an_8_8_bolt_is_unchanged():
    assert _run(bolt_area_mm2=245, shear_strength_mpa=800, code="Eurocode 3")["capacity_kn"] == 94.08


def test_an_fub_two_classes_share_keeps_the_default_and_says_so():
    r = _run(bolt_area_mm2=157, shear_strength_mpa=400, code="EN 1993-1-8")
    assert r["alpha_v"] == 0.6 and "not given" in r["note"]
