"""Oracle tests for the structural-steel additive formulas (dual-code).

Every value is hand-derived from the cited clause and shown in the comment so a
reviewer can check it against AISC 360-16 / EN 1993 without running the code.
Routed through the live CALCULATORS registry (the real dispatch surface).
"""
import pytest

from app.lib.construction_formulas import CALCULATORS


# ── steel_tension_capacity ─────────────────────────────────────────────────

def test_steel_tension_aisc_rupture_governs():
    # AISC §D2: yield 0.90*345*3000 = 931,500 N = 931.5 kN;
    # rupture 0.75*450*2550 = 860,625 N = 860.63 kN -> rupture governs.
    r = CALCULATORS["steel_tension_capacity"](
        gross_area_mm2=3000, net_area_mm2=2550, fy_mpa=345, fu_mpa=450, code="aci")
    assert r["yield_capacity_kn"] == pytest.approx(931.5, abs=0.1)
    assert r["rupture_capacity_kn"] == pytest.approx(860.63, abs=0.05)
    assert r["capacity_kn"] == pytest.approx(860.63, abs=0.05)
    assert r["governing_limit_state"] == "rupture"


def test_steel_tension_eurocode_rupture_governs():
    # EN 1993-1-1 §6.2.3: Npl = 3000*355/1.0 = 1065 kN;
    # Nu = 0.9*2550*490/1.25 = 899,640 N = 899.64 kN -> rupture governs.
    r = CALCULATORS["steel_tension_capacity"](
        gross_area_mm2=3000, net_area_mm2=2550, fy_mpa=355, fu_mpa=490, code="eurocode")
    assert r["yield_capacity_kn"] == pytest.approx(1065.0, abs=0.1)
    assert r["rupture_capacity_kn"] == pytest.approx(899.64, abs=0.05)
    assert r["capacity_kn"] == pytest.approx(899.64, abs=0.05)


def test_steel_tension_yield_can_govern():
    # Large net area -> yield governs. Ae=Ag: yield 931.5 < rupture 0.75*450*3000=1012.5.
    r = CALCULATORS["steel_tension_capacity"](
        gross_area_mm2=3000, fy_mpa=345, fu_mpa=450, code="aci")
    assert r["governing_limit_state"] == "yield"
    assert r["capacity_kn"] == pytest.approx(931.5, abs=0.1)


# ── bolt_shear_capacity ────────────────────────────────────────────────────

def test_bolt_shear_aisc():
    # AISC §J3.6: 0.75*372*380*1 = 106,020 N = 106.02 kN.
    r = CALCULATORS["bolt_shear_capacity"](
        bolt_area_mm2=380, shear_strength_mpa=372, n_shear_planes=1, code="aci")
    assert r["capacity_kn"] == pytest.approx(106.02, abs=0.05)


def test_bolt_shear_eurocode_8_8():
    # EN 1993-1-8 Table 3.4: 0.6*800*303*1/1.25 = 116,352 N = 116.35 kN (class 8.8).
    r = CALCULATORS["bolt_shear_capacity"](
        bolt_area_mm2=303, shear_strength_mpa=800, n_shear_planes=1,
        code="eurocode", alpha_v=0.6)
    assert r["capacity_kn"] == pytest.approx(116.35, abs=0.05)


def test_bolt_shear_two_planes_doubles():
    r1 = CALCULATORS["bolt_shear_capacity"](bolt_area_mm2=380, shear_strength_mpa=372,
                                            n_shear_planes=1, code="aci")
    r2 = CALCULATORS["bolt_shear_capacity"](bolt_area_mm2=380, shear_strength_mpa=372,
                                            n_shear_planes=2, code="aci")
    assert r2["capacity_kn"] == pytest.approx(2 * r1["capacity_kn"], abs=0.01)


# ── weld_capacity ──────────────────────────────────────────────────────────

def test_weld_aisc():
    # AISC §J2.4: throat=0.707*6=4.242; 0.75*0.6*490*4.242 = 935.36 N/mm;
    # x100mm = 93,536 N = 93.54 kN.
    r = CALCULATORS["weld_capacity"](
        leg_size_mm=6, length_mm=100, electrode_strength_mpa=490, code="aci")
    assert r["effective_throat_mm"] == pytest.approx(4.242, abs=0.001)
    assert r["capacity_kn"] == pytest.approx(93.54, abs=0.05)


def test_weld_eurocode_s355():
    # EN 1993-1-8 §4.5.3.3: fvw,d = 490/(sqrt3*0.9*1.25) = 251.47 MPa;
    # x throat 4.242 = 1066.7 N/mm; x100 = 106.67 kN.
    r = CALCULATORS["weld_capacity"](
        leg_size_mm=6, length_mm=100, electrode_strength_mpa=490,
        code="eurocode", beta_w=0.9)
    assert r["capacity_kn"] == pytest.approx(106.67, abs=0.1)
