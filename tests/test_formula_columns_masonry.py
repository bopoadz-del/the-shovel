"""Oracle tests for column axial capacity + masonry wall capacity (with
slenderness, dual-code). Values hand-derived from ACI 318-19/EN 1992 and
TMS 402/EN 1996 and shown in comments."""
import pytest

from app.lib.construction_formulas import CALCULATORS


# ── column_axial_capacity ──────────────────────────────────────────────────

def test_column_axial_aci_short_capacity_and_slenderness():
    # P0 = 0.85*30*156000 + 420*4000 = 5,658,000 N; Pn,max=0.80*P0;
    # phi.Pn = 0.65*4,526,400 = 2,942,160 N = 2942.2 kN.
    # slenderness = 3000/115.47 = 25.98 (> 22 -> slender).
    # Ec=4700*sqrt(30)=25743; Ig=160000*115.47^2=2.1333e9; EI=0.4*Ec*Ig;
    # Pc = pi^2*EI/3000^2 = 24,091 kN.
    r = CALCULATORS["column_axial_capacity"](
        gross_area_mm2=160000, steel_area_mm2=4000, fc_mpa=30, fy_mpa=420,
        unsupported_length_mm=3000, radius_gyration_mm=115.47, code="aci")
    assert r["axial_capacity_kn"] == pytest.approx(2942.2, abs=0.5)
    assert r["slenderness_ratio"] == pytest.approx(25.98, abs=0.05)
    assert r["is_slender"] is True
    assert r["euler_critical_load_kn"] == pytest.approx(24091, abs=30)


def test_column_axial_aci_magnifier():
    # Pu=2000 kN, Cm=1.0, Pc=24091: delta = 1/(1 - 2000/(0.75*24091)) = 1.124.
    r = CALCULATORS["column_axial_capacity"](
        gross_area_mm2=160000, steel_area_mm2=4000, fc_mpa=30, fy_mpa=420,
        unsupported_length_mm=3000, radius_gyration_mm=115.47, code="aci",
        applied_load_kn=2000, cm=1.0)
    assert r["moment_magnifier_delta_ns"] == pytest.approx(1.124, abs=0.01)


def test_column_axial_eurocode():
    # NRd = fcd*Ac + fyd*As = 20*156000 + 365.22*4000 = 4,580,870 N = 4580.9 kN.
    r = CALCULATORS["column_axial_capacity"](
        gross_area_mm2=160000, steel_area_mm2=4000, fc_mpa=30, fy_mpa=420,
        unsupported_length_mm=3000, radius_gyration_mm=115.47, code="eurocode")
    assert r["axial_capacity_kn"] == pytest.approx(4580.9, abs=0.5)
    assert r["slenderness_ratio"] == pytest.approx(25.98, abs=0.05)


# ── masonry_wall_capacity ──────────────────────────────────────────────────

def test_masonry_tms_slender():
    # r = 190/sqrt(12) = 54.85; h/r = 54.70 <= 99;
    # R = 1-(3000/(140*54.85))^2 = 1-0.1526 = 0.8474;
    # Pa = 0.25*10*190000 * 0.8474 = 402.5 kN.
    r = CALCULATORS["masonry_wall_capacity"](
        masonry_strength_mpa=10, net_area_mm2=190000, height_mm=3000,
        thickness_mm=190, code="aci")
    assert r["slenderness_reduction_R"] == pytest.approx(0.8474, abs=0.002)
    assert r["capacity_kn"] == pytest.approx(402.5, abs=1.0)


def test_masonry_eurocode_annex_g():
    # fd=10/2.7=3.704; lam_c=(3000/190)*sqrt(10/10000)=0.4992; emk/t=0.05;
    # A1=0.9; u=(0.4992-0.063)/(0.73-0.0585)=0.6497; Phi_m=0.9*exp(-u^2/2)=0.7288;
    # NRd=0.7288*190000*3.704 = 512.9 kN.
    r = CALCULATORS["masonry_wall_capacity"](
        masonry_strength_mpa=10, net_area_mm2=190000, height_mm=3000,
        thickness_mm=190, code="eurocode", gamma_m=2.7)
    assert r["phi_reduction"] == pytest.approx(0.7288, abs=0.003)
    assert r["capacity_kn"] == pytest.approx(512.9, abs=1.5)
