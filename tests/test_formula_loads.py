"""Oracle tests for the design-load additive formulas (dual-code).

Values hand-derived from ASCE 7-16 / EN 1991/1998 and shown in comments.
"""
import pytest

from app.lib.construction_formulas import CALCULATORS


# ── wind_pressure ──────────────────────────────────────────────────────────

def test_wind_pressure_asce():
    # ASCE 7-16 §26.10: 0.613*1*1*0.85*50^2 = 0.613*0.85*2500 = 1302.6 Pa.
    r = CALCULATORS["wind_pressure"](wind_speed_m_s=50, code="aci",
                                     kz=1.0, kzt=1.0, kd=0.85)
    assert r["velocity_pressure_pa"] == pytest.approx(1302.6, abs=0.5)


def test_wind_pressure_eurocode():
    # EN 1991-1-4: qb = 0.5*1.25*50^2 = 1562.5 Pa.
    r = CALCULATORS["wind_pressure"](wind_speed_m_s=50, code="eurocode")
    assert r["velocity_pressure_pa"] == pytest.approx(1562.5, abs=0.5)


# ── seismic_base_shear ─────────────────────────────────────────────────────

def test_seismic_base_shear_asce():
    # ASCE 7-16 §12.8: Cs = 1.0*1.0/8 = 0.125; V = 0.125*10000 = 1250 kN.
    r = CALCULATORS["seismic_base_shear"](seismic_weight_kn=10000, code="aci",
                                          sds=1.0, r=8.0, ie=1.0)
    assert r["seismic_response_coefficient_cs"] == pytest.approx(0.125, abs=0.0005)
    assert r["base_shear_kn"] == pytest.approx(1250.0, abs=0.5)


def test_seismic_base_shear_eurocode():
    # EN 1998-1: Cs = (0.3*1.2*2.5/3.9)*0.85 = 0.23077*0.85 = 0.19615;
    # V = 0.19615*10000 = 1961.5 kN.
    r = CALCULATORS["seismic_base_shear"](
        seismic_weight_kn=10000, code="eurocode",
        ag_g=0.3, soil_factor_s=1.2, behaviour_factor_q=3.9, lambda_factor=0.85)
    assert r["seismic_response_coefficient_cs"] == pytest.approx(0.1962, abs=0.001)
    assert r["base_shear_kn"] == pytest.approx(1961.5, abs=1.0)


# ── live_load_reduction ────────────────────────────────────────────────────

def test_live_load_reduction_asce():
    # ASCE 7-16 §4.7.2: KLL*AT = 4*40 = 160 >= 37.2; factor = 0.25 + 4.57/sqrt(160)
    # = 0.25 + 0.3613 = 0.6113; L = 4.79*0.6113 = 2.928 kN/m2.
    r = CALCULATORS["live_load_reduction"](
        base_live_load_kn_m2=4.79, tributary_area_m2=40, code="aci", kll=4)
    assert r["reduction_factor"] == pytest.approx(0.6113, abs=0.001)
    assert r["reduced_live_load_kn_m2"] == pytest.approx(2.928, abs=0.005)


def test_live_load_reduction_asce_no_reduction_small_area():
    # KLL*AT = 4*8 = 32 < 37.2 -> no reduction.
    r = CALCULATORS["live_load_reduction"](
        base_live_load_kn_m2=4.79, tributary_area_m2=8, code="aci", kll=4)
    assert r["reduction_factor"] == 1.0
    assert r["reduced_live_load_kn_m2"] == pytest.approx(4.79, abs=0.001)


def test_live_load_reduction_eurocode():
    # EN 1991-1-1 §6.3.1.2: alphaA = (5/7)*0.7 + 10/40 = 0.5 + 0.25 = 0.75;
    # L = 0.75*4.79 = 3.593 kN/m2.
    r = CALCULATORS["live_load_reduction"](
        base_live_load_kn_m2=4.79, tributary_area_m2=40, code="eurocode", psi0=0.7)
    assert r["reduction_factor"] == pytest.approx(0.75, abs=0.001)
    assert r["reduced_live_load_kn_m2"] == pytest.approx(3.593, abs=0.005)
