"""Phase 1 formula audit — calculators A–F (#1–44 of available_calculations()).

Each test derives the expected figure by hand from the cited source, then
compares ``run_calculation(name, params)["result"]``. A wrong formula that
still returns a number must fail. One test per calculator; a second case in
the same test covers a boundary or a branch change. Bad input must come back
as ``status == "error"`` with a useful message.

This file owns the A–F half. Do not put these tests in the sibling G–Z file.
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import available_calculations, run_calculation
from tests.example_project_rates import (
    CONCRETE_RATES, ELECTRICAL_STAGE_RATES, FORMWORK_RATES, REBAR_RATES,
)

# Alphabetical #1–44 of available_calculations() at audit time. The two
# point-load deflection calculators joined the slice with SET5 E3.
_AUDIT_A_TO_F = [
    "backfill_volume",
    "beam_deflection_cantilever_point_load",
    "beam_deflection_cantilever_udl",
    "beam_deflection_ss_point_load_midspan",
    "beam_deflection_ss_udl",
    "beam_moment_fixed_udl",
    "beam_moment_point_load",
    "beam_moment_simple",
    "beam_shear_simple",
    "bim_clash_tolerance",
    "bolt_shear_capacity",
    "calculate_evm",
    "calculate_interim_payment",
    "calculate_payment",
    "carbon_footprint_concrete",
    "column_axial_capacity",
    "compaction_control",
    "composite_column_design",
    "concrete_curing_time",
    "concrete_cylinders",
    "concrete_maturity_strength",
    "concrete_mix_design_sg",
    "concrete_mix_proportions",
    "concrete_mix_slip_form",
    "concrete_shrinkage",
    "concrete_thermal_cracking_check",
    "concrete_volume",
    "cost_buildup_concrete",
    "cost_buildup_formwork",
    "cost_buildup_rebar",
    "cost_per_area",
    "crane_cost_estimate",
    "crane_lift_capacity",
    "crane_planning",
    "critical_path_float",
    "cut_fill_balance",
    "delay_damages_daily",
    "dewatering_uplift_check",
    "dewatering_well_point_spacing",
    "diaphragm_wall_panel_volume",
    "electrical_installation_sequence",
    "evaluate_tender",
    "excavation_volume",
    "fall_arrest_force",
]


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env["status"] == "success", env
    assert isinstance(env.get("result"), dict)
    return env["result"]


def _err(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env["status"] == "error", env
    assert isinstance(env.get("error"), str) and env["error"].strip()
    return env


def test_audit_covers_the_leading_names():
    names = available_calculations()
    assert names[:len(_AUDIT_A_TO_F)] == _AUDIT_A_TO_F


# ── 1 backfill_volume ──────────────────────────────────────────────────────

def test_audit_backfill_volume():
    """Geometry / soil swell: void = excavation − structure;
    loose = void × (1 + swell).
    Hand: 600 − 200 = 400 m³ void; × 1.20 = 480 m³ loose.
    Boundary: structure ≥ excavation → void 0, loose 0 (not a negative import).
    """
    r = _ok("backfill_volume", {"excavation_bank_m3": 600, "structure_volume_m3": 200})
    assert r["void_volume_m3"] == pytest.approx(400.0, abs=0.01)
    assert r["loose_backfill_needed_m3"] == pytest.approx(480.0, abs=0.01)

    edge = _ok("backfill_volume", {"excavation_bank_m3": 200, "structure_volume_m3": 200})
    assert edge["void_volume_m3"] == pytest.approx(0.0, abs=1e-9)
    assert edge["loose_backfill_needed_m3"] == pytest.approx(0.0, abs=1e-9)

    _err("backfill_volume", {"excavation_bank_m3": -1, "structure_volume_m3": 10})


# ── 2–3 beam deflection (mechanics of materials) ───────────────────────────

def test_audit_beam_deflection_cantilever_udl():
    """Cantilever UDL: δ = wL⁴ / (8EI). Units: kN/m ≡ N/mm, L in mm, E in MPa,
    I in mm⁴ → δ in mm.
    Hand: w=10, L=5000 mm, E=30000, I=1e9
          10 × 5000⁴ / (8 × 30000 × 1e9) = 6.25e15 / 2.4e14 = 26.0417 mm.
    SI check: 10000 N/m × 5⁴ / (8 × 30e9 × 1e-3) = 0.02604 m = 26.04 mm.
    """
    r = _ok("beam_deflection_cantilever_udl",
            {"w_kn_m": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert r["value"] == pytest.approx(26.04, abs=0.02)

    zero = _ok("beam_deflection_cantilever_udl",
               {"w_kn_m": 10, "span_m": 0, "ec_mpa": 30000, "i_mm4": 1e9})
    assert zero["value"] == pytest.approx(0.0, abs=1e-9)

    _err("beam_deflection_cantilever_udl",
         {"w_kn_m": 10, "span_m": 5, "ec_mpa": 0, "i_mm4": 1e9})


def test_audit_beam_deflection_cantilever_point_load():
    """Cantilever, load at the tip: δ = PL³ / (3EI). Same units, P in kN → N.
    Hand: P=10 kN = 10,000 N, L=5000 mm, E=30000 MPa, I=1e9 mm⁴
          10000 × 5000³ / (3 × 30000 × 1e9) = 1.25e15 / 9e13 = 13.889 mm.
    SI check: 10 kN × 5³ / (3 × 30e9 × 1e-3) = 0.01389 m = 13.89 mm.

    Against the UDL sibling on the same numbers: 26.04 mm. Substituting a
    point load into wL⁴/8EI is what produced SET5 E3's 16.0 mm for a case
    whose answer is 10.67 mm.
    """
    r = _ok("beam_deflection_cantilever_point_load",
            {"p_kn": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert r["value"] == pytest.approx(13.89, abs=0.02)

    zero = _ok("beam_deflection_cantilever_point_load",
               {"p_kn": 0, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert zero["value"] == pytest.approx(0.0, abs=1e-9)

    _err("beam_deflection_cantilever_point_load",
         {"p_kn": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 0})


def test_audit_beam_deflection_ss_point_load_midspan():
    """Simply supported, load at midspan: δ = PL³ / (48EI).
    Hand: 10000 × 5000³ / (48 × 30000 × 1e9) = 1.25e15 / 1.44e15 = 0.868 mm.
    Ratio cantilever/SS for the same P and L = 48/3 = 16; 13.889/0.868 = 16.0.

    A second moment given in m⁴ is refused rather than used: 1e-3 mm⁴ is no
    section, and taking it at face value overstates δ by 1e12.
    """
    r = _ok("beam_deflection_ss_point_load_midspan",
            {"p_kn": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert r["value"] == pytest.approx(0.87, abs=0.01)

    cantilever = _ok("beam_deflection_cantilever_point_load",
                     {"p_kn": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert cantilever["value"] / r["value"] == pytest.approx(16.0, abs=0.05)

    _err("beam_deflection_ss_point_load_midspan",
         {"p_kn": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e-3})


def test_audit_beam_deflection_ss_udl():
    """Simply-supported UDL: δ = 5wL⁴ / (384EI). Same units as the cantilever.
    Hand: 5 × 10 × 5000⁴ / (384 × 30000 × 1e9) = 3.125e16 / 1.152e16 = 2.712 mm.
    Ratio cantilever/SS = (1/8) / (5/384) = 9.6; 26.04/2.71 = 9.61.
    """
    r = _ok("beam_deflection_ss_udl",
            {"w_kn_m": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 1e9})
    assert r["value"] == pytest.approx(2.71, abs=0.02)

    _err("beam_deflection_ss_udl",
         {"w_kn_m": 10, "span_m": 5, "ec_mpa": 30000, "i_mm4": 0})


# ── 4–7 beam statics ───────────────────────────────────────────────────────

def test_audit_beam_moment_fixed_udl():
    """Fixed-fixed UDL: hogging wL²/12 at supports, sagging wL²/24 at midspan.
    Hand: w=20 kN/m, L=6 m → 20×36/12 = 60 kN·m; 20×36/24 = 30 kN·m.
    """
    r = _ok("beam_moment_fixed_udl", {"udl_w_kn_m": 20, "span_m": 6})
    assert r["support_moment_kn_m"] == pytest.approx(60.0, abs=0.01)
    assert r["midspan_moment_kn_m"] == pytest.approx(30.0, abs=0.01)
    assert r["max_moment_kn_m"] == pytest.approx(60.0, abs=0.01)

    _err("beam_moment_fixed_udl", {"udl_w_kn_m": 20, "span_m": 0})


def test_audit_beam_moment_point_load():
    """SS point load: central M = PL/4; offset M = Pab/L.
    Hand: P=50 kN, L=6 m → 50×6/4 = 75 kN·m.
    Offset a=2 m, b=4 m → 50×2×4/6 = 66.667 kN·m.
    """
    mid = _ok("beam_moment_point_load", {"point_load_kn": 50, "span_m": 6})
    assert mid["max_moment_kn_m"] == pytest.approx(75.0, abs=0.01)

    off = _ok("beam_moment_point_load",
              {"point_load_kn": 50, "span_m": 6, "distance_from_left_m": 2})
    assert off["max_moment_kn_m"] == pytest.approx(66.667, abs=0.01)

    _err("beam_moment_point_load",
         {"point_load_kn": 50, "span_m": 6, "distance_from_left_m": 7})


def test_audit_beam_moment_simple():
    """SS UDL: Mmax = wL²/8 at midspan.
    Hand: 20 × 36 / 8 = 90 kN·m.
    """
    r = _ok("beam_moment_simple", {"udl_w_kn_m": 20, "span_m": 6})
    assert r["max_moment_kn_m"] == pytest.approx(90.0, abs=0.01)

    _err("beam_moment_simple", {"udl_w_kn_m": 20, "span_m": -1})


def test_audit_beam_shear_simple():
    """SS shear: V = wL/2 + P/2 at the support.
    Hand: 20×6/2 = 60 kN. With P=40: 60 + 20 = 80 kN.
    """
    r = _ok("beam_shear_simple", {"udl_w_kn_m": 20, "span_m": 6})
    assert r["max_shear_kn"] == pytest.approx(60.0, abs=0.01)

    both = _ok("beam_shear_simple",
               {"udl_w_kn_m": 20, "span_m": 6, "central_point_load_kn": 40})
    assert both["max_shear_kn"] == pytest.approx(80.0, abs=0.01)

    _err("beam_shear_simple", {"udl_w_kn_m": -1, "span_m": 6})


# ── 8 bim_clash_tolerance ──────────────────────────────────────────────────

def test_audit_bim_clash_tolerance():
    """Indicative BEP lookup (not a standard constant): LOD 350 → ±12 mm;
    LOD 200 → ±50 mm. Unknown LOD must error, not invent a tolerance.
    """
    r = _ok("bim_clash_tolerance", {"lod": 350})
    assert r["clash_tolerance_mm"] == pytest.approx(12.0, abs=1e-9)

    lo = _ok("bim_clash_tolerance", {"lod": 200})
    assert lo["clash_tolerance_mm"] == pytest.approx(50.0, abs=1e-9)

    _err("bim_clash_tolerance", {"lod": 999})


# ── 9 bolt_shear_capacity ──────────────────────────────────────────────────

def test_audit_bolt_shear_capacity():
    """AISC 360-16 §J3.6: φRn = 0.75 × Fnv × Ab × n.
    Hand: 0.75 × 372 × 380 × 1 = 106,020 N = 106.02 kN.
    Two planes doubles it. EN 1993-1-8 Table 3.4:
    0.6 × 800 × 303 / 1.25 = 116,352 N = 116.35 kN.
    """
    aisc = _ok("bolt_shear_capacity",
               {"bolt_area_mm2": 380, "shear_strength_mpa": 372,
                "n_shear_planes": 1, "code": "aci"})
    assert aisc["capacity_kn"] == pytest.approx(106.02, abs=0.05)

    two = _ok("bolt_shear_capacity",
              {"bolt_area_mm2": 380, "shear_strength_mpa": 372,
               "n_shear_planes": 2, "code": "aci"})
    assert two["capacity_kn"] == pytest.approx(212.04, abs=0.05)

    ec = _ok("bolt_shear_capacity",
             {"bolt_area_mm2": 303, "shear_strength_mpa": 800,
              "n_shear_planes": 1, "code": "eurocode", "alpha_v": 0.6})
    assert ec["capacity_kn"] == pytest.approx(116.35, abs=0.05)

    _err("bolt_shear_capacity", {"bolt_area_mm2": 0, "shear_strength_mpa": 372})


# ── 10 calculate_evm ───────────────────────────────────────────────────────

def test_audit_calculate_evm():
    """PMBOK / PMI: SPI = EV/PV, CPI = EV/AC, SV = EV−PV, CV = EV−AC,
    EAC = BAC/CPI from the *unrounded* CPI.
    Hand: PV=500k, EV=400k, AC=450k, BAC=1e6
          CPI = 400/450 = 0.888… → display 0.889
          SPI = 0.800; CV = −50,000; SV = −100,000
          EAC = 1e6 × 450/400 = 1,125,000; ETC = 675,000; VAC = −125,000.
    Missing AC must error (no invented actuals).
    """
    r = _ok("calculate_evm",
            {"bac": 1_000_000, "bcwp": 400_000, "bcws": 500_000, "acwp": 450_000})
    assert r["CPI"] == pytest.approx(0.889, abs=0.001)
    assert r["SPI"] == pytest.approx(0.800, abs=0.001)
    assert r["CV"] == pytest.approx(-50_000, abs=0.01)
    assert r["SV"] == pytest.approx(-100_000, abs=0.01)
    assert r["EAC"] == pytest.approx(1_125_000.00, abs=0.02)
    assert r["ETC"] == pytest.approx(675_000.00, abs=0.02)
    assert r["VAC"] == pytest.approx(-125_000.00, abs=0.02)
    assert r["status"]["cost"] == "OVER BUDGET"
    assert r["status"]["schedule"] == "BEHIND"

    _err("calculate_evm", {"pv": 100_000, "ev": 90_000})  # AC missing
    _err("calculate_evm", {"pv": -1, "ev": 90_000, "ac": 95_000})


# ── 11 calculate_interim_payment ───────────────────────────────────────────

def test_audit_calculate_interim_payment():
    """FIDIC Red Book 14.3: net = gross × (1 − retention%).
    Hand: 900,000 × 10% = 90,000 held; net = 810,000.
    Boundary: 0% retention → net = gross.
    """
    r = _ok("calculate_interim_payment",
            {"gross_valuation": 900_000, "retention_percent": 10})
    assert r["retention_amount"] == pytest.approx(90_000.00, abs=0.01)
    assert r["net_payment"] == pytest.approx(810_000.00, abs=0.01)

    none = _ok("calculate_interim_payment",
               {"gross_valuation": 900_000, "retention_percent": 0})
    assert none["net_payment"] == pytest.approx(900_000.00, abs=0.01)

    _err("calculate_interim_payment", {"gross_valuation": -1, "retention_percent": 10})
    _err("calculate_interim_payment", {"gross_valuation": 100, "retention_percent": 150})


# ── 12 calculate_payment ───────────────────────────────────────────────────

def test_audit_calculate_payment():
    """Interim payment procedure / FIDIC 14.3 this-period: retention = certified × rate;
    net = certified − retention. Default rate 5% (fraction).
    Hand: certified 90,000 × 0.05 = 4,500 held; net = 85,500;
          disputed = 100,000 − 90,000 = 10,000.
    Boundary: previous certified 50,000 → cumulative 140,000;
              contract 1,000,000 → 14.0% complete.
    """
    r = _ok("calculate_payment",
            {"claimed_amount": 100_000, "certified_amount": 90_000})
    assert r["retention_held"] == pytest.approx(4_500.00, abs=0.01)
    assert r["net_payment_due"] == pytest.approx(85_500.00, abs=0.01)
    assert r["disputed_amount"] == pytest.approx(10_000.00, abs=0.01)
    assert r["retention_rate_pct"] == pytest.approx(5.0, abs=1e-9)

    cum = _ok("calculate_payment", {
        "claimed_amount": 100_000, "certified_amount": 90_000,
        "retention_rate": 0.05, "cumulative_previous_certified": 50_000,
        "contract_value": 1_000_000,
    })
    assert cum["cumulative_certified"] == pytest.approx(140_000.00, abs=0.01)
    assert cum["percent_complete"] == pytest.approx(14.0, abs=0.05)

    _err("calculate_payment", {"claimed_amount": -1, "certified_amount": 90_000})
    _err("calculate_payment",
         {"claimed_amount": 100, "certified_amount": 90, "retention_rate": 5})


# ── 13 carbon_footprint_concrete ───────────────────────────────────────────

def test_audit_carbon_footprint_concrete():
    """ICE / EPD indicative table: C30 = 320 kgCO2e/m³.
    Hand: 100 × 320 = 32,000 kg = 32.0 t. Override factor 200 → 50 × 200 = 10,000 kg.
    """
    r = _ok("carbon_footprint_concrete", {"volume_m3": 100, "grade": "c30"})
    assert r["total_kgco2e"] == pytest.approx(32_000.0, abs=0.1)
    assert r["total_tco2e"] == pytest.approx(32.0, abs=0.001)

    ov = _ok("carbon_footprint_concrete",
             {"volume_m3": 50, "embodied_kgco2e_m3": 200})
    assert ov["total_kgco2e"] == pytest.approx(10_000.0, abs=0.1)

    _err("carbon_footprint_concrete", {"volume_m3": -1, "grade": "c30"})


# ── 14 column_axial_capacity ───────────────────────────────────────────────

def test_audit_column_axial_capacity():
    """ACI 318-19 §22.4: P0 = 0.85 fc (Ag−Ast) + fy Ast;
    Pn,max = 0.80 P0 (tied); φPn = 0.65 Pn,max.
    Hand: Ag=160000, Ast=4000, fc=30, fy=420
          P0 = 0.85×30×156000 + 420×4000 = 5,658,000 N
          φPn = 0.65×0.80×5,658,000 = 2,942,160 N = 2942.2 kN
          klu/r = 3000/115.47 = 25.98 > 22 → slender.
    EC2: NRd = (30/1.5)×156000 + (420/1.15)×4000 = 4,580,870 N = 4580.9 kN.
    """
    aci = _ok("column_axial_capacity", {
        "gross_area_mm2": 160000, "steel_area_mm2": 4000, "fc_mpa": 30,
        "fy_mpa": 420, "unsupported_length_mm": 3000, "radius_gyration_mm": 115.47,
        "code": "aci",
    })
    assert aci["axial_capacity_kn"] == pytest.approx(2942.2, abs=0.5)
    assert aci["slenderness_ratio"] == pytest.approx(25.98, abs=0.05)
    assert aci["is_slender"] is True

    ec = _ok("column_axial_capacity", {
        "gross_area_mm2": 160000, "steel_area_mm2": 4000, "fc_mpa": 30,
        "fy_mpa": 420, "unsupported_length_mm": 3000, "radius_gyration_mm": 115.47,
        "code": "eurocode",
    })
    assert ec["axial_capacity_kn"] == pytest.approx(4580.9, abs=0.5)

    _err("column_axial_capacity", {
        "gross_area_mm2": 1000, "steel_area_mm2": 1000, "fc_mpa": 30,
        "fy_mpa": 420, "unsupported_length_mm": 3000, "radius_gyration_mm": 50,
    })


# ── 15 compaction_control ──────────────────────────────────────────────────

def test_audit_compaction_control():
    """ASTM D698/D1557: compaction % = field MDD / lab MDD × 100.
    Hand: 1.90/2.00 × 100 = 95.0% → pass at 95%. 1.80/2.00 = 90% → fail.
    """
    r = _ok("compaction_control",
            {"field_dry_density": 1.90, "max_dry_density": 2.00})
    assert r["compaction_percent"] == pytest.approx(95.0, abs=0.01)
    assert r["passed"] is True

    fail = _ok("compaction_control",
               {"field_dry_density": 1.80, "max_dry_density": 2.00})
    assert fail["compaction_percent"] == pytest.approx(90.0, abs=0.01)
    assert fail["passed"] is False

    _err("compaction_control",
         {"field_dry_density": 1.90, "max_dry_density": 0})


# ── 16 composite_column_design ─────────────────────────────────────────────

def test_audit_composite_column_design():
    """EN 1994-1-1 §6.7.3.2: Npl,Rd = 0.85*Ag*fck/1.5 when no steel is given.
    Hand: D=800 mm, fck=60, Ag=π×400²
          N = 0.85×Ag×60/1.5/1000 = 17,090 kN.
    A section mass is not an area, so 2 t does not raise it by 10 percent.
    """
    r = _ok("composite_column_design",
            {"axial_load_kn": 5000, "column_diameter_mm": 800})
    expect = 0.85 * math.pi * 400.0 ** 2 * 60 / 1.5 / 1000
    assert r["option_a"]["capacity_kn"] == pytest.approx(expect, abs=1)
    assert r["option_b"]["capacity_kn"] == pytest.approx(expect, abs=1)
    assert r["recommended"] == "A or B"

    ib = _ok("composite_column_design",
             {"axial_load_kn": 5000, "column_diameter_mm": 800,
              "use_i_beam": True, "i_beam_weight_t": 2.0})
    assert ib["option_a"]["capacity_kn"] == pytest.approx(expect, abs=1)

    _err("composite_column_design",
         {"axial_load_kn": 5000, "column_diameter_mm": 0})


# ── 17–19 concrete QC / maturity ───────────────────────────────────────────

def test_audit_concrete_curing_time():
    """ACI 209R inverse: t = a·frac / (1 − b·frac), Type I moist a=4, b=0.85.
    Hand: 4×0.70 / (1 − 0.85×0.70) = 2.8 / 0.405 = 6.9136 days.
    50%: 4×0.50 / (1 − 0.425) = 2.0 / 0.575 = 3.478 days.
    """
    r = _ok("concrete_curing_time", {"target_strength_fraction": 0.70})
    assert r["days_to_target"] == pytest.approx(6.91, abs=0.05)

    half = _ok("concrete_curing_time", {"target_strength_fraction": 0.50})
    assert half["days_to_target"] == pytest.approx(3.48, abs=0.05)

    _err("concrete_curing_time", {"target_strength_fraction": 0})
    _err("concrete_curing_time", {"target_strength_fraction": 1.20})


def test_audit_concrete_cylinders():
    """ASTM C39 / BS EN 12390-3: f = P/A, A = π/4 d².
    Hand: A = π/4 × 150² = 17,671.46 mm²; 530,000 / 17,671.46 = 29.99 MPa.
    100 mm cylinder, 200 kN: A = 7853.98; 200,000/7853.98 = 25.46 MPa.
    """
    r = _ok("concrete_cylinders",
            {"failure_load_kn": 530, "cylinder_diameter_mm": 150})
    assert r["compressive_strength_mpa"] == pytest.approx(30.0, abs=0.05)

    small = _ok("concrete_cylinders",
                {"failure_load_kn": 200, "cylinder_diameter_mm": 100})
    assert small["compressive_strength_mpa"] == pytest.approx(25.46, abs=0.05)

    _err("concrete_cylinders",
         {"failure_load_kn": 530, "cylinder_diameter_mm": 0})


def test_audit_concrete_maturity_strength():
    """Nurse-Saul MI = Σ max(0, T−T0)·Δt, T0=−10 °C.
    Equivalent age te_days = MI / (Tref−T0) / 24, Tref=20 °C.
    ACI 209R Type I moist: f/f28 = te / (4 + 0.85 te).
    Hand (18 h: 20/22/25 °C × 6 h):
          MI = 30×6 + 32×6 + 35×6 = 582 °C·h
          te = 582/30/24 = 0.80833 d
          ratio = 0.80833 / (4 + 0.68608) = 0.17246 → 6.9 MPa (17.2% of 40).
    Boundary (7 d at 20 °C): MI = 30×168 = 5040; te = 7 d;
          ratio = 7 / (4 + 5.95) = 0.7035 → 28.1 MPa (70.4%).
    The old (1055+625T)/(1225 T) ratio ignored MI and returned ~55% for
    both 18 h and 7 d — a wrong formula that still produced a number.
    """
    early = _ok("concrete_maturity_strength", {
        "temperature_history_c": [20, 22, 25],
        "time_intervals_hours": [6, 6, 6],
    })
    assert early["maturity_index_c_hrs"] == pytest.approx(582.0, abs=0.5)
    assert early["predicted_strength_n_mm2"] == pytest.approx(6.9, abs=0.1)
    assert early["percent_of_28d"] == pytest.approx(17.2, abs=0.2)
    # A time-independent ~55% result is the defect this test exists to catch.
    assert early["percent_of_28d"] < 30.0

    week = _ok("concrete_maturity_strength", {
        "temperature_history_c": [20],
        "time_intervals_hours": [168],
    })
    assert week["maturity_index_c_hrs"] == pytest.approx(5040.0, abs=0.5)
    assert week["predicted_strength_n_mm2"] == pytest.approx(28.1, abs=0.15)
    assert week["percent_of_28d"] == pytest.approx(70.4, abs=0.2)

    _err("concrete_maturity_strength", {
        "temperature_history_c": [20, 22],
        "time_intervals_hours": [6],
    })
    _err("concrete_maturity_strength", {
        "temperature_history_c": [],
        "time_intervals_hours": [],
    })


# ── 20–22 mix design ───────────────────────────────────────────────────────

def test_audit_concrete_mix_design_sg():
    """Absolute-volume method: C = 1 / (w/c + 1/SGc + fa/SGf + ca/SGcr) tonnes.
    Hand: w/c=0.48, SGc=3.15, SGf=2.67, SGcr=2.54, fa=2, ca=4
          denom = 0.48 + 0.31746 + 0.74906 + 1.57480 = 3.12132
          C = 0.32038 t → 320 kg; W = 154 L; FA = 640; CA = 1280.
    dune_sand_pct ≠ 0 is not implemented (no dune SG) and must error.
    """
    r = _ok("concrete_mix_design_sg", {"w_c_ratio": 0.48})
    assert r["cement_kg_m3"] == pytest.approx(320.0, abs=0.5)
    assert r["water_litres_m3"] == pytest.approx(154.0, abs=0.5)
    assert r["fine_aggregate_kg_m3"] == pytest.approx(640.0, abs=0.5)
    assert r["coarse_aggregate_kg_m3"] == pytest.approx(1280.0, abs=0.5)
    assert r["total_weight_kg_m3"] == pytest.approx(2394.0, abs=1.0)

    _err("concrete_mix_design_sg", {"w_c_ratio": 0})
    _err("concrete_mix_design_sg", {"w_c_ratio": 0.48, "dune_sand_pct": 0.3})


def test_audit_concrete_mix_proportions():
    """PE sheet: dry = wet × 1.54; constituent = dry × (parts / Σ parts).
    Hand: wet=10, 1:2:4 → dry=15.4; cement=15.4×1/7=2.2;
          sand=4.4; agg=8.8.
    Missing parts must error (no invented mix).
    """
    r = _ok("concrete_mix_proportions", {
        "wet_volume": 10, "cement_parts": 1, "sand_parts": 2, "aggregate_parts": 4,
    })
    assert r["dry_volume"] == pytest.approx(15.4, abs=1e-6)
    assert r["cement_volume"] == pytest.approx(2.2, abs=1e-6)
    assert r["sand_volume"] == pytest.approx(4.4, abs=1e-6)
    assert r["aggregate_volume"] == pytest.approx(8.8, abs=1e-6)

    _err("concrete_mix_proportions", {"wet_volume": 10})


def test_audit_concrete_mix_slip_form():
    """Slip-form mix is absolute-volume at w/c=0.42, 1:2:2.6.
    Hand: denom = 0.42 + 0.31746 + 0.74906 + 1.02362 = 2.51014
          C = 0.3984 t → 398 kg; W = 167 L; FA = 796; CA = 1034.8 → 1035.
    Documented extras (retarder 3.8/2.9 L, slump 150±30, 32 °C) are
    unsourced SPEC-C900 constants — locked, not re-derived.
    """
    r = _ok("concrete_mix_slip_form", {})
    assert r["cement_kg_m3"] == pytest.approx(398.0, abs=0.5)
    assert r["water_litres_m3"] == pytest.approx(167.0, abs=0.5)
    assert r["fine_aggregate_kg_m3"] == pytest.approx(796.0, abs=0.5)
    assert r["coarse_aggregate_kg_m3"] == pytest.approx(1035.0, abs=0.5)
    assert r["w_c_ratio"] == pytest.approx(0.42, abs=1e-9)
    assert r["retarder_20c_lit_m3"] == pytest.approx(3.8, abs=1e-9)


# ── 23–25 shrinkage / thermal / volume ─────────────────────────────────────

def test_audit_concrete_shrinkage():
    """ACI 209R hyperbolic: εsh(t) = t/(f+t) × εsh,ult, f=35 d, εult=780 με.
    Hand: 365/(35+365)×780 = 0.9125×780 = 711.75 με.
    t=0 → 0 με.
    """
    r = _ok("concrete_shrinkage", {"time_days": 365})
    assert r["shrinkage_microstrain"] == pytest.approx(711.75, abs=0.5)

    zero = _ok("concrete_shrinkage", {"time_days": 0})
    assert zero["shrinkage_microstrain"] == pytest.approx(0.0, abs=1e-9)

    _err("concrete_shrinkage", {"time_days": -1})


def test_audit_concrete_thermal_cracking_check():
    """CIRIA C660/C766 + common mass-concrete specs: core ≤ 70 °C, ΔT ≤ 20 °C.
    Hand: 60/40 → ΔT=20, both OK, risk False.
    Boundary: 70/50 → ΔT=20, still OK; 70.1 or ΔT=20.1 → risk True.
    """
    ok = _ok("concrete_thermal_cracking_check",
             {"core_temp_c": 60, "surface_temp_c": 40})
    assert ok["delta_t_c"] == pytest.approx(20.0, abs=0.05)
    assert ok["core_ok"] is True
    assert ok["delta_ok"] is True
    assert ok["thermal_cracking_risk"] is False

    edge = _ok("concrete_thermal_cracking_check",
               {"core_temp_c": 70, "surface_temp_c": 50})
    assert edge["thermal_cracking_risk"] is False

    hot = _ok("concrete_thermal_cracking_check",
              {"core_temp_c": 75, "surface_temp_c": 50})
    assert hot["delta_t_c"] == pytest.approx(25.0, abs=0.05)
    assert hot["thermal_cracking_risk"] is True


def test_audit_concrete_volume():
    """Geometry: rectangular L×W×T; headline includes documented 5% waste.
    Hand: 10×5×0.3 = 15.0 m³ net; ×1.05 = 15.75 m³.
    Cylinder: π(0.6/2)²×10 = 2.827 m³ net.
    """
    r = _ok("concrete_volume",
            {"length_m": 10, "width_m": 5, "thickness_m": 0.3})
    assert r["net_volume_m3"] == pytest.approx(15.0, abs=0.001)
    assert r["volume_m3"] == pytest.approx(15.75, abs=0.001)

    cyl = _ok("concrete_volume",
              {"shape": "cylinder", "diameter_m": 0.6, "height_m": 10})
    assert cyl["net_volume_m3"] == pytest.approx(math.pi * 0.09 * 10, abs=0.001)

    _err("concrete_volume",
         {"length_m": -1, "width_m": 5, "thickness_m": 0.3})


# ── 26–29 cost ─────────────────────────────────────────────────────────────

def test_audit_cost_buildup_concrete():
    """Documented GCC build-up (rates are parameters / fallbacks).
    Hand: cement 0.4×190=76; agg 1.839×29=53.331; water 0.16×10=1.6;
          MS 12×1.5=18; plast 5×2.4=12 → material 160.931
          waste is on material only: 160.931×1.03 = 165.759
          direct = 165.759+39.2+8+3+2.5+4 = 222.459
          sell = 222.459×1.18/0.85 = 308.83 SAR/m³
          ×100 m³ = 30,883.
    """
    r = _ok("cost_buildup_concrete", {"quantity_m3": 100, **CONCRETE_RATES})
    assert r["material_cost_per_m3"] == pytest.approx(160.93, abs=0.01)
    assert r["selling_price_per_m3"] == pytest.approx(308.83, abs=0.1)
    assert r["total_project_value"] == pytest.approx(30883, abs=2)

    _err("cost_buildup_concrete", {"quantity_m3": -1})


def test_audit_cost_buildup_formwork():
    """Documented: shuttering = supply/6 uses; scaffold = daily × cycle;
    labour = mhr × rate; plant = crane_rate / output; then ×1.18/0.85.
    Hand: 55/6 = 9.1667; scaffold 2.15×10 = 21.50; labour 9×4.1 = 36.90
          plant 134.6/30 = 4.4867; direct = 72.0533
          sell = 72.0533 × 1.18 / 0.85 = 100.027 SAR/m²
          ×500 m² = 50,013.5 → 50,013.
    """
    r = _ok("cost_buildup_formwork", {"area_m2": 500, **FORMWORK_RATES})
    assert r["shuttering_per_m2"] == pytest.approx(55 / 6, abs=0.01)
    assert r["scaffolding_per_m2"] == pytest.approx(21.50, abs=0.01)
    assert r["labour_per_m2"] == pytest.approx(36.90, abs=0.01)
    assert r["selling_price_per_m2"] == pytest.approx(100.03, abs=0.02)
    assert r["total_for_area"] == pytest.approx(50013, abs=2)

    _err("cost_buildup_formwork", {"area_m2": -1})


def test_audit_cost_buildup_rebar():
    """Documented: material = qty_t × price × 1.10; then + labour + crane,
    ×1.18/0.85. Hand for 1 t @ 2600: material 2860;
    labour 90×4.1=369; crane 2×134.6=269.2; direct 3498.2
    sell = 3498.2 × 1.18 / 0.85 = 4856 SAR/t.
    """
    r = _ok("cost_buildup_rebar", {"quantity_kg": 1000, **REBAR_RATES})
    assert r["material_per_t"] == pytest.approx(2860, abs=1)
    assert r["selling_price_per_t"] == pytest.approx(4856, abs=2)

    real = _ok("cost_buildup_rebar",
               {**REBAR_RATES, "quantity_kg": 1000, "material_price_per_t": 3000})
    assert real["material_per_t"] == pytest.approx(3300, abs=1)

    _err("cost_buildup_rebar", {"quantity_kg": 0})


def test_audit_cost_per_area():
    """Arithmetic: cost/area. Hand: 1,000,000 / 5000 = 200.
    Zero or negative area must error — never a silent 0.00.
    """
    r = _ok("cost_per_area", {"total_cost": 1_000_000, "area": 5000})
    assert r["cost_per_area"] == pytest.approx(200.0, abs=0.01)

    env = _err("cost_per_area", {"total_cost": 1000, "area": 0})
    assert "area" in env["error"].lower()
    _err("cost_per_area", {"total_cost": 1000, "area": -5})


# ── 30–32 crane ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rates,cur,per,total,mob,demob,grand", [
    ((18000, 3500, 5600), "SAR", 27100, 650400, 18000, 13500, 681900),
    ((10000, 2000, 3000), "USD", 15000, 360000, 10000, 7500, 377500),
])
def test_audit_crane_cost_estimate(rates, cur, per, total, mob, demob, grand):
    """Caller-supplied project rates (no built-in table, no currency).
    Hand: per = dry + op + riggers; total = per x 2 cranes x 12 months;
          mob = 2 x dry x 0.5; demob = 0.75 x mob.
    """
    dry_r, op_r, rig_r = rates
    base = {"num_cranes": 2, "crane_capacity_tons": 50, "duration_months": 12,
            "dry_hire_rate_per_month": dry_r, "currency": cur}
    r = _ok("crane_cost_estimate",
            {**base, "operator_rate_per_month": op_r, "riggers_rate_per_month": rig_r})
    assert r["dry_hire_per_month"] == dry_r
    assert r["per_crane_monthly"] == pytest.approx(per, abs=0.5)
    assert r["total_project_cost"] == pytest.approx(total, abs=1)
    assert r["mobilization"] == pytest.approx(mob, abs=1)
    assert r["demobilization"] == pytest.approx(demob, abs=1)
    assert r["grand_total"] == pytest.approx(grand, abs=1)
    assert r["currency"] == cur
    assert not any(k.endswith("_sar") or "_sar_" in k for k in r)

    dry = _ok("crane_cost_estimate",
              {**base, "include_operator": False, "include_riggers": False})
    assert dry["per_crane_monthly"] == pytest.approx(dry_r, abs=0.5)

    _err("crane_cost_estimate", {**base, "num_cranes": -1})


@pytest.mark.parametrize("params", [
    {"num_cranes": 2, "crane_capacity_tons": 50, "duration_months": 12},
    {"num_cranes": 1, "crane_capacity_tons": 300, "duration_months": 3},
])
def test_crane_cost_without_a_rate_asks_for_it(params):
    """No silent built-in rate: the dry hire rate is a required input."""
    out = run_calculation("crane_cost_estimate", params)
    assert out["status"] == "error"
    assert "dry_hire_rate_per_month" in str(out)


@pytest.mark.parametrize("missing", ["operator_rate_per_month", "riggers_rate_per_month"])
def test_crane_cost_included_crew_without_a_rate_asks_for_it(missing):
    params = {"num_cranes": 2, "crane_capacity_tons": 50, "duration_months": 12,
              "dry_hire_rate_per_month": 18000,
              "operator_rate_per_month": 3500, "riggers_rate_per_month": 5600}
    params.pop(missing)
    out = run_calculation("crane_cost_estimate", params)
    assert out["status"] == "error"
    assert missing in str(out)


def test_audit_crane_lift_capacity():
    """Lift plan: net = chart − deductions; util = load/net; pass ≤ 85%.
    Hand: 50 − 3 = 47 t; 38/47 = 80.851% → pass.
    45/47 = 95.74% → fail. Deductions ≥ chart must error.
    """
    r = _ok("crane_lift_capacity",
            {"chart_capacity_t": 50, "deductions_t": 3, "load_t": 38})
    assert r["net_capacity_t"] == pytest.approx(47.0, abs=0.01)
    assert r["utilization_percent"] == pytest.approx(80.85, abs=0.1)
    assert r["passed"] is True

    fail = _ok("crane_lift_capacity",
               {"chart_capacity_t": 50, "deductions_t": 3, "load_t": 45})
    assert fail["passed"] is False

    _err("crane_lift_capacity",
         {"chart_capacity_t": 50, "deductions_t": 50, "load_t": 10})


@pytest.mark.parametrize("rate,cur", [(18000, "SAR"), (9500, "AED")])
def test_audit_crane_planning(rate, cur):
    """Documented: lifts/h = 60/cycle; daily t = cap×util × lifts/h × h × shifts.
    Hand: 60/20=3 lifts/h; 50×0.65=32.5 t/lift; 3×9.5×2=57 lifts/d
          daily = 32.5×57 = 1,852.5 t; monthly = 1,852.5×26 = 48,165 t
          demand 1,000 → 1 crane. Demand 48,166 → 2 cranes.
    Cost only from the caller's rate.
    """
    r = _ok("crane_planning",
            {"total_lift_demand_tons": 1000, "crane_capacity_tons": 50,
             "dry_hire_rate_per_month": rate, "currency": cur})
    assert r["lifts_per_hour_per_crane"] == pytest.approx(3.0, abs=0.05)
    assert r["daily_tonnage_per_crane"] == pytest.approx(1852.5, abs=0.1)
    assert r["cranes_required"] == 1
    assert r["monthly_rate"] == rate
    assert r["total_monthly_cost"] == rate
    assert r["currency"] == cur

    two = _ok("crane_planning",
              {"total_lift_demand_tons": 48166, "crane_capacity_tons": 50,
               "dry_hire_rate_per_month": rate})
    assert two["cranes_required"] == 2
    assert two["total_monthly_cost"] == 2 * rate

    _err("crane_planning",
         {"total_lift_demand_tons": 1000, "crane_capacity_tons": 50,
          "cycle_time_minutes": 0})


@pytest.mark.parametrize("cap", [50, 300])
def test_crane_planning_without_a_rate_gives_count_and_no_invented_cost(cap):
    r = _ok("crane_planning", {"total_lift_demand_tons": 1000, "crane_capacity_tons": cap})
    assert r["cranes_required"] >= 1
    assert r["monthly_rate"] is None and r["total_monthly_cost"] is None
    assert any("rate" in n for n in r["notes"])


# ── 33–35 planning / earthwork / FIDIC ─────────────────────────────────────

def test_audit_critical_path_float():
    """CPM / PMBOK: TF = LS − ES (= LF − EF). TF ≤ 0 is critical,
    including negative float.
    Hand: LS=8, ES=5 → TF=3, not critical.
    LS=ES=5 → TF=0, critical. LS=2, ES=5 → TF=−3, critical.
    """
    r = _ok("critical_path_float",
            {"early_start": 5, "early_finish": 10, "late_start": 8, "late_finish": 13})
    assert r["total_float"] == pytest.approx(3.0, abs=0.001)
    assert r["is_critical"] is False

    zero = _ok("critical_path_float",
               {"early_start": 5, "early_finish": 10, "late_start": 5, "late_finish": 10})
    assert zero["total_float"] == pytest.approx(0.0, abs=0.001)
    assert zero["is_critical"] is True

    neg = _ok("critical_path_float",
              {"early_start": 5, "early_finish": 10, "late_start": 2, "late_finish": 7})
    assert neg["total_float"] == pytest.approx(-3.0, abs=0.001)
    assert neg["is_critical"] is True

    _err("critical_path_float",
         {"early_start": 10, "early_finish": 5, "late_start": 8, "late_finish": 13})


def test_audit_cut_fill_balance():
    """Mass balance: balance = cut − fill (bank). Loose haul = |bal|×(1+bulk).
    Hand: 5000 − 3500 = +1500 surplus; loose 1500×1.25 = 1875.
    Equal cut/fill → balanced, haul 0. Fill > cut → import_deficit.
    """
    r = _ok("cut_fill_balance", {"cut_volume_m3": 5000, "fill_volume_m3": 3500})
    assert r["balance_bank_m3"] == pytest.approx(1500.0, abs=0.01)
    assert r["status"] == "export_surplus"
    assert r["haul_loose_m3"] == pytest.approx(1875.0, abs=0.01)

    bal = _ok("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1000})
    assert bal["status"] == "balanced"
    assert bal["haul_loose_m3"] == pytest.approx(0.0, abs=1e-9)

    deficit = _ok("cut_fill_balance", {"cut_volume_m3": 1000, "fill_volume_m3": 1500})
    assert deficit["status"] == "import_deficit"
    assert deficit["balance_bank_m3"] == pytest.approx(-500.0, abs=0.01)

    _err("cut_fill_balance", {"cut_volume_m3": -1, "fill_volume_m3": 10})


def test_audit_delay_damages_daily():
    """FIDIC 8.8: daily = rate% × Accepted Contract Amount.
    Hand: 0.1% × 1,754,504,456.25 = 1,754,504.45625 → 1,754,504.46 / day.
    Zero rate → 0 / day (valid). Negative rate must error.
    """
    r = _ok("delay_damages_daily", {
        "rate_percent": 0.1, "contract_amount": 1_754_504_456.25, "currency": "SAR",
    })
    assert r["daily_amount"] == pytest.approx(1_754_504.46, abs=0.005)

    zero = _ok("delay_damages_daily",
               {"rate_percent": 0, "contract_amount": 1_000_000, "currency": "SAR"})
    assert zero["daily_amount"] == pytest.approx(0.0, abs=1e-9)

    _err("delay_damages_daily",
         {"rate_percent": -0.1, "contract_amount": 1_000_000})


# ── 36–38 dewatering / D-wall ──────────────────────────────────────────────

def test_audit_dewatering_uplift_check():
    """Hydrostatics: uplift = h × γw (T/m²). Counter = raft×γc + floors×t×γc.
    FOS = counter / uplift; stop if FOS ≥ 1.25.
    Hand: h=23, raft=2, 5 floors × 0.3, γc=2.5
          uplift=23; counter=5.0+3.75=8.75; FOS=0.380
          min floors: ceil((23×1.25 − 5.0)/0.75) = ceil(31.667) = 32.
    Zero water: no uplift → FOS = ∞, can_stop True. Silent FOS=0 was wrong.
    """
    r = _ok("dewatering_uplift_check",
            {"water_depth": 23, "raft_thickness": 2.0, "floor_count": 5})
    assert r["uplift_force_t_m2"] == pytest.approx(23.0, abs=0.001)
    assert r["counter_weight_t_m2"] == pytest.approx(8.75, abs=0.001)
    assert r["fos"] == pytest.approx(0.380, abs=0.001)
    assert r["can_stop"] is False
    assert r["needs_tension_piles"] is True
    assert r["min_floors_for_stop"] == 32

    dry = _ok("dewatering_uplift_check",
              {"water_depth": 0, "raft_thickness": 2.0, "floor_count": 5})
    assert dry["uplift_force_t_m2"] == pytest.approx(0.0, abs=1e-9)
    assert dry["fos"] == float("inf") or dry["fos"] > 1e6
    assert dry["can_stop"] is True
    assert dry["needs_tension_piles"] is False

    _err("dewatering_uplift_check",
         {"water_depth": -1, "raft_thickness": 2.0, "floor_count": 5})


def test_audit_dewatering_well_point_spacing():
    """Practice table + 5 m suction/stage. k=2e-3 m/s → coarse sand, 1.5 m.
    Drawdown 12 m → ceil(12/5)=3 stages. k=5e-5 → fine sand, 0.9 m.
    """
    r = _ok("dewatering_well_point_spacing",
            {"soil_permeability_m_s": 2e-3, "required_drawdown_m": 12})
    assert r["soil_type"] == "coarse sand/gravel"
    assert r["well_point_spacing_m"] == pytest.approx(1.5, abs=1e-9)
    assert r["stages_needed"] == 3

    fine = _ok("dewatering_well_point_spacing",
               {"soil_permeability_m_s": 5e-5, "required_drawdown_m": 4})
    assert fine["soil_type"] == "fine sand"
    assert fine["well_point_spacing_m"] == pytest.approx(0.9, abs=1e-9)
    assert fine["stages_needed"] == 1

    _err("dewatering_well_point_spacing",
         {"soil_permeability_m_s": 0, "required_drawdown_m": 12})


def test_audit_diaphragm_wall_panel_volume():
    """Geometry + 10% tremie waste: V = L × t × D.
    Hand: 6×0.8×20 = 96 m³/panel; ×3 = 288; ×1.10 = 316.8.
    """
    r = _ok("diaphragm_wall_panel_volume",
            {"panel_length": 6, "wall_thickness": 0.8,
             "excavation_depth": 20, "panel_count": 3})
    assert r["volume_per_panel_m3"] == pytest.approx(96.0, abs=0.01)
    assert r["total_volume_m3"] == pytest.approx(288.0, abs=0.01)
    assert r["volume_with_waste_m3"] == pytest.approx(316.8, abs=0.01)

    _err("diaphragm_wall_panel_volume",
         {"panel_length": 0, "wall_thickness": 0.8, "excavation_depth": 20})


# ── 39–42 MEP / tender / excavation / fall arrest ──────────────────────────

def test_audit_electrical_installation_sequence():
    """Documented productivity (m²/day), sequential sum of ceil(area/rate).
    Hand: 1000×3 = 3000 m²
          ceil(3000/50)=60, /75=40, /100=30, /100=30, /50=60, /75=40, /30=100
          total = 360 days.
    Rates are unsourced rules of thumb — arithmetic of the documented table.
    """
    r = _ok("electrical_installation_sequence",
            {"floor_area_m2": 1000, "num_floors": 3, **ELECTRICAL_STAGE_RATES})
    assert r["total_area_m2"] == pytest.approx(3000, abs=1e-9)
    assert r["stages"]["1st_fix_conduit_boxes"] == 60
    assert r["stages"]["commissioning"] == 100
    assert r["total_days"] == 360

    one = _ok("electrical_installation_sequence",
              {"floor_area_m2": 50, "num_floors": 1, **ELECTRICAL_STAGE_RATES})
    assert one["stages"]["1st_fix_conduit_boxes"] == 1
    assert one["total_days"] == 1 + 1 + 1 + 1 + 1 + 1 + 2  # 50/30 → 2

    _err("electrical_installation_sequence",
         {"floor_area_m2": 0, "num_floors": 3})


def test_audit_evaluate_tender():
    """Tender analysis weighted score: 0.45 tech + 0.45 comm + 0.07 HSE + 0.03 local.
    Hand:
      A: 85×0.45 + 75×0.45 + 90×0.07 + 60×0.03 = 80.10
      B: 78×0.45 + 82×0.45 + 85×0.07 + 70×0.03 = 80.05
      C: 70×0.45 + 88×0.45 + 80×0.07 + 75×0.03 = 78.95
    A ranks 1st. Empty tenderer list must error.
    """
    r = _ok("evaluate_tender", {"tenderers": [
        {"name": "Bidder A", "technical_score": 85, "commercial_score": 75,
         "hse_score": 90, "local_content_score": 60},
        {"name": "Bidder B", "technical_score": 78, "commercial_score": 82,
         "hse_score": 85, "local_content_score": 70},
        {"name": "Bidder C", "technical_score": 70, "commercial_score": 88,
         "hse_score": 80, "local_content_score": 75},
    ]})
    ranked = r["ranked_tenderers"]
    assert ranked[0]["name"] == "Bidder A"
    assert ranked[0]["weighted_total"] == pytest.approx(80.10, abs=0.01)
    assert ranked[1]["weighted_total"] == pytest.approx(80.05, abs=0.01)
    assert ranked[2]["weighted_total"] == pytest.approx(78.95, abs=0.01)
    assert r["recommended"]["name"] == "Bidder A"

    _err("evaluate_tender", {"tenderers": []})


def test_audit_excavation_volume():
    """Geometry + bulking: bank = L×W×D; loose = bank × (1+0.25).
    Hand: 20×10×3 = 600 bank; ×1.25 = 750 loose.
    Zero depth → 0 bank (valid empty trench). Negative dim must error.
    """
    r = _ok("excavation_volume",
            {"length_m": 20, "width_m": 10, "depth_m": 3})
    assert r["bank_volume_m3"] == pytest.approx(600.0, abs=0.01)
    assert r["loose_volume_m3"] == pytest.approx(750.0, abs=0.01)

    z = _ok("excavation_volume",
            {"length_m": 20, "width_m": 10, "depth_m": 0})
    assert z["bank_volume_m3"] == pytest.approx(0.0, abs=1e-9)

    _err("excavation_volume",
         {"length_m": 20, "width_m": 10, "depth_m": -1})


def test_audit_fall_arrest_force():
    """Energy balance (OSHA 1926.502 family): F = mg (1 + H/d), cap 8 kN.
    Hand: m=100 kg, g=9.81, H=1.8, d=1.0
          F = 981 × 2.8 = 2,746.8 N = 2.75 kN (< 8 kN).
    H=6 m, d=0.5: 981 × (1+12) = 12,753 N = 12.75 kN, exceeds.
    d=0 must error (not silent inf).
    """
    r = _ok("fall_arrest_force",
            {"worker_mass_kg": 100, "free_fall_m": 1.8, "deceleration_distance_m": 1.0})
    assert r["max_arrest_force_kn"] == pytest.approx(2.75, abs=0.02)
    assert r["within_osha_limit"] is True

    hard = _ok("fall_arrest_force",
               {"worker_mass_kg": 100, "free_fall_m": 6.0, "deceleration_distance_m": 0.5})
    assert hard["max_arrest_force_kn"] == pytest.approx(12.75, abs=0.05)
    assert hard["within_osha_limit"] is False

    _err("fall_arrest_force",
         {"worker_mass_kg": 100, "free_fall_m": 1.8, "deceleration_distance_m": 0})
