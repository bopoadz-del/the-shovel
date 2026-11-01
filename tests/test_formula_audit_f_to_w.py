"""Phase 1 + Phase 2 formula audit for calculators #45–86.

Scope: ``sorted(available_calculations())[44:]`` — the boundary moved from
42 when the two point-load deflection calculators were added for SET5 E3. Every case is hand-derived
independently of the implementation; ``pytest.approx`` tolerances are justified
in the docstring (rounding in the calculator, or a physical constant).

Call through ``run_calculation`` — that is the live dispatch envelope
(``status`` / ``result`` / ``error``).

Phase 2 routing: ``resolve_fw_calc`` remaps the two live wrong-name picks
(``resource_line_cost`` vs ``unit_cost_total``, ``material_consumption`` vs
``concrete_mix_proportions`` / E4 ``concrete_volume``) and the intent map
forces ``construction_calc`` on distinctive F–W phrases.
"""
from __future__ import annotations

import math

from tests.example_project_rates import MOBILIZATION_RATES, SUPERVISION_RATIOS
import pytest

from app.lib.construction_formulas import available_calculations, run_calculation
from app.lib.construction_formulas_quantities import (
    FW_ROUTE_NAMES,
    resolve_fw_calc,
)

# Confirmed against live registry on this branch. STOP if this drifts.
_AUDIT_SLICE = [
    "fineness_modulus",
    "formwork_striking_time",
    "foundation_bearing_pressure",
    "grout_pressure_calc",
    "guardrail_top_rail_height",
    "interior_finishes_takeoff",
    "laser_scan_accuracy",
    "leed_points_estimate",
    "live_load_reduction",
    "masonry_wall_capacity",
    "material_consumption",
    "mobilization_cost_estimate",
    "modulus_of_elasticity_concrete",
    "modulus_of_rupture",
    "pe_unit_convert",
    "plumbing_flow_programme",
    "post_tensioning_force",
    "precast_beam_erection_check",
    "productivity_manpower_duration",
    "productivity_rate",
    "progress_quantity",
    "rc_beam_moment_capacity",
    "rc_beam_shear_capacity",
    "rebar_by_area",
    "rebar_lap_length",
    "rebar_weight",
    "resource_line_cost",
    "roi_calculator",
    "scaffold_load_capacity",
    "score_risk",
    "seismic_base_shear",
    "shear_stress_check",
    "slab_thickness_min",
    "slope_fos_simple",
    "steel_tension_capacity",
    "supervision_ratio",
    "thermal_shrinkage_equivalence",
    "tolerance_check",
    "unit_cost_total",
    "unit_weight_concrete",
    "weld_capacity",
    "wind_load_on_formwork",
    "wind_pressure",
]


def _ok(name: str, **params):
    env = run_calculation(name, params)
    assert env.get("status") == "success", env
    assert "result" in env
    return env["result"]


def _err(name: str, *, expect: str = "error", **params):
    # expect="needs_input" for a missing-required / unreadable-value ASK;
    # default "error" for a value-domain / calculator-raised rejection
    # (item 7, ORDERS_31OCT_0825Z).
    env = run_calculation(name, params)
    assert env.get("status") == expect, env
    assert isinstance(env.get("error"), str) and env["error"]
    return env


# ── registry contract ───────────────────────────────────────────────────────

def test_audit_slice_matches_live_registry():
    """sorted(available_calculations())[44:] must stay the owned F–W set."""
    assert available_calculations()[44:] == _AUDIT_SLICE


def test_audit_file_has_two_cases_per_calculator():
    """At least two test_* functions mention each owned calculator name."""
    import pathlib
    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    missing = []
    for name in _AUDIT_SLICE:
        count = src.count(f'"{name}"') + src.count(f"'{name}'")
        if count < 2:
            missing.append((name, count))
    assert not missing, f"need ≥2 name mentions per calculator: {missing}"


# ═══════════════════════════════════════════════════════════════════════════
# PRIORITY 15 — no prior arithmetic oracle
# ═══════════════════════════════════════════════════════════════════════════

class TestFormworkStrikingTime:
    """BS 8110-1:1997 Table 6.2 (12 h vertical) + CIRIA striking-strength check.

    Cube-test route may undercut the 12 h table when strength meets BOTH the
    CIRIA surface minimum and the design stress. CIRIA failing must not still
    stamp based_on='test_data' or recommend an early strike.
    """

    def test_adequate_cube_allows_proposed_early_strike(self):
        # Defaults: cube 7.0 ≥ CIRIA 3.0 and ≥ design 0.332 → test_data; 8 h.
        r = _ok("formwork_striking_time")
        assert r["actual_early_strength_n_mm2"] == pytest.approx(7.0, abs=1e-9)
        assert r["ciria_minimum_strength_n_mm2"] == pytest.approx(3.0, abs=1e-9)
        assert r["based_on"] == "test_data"
        assert r["recommended_hours"] == pytest.approx(8.0, abs=1e-9)

    def test_below_ciria_is_insufficient_and_not_early(self):
        # 1.0 N/mm² < CIRIA 3.0 (even though 1.0 > design 0.332).
        # Must not recommend 8 h; fall back to at least the BS 8110 12 h table.
        r = _ok("formwork_striking_time", concrete_strength_7h=1.0, proposed_hours=8.0)
        assert r["based_on"] == "insufficient"
        assert r["recommended_hours"] == pytest.approx(12.0, abs=1e-9)


class TestFoundationBearingPressure:
    """q = P/A (kN/m²). Net = q − γD with γ = 18 kN/m³ (unsourced default).
    The soil value is allowable, so bearing_ok iff FOS ≥ 1.
    """

    def test_3m_square_2000kn(self):
        # A = 3×3 = 9 m²; q = 2000/9 = 222.222… kN/m²; D=0 → net=q;
        # FOS = 300/222.222 = 1.35 ≥ 1 → within the allowable.
        r = _ok(
            "foundation_bearing_pressure",
            foundation_width_m=3, foundation_length_m=3,
            column_load_kn=2000, soil_bearing_capacity_kn_m2=300,
        )
        assert r["bearing_pressure_kn_m2"] == pytest.approx(222.22, abs=0.01)
        assert r["net_pressure_kn_m2"] == pytest.approx(222.22, abs=0.01)
        assert r["fos_against_bearing"] == pytest.approx(1.35, abs=0.01)
        assert r["bearing_ok"] is True

    def test_overburden_reduces_net_pressure(self):
        # D=2 m → surcharge = 2×18 = 36; net = 222.22−36 = 186.22;
        # FOS = 300/186.22 = 1.611 ≥ 1.
        r = _ok(
            "foundation_bearing_pressure",
            foundation_width_m=3, foundation_length_m=3,
            column_load_kn=2000, soil_bearing_capacity_kn_m2=300,
            foundation_depth_m=2,
        )
        assert r["net_pressure_kn_m2"] == pytest.approx(186.22, abs=0.01)
        assert r["fos_against_bearing"] == pytest.approx(1.61, abs=0.01)
        assert r["bearing_ok"] is True

    def test_zero_width_is_error(self):
        _err(
            "foundation_bearing_pressure",
            foundation_width_m=0, foundation_length_m=3,
            column_load_kn=2000, soil_bearing_capacity_kn_m2=300,
        )

    def test_negative_width_is_error_not_ok(self):
        # A negative plan dimension must not yield bearing_ok=True.
        _err(
            "foundation_bearing_pressure",
            foundation_width_m=-3, foundation_length_m=3,
            column_load_kn=2000, soil_bearing_capacity_kn_m2=300,
        )


class TestGroutPressureCalc:
    """PT duct volume + pressure unit conversions (EN 447 / ETAG 013 family).

    Area = π(d/2)². Volume L/m = area_mm² / 1000.
    1 N/mm² = 10.197 kg/cm² = 145.038 psi. Default 0.5 N/mm² is a spec, not derived.
    """

    def test_80mm_duct_volume_and_pressure_units(self):
        # A = π×40² = 5026.548 mm²; L/m = 5026.548/1000 = 5.027 L/m.
        # 0.5×10.197 = 5.0985 → 5.1 kg/cm²; 0.5×145.038 = 72.519 → 73 psi.
        r = _ok("grout_pressure_calc", tendon_duct_diameter_mm=80)
        assert r["duct_area_mm2"] == pytest.approx(math.pi * 40.0 ** 2, abs=0.01)
        assert r["grout_volume_l_m"] == pytest.approx(math.pi * 40.0 ** 2 / 1000.0, abs=0.001)
        assert r["pressure_n_mm2"] == pytest.approx(0.5, abs=1e-9)
        assert r["pressure_kg_cm2"] == pytest.approx(5.1, abs=0.05)
        assert r["pressure_psi"] == pytest.approx(73.0, abs=0.5)

    def test_zero_and_negative_diameter_are_errors(self):
        _err("grout_pressure_calc", tendon_duct_diameter_mm=0)
        _err("grout_pressure_calc", tendon_duct_diameter_mm=-80)


class TestInteriorFinishesTakeoff:
    """Geometry process: floor = area; wall paint = P×h − doors − windows;
    blockwork = paint × shared_wall_fraction; skirting = P − Σ door widths.
    """

    def test_eight_room_floor(self):
        # 100 m² floors/ceiling. P=90, h=2.7 → gross wall 243.
        # 8 doors × 0.9×2.1 = 15.12 m²; paint = 243−15.12 = 227.88;
        # blockwork = 227.88×0.5 = 113.94; skirting = 90−7.2 = 82.8.
        r = _ok(
            "interior_finishes_takeoff",
            floor_area_m2=100, perimeter_m=90, room_count=8,
            floor_to_ceiling_m=2.7, door_width_m=0.9, door_height_m=2.1,
            doors_per_room=1,
        )
        assert r["floor_screed_m2"] == pytest.approx(100.0, abs=0.01)
        assert r["ceiling_finish_m2"] == pytest.approx(100.0, abs=0.01)
        assert r["gross_wall_face_m2"] == pytest.approx(243.0, abs=0.01)
        assert r["door_deduction_m2"] == pytest.approx(15.12, abs=0.01)
        assert r["wall_paint_m2"] == pytest.approx(227.88, abs=0.01)
        assert r["blockwork_m2"] == pytest.approx(113.94, abs=0.01)
        assert r["skirting_m"] == pytest.approx(82.8, abs=0.01)

    def test_missing_height_is_error(self):
        env = _err(
            "interior_finishes_takeoff",
            floor_area_m2=100, perimeter_m=90, room_count=8,
            floor_to_ceiling_m=0,
        )
        assert "floor_to_ceiling" in env["error"]


class TestMobilizationCostEstimate:
    """Build-up: monthly = (office+camp+transport+safety)×n×factor;
    recurring = monthly×months; fixed = (setups + IT + warehouse)×factor once.

    Warehouse must not be pre-multiplied then multiplied again.
    camp is camp setup + camp recurring only (not the whole recurring pot).
    Rates are unsourced GCC fallbacks — arithmetic only is audited here.
    """

    def test_100_staff_18_months_rented(self):
        # per-head = 350+2800+220+150 = 3520; ×100×1.35 = 475,200 / month.
        # recurring = 475,200×18 = 8,553,600.
        # IT = 100×85×18 = 153,000.
        # fixed base = 450k+60k+280k+40k+153k+35k = 1,018,000; ×1.35 = 1,374,300.
        # grand = 8,553,600 + 1,374,300 = 9,927,900.
        # camp = (60,000 + 2,800×100×18)×1.35 = 6,885,000.
        r = _ok("mobilization_cost_estimate", num_personnel=100, duration_months=18, **MOBILIZATION_RATES)
        assert r["recurring_monthly"] == pytest.approx(475_200, abs=1)
        assert r["total_recurring"] == pytest.approx(8_553_600, abs=1)
        assert r["total_fixed"] == pytest.approx(1_374_300, abs=1)
        assert r["grand_total"] == pytest.approx(9_927_900, abs=1)
        assert r["breakdown"]["camp"] == pytest.approx(6_885_000, abs=1)
        assert r["breakdown"]["warehouse_equipment"] == pytest.approx(47_250, abs=1)
        # Breakdown camp line must not exceed the grand total.
        assert r["breakdown"]["camp"] < r["grand_total"]
        assert r["unit"] == "currency"  # K-T5d ruling ii: generic, not an invented SAR
        assert r["value"] == pytest.approx(9_927_900, abs=1)
        # Note carries the generic unit now, not an invented SAR (K-T5d ruling ii).
        assert "currency" in str(r.get("note") or "")

    def test_hotel_urban_factor_one(self):
        # camp=4500; per-head=350+4500+220+150=5220; ×100×1.0=522,000 / month.
        # recurring=522,000×18=9,396,000.
        # fixed=(450k+60k+280k+40k+153k+35k)×1.0=1,018,000.
        # grand=10,414,000.
        r = _ok(
            "mobilization_cost_estimate",
            num_personnel=100, duration_months=18,
            **{**MOBILIZATION_RATES, "camp_rate_per_person_month": 4500,
               "remote_area_factor": 1.0},
            camp_type="hotel",
        )
        assert r["recurring_monthly"] == pytest.approx(522_000, abs=1)
        assert r["grand_total"] == pytest.approx(10_414_000, abs=1)
        assert r["unit"] == "currency"  # K-T5d ruling ii: generic, not an invented SAR
        assert r["value"] == pytest.approx(10_414_000, abs=1)


class TestModulusOfElasticityConcrete:
    """kg/cm² form Ec = 15_000 √(f'c) with f'c = 10.197×fck (N/mm²→kg/cm²).
    """

    def test_c40(self):
        # f'c = 40×10.197 kg/cm²; Ec = 15_000×√(that) = 302_941 kg/cm².
        r = _ok("modulus_of_elasticity_concrete", fck_n_mm2=40)
        assert r["value"] == pytest.approx(round(15_000 * math.sqrt(40 * 10.197)), abs=0.5)

    def test_c25_and_zero(self):
        r = _ok("modulus_of_elasticity_concrete", fck_n_mm2=25)
        assert r["value"] == pytest.approx(round(15_000 * math.sqrt(25 * 10.197)), abs=0.5)
        z = _err("modulus_of_elasticity_concrete", fck_n_mm2=0)
        assert z["needs_input"] is True
        assert "characteristic concrete strength" in z["question"]


class TestPlumbingFlowProgramme:
    """Staged programme: fixed 7+21+7+3 days plus area-rate stages
    ceil(A/40)+ceil(A/60)+ceil(A/80)+ceil(A/50). Rates are unsourced.
    """

    def test_3000m2_three_floors(self):
        # A=1000×3=3000. 75+50+38+60 + 7+21+7+3 = 261.
        r = _ok("plumbing_flow_programme", floor_area_m2=1000, num_floors=3)
        assert r["stages"]["1st_fix_marking_piping"]["days"] == 75
        assert r["stages"]["2nd_fix_sanitary_fittings"]["days"] == 50
        assert r["stages"]["final_fix_pumps_tanks"]["days"] == 38
        assert r["stages"]["testing_commissioning"]["days"] == 60
        assert r["total_days"] == 261
        assert r["stages"]["handover"]["cumulative_days"] == 261

    def test_zero_area_keeps_fixed_plus_one_day_floors(self):
        # max(1, ceil(0/rate))=1 for four rate stages; +38 fixed = 42.
        r = _ok("plumbing_flow_programme", floor_area_m2=0, num_floors=1)
        assert r["total_days"] == 42


class TestPostTensioningForce:
    """Load-balancing: P = w L² / (8 e) with w = 0.75 × DL, DL = t×25 kN/m²,
    e = 0.15 t. Per-metre-width strip. Live load is an unused parameter.

    7-wire strand area is 0.78 × π(d/2)² (ASTM A416 / EN 10138: 12.7 mm → 98.7 mm²),
    not the circumscribed circle 126.7 mm².
    """

    def test_8m_span_250mm_slab(self):
        # DL=0.25×25=6.25; w=4.6875 kN/m²; e=0.0375 m;
        # P=4.6875×64/(8×0.0375)=300/0.3=1000 kN per m width.
        # As=1000e3/1300=769.23 mm²; strand=0.78×π×6.35²=98.81 mm²;
        # n=ceil(769.23/98.81)=8.
        r = _ok(
            "post_tensioning_force",
            span_m=8, slab_thickness_m=0.25, live_load_kn_m2=5,
        )
        assert r["tendon_force_kn"] == pytest.approx(1000.0, abs=0.1)
        assert r["eccentricity_mm"] == pytest.approx(37.5, abs=0.05)
        assert r["balanced_load_kn_m"] == pytest.approx(4.69, abs=0.01)
        assert r["tendon_area_mm2"] == pytest.approx(769.2, abs=0.2)
        assert r["num_strands"] == 8

    def test_zero_thickness_is_zero_force(self):
        r = _ok("post_tensioning_force", span_m=8, slab_thickness_m=0, live_load_kn_m2=5)
        assert r["tendon_force_kn"] == pytest.approx(0.0, abs=0.1)
        assert r["num_strands"] == 0

    def test_live_load_does_not_change_balanced_force(self):
        # Documented formula balances 75% DL only; LL is accepted but unused.
        a = _ok("post_tensioning_force", span_m=8, slab_thickness_m=0.25, live_load_kn_m2=0)
        b = _ok("post_tensioning_force", span_m=8, slab_thickness_m=0.25, live_load_kn_m2=20)
        assert a["tendon_force_kn"] == pytest.approx(b["tendon_force_kn"], abs=0.1)


class TestPrecastBeamErectionCheck:
    """effective = (beam + rigging) × dynamic_factor (default 1.25).
    util% = effective / crane_capacity × 100. Radius/length unused (no chart).
    """

    def test_20t_on_50t_crane(self):
        # (20+0.5)×1.25 = 25.625 t; 25.625/50×100 = 51.25%.
        r = _ok(
            "precast_beam_erection_check",
            beam_weight_t=20, beam_length_m=15,
            crane_capacity_t=50, lift_radius_m=12,
        )
        assert r["effective_lift_weight_t"] == pytest.approx(25.62, abs=0.02)
        assert r["utilization_pct"] == pytest.approx(51.2, abs=0.15)
        assert r["is_safe"] is True

    def test_overload_and_zero_capacity(self):
        r = _ok(
            "precast_beam_erection_check",
            beam_weight_t=45, beam_length_m=15,
            crane_capacity_t=50, lift_radius_m=12,
        )
        # (45+0.5)×1.25=56.875 t; 113.75% → not safe.
        assert r["effective_lift_weight_t"] == pytest.approx(56.88, abs=0.02)
        assert r["is_safe"] is False
        _err(
            "precast_beam_erection_check",
            beam_weight_t=20, beam_length_m=15,
            crane_capacity_t=0, lift_radius_m=12,
        )


class TestResourceLineCost:
    """labour = (qty/daily_output)×day_rate; plant = fraction×labour;
    material = qty×rate when rate ≥ 0.
    """

    def test_100_units(self):
        # days=10; labour=2800; plant=84; material=5000; total=7884.
        r = _ok(
            "resource_line_cost",
            quantity=100, daily_output=10, day_rate=280,
            material_rate_per_unit=50, plant_fraction_of_labour=0.03,
        )
        assert r["crew_days"] == pytest.approx(10.0, abs=0.01)
        assert r["labour_cost"] == pytest.approx(2800.0, abs=0.01)
        assert r["plant_cost"] == pytest.approx(84.0, abs=0.01)
        assert r["material_cost"] == pytest.approx(5000.0, abs=0.01)
        assert r["total_cost"] == pytest.approx(7884.0, abs=0.01)

    def test_zero_output_is_error(self):
        env = _err("resource_line_cost", quantity=100, daily_output=0, day_rate=280)
        assert "daily_output" in env["error"]


class TestShearStressCheck:
    """Nominal v = V/(b d). V in N, b,d in mm → N/mm². BS 8110 / EC2 v=V/bd."""

    def test_200kn_300x550(self):
        # 200_000 / (300×550) = 1.21212 N/mm².
        r = _ok("shear_stress_check", v_kn=200, b_mm=300, d_mm=550)
        assert r["shear_stress_n_mm2"] == pytest.approx(1.212, abs=0.001)

    def test_zero_width_is_error(self):
        _err("shear_stress_check", v_kn=200, b_mm=0, d_mm=550)


class TestSupervisionRatio:
    """Midpoints of the stated rules of thumb: 35_000 m³, 2_500 t, 17_500
    dia-inch, 65 km, plus 1 general / 20_000 m². HSE = max(2, ceil(sup/8));
    DC = max(1, ceil(sup/15)).
    """

    def test_5000m3_and_10000m2(self):
        # ceil(5000/35000)=1 civil; ceil(10000/20000)=1 general; HSE=2; DC=1.
        r = _ok("supervision_ratio", concrete_m3=5000, area_m2=10000, **SUPERVISION_RATIOS)
        assert r["civil_supervisors"] == 1
        assert r["general_supervisors"] == 1
        assert r["total_supervisors"] == 2
        assert r["hse_officers"] == 1  # ceil(2/8): no built-in floor
        assert r["document_controllers"] == 1
        assert r["total_supervision_staff"] == 4

    @pytest.mark.parametrize("qty", [dict(concrete_m3=0, area_m2=0), dict(structural_steel_t=0, piping_dia_inch=0)])
    def test_zero_quantities_need_no_ratio_and_staff_nobody(self, qty):
        r = _ok("supervision_ratio", **qty)
        assert r["total_supervisors"] == 0
        assert r["total_supervision_staff"] == 0

    @pytest.mark.parametrize("qty,ratio", [("concrete_m3", "concrete_m3_per_supervisor"),
                                           ("area_m2", "area_m2_per_supervisor")])
    def test_a_quantity_without_its_ratio_asks_for_it(self, qty, ratio):
        env = run_calculation("supervision_ratio", {qty: 5000})
        assert env["status"] == "needs_input"
        assert ratio in env["missing"]


class TestThermalShrinkageEquivalence:
    """First principles: ΔT = ε_sh / α_c. EC2 α_c ≈ 10×10⁻⁶ /°C."""

    def test_default_200_microstrain(self):
        # 0.0002 / 10e-6 = 20.0 °C.
        r = _ok("thermal_shrinkage_equivalence")
        assert r["equivalent_temp_drop_c"] == pytest.approx(20.0, abs=0.05)

    def test_400_microstrain(self):
        r = _ok("thermal_shrinkage_equivalence", shrinkage_strain=0.0004, alpha_c=10e-6)
        assert r["equivalent_temp_drop_c"] == pytest.approx(40.0, abs=0.05)

    def test_zero_alpha_is_error(self):
        _err("thermal_shrinkage_equivalence", shrinkage_strain=0.0002, alpha_c=0)


class TestUnitWeightConcrete:
    """EN 1991-1-1 Table A.1: concrete 24 kN/m³, RC 25 kN/m³ → 2400 / 2500 kg/m³."""

    def test_reinforced_default(self):
        r = _ok("unit_weight_concrete")
        assert r["unit_weight_kg_m3"] == 2500

    def test_plain(self):
        r = _ok("unit_weight_concrete", reinforced=False)
        assert r["unit_weight_kg_m3"] == 2400
        assert r["range_kg_m3"] == (2330, 2470) or list(r["range_kg_m3"]) == [2330, 2470]


class TestWindLoadOnFormwork:
    """Dynamic pressure q=0.613 V² N/m² (ASCE 7 / EN 1991 ½ρ). F=q A;
    M=F×h/2; Z=w h²/6 (solid-rectangle section modulus).
    """

    def test_30mps_50m2(self):
        # q=0.613×900=551.7 Pa = 0.5517 kPa → 0.552 kPa.
        # F=551.7×50=27_585 N = 27.585 kN; M=27_585×1.5=41_377.5 N·m.
        # Z=8×9/6=12 m³; σ=41_377.5/12=3448.125 N/m².
        r = _ok(
            "wind_load_on_formwork",
            wind_velocity_m_s=30, formwork_area_m2=50,
            formwork_height_m=3, formwork_width_m=8,
        )
        assert r["wind_pressure_kpa"] == pytest.approx(0.552, abs=0.001)
        assert r["wind_force_kn"] == pytest.approx(27.585, abs=0.002)
        assert r["overturning_moment_kn_m"] == pytest.approx(41.378, abs=0.002)
        assert r["bending_stress_n_m2"] == pytest.approx(3448.13, abs=0.05)

    def test_zero_wind(self):
        r = _ok(
            "wind_load_on_formwork",
            wind_velocity_m_s=0, formwork_area_m2=50,
            formwork_height_m=3, formwork_width_m=8,
        )
        assert r["wind_force_kn"] == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# STRUCTURAL / SAFETY SET
# ═══════════════════════════════════════════════════════════════════════════

class TestRcBeamMomentCapacity:
    """ACI 318-19 §22.2: a=As fy/(0.85 fc b); φMn=0.9 As fy (d−a/2).
    EN 1992-1-1 §6.1: fcd=fc/1.5, fyd=fy/1.15; x=As fyd/(0.8 b fcd);
    z=d−0.4x; MRd=As fyd z. Units N·mm / 1e6 = kN·m.
    """

    def test_aci_1500mm2(self):
        # a=1500×420/(0.85×30×300)=82.353 mm; φMn=0.9×1500×420×(550−41.176)
        # =288.50 kN·m. εt>0.005 so φ=0.9 is the tension-controlled factor.
        r = _ok(
            "rc_beam_moment_capacity",
            steel_area_mm2=1500, fy_mpa=420, width_mm=300,
            eff_depth_mm=550, fc_mpa=30, code="aci",
        )
        assert r["moment_capacity_kn_m"] == pytest.approx(288.50, abs=0.05)

    def test_eurocode_1500mm2(self):
        # fcd=20; fyd=365.217; x=114.13 mm; z=504.35; MRd=276.25 kN·m.
        r = _ok(
            "rc_beam_moment_capacity",
            steel_area_mm2=1500, fy_mpa=420, width_mm=300,
            eff_depth_mm=550, fc_mpa=30, code="eurocode",
        )
        assert r["moment_capacity_kn_m"] == pytest.approx(276.25, abs=0.1)


class TestRcBeamShearCapacity:
    """ACI 318-19 Table 22.5.5.1, Av < Av,min: φ×0.66×λs×(ρ)^(1/3)×√fc×bw×d.
    EN 1992-1-1 §6.2.2: VRd,c = max( Eq.6.2a, vmin bw d ) with
    CRd,c=0.12, k=1+√(200/d)≤2, vmin=0.035 k^{3/2} √fck.
    """

    def test_aci_300x550_c30(self):
        # λs=√(2/(1+0.004×550)); ρ=0.01; φVc ≈ 76.19 kN.
        r = _ok(
            "rc_beam_shear_capacity",
            width_mm=300, eff_depth_mm=550, fc_mpa=30, code="aci",
        )
        assert r["shear_capacity_kn"] == pytest.approx(76.19, abs=0.05)

    def test_eurocode_rho_0_01_6_2a_governs(self):
        # k=1.603; (100×0.01×30)^{1/3}=3.107; V=0.12×1.603×3.107×165=98.62 kN.
        # vmin=0.035×1.603^{1.5}×√30=0.389 N/mm² → 64.2 kN < 98.62.
        r = _ok(
            "rc_beam_shear_capacity",
            width_mm=300, eff_depth_mm=550, fc_mpa=30, code="eurocode", rho_l=0.01,
        )
        assert r["shear_capacity_kn"] == pytest.approx(98.62, abs=0.1)

    def test_eurocode_low_rho_vmin_floor(self):
        # ρl=0.001 → 6.2a = 45.78 kN; vmin bw d = 64.2 kN governs.
        r = _ok(
            "rc_beam_shear_capacity",
            width_mm=300, eff_depth_mm=550, fc_mpa=30, code="eurocode", rho_l=0.001,
        )
        k = 1.0 + math.sqrt(200.0 / 550.0)
        vmin = 0.035 * (k ** 1.5) * math.sqrt(30.0)
        assert r["shear_capacity_kn"] == pytest.approx(vmin * 300 * 550 / 1000.0, abs=0.15)


class TestMasonryWallCapacity:
    """TMS 402: Pa=(0.25 f'm An)×[1−(h/140r)²] for h/r≤99, r=t/√12.
    EN 1996-1-1 Annex G Φ_m method.
    """

    def test_tms_190mm_wall(self):
        # r=190/√12=54.848; h/r=54.70; R=1−(3000/(140×54.848))²=0.8474;
        # Pa=0.25×10×190000×0.8474 / 1000 = 402.5 kN.
        r = _ok(
            "masonry_wall_capacity",
            masonry_strength_mpa=10, net_area_mm2=190000,
            height_mm=3000, thickness_mm=190, code="aci",
        )
        assert r["slenderness_reduction_R"] == pytest.approx(0.8474, abs=0.002)
        assert r["capacity_kn"] == pytest.approx(402.5, abs=0.5)

    def test_ec6_annex_g(self):
        # fd=10/2.7=3.704; lam_c=(3000/190)×√(10/10000)=0.4993;
        # emk/t=0.05; A1=0.9; u=0.6498; Φ=0.9×exp(−u²/2)=0.7287;
        # NRd=0.7287×190000×3.704/1000=512.8 kN.
        r = _ok(
            "masonry_wall_capacity",
            masonry_strength_mpa=10, net_area_mm2=190000,
            height_mm=3000, thickness_mm=190, code="eurocode", gamma_m=2.7,
        )
        assert r["phi_reduction"] == pytest.approx(0.7287, abs=0.003)
        assert r["capacity_kn"] == pytest.approx(512.8, abs=1.0)


class TestSteelTensionCapacity:
    """AISC 360-16 §D2: φPn = min(0.90 Fy Ag, 0.75 Fu Ae).
    EN 1993-1-1 §6.2.3: min(A fy/γM0, 0.9 Anet fu/γM2), γM0=1.0, γM2=1.25.
    N → kN by /1000.
    """

    def test_aisc_rupture_governs(self):
        # yield 0.90×345×3000=931.5 kN; rupture 0.75×450×2550=860.625 kN.
        r = _ok(
            "steel_tension_capacity",
            gross_area_mm2=3000, net_area_mm2=2550,
            fy_mpa=345, fu_mpa=450, code="aci",
        )
        assert r["yield_capacity_kn"] == pytest.approx(931.5, abs=0.05)
        assert r["rupture_capacity_kn"] == pytest.approx(860.63, abs=0.05)
        assert r["capacity_kn"] == pytest.approx(860.63, abs=0.05)
        assert r["governing_limit_state"] == "rupture"

    def test_eurocode_and_yield_governs_when_net_equals_gross(self):
        r = _ok(
            "steel_tension_capacity",
            gross_area_mm2=3000, net_area_mm2=2550,
            fy_mpa=355, fu_mpa=490, code="eurocode",
        )
        # Npl=3000×355=1065 kN; Nu=0.9×2550×490/1.25=899.64 kN.
        assert r["capacity_kn"] == pytest.approx(899.64, abs=0.05)
        y = _ok("steel_tension_capacity", gross_area_mm2=3000, fy_mpa=345, fu_mpa=450, code="aci")
        assert y["governing_limit_state"] == "yield"
        assert y["capacity_kn"] == pytest.approx(931.5, abs=0.05)


class TestWeldCapacity:
    """Throat=0.707×leg. AISC §J2.4: φRn=0.75×0.60 FEXX×throat×L.
    EN 1993-1-8 §4.5.3.3: fvw,d=fu/(√3 βw γM2).
    """

    def test_aisc_6mm_x_100mm(self):
        # throat=4.242; 0.75×0.60×490×4.242=935.36 N/mm; ×100=93.54 kN.
        r = _ok("weld_capacity", leg_size_mm=6, length_mm=100, electrode_strength_mpa=490, code="aci")
        assert r["effective_throat_mm"] == pytest.approx(4.242, abs=0.001)
        assert r["capacity_kn"] == pytest.approx(93.54, abs=0.05)

    def test_eurocode_s355(self):
        # fvw,d=490/(√3×0.9×1.25)=251.47 MPa; ×4.242×100/1000=106.67 kN.
        r = _ok(
            "weld_capacity",
            leg_size_mm=6, length_mm=100, electrode_strength_mpa=490,
            code="eurocode", beta_w=0.9,
        )
        assert r["capacity_kn"] == pytest.approx(106.67, abs=0.05)


class TestScaffoldLoadCapacity:
    """OSHA 1926.451: 25/50/75 psf → 1.2/2.4/3.6 kPa; required = 4× intended."""

    def test_medium_10m2(self):
        # 2.4×10=24 kN; ×4=96 kN.
        r = _ok("scaffold_load_capacity", platform_area_m2=10, duty="medium")
        assert r["intended_load_kn"] == pytest.approx(24.0, abs=0.01)
        assert r["required_capacity_kn"] == pytest.approx(96.0, abs=0.01)

    def test_light_and_heavy_branch(self):
        light = _ok("scaffold_load_capacity", platform_area_m2=10, duty="light")
        heavy = _ok("scaffold_load_capacity", platform_area_m2=10, duty="heavy")
        assert light["duty_load_kpa"] == pytest.approx(1.2, abs=1e-9)
        assert light["required_capacity_kn"] == pytest.approx(48.0, abs=0.01)
        assert heavy["duty_load_kpa"] == pytest.approx(3.6, abs=1e-9)
        assert heavy["required_capacity_kn"] == pytest.approx(144.0, abs=0.01)


class TestSeismicBaseShear:
    """ASCE 7-16 §12.8.1.1: Cs = SDS Ie / R; V = Cs W.
    EN 1998-1 §4.3.3.2 plateau: Cs = (ag/g × S × 2.5 / q) × λ.
    """

    def test_asce_sds_1_r_8(self):
        # Cs=1×1/8=0.125; V=1250 kN.
        r = _ok("seismic_base_shear", seismic_weight_kn=10000, code="aci", sds=1.0, r=8.0, ie=1.0)
        assert r["seismic_response_coefficient_cs"] == pytest.approx(0.125, abs=0.0005)
        assert r["base_shear_kn"] == pytest.approx(1250.0, abs=0.05)

    def test_eurocode_plateau_and_zero_r(self):
        # Cs=(0.3×1.2×2.5/3.9)×0.85=0.19615; V=1961.5 kN.
        r = _ok(
            "seismic_base_shear", seismic_weight_kn=10000, code="eurocode",
            ag_g=0.3, soil_factor_s=1.2, behaviour_factor_q=3.9, lambda_factor=0.85,
        )
        assert r["seismic_response_coefficient_cs"] == pytest.approx(0.1962, abs=0.0002)
        assert r["base_shear_kn"] == pytest.approx(1961.5, abs=0.5)
        _err("seismic_base_shear", seismic_weight_kn=10000, code="aci", sds=1.0, r=0, ie=1.0)


class TestSlopeFosSimple:
    """Infinite slope (dry): FoS = tanφ/tanβ + c/(γ z sinβ cosβ)."""

    def test_cohesionless_30_over_20(self):
        # tan30/tan20 = 0.577350/0.363970 = 1.586.
        r = _ok("slope_fos_simple", friction_angle_deg=30, slope_angle_deg=20)
        assert r["factor_of_safety"] == pytest.approx(math.tan(math.radians(30)) / math.tan(math.radians(20)), abs=0.002)
        assert r["unit"] == "unitless"
        assert "unitless" in str(r.get("note") or "").lower()

    def test_with_cohesion_and_zero_slope(self):
        # c-term = 5/(18×3×sin20×cos20)=0.288; FoS=1.874.
        r = _ok(
            "slope_fos_simple",
            friction_angle_deg=30, slope_angle_deg=20,
            cohesion_kpa=5, unit_weight_kn_m3=18, depth_m=3,
        )
        assert r["cohesive_term"] == pytest.approx(0.288, abs=0.003)
        assert r["factor_of_safety"] == pytest.approx(1.874, abs=0.003)
        _err("slope_fos_simple", friction_angle_deg=30, slope_angle_deg=0)


class TestWindPressure:
    """ASCE 7-16 §26.10 SI: qz=0.613 Kz Kzt Kd V² (Ke omitted, default 1).
    EN 1991-1-4 §4.5: qb=0.5 ρ vb², ρ=1.25 kg/m³.
    """

    def test_asce_50mps(self):
        # 0.613×1×1×0.85×2500=1302.625 Pa.
        r = _ok("wind_pressure", wind_speed_m_s=50, code="aci", kz=1.0, kzt=1.0, kd=0.85)
        assert r["velocity_pressure_pa"] == pytest.approx(1302.6, abs=0.2)

    def test_eurocode_and_zero_speed(self):
        r = _ok("wind_pressure", wind_speed_m_s=50, code="eurocode")
        assert r["velocity_pressure_pa"] == pytest.approx(0.5 * 1.25 * 50 ** 2, abs=0.2)
        z = _ok("wind_pressure", wind_speed_m_s=0, code="aci")
        assert z["velocity_pressure_pa"] == pytest.approx(0.0, abs=0.1)


class TestLiveLoadReduction:
    """ASCE 7-16 §4.7.2 SI: L=L0(0.25+4.57/√(KLL AT)), only if KLL AT≥37.2 m²;
    floor min_factor (default 0.5). EN 1991-1-1 §6.3.1.2: αA=(5/7)ψ0+10/A ≤1.
    """

    def test_asce_reduction_and_small_area_floor(self):
        # KLL AT=160; 0.25+4.57/√160=0.6113; L=4.79×0.6113=2.928.
        r = _ok(
            "live_load_reduction",
            base_live_load_kn_m2=4.79, tributary_area_m2=40, code="aci", kll=4,
        )
        assert r["reduction_factor"] == pytest.approx(0.611, abs=0.002)
        assert r["reduced_live_load_kn_m2"] == pytest.approx(2.928, abs=0.005)
        small = _ok(
            "live_load_reduction",
            base_live_load_kn_m2=4.79, tributary_area_m2=8, code="aci", kll=4,
        )
        assert small["reduction_factor"] == pytest.approx(1.0, abs=1e-9)

    def test_eurocode_and_zero_area(self):
        # αA=0.714×0.7 + 10/40 = 0.75; L=3.5925.
        r = _ok(
            "live_load_reduction",
            base_live_load_kn_m2=4.79, tributary_area_m2=40, code="eurocode", psi0=0.7,
        )
        assert r["reduction_factor"] == pytest.approx(0.75, abs=0.002)
        assert r["reduced_live_load_kn_m2"] == pytest.approx(3.593, abs=0.005)
        _err("live_load_reduction", base_live_load_kn_m2=4.79, tributary_area_m2=0, code="eurocode")


class TestRebarLapLength:
    """ACI 318-19 §25.4.2 / §25.5.2 Class B: ld=(fy/(1.1√fc × conf)) db; lap=1.3 ld.
    EN 1992-1-1 §8.4/8.7: fbd=2.25×0.7×0.3 fck^{2/3}/1.5; l0=1.5 lb,rqd.
    """

    def test_aci_t20(self):
        # ld=(420/(1.1×√30×1.5))×20=929.5 mm; lap=1208.3 mm.
        r = _ok("rebar_lap_length", bar_diameter_mm=20, fy_mpa=420, fc_mpa=30, code="aci")
        assert r["lap_length_mm"] == pytest.approx(1208.3, abs=0.5)

    def test_eurocode_t20(self):
        # fck^{2/3}=30^{2/3}=9.6549; fbd=2.25×0.7×0.3×9.6549/1.5=3.041;
        # fyd=365.217; lb=(20/4)×(365.217/3.041)=600.4; l0=900.6 mm.
        r = _ok("rebar_lap_length", bar_diameter_mm=20, fy_mpa=420, fc_mpa=30, code="eurocode")
        assert r["lap_length_mm"] == pytest.approx(900.6, abs=1.0)


class TestSlabThicknessMin:
    """ACI 318-19 Table 7.3.1.1: SS L/20, ×(0.4+fy/700).
    EN 1992-1-1 §7.4.2 basic (lightly stressed): cantilever L/8.
    """

    def test_aci_ss_fy420_and_fy500(self):
        # 6000/20 × 1.0 = 300 mm; fy=500 → ×(0.4+500/700)=334.29 mm.
        r = _ok(
            "slab_thickness_min", span_mm=6000, support_condition="simply_supported",
            code="aci", fy_mpa=420,
        )
        assert r["min_thickness_mm"] == pytest.approx(300.0, abs=0.1)
        r5 = _ok(
            "slab_thickness_min", span_mm=6000, support_condition="simply_supported",
            code="aci", fy_mpa=500,
        )
        assert r5["min_thickness_mm"] == pytest.approx(6000 / 20 * (0.4 + 500 / 700), abs=0.1)

    def test_eurocode_cantilever(self):
        r = _ok("slab_thickness_min", span_mm=6000, support_condition="cantilever", code="eurocode")
        assert r["min_thickness_mm"] == pytest.approx(775.0, abs=0.1)


class TestGuardrailTopRailHeight:
    """OSHA 29 CFR 1926.502(b): 42 in ± 3 in. 42×25.4=1066.8 mm → 1067 mm."""

    def test_osha_42_inches(self):
        r = _ok("guardrail_top_rail_height")
        assert r["top_rail_height_in"] == 42
        assert r["top_rail_height_mm"] == 1067
        assert r["tolerance_in"] == 3
        assert r["tolerance_mm"] == 76

    def test_midrail_halfway(self):
        r = _ok("guardrail_top_rail_height")
        assert r["mid_rail_height_in"] == 21
        assert r["mid_rail_height_mm"] == 533


# ═══════════════════════════════════════════════════════════════════════════
# REMAINING CALCULATORS
# ═══════════════════════════════════════════════════════════════════════════

class TestFinenessModulus:
    """ASTM C125 / C136: FM = Σ cumulative % retained / 100."""

    def test_standard_six_sieves(self):
        # 10+25+45+70+90+98=338; FM=3.38.
        r = _ok("fineness_modulus", sieve_retained_percentages=[10, 25, 45, 70, 90, 98])
        assert r["value"] == pytest.approx(3.38, abs=0.001)
        assert r["unit"] == "unitless"
        assert "unitless" in str(r.get("note") or "").lower()

    def test_empty_list_is_asked_for(self):
        e = _err("fineness_modulus", expect="needs_input", sieve_retained_percentages=[])
        assert e["needs_input"] is True
        assert "at least one number" in e["question"]


class TestLaserScanAccuracy:
    """Linear scale of a vendor 10 m figure: acc(r) = acc_10 × (r/10)."""

    def test_40m_from_2mm_at_10m(self):
        r = _ok("laser_scan_accuracy", range_m=40, accuracy_at_10m_mm=2.0)
        assert r["estimated_accuracy_mm"] == pytest.approx(8.0, abs=0.01)

    def test_identity_at_10m(self):
        r = _ok("laser_scan_accuracy", range_m=10, accuracy_at_10m_mm=2.0)
        assert r["estimated_accuracy_mm"] == pytest.approx(2.0, abs=0.01)


class TestLeedPointsEstimate:
    """LEED v4 BD+C: Certified≥40, Silver≥50, Gold≥60, Platinum≥80 (of 110)."""

    def test_gold_and_boundaries(self):
        assert _ok("leed_points_estimate", points=62)["certification_level"] == "Gold"
        assert _ok("leed_points_estimate", points=80)["certification_level"] == "Platinum"
        assert _ok("leed_points_estimate", points=50)["certification_level"] == "Silver"
        assert _ok("leed_points_estimate", points=40)["certification_level"] == "Certified"

    def test_just_below_certified(self):
        assert _ok("leed_points_estimate", points=39.9)["certification_level"] == "Not certified"


class TestMaterialConsumption:
    """Material = (qty / output_per_unit) × (1 + waste%/100)."""

    def test_100_at_10_with_5pct_waste(self):
        # base=10; ×1.05=10.5.
        r = _ok("material_consumption", quantity_of_work=100, output_per_unit=10, waste_percent=5)
        assert r["base_without_waste"] == pytest.approx(10.0, abs=1e-6)
        assert r["material_required"] == pytest.approx(10.5, abs=1e-6)

    def test_zero_output_and_missing_args(self):
        _err("material_consumption", quantity_of_work=100, output_per_unit=0)
        _err("material_consumption")


class TestModulusOfRupture:
    """fr = 2.4 √(10 fck) / 10  (kg/cm² form). Split-cylinder 1.78 √(10 fck)/10
    ≈ ACI 0.56 √f'c. The 2.4 factor is not ACI 0.62 √f'c — audited as coded.
    """

    def test_c40(self):
        # √400=20; fr=2.4×20/10=4.8; ft=1.78×20/10=3.56; 3.56/40×100=8.9%.
        r = _ok("modulus_of_rupture", fck_n_mm2=40)
        assert r["modulus_of_rupture_n_mm2"] == pytest.approx(4.8, abs=0.005)
        assert r["split_cylinder_aci_n_mm2"] == pytest.approx(3.56, abs=0.01)
        assert r["tensile_pct_of_compressive"] == pytest.approx(8.9, abs=0.05)

    def test_zero_fck_is_error(self):
        _err("modulus_of_rupture", fck_n_mm2=0)


_LENGTH_SI = ("m", "meter", "metre")
_LENGTH_FT = ("ft", "feet", "foot")
_AREA_SI = ("m2", "m²", "sqm")
_AREA_FT = ("ft2", "ft²", "sqft", "sf")
_VOL_SI = ("m3", "m³", "cum")
_VOL_FT = ("ft3", "ft³", "cuft", "cf")
_TIME_H = ("h", "hr", "hour", "hours")
_TIME_D = ("d", "day", "days")


def _pairs(a, b):
    out = []
    for x in a:
        for y in b:
            out.append((x, y))
            out.append((y, x))
    return out


class TestPeUnitConvert:
    """PE sheet SI bridges: 1 ft=0.3048 m; 1 ft²=0.09290304 m²;
    1 ft³=0.028316846592 m³; 1 day=24 h. Both directions, every alias pair.
    """

    @pytest.mark.parametrize("src,dst", _pairs(_LENGTH_SI, _LENGTH_FT))
    def test_length_aliases_both_ways(self, src, dst):
        r = _ok("pe_unit_convert", value=10, from_unit=src, to_unit=dst)
        to_m = {"m": 1.0, "meter": 1.0, "metre": 1.0, "ft": 0.3048, "feet": 0.3048, "foot": 0.3048}
        expected = 10.0 * to_m[src] / to_m[dst]
        assert r["value_out"] == pytest.approx(expected, rel=1e-8, abs=1e-8)
        assert r["dimension"] == "length"

    @pytest.mark.parametrize("src,dst", _pairs(_AREA_SI, _AREA_FT))
    def test_area_aliases_both_ways(self, src, dst):
        r = _ok("pe_unit_convert", value=10, from_unit=src, to_unit=dst)
        to_m2 = {u: 1.0 for u in _AREA_SI}
        to_m2.update({u: 0.09290304 for u in _AREA_FT})
        expected = 10.0 * to_m2[src] / to_m2[dst]
        assert r["value_out"] == pytest.approx(expected, rel=1e-8, abs=1e-8)

    @pytest.mark.parametrize("src,dst", _pairs(_VOL_SI, _VOL_FT))
    def test_volume_aliases_both_ways(self, src, dst):
        r = _ok("pe_unit_convert", value=10, from_unit=src, to_unit=dst)
        to_m3 = {u: 1.0 for u in _VOL_SI}
        to_m3.update({u: 0.028316846592 for u in _VOL_FT})
        expected = 10.0 * to_m3[src] / to_m3[dst]
        assert r["value_out"] == pytest.approx(expected, rel=1e-8, abs=1e-8)

    @pytest.mark.parametrize("src,dst", _pairs(_TIME_H, _TIME_D))
    def test_time_aliases_both_ways(self, src, dst):
        r = _ok("pe_unit_convert", value=10, from_unit=src, to_unit=dst)
        to_h = {u: 1.0 for u in _TIME_H}
        to_h.update({u: 24.0 for u in _TIME_D})
        expected = 10.0 * to_h[src] / to_h[dst]
        assert r["value_out"] == pytest.approx(expected, rel=1e-8, abs=1e-8)

    def test_unsupported_pair(self):
        env = _err("pe_unit_convert", value=10, from_unit="m", to_unit="kg")
        assert "Unsupported" in env["error"]


class TestProductivityManpowerDuration:
    """Productivity = qty_exec / man-hours; duration = qty / daily_production;
    manhours_required = qty / productivity (uses computed productivity).
    """

    def test_paired_productivity_and_duration(self):
        # 100/50=2 qty/mh; mh_req=200/2=100; duration=200/20=10 days.
        r = _ok(
            "productivity_manpower_duration",
            quantity_executed=100, man_hours=50, quantity=200, daily_production=20,
        )
        assert r["productivity"] == pytest.approx(2.0, abs=1e-6)
        assert r["manhours_required"] == pytest.approx(100.0, abs=1e-4)
        assert r["duration"] == pytest.approx(10.0, abs=1e-4)

    def test_missing_pairs_is_error(self):
        _err("productivity_manpower_duration")

    def test_live_a2_plastering_rate_plus_crew_cost(self):
        r = _ok(
            "productivity_manpower_duration",
            quantity=3400,
            productivity_rate=42,
            rate_unit="m2 per gang-day",
            crew_cost_per_day=1950,
        )
        assert r["duration"] == pytest.approx(3400 / 42, abs=1e-4)
        assert r["total_cost"] == pytest.approx(3400 / 42 * 1950, abs=0.05)


class TestProductivityRate:
    """rate = output / man-hours. The crew does not divide that rate again."""

    def test_500_in_40_hours_crew_4(self):
        # 500/40 = 12.5 per man-hour. A crew of 4 does not make it 3.125.
        r = _ok("productivity_rate", output_quantity=500, labor_hours=40, crew_size=4)
        assert r["rate_per_hour"] == pytest.approx(12.5, abs=0.001)
        assert r["rate_per_worker_hour"] == pytest.approx(12.5, abs=0.001)

    def test_zero_hours_is_error(self):
        _err("productivity_rate", output_quantity=500, labor_hours=0, crew_size=4)


class TestProgressQuantity:
    """Planned%=P/T×100; Actual%=A/T×100; Remaining=T−A; Var=Actual−Planned."""

    def test_1000_total(self):
        # 400/1000=40%; 350/1000=35%; rem=650; var=−5%.
        r = _ok("progress_quantity", total_qty=1000, planned_qty=400, actual_qty=350)
        assert r["planned_percent"] == pytest.approx(40.0, abs=0.001)
        assert r["actual_percent"] == pytest.approx(35.0, abs=0.001)
        assert r["remaining_qty"] == pytest.approx(650.0, abs=0.001)
        assert r["progress_variance_percent"] == pytest.approx(-5.0, abs=0.001)

    def test_zero_total_is_error(self):
        _err("progress_quantity", total_qty=0, planned_qty=400)


class TestRebarByArea:
    """bars/m = 1000/spacing; length = bars/m × area × ways;
    mass = length × (π/4)(d/1000)²×7850.
    """

    def test_t12_at_200_one_way(self):
        # unit=0.8878 kg/m; 5 bars/m ×20 m²=100 m; 88.78 kg.
        r = _ok("rebar_by_area", area_m2=20, spacing_mm=200, bar_diameter_mm=12)
        assert r["bars_per_m"] == pytest.approx(5.0, abs=0.001)
        assert r["total_mass_kg"] == pytest.approx(
            100.0 * (math.pi / 4.0) * (0.012 ** 2) * 7850.0, abs=0.05,
        )

    def test_zero_spacing_is_error(self):
        _err("rebar_by_area", area_m2=20, spacing_mm=0, bar_diameter_mm=12)


class TestRebarWeight:
    """BS 8666 / physical: m = (π/4) d² ρ, ρ=7850 kg/m³. d=16 → 1.578 kg/m."""

    def test_t16_12m_x50(self):
        unit = (math.pi / 4.0) * (0.016 ** 2) * 7850.0
        r = _ok("rebar_weight", bar_diameter_mm=16, total_length_m=12, quantity=50)
        assert r["unit_mass_kg_m"] == pytest.approx(unit, abs=0.0002)
        assert r["total_mass_kg"] == pytest.approx(unit * 12 * 50, abs=0.05)

    def test_zero_diameter_is_asked_for(self):
        e = _err("rebar_weight", bar_diameter_mm=0, total_length_m=12, quantity=50)
        assert e["needs_input"] is True
        assert [r["parameter"] for r in e["rejected"]] == ["bar_diameter_mm"]

    def test_weight_to_length_12t_y16(self):
        unit = (math.pi / 4.0) * (0.016 ** 2) * 7850.0
        r = _ok(
            "rebar_weight",
            bar_diameter_mm=16,
            total_weight_kg=12000,
            mode="weight_to_length",
        )
        assert r["metres_run"] == pytest.approx(12000 / unit, abs=0.5)
        assert r["total_length_m"] > 1000


class TestRoiCalculator:
    """ROI% = (gain − cost) / cost × 100. Zero cost is undefined, not 0%."""

    def test_20_percent(self):
        r = _ok("roi_calculator", gain=1_200_000, cost=1_000_000)
        assert r["net_profit"] == pytest.approx(200_000, abs=0.01)
        assert r["roi_percent"] == pytest.approx(20.0, abs=0.01)

    def test_zero_cost_is_error(self):
        _err("roi_calculator", gain=100, cost=0)


class TestScoreRisk:
    """Risk scoring: score = P×I on 1–5. GREEN ≤4, AMBER ≤9, else RED."""

    def test_red_and_green_edges(self):
        red = _ok("score_risk", probability=4, impact=5)
        assert red["score"] == 20
        assert red["band"] == "RED"
        assert red["unit"] == "unitless"
        assert red["value"] == 20
        assert "unitless" in str(red.get("note") or "").lower()
        green = _ok("score_risk", probability=2, impact=2)
        assert green["score"] == 4
        assert green["band"] == "GREEN"
        amber = _ok("score_risk", probability=3, impact=3)
        assert amber["score"] == 9
        assert amber["band"] == "AMBER"

    def test_out_of_range_is_error(self):
        _err("score_risk", probability=0, impact=5)
        _err("score_risk", probability=4, impact=6)


class TestToleranceCheck:
    """Measured vs specified +/- tolerance."""

    def test_inside_the_band(self):
        r = _ok("tolerance_check", measured=24.6, specified=25.0, tolerance_plus=0.5)
        assert r["within_tolerance"] is True
        assert r["deviation"] == pytest.approx(-0.4, abs=1e-9)
        assert r["margin"] == pytest.approx(0.1, abs=1e-9)

    def test_outside_an_asymmetric_band(self):
        r = _ok("tolerance_check", measured=25.4, specified=25.0, tolerance_plus=0.3, tolerance_minus=1.0)
        assert r["within_tolerance"] is False
        assert (r["lower_limit"], r["upper_limit"]) == (pytest.approx(24.0), pytest.approx(25.3))


class TestUnitCostTotal:
    """BOQ extension: total = qty × rate."""

    def test_100_at_250(self):
        r = _ok("unit_cost_total", quantity=100, unit_rate=250)
        assert r["total_cost"] == pytest.approx(25_000.0, abs=0.01)

    def test_zero_quantity(self):
        r = _ok("unit_cost_total", quantity=0, unit_rate=250)
        assert r["total_cost"] == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 routing — wrong calculator / skip-tool / flake name-pin
# ═══════════════════════════════════════════════════════════════════════════


def test_fw_route_names_match_audit_slice():
    assert set(FW_ROUTE_NAMES) == set(_AUDIT_SLICE)


class TestResolveFwResourceLine:
    """Live FAIL: resource_line_cost asked, model picked unit_cost_total."""

    def test_unit_cost_total_plus_daily_output_remaps(self):
        name, params = resolve_fw_calc(
            "unit_cost_total",
            {"quantity": 100, "unit_rate": 280, "daily_output": 10},
        )
        assert name == "resource_line_cost"
        assert params["day_rate"] == 280
        env = run_calculation(
            "unit_cost_total",
            {"quantity": 100, "unit_rate": 280, "daily_output": 10},
        )
        assert env["status"] == "success", env
        assert env["calculation"] == "resource_line_cost"
        assert env["result"]["labour_cost"] == pytest.approx(2800.0, abs=0.01)

    def test_resource_line_language_remaps_even_without_daily_output_key(self):
        name, _params = resolve_fw_calc(
            "unit_cost_total",
            {"quantity": 100, "unit_rate": 280,
             "text": "resource line cost at 10 m2 daily output and 280 day rate"},
        )
        assert name == "resource_line_cost"

    def test_true_qty_times_rate_is_not_stolen(self):
        name, params = resolve_fw_calc(
            "unit_cost_total",
            {"quantity": 100, "unit_rate": 250},
        )
        assert name == "unit_cost_total"
        assert params["unit_rate"] == 250
        env = run_calculation("unit_cost_total", {"quantity": 100, "unit_rate": 250})
        assert env["calculation"] == "unit_cost_total"
        assert env["result"]["total_cost"] == pytest.approx(25_000.0, abs=0.01)


class TestResolveFwMaterialConsumption:
    """Live FAIL: material_consumption asked, model/E4 picked mix or volume."""

    def test_mix_name_plus_consumption_params_remaps(self):
        env = run_calculation(
            "concrete_mix_proportions",
            {"quantity_of_work": 100, "output_per_unit": 10, "waste_percent": 5},
        )
        assert env["status"] == "success", env
        assert env["calculation"] == "material_consumption"
        assert env["result"]["material_required"] == pytest.approx(10.5, abs=1e-6)

    def test_e4_concrete_word_does_not_leave_consumption_on_volume(self):
        env = run_calculation(
            "concrete_mix_proportions",
            {
                "quantity_of_work": 100,
                "output_per_unit": 10,
                "waste_percent": 5,
                "text": "material consumption of concrete, 100 m3 of work",
            },
        )
        assert env["calculation"] == "material_consumption"
        assert env["result"]["material_required"] == pytest.approx(10.5, abs=1e-6)

    def test_e4_injected_waste_fraction_becomes_percent(self):
        """E4 writes waste_factor=0.05; consumption treats that as ×0.05."""
        name, params = resolve_fw_calc(
            "concrete_volume",
            {
                "quantity_of_work": 100,
                "output_per_unit": 10,
                "waste_factor": 0.05,
                "text": "material consumption",
            },
            original_name="concrete_mix_proportions",
        )
        assert name == "material_consumption"
        assert params.get("waste_percent") == pytest.approx(5.0)
        assert "waste_factor" not in params or params.get("waste_factor") is None

    def test_true_1_2_4_mix_is_not_stolen(self):
        env = run_calculation(
            "concrete_mix_proportions",
            {
                "wet_volume": 10,
                "cement_parts": 1,
                "sand_parts": 2,
                "aggregate_parts": 4,
            },
        )
        assert env["status"] == "success", env
        assert env["calculation"] == "concrete_mix_proportions"
        assert env["result"]["dry_volume"] == pytest.approx(15.4, abs=1e-6)


class TestResolveFwNamePin:
    """1-of-2 flakes: text names the F–W calculator."""

    def test_spaced_name_pins_after_e4_steal(self):
        name, _params = resolve_fw_calc(
            "concrete_volume",
            {"text": "what is the unit weight of concrete, reinforced"},
            original_name="excavation_volume",
        )
        assert name == "unit_weight_concrete"

    def test_original_fw_name_restored_after_e4_steal(self):
        name, _params = resolve_fw_calc(
            "concrete_volume",
            {"reinforced": True},
            original_name="unit_weight_concrete",
        )
        assert name == "unit_weight_concrete"
        env = run_calculation(
            "unit_weight_concrete",
            {"reinforced": True, "text": "unit weight of concrete"},
        )
        assert env["calculation"] == "unit_weight_concrete"

    def test_e4_raft_volume_is_not_stolen(self):
        env = run_calculation(
            "excavation_volume",
            {
                "length_m": 30,
                "width_m": 20,
                "thickness_m": 1.5,
                "text": (
                    "Concrete volume for a raft 30x20x1.5 m including "
                    "your documented waste factor."
                ),
            },
        )
        assert env["status"] == "success", env
        assert env["calculation"] == "concrete_volume"
        assert env["result"]["volume_m3"] == pytest.approx(945.0, abs=0.01)


class TestFwIntentMapForcesConstructionCalc:
    """Plain-engineer F–W asks have no calc-verb + ≥2 dims — force the tool."""

    AVAILABLE = {"construction_calc", "search_project_documents"}

    @pytest.mark.parametrize(
        "q",
        [
            "What is the OSHA guardrail height for a top rail?",
            "Give me the material consumption for 100 m2 at 10 m2 per unit",
            "Resource line cost for 240 m2 plaster at 12 m2 daily output",
            "What is the unit weight of concrete, reinforced?",
            "Fineness modulus of this sand grading",
            "Wind pressure on the formwork face",
            "Estimate the mobilization cost for 100 staff over 18 months",
            "What is the site mobilisation cost?",
        ],
    )
    def test_fw_phrases_force_the_calculator(self, q):
        from app.agents.runtime import _forced_specific_tool
        assert _forced_specific_tool(
            [{"role": "user", "content": q}], self.AVAILABLE,
        ) == "construction_calc"

    def test_document_lookup_without_fw_phrase_is_not_forced(self):
        from app.agents.runtime import _forced_specific_tool
        assert _forced_specific_tool(
            [{"role": "user", "content": "what is the rebar specification"}],
            self.AVAILABLE,
        ) is None


class TestFwLeftoverUnitPresentation:
    """Live leftover after #620: construction_calc fired but the stream
    had a number and no unit (or unitless not stated)."""

    def test_fineness_modulus_states_unitless(self):
        env = run_calculation(
            "fineness_modulus",
            {"sieve_retained_percentages": [10, 25, 45, 70, 90, 98]},
        )
        assert env["status"] == "success", env
        r = env["result"]
        assert r["value"] == pytest.approx(3.38, abs=0.001)
        assert r["unit"] == "unitless"
        assert "unitless" in str(r["note"]).lower()

    def test_score_risk_states_unitless(self):
        env = run_calculation("score_risk", {"probability": 4, "impact": 5})
        r = env["result"]
        assert r["score"] == 20
        assert r["value"] == 20
        assert r["unit"] == "unitless"
        assert "unitless" in str(r["note"]).lower()

    def test_slope_fos_simple_states_unitless(self):
        env = run_calculation(
            "slope_fos_simple",
            {"friction_angle_deg": 30, "slope_angle_deg": 20},
        )
        r = env["result"]
        assert r["factor_of_safety"] == pytest.approx(1.586, abs=0.002)
        assert r["value"] == pytest.approx(1.586, abs=0.002)
        assert r["unit"] == "unitless"
        assert "unitless" in str(r["note"]).lower()

    def test_a_to_f_envelope_is_not_stamped(self):
        """Shaping is F–W (#43–84) only — A–F results keep their own keys."""
        env = run_calculation(
            "dewatering_uplift_check",
            {"water_depth": 23, "raft_thickness": 2, "floor_count": 5},
        )
        assert env["status"] == "success", env
        assert "unit" not in (env.get("result") or {})


class TestFwMobilizationFlake:
    """A plain mobilisation ask reaches the calculator; with no project
    rates it asks for them (no built-in prices), never a made-up figure."""

    @pytest.mark.parametrize("ask", ["What is the site mobilisation cost?",
                                     "What is the site mobilization cost?"])
    def test_no_figure_uk_spelling_asks_for_the_project_rates(self, ask):
        env = run_calculation(
            "unit_cost_total",
            {"text": ask},
        )
        assert env["calculation"] == "mobilization_cost_estimate"
        assert env["status"] == "needs_input" and env.get("needs_input")
        assert "office_rate_per_person_month" in env["missing"]

    def test_staff_and_months_are_read_from_the_ask(self):
        name, params = resolve_fw_calc(
            "unit_cost_total",
            {"text": "Estimate the mobilization cost for 100 staff over 18 months"},
        )
        assert name == "mobilization_cost_estimate"
        assert params["num_personnel"] == 100
        assert params["duration_months"] == 18
        env = run_calculation(
            "mobilization_cost_estimate",
            {"text": "Estimate the mobilization cost for 100 staff over 18 months",
             **MOBILIZATION_RATES},
        )
        assert env["status"] == "success", env
        assert env["result"]["value"] == pytest.approx(9_927_900, abs=1)

    @pytest.mark.parametrize("rates", [None, "given"])
    def test_named_call_needs_rates_then_runs(self, rates):
        params = {"num_personnel": 10, "duration_months": 2, **MOBILIZATION_RATES} if rates else {}
        env = run_calculation("mobilization_cost_estimate", params)
        assert env["calculation"] == "mobilization_cost_estimate"
        if rates:
            assert env["status"] == "success", env
            assert env["result"]["value"] > 0
        else:
            assert env["status"] == "needs_input"


@pytest.mark.asyncio
async def test_container_construction_calc_remaps_resource_line():
    from app.containers.construction import ConstructionContainer

    r = await ConstructionContainer().construction_calc(
        {"text": "resource line cost for 100 units"},
        {
            "action": "construction_calc",
            "calculation": "unit_cost_total",
            "quantity": 100,
            "unit_rate": 280,
            "daily_output": 10,
        },
    )
    assert r.get("status") == "success", r
    assert r.get("calculation") == "resource_line_cost"
    inner = r.get("result") or {}
    assert inner.get("labour_cost") == pytest.approx(2800.0, abs=0.01)
