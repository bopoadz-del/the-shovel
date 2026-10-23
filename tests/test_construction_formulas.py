"""Verification of the construction_formulas library (integrated from the
operator's SPEC-C900 formula upgrade). Every expected value below is hand-
computed from the formula definition, so this proves the maths — not just that
the functions run. Also pins the rates-as-parameters contract that keeps cost
build-ups grounded (real rates flow in from RAG; hardcoded defaults are
indicative fallbacks only).
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf


def test_dewatering_uplift_check_cannot_stop():
    # uplift = 23; counter = 2.0*2.5 + 5*0.3*2.5 = 5.0 + 3.75 = 8.75; FOS = 0.380
    r = cf.dewatering_uplift_check(water_depth=23, raft_thickness=2.0, floor_count=5)
    assert r.uplift_force_t_m2 == 23.0
    assert r.counter_weight_t_m2 == 8.75
    assert r.fos == pytest.approx(0.380, abs=0.001)
    assert r.can_stop is False
    assert r.needs_tension_piles is True
    # min floors: ceil((23*1.25 - 5.0)/0.75) = ceil(31.67) = 32
    assert r.min_floors_for_stop == 32


def test_diaphragm_wall_volume_with_tremie_waste():
    r = cf.diaphragm_wall_panel_volume(panel_length=6, wall_thickness=0.8,
                                       excavation_depth=20, panel_count=3)
    assert r["volume_per_panel_m3"] == 96.0
    assert r["total_volume_m3"] == 288.0
    assert r["volume_with_waste_m3"] == 316.8  # 288 * 1.10


def test_well_point_spacing_by_permeability():
    r = cf.dewatering_well_point_spacing(soil_permeability_m_s=2e-3, required_drawdown_m=12)
    assert r["soil_type"] == "coarse sand/gravel"
    assert r["well_point_spacing_m"] == 1.5
    assert r["stages_needed"] == 3  # ceil(12/5)


def test_fineness_modulus():
    assert cf.fineness_modulus([10, 25, 45, 70, 90, 98]) == 3.38  # 338/100


def test_cost_buildup_concrete_arithmetic():
    # material = 0.4*190 + 1.839*29 + 0.16*10 + 12*1.5 + 5*2.4 = 160.93
    r = cf.cost_buildup_concrete(quantity_m3=100)
    assert r["material_cost_sar_m3"] == pytest.approx(160.93, abs=0.01)
    # waste on material only: 160.931*1.03 + 39.2+8+3+2.5+4 = 222.459
    # *1.18 indirect / 0.85 markup = 308.83 SAR/m3; ×100 = 30883
    assert r["selling_price_sar_m3"] == pytest.approx(308.83, abs=0.1)
    assert r["total_project_value_sar"] == pytest.approx(30883, abs=2)


def test_cost_buildup_rebar_arithmetic():
    r = cf.cost_buildup_rebar(quantity_kg=1000)
    # material = 1t * 2600 * 1.10 = 2860
    assert r["material_sar_t"] == 2860
    # (2860 + 90*4.1 + 2*134.6) * 1.18 / 0.85 = 4856
    assert r["selling_price_sar_t"] == pytest.approx(4856, abs=2)


def test_rates_are_parameters_not_hardcoded():
    # The RAG-grounding contract: a real rate passed in flows through the maths.
    base = cf.cost_buildup_rebar(quantity_kg=1000)
    with_real_rate = cf.cost_buildup_rebar(quantity_kg=1000, material_price_sar_t=3000)
    assert with_real_rate["material_sar_t"] == 3300  # 1t * 3000 * 1.10
    assert with_real_rate["material_sar_t"] != base["material_sar_t"]


def test_modulus_of_rupture_runs_and_is_sane():
    r = cf.modulus_of_rupture(fck_n_mm2=40)
    # fr ~ 0.7*sqrt(fck) family — just assert a positive, plausible value + keys.
    assert isinstance(r, dict)
    assert any(isinstance(v, (int, float)) and v > 0 for v in r.values())


def test_library_surface_is_broad():
    # Sanity: the integrated library exposes the full calculator set.
    for fn in ("dewatering_uplift_check", "concrete_mix_design_sg", "crane_planning",
               "crane_cost_estimate", "cost_buildup_formwork", "mobilization_cost_estimate",
               "beam_deflection_ss_udl", "foundation_bearing_pressure"):
        assert callable(getattr(cf, fn)), fn


# Valid inputs for every public calculator — smoke-executes each body so the
# whole library is exercised (each returns a dict/dataclass, never raises).
import dataclasses  # noqa: E402

_SMOKE_INPUTS = {
    "beam_deflection_cantilever_point_load": dict(p_kn=10, span_m=5, ec_mpa=30000, i_mm4=1e9),
    "beam_deflection_cantilever_udl": dict(w_kn_m=10, span_m=5, ec_mpa=30000, i_mm4=1e9),
    "beam_deflection_ss_point_load_midspan": dict(p_kn=10, span_m=5, ec_mpa=30000, i_mm4=1e9),
    "beam_deflection_ss_udl": dict(w_kn_m=10, span_m=5, ec_mpa=30000, i_mm4=1e9),
    "composite_column_design": dict(axial_load_kn=5000, column_diameter_mm=800),
    "concrete_maturity_strength": dict(temperature_history_c=[20, 22, 25],
                                       time_intervals_hours=[6, 6, 6]),
    "concrete_mix_design_sg": dict(w_c_ratio=0.48),
    "concrete_mix_slip_form": dict(),
    "tolerance_check": dict(measured=10.2, specified=10.0, tolerance_plus=0.5),
    "concrete_thermal_cracking_check": dict(core_temp_c=60, surface_temp_c=40),
    "cost_buildup_concrete": dict(quantity_m3=100),
    "cost_buildup_formwork": dict(area_m2=500),
    "cost_buildup_rebar": dict(quantity_kg=1000),
    "crane_cost_estimate": dict(num_cranes=2, crane_capacity_tons=50, duration_months=12),
    "crane_planning": dict(total_lift_demand_tons=1000, crane_capacity_tons=50),
    "dewatering_uplift_check": dict(water_depth=23, raft_thickness=2, floor_count=5),
    "dewatering_well_point_spacing": dict(soil_permeability_m_s=2e-3, required_drawdown_m=12),
    "diaphragm_wall_panel_volume": dict(panel_length=6, wall_thickness=0.8, excavation_depth=20),
    "electrical_installation_sequence": dict(floor_area_m2=1000, num_floors=3),
    "fineness_modulus": dict(sieve_retained_percentages=[10, 25, 45, 70, 90, 98]),
    "formwork_striking_time": dict(),
    "foundation_bearing_pressure": dict(foundation_width_m=3, foundation_length_m=3,
                                        column_load_kn=2000, soil_bearing_capacity_kn_m2=300),
    "grout_pressure_calc": dict(tendon_duct_diameter_mm=80),
    "mobilization_cost_estimate": dict(num_personnel=100, duration_months=18),
    "modulus_of_elasticity_concrete": dict(fck_n_mm2=40),
    "modulus_of_rupture": dict(fck_n_mm2=40),
    "plumbing_flow_programme": dict(floor_area_m2=1000, num_floors=3),
    "post_tensioning_force": dict(span_m=8, slab_thickness_m=0.25, live_load_kn_m2=5),
    "precast_beam_erection_check": dict(beam_weight_t=20, beam_length_m=15,
                                        crane_capacity_t=50, lift_radius_m=12),
    "shear_stress_check": dict(v_kn=200, b_mm=300, d_mm=550),
    "supervision_ratio": dict(concrete_m3=5000, area_m2=10000),
    "thermal_shrinkage_equivalence": dict(),
    "unit_weight_concrete": dict(),
    "wind_load_on_formwork": dict(wind_velocity_m_s=30, formwork_area_m2=50,
                                  formwork_height_m=3, formwork_width_m=8),
    # Referenced from construction_knowledge (reporting/commercial/procurement/risk).
    "calculate_evm": dict(bac=1_000_000, bcwp=400_000, bcws=500_000, acwp=450_000),
    "calculate_payment": dict(claimed_amount=100_000, certified_amount=90_000),
    "evaluate_tender": dict(tenderers=[
        {"name": "A", "technical_score": 80, "commercial_score": 70,
         "hse_score": 90, "local_content_score": 50}]),
    "score_risk": dict(probability=4, impact=5),
    # Referenced from construction_formulas_additions (post-live-chat Q2/Q12).
    "guardrail_top_rail_height": dict(),
    "calculate_interim_payment": dict(gross_valuation=900_000, retention_percent=10),
    # Additive discipline library — structural steel (dual-code).
    "steel_tension_capacity": dict(gross_area_mm2=3000, net_area_mm2=2550,
                                   fy_mpa=345, fu_mpa=450, code="aci"),
    "bolt_shear_capacity": dict(bolt_area_mm2=380, shear_strength_mpa=372,
                                n_shear_planes=1, code="aci"),
    "weld_capacity": dict(leg_size_mm=6, length_mm=100,
                          electrode_strength_mpa=490, code="aci"),
    # Additive discipline library — design loads (dual-code).
    "wind_pressure": dict(wind_speed_m_s=50, code="aci"),
    "seismic_base_shear": dict(seismic_weight_kn=10000, code="aci",
                               sds=1.0, r=8.0, ie=1.0),
    "live_load_reduction": dict(base_live_load_kn_m2=4.79, tributary_area_m2=40,
                                code="aci", kll=4),
    # Additive discipline library — quantities (geometry).
    "concrete_volume": dict(length_m=10, width_m=5, thickness_m=0.3),
    "rebar_weight": dict(bar_diameter_mm=16, total_length_m=12, quantity=50),
    "rebar_by_area": dict(area_m2=20, spacing_mm=200, bar_diameter_mm=12),
    "interior_finishes_takeoff": dict(
        floor_area_m2=100, perimeter_m=90, room_count=8, floor_to_ceiling_m=2.7,
        door_width_m=0.9, door_height_m=2.1, doors_per_room=1),
    "resource_line_cost": dict(
        quantity=100, daily_output=10, day_rate=280,
        material_rate_per_unit=50, plant_fraction_of_labour=0.03),
    # Additive discipline library — earthwork/geotech.
    "excavation_volume": dict(length_m=20, width_m=10, depth_m=3),
    "backfill_volume": dict(excavation_bank_m3=600, structure_volume_m3=200),
    "cut_fill_balance": dict(cut_volume_m3=5000, fill_volume_m3=3500),
    "compaction_control": dict(field_dry_density=1.90, max_dry_density=2.00),
    "slope_fos_simple": dict(friction_angle_deg=30, slope_angle_deg=20),
    # Additive discipline library — beam analysis (statics).
    "beam_moment_simple": dict(udl_w_kn_m=20, span_m=6),
    "beam_moment_point_load": dict(point_load_kn=50, span_m=6),
    "beam_shear_simple": dict(udl_w_kn_m=20, span_m=6),
    "beam_moment_fixed_udl": dict(udl_w_kn_m=20, span_m=6),
    # Additive discipline library — RC design (dual-code).
    "rc_beam_moment_capacity": dict(steel_area_mm2=1500, fy_mpa=420, width_mm=300,
                                    eff_depth_mm=550, fc_mpa=30, code="aci"),
    "rc_beam_shear_capacity": dict(width_mm=300, eff_depth_mm=550, fc_mpa=30, code="aci"),
    "slab_thickness_min": dict(span_mm=6000, support_condition="simply_supported",
                               code="aci", fy_mpa=420),
    "rebar_lap_length": dict(bar_diameter_mm=20, fy_mpa=420, fc_mpa=30, code="aci"),
    # Additive discipline library — QC.
    "concrete_cylinders": dict(failure_load_kn=530, cylinder_diameter_mm=150),
    "concrete_shrinkage": dict(time_days=365),
    "concrete_curing_time": dict(target_strength_fraction=0.70),
    # Additive discipline library — commercial / PM.
    "roi_calculator": dict(gain=1_200_000, cost=1_000_000),
    "unit_cost_total": dict(quantity=100, unit_rate=250),
    "delay_damages_daily": dict(
        rate_percent=0.1, contract_amount=1_754_504_456.25, currency="SAR",
    ),
    "cost_per_area": dict(total_cost=1_000_000, area=5000),
    "productivity_rate": dict(output_quantity=500, labor_hours=40, crew_size=4),
    # Additive discipline library — planning.
    "critical_path_float": dict(early_start=5, early_finish=10, late_start=8, late_finish=13),
    "progress_quantity": dict(total_qty=1000, planned_qty=400, actual_qty=350),
    "productivity_manpower_duration": dict(
        quantity_executed=100, man_hours=50, quantity=200, daily_production=20,
    ),
    "pe_unit_convert": dict(value=10, from_unit="m", to_unit="ft"),
    "material_consumption": dict(quantity_of_work=100, output_per_unit=10, waste_percent=5),
    "concrete_mix_proportions": dict(
        wet_volume=10, cement_parts=1, sand_parts=2, aggregate_parts=4,
    ),
    # Additive discipline library — safety.
    "scaffold_load_capacity": dict(platform_area_m2=10, duty="medium"),
    "fall_arrest_force": dict(worker_mass_kg=100, free_fall_m=1.8, deceleration_distance_m=1.0),
    "crane_lift_capacity": dict(chart_capacity_t=50, deductions_t=3, load_t=38),
    # Additive discipline library — reference tables (cited lookups, not physics).
    "carbon_footprint_concrete": dict(volume_m3=100, grade="c30"),
    "leed_points_estimate": dict(points=62),
    "bim_clash_tolerance": dict(lod=350),
    "laser_scan_accuracy": dict(range_m=40, accuracy_at_10m_mm=2.0),
    # Additive discipline library — columns + masonry (with slenderness, dual-code).
    "column_axial_capacity": dict(gross_area_mm2=160000, steel_area_mm2=4000,
                                  fc_mpa=30, fy_mpa=420, unsupported_length_mm=3000,
                                  radius_gyration_mm=115.47, code="aci"),
    "masonry_wall_capacity": dict(masonry_strength_mpa=10, net_area_mm2=190000,
                                  height_mm=3000, thickness_mm=190, code="aci"),
}


@pytest.mark.parametrize("name", sorted(_SMOKE_INPUTS))
def test_every_calculator_runs(name):
    # Call through the registry (the real dispatch surface) — covers both the
    # module's own calculators and the ones referenced from construction_knowledge.
    result = cf.CALCULATORS[name](**_SMOKE_INPUTS[name])
    assert result is not None
    assert isinstance(result, (dict, float, int)) or dataclasses.is_dataclass(result), name


def test_smoke_inputs_cover_all_public_calculators():
    # Guard: if a new calculator is added to the dispatch registry, this fails
    # until it gets a smoke input. Checks the registry (the real tool surface),
    # so the dispatch helpers (run_calculation/available_calculations) are excluded.
    missing = set(cf.CALCULATORS) - set(_SMOKE_INPUTS)
    assert not missing, f"calculators missing smoke inputs: {sorted(missing)}"
