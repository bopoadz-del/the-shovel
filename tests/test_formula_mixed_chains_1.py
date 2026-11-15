"""Formula mixed scenarios, part 1: CI chains, no model.

From the formula test plan, part 3 (mixed scenarios). Each chain calls the
calculators in order through ``run_calculation`` (the entry point
``construction_calc`` uses). The named output of one step is typed into the
named input of the next. Every intermediate value in the plan's "Hand" line is
asserted, not only the final value.

Every expected value is a HAND calculation from the published definition,
written beside its row; none was copied from the product's output. Each
chain runs on two inputs: the plan's case and a second invented case worked
by hand the same way. Plan tolerance: 0.5 % on any number unless stated.

Chains here: M01, M02, M08, M12, M14 (with its pass twin) and M24.
All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 5e-3  # plan tolerance for mixed scenarios: +-0.5 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", (name, env)
    return env["result"]


# ------------------------------------------------- M01 slab -> order -> cost
# net = L x W x t; order = net x (1 + waste); cost = order x rate.

@pytest.mark.parametrize("dims, waste, rate, net, order, cost", [
    ((24, 12, 0.25), 0.05, 310, 72.0, 75.6, 23_436.00),  # 72; 72 x 1.05 = 75.6; 75.6 x 310 = 23,436
    ((10, 8, 0.2), 0.10, 280, 16.0, 17.6, 4_928.00),     # 16; 16 x 1.10 = 17.6; 17.6 x 280 = 4,928
])
def test_m01_slab_order_with_waste_then_cost(dims, waste, rate, net, order, cost):
    length, width, thick = dims
    vol = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                  "waste_factor": waste})
    assert vol["net_volume_m3"] == pytest.approx(net, rel=_REL)
    assert vol["volume_with_waste_m3"] == pytest.approx(order, rel=_REL)
    priced = _ok("unit_cost_total", {"quantity": vol["volume_with_waste_m3"], "unit_rate": rate})
    assert priced["total_cost"] == pytest.approx(cost, rel=_REL)


@pytest.mark.parametrize("dims, rate, net, cost", [
    ((24, 12, 0.25), 310, 72.0, 22_320.00),  # waste 0: order = net = 72; 72 x 310 = 22,320
    ((10, 8, 0.2), 280, 16.0, 4_480.00),     # 16 x 280 = 4,480
])
def test_m01_twin_zero_waste_orders_the_net_volume(dims, rate, net, cost):
    length, width, thick = dims
    vol = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                  "waste_factor": 0})
    assert vol["volume_with_waste_m3"] == pytest.approx(net, rel=_REL)
    priced = _ok("unit_cost_total", {"quantity": vol["volume_with_waste_m3"], "unit_rate": rate})
    assert priced["total_cost"] == pytest.approx(cost, rel=_REL)


# -------------------------------------------- M02 mesh -> kg -> t -> cost
# bars per m = 1000 / spacing; length = area x bars per m x 2 (both ways, no edge bar);
# unit mass = pi/4 x d^2 x 7,850; t = kg / 1,000; cost = t x rate per t.

@pytest.mark.parametrize("area, spacing, d, rate, length, kg, t, cost", [
    (72, 150, 12, 3_400, 960.0, 852.30, 0.85230, 2_897.83),  # 6.6667/m; 960 m x 0.887814 = 852.30 kg
    (50, 200, 10, 3_000, 500.0, 308.27, 0.30827, 924.81),    # 5/m; 500 m x 0.616538 = 308.27 kg
])
def test_m02_mesh_mass_converted_to_tonnes_then_priced(area, spacing, d, rate, length, kg, t, cost):
    mesh = _ok("rebar_by_area", {"area_m2": area, "spacing_mm": spacing, "bar_diameter_mm": d,
                                 "both_ways": True})
    assert mesh["total_bar_length_m"] == pytest.approx(length, rel=_REL)
    assert mesh["total_mass_kg"] == pytest.approx(kg, rel=_REL)
    conv = _ok("pe_unit_convert", {"value": mesh["total_mass_kg"], "from_unit": "kg", "to_unit": "t"})
    assert conv["value_out"] == pytest.approx(t, rel=_REL)
    priced = _ok("unit_cost_total", {"quantity": conv["value_out"], "unit_rate": rate})
    # A price per tonne applied to kilograms (852.30 x 3,400 = 2.9 million) is a hard FAIL.
    assert priced["total_cost"] == pytest.approx(cost, rel=_REL)


# ------------------------------- M08 imperial trench -> backfill -> cost
# bank = L x W x D (ft3 x 0.0283168 = m3); void = bank - structure;
# loose backfill = void x (1 + swell); cost = loose x rate; 1 yd3 = 0.764555 m3.

@pytest.mark.parametrize("dims, structure, swell, rate, bank, void, loose, cost, yd3", [
    (("150 ft", "3 ft", "6 ft"), 12, 0.30, 18,
     76.455, 64.455, 83.792, 1_508.26, 100.0),   # 2,700 ft3 = 76.455 m3; -12; x 1.30; x 18; = 100 yd3
    (("100 ft", "2 ft", "5 ft"), 5, 0.25, 20,
     28.317, 23.317, 29.146, 582.92, 37.037),    # 1,000 ft3 = 28.317 m3; -5; x 1.25; x 20; = 37.04 yd3
])
def test_m08_imperial_trench_backfill_import_cost(dims, structure, swell, rate,
                                                  bank, void, loose, cost, yd3):
    length, width, depth = dims
    dig = _ok("excavation_volume", {"length_m": length, "width_m": width, "depth_m": depth,
                                    "bulking_factor": swell})
    # A feet value used as metres (150 x 3 x 6 = 2,700 "m3") is a hard FAIL.
    assert dig["bank_volume_m3"] == pytest.approx(bank, rel=_REL)
    fill = _ok("backfill_volume", {"excavation_bank_m3": dig["bank_volume_m3"],
                                   "structure_volume_m3": structure, "swell_factor": swell})
    assert fill["void_volume_m3"] == pytest.approx(void, rel=_REL)
    assert fill["loose_backfill_needed_m3"] == pytest.approx(loose, rel=_REL)
    priced = _ok("unit_cost_total", {"quantity": fill["loose_backfill_needed_m3"], "unit_rate": rate})
    assert priced["total_cost"] == pytest.approx(cost, rel=_REL)
    conv = _ok("pe_unit_convert", {"value": dig["bank_volume_m3"], "from_unit": "m3", "to_unit": "yd3"})
    assert conv["value_out"] == pytest.approx(yd3, rel=_REL)


# ------------------------------- M12 cylinder -> Ec -> deflection check
# f = P / (pi/4 d^2); Ec = 4700 sqrt(f'c specified) (ACI); delta = 5 w L^4 / (384 Ec I); limit L/360.

@pytest.mark.parametrize("p, dia, spec, w, span, i, strength, ec, delta, limit", [
    (720, 150, 35, 20, 6, 5.4e9, 40.74, 27_805.6, 2.248, 16.667),  # 720,000 / 17,671.5; 4700 x 5.9161
    (300, 100, 30, 15, 5, 2.0e9, 38.20, 25_743.0, 2.371, 13.889),  # 300,000 / 7,854.0; 4700 x 5.4772
])
def test_m12_cylinder_then_modulus_then_deflection(p, dia, spec, w, span, i,
                                                   strength, ec, delta, limit):
    cyl = _ok("concrete_cylinders", {"failure_load_kn": p, "cylinder_diameter_mm": dia})
    assert cyl["compressive_strength_mpa"] == pytest.approx(strength, rel=_REL)
    assert cyl["compressive_strength_mpa"] >= spec  # the test result meets the specified f'c
    # Ec from the SPECIFIED strength, not the tested one (the request says so).
    mod = _ok("modulus_of_elasticity_concrete", {"fck_n_mm2": spec, "code": "aci"})
    assert mod["value"] == pytest.approx(ec, rel=_REL)
    defl = _ok("beam_deflection_ss_udl", {"w_kn_m": w, "span_m": span, "ec_mpa": mod["value"],
                                          "i_mm4": i})
    assert defl["value"] == pytest.approx(delta, rel=_REL)
    allowed = span * 1000 / 360  # L/360 in mm
    assert allowed == pytest.approx(limit, rel=_REL)
    assert defl["value"] <= allowed  # deflection check: OK


# ------------------------------ M14 compaction fails -> rework -> cost
# % = field / MDD x 100; rework volume = L x W x layer; crew-days = volume / output;
# labour = crew-days x day rate; plant = fraction x labour; total = labour + plant.

@pytest.mark.parametrize("field, mdd, pct, dims, output, day_rate, plant_f, vol, days, labour, plant, total", [
    (1.82, 1.95, 93.33, (40, 20, 0.3), 120, 1_800, 0.5,
     240.0, 2.0, 3_600, 1_800, 5_400),   # 93.33 % < 95 FAIL; 240 m3; 2 d; 3,600; 1,800; 5,400
    (1.70, 1.85, 91.89, (30, 10, 0.25), 50, 2_000, 0.4,
     75.0, 1.5, 3_000, 1_200, 4_200),    # 91.89 % < 95 FAIL; 75 m3; 1.5 d; 3,000; 1,200; 4,200
])
def test_m14_failed_compaction_prices_the_rework(field, mdd, pct, dims, output, day_rate, plant_f,
                                                 vol, days, labour, plant, total):
    test = _ok("compaction_control", {"field_dry_density": field, "max_dry_density": mdd,
                                      "required_compaction_percent": 95})
    assert test["compaction_percent"] == pytest.approx(pct, rel=_REL)
    assert test["passed"] is False
    length, width, layer = dims
    dig = _ok("excavation_volume", {"length_m": length, "width_m": width, "depth_m": layer,
                                    "bulking_factor": 0})
    assert dig["bank_volume_m3"] == pytest.approx(vol, rel=_REL)
    cost = _ok("resource_line_cost", {"quantity": dig["bank_volume_m3"], "daily_output": output,
                                      "day_rate": day_rate, "plant_fraction_of_labour": plant_f})
    assert cost["crew_days"] == pytest.approx(days, rel=_REL)
    assert cost["labour_cost"] == pytest.approx(labour, rel=_REL)
    assert cost["plant_cost"] == pytest.approx(plant, rel=_REL)
    assert cost["total_cost"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("field, mdd, pct", [
    (1.88, 1.95, 96.41),  # 1.88 / 1.95 = 96.41 % >= 95: PASS, no rework
    (1.80, 1.85, 97.30),  # 1.80 / 1.85 = 97.30 % >= 95: PASS, no rework
])
def test_m14_twin_passed_compaction_needs_no_rework(field, mdd, pct):
    test = _ok("compaction_control", {"field_dry_density": field, "max_dry_density": mdd,
                                      "required_compaction_percent": 95})
    assert test["compaction_percent"] == pytest.approx(pct, rel=_REL)
    assert test["passed"] is True  # the chain stops here: no rework volume, no cost


# ------------------------------- M24 slabs -> seismic weight -> base shear
# W = floors x (L x W x t) x unit weight; Cs = SDS x Ie / R; V = Cs x W.

@pytest.mark.parametrize("floors, dims, gamma, sds, r_, ie, vol, weight, cs, v", [
    (5, (30, 20, 0.25), 25, 0.9, 5, 1.0, 150.0, 18_750, 0.18, 3_375),  # 150 m3 x 25 x 5; 0.9/5; 3,375 kN
    (4, (20, 15, 0.2), 24, 0.6, 6, 1.25, 60.0, 5_760, 0.125, 720),     # 60 m3 x 24 x 4; 0.6x1.25/6; 720 kN
])
def test_m24_slab_quantity_to_seismic_base_shear(floors, dims, gamma, sds, r_, ie, vol, weight, cs, v):
    length, width, thick = dims
    slab = _ok("concrete_volume", {"length_m": length, "width_m": width, "thickness_m": thick,
                                   "waste_factor": 0})
    assert slab["net_volume_m3"] == pytest.approx(vol, rel=_REL)
    w = slab["net_volume_m3"] * gamma * floors  # arithmetic step of the chain
    assert w == pytest.approx(weight, rel=_REL)
    shear = _ok("seismic_base_shear", {"seismic_weight_kn": w, "sds": sds, "r": r_, "ie": ie})
    assert shear["seismic_response_coefficient_cs"] == pytest.approx(cs, rel=_REL)
    assert shear["base_shear_kn"] == pytest.approx(v, rel=_REL)
