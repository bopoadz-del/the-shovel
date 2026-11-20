"""Formula Layer 1, batch 12 (exam 5): guardrail, PT grout, crane hire, scan accuracy,
material consumption, concrete cost build-up.

Same rules as batches 1-11 (tests/test_formula_layer1_batch1.py):
  T  typical;
  U  different units that must convert to the formula's own unit;
  M  one required input missing -- the calculator must ASK for it and must
     NOT return the figure it would have needed that input for;
  B  boundary -- zero, a value on a limit or cap, a value past it.

Every expected value is a HAND calculation, written in the comment beside
its parametrize row:
  * guardrail top rail           OSHA 29 CFR 1926.502(b): top edge 42 in
                                 +/- 3 in; mid-rail at half height;
                                 1 in = 25.4 mm (exact)
  * PT grout duct                area = pi (d/2)^2 mm2; grout per metre =
                                 area x 1000 mm / 1e6 = litres per m;
                                 1 psi = 0.006894757 N/mm2,
                                 1 kgf/cm2 = 0.0980665 N/mm2
  * crane hire cost              per crane month = (dry + operator +
                                 riggers) x remote factor; total = per x
                                 cranes x months; mobilisation = cranes x
                                 dry x 0.5; demobilisation = 0.75 x
                                 mobilisation (ratios stated in the
                                 calculator); grand = total + mob + demob
  * laser scan accuracy          stated accuracy at 10 m x range / 10
                                 (the calculator's stated linear rule)
  * material consumption         material = quantity / output per unit x
                                 waste factor; factor = 1 + waste % / 100
  * concrete cost build-up       material/m3 = cement kg/1000 x price/t +
    (per m3)                     1.839 t aggregate x price/t + 0.16 m3
                                 water x price + microsilica kg x price +
                                 plasticizer l x price (batch masses stated
                                 in the calculator); direct = material x
                                 (1 + waste) + plant + labour + erection +
                                 power + indirect line; sell = direct x
                                 (1 + indirect) / (1 - markup); total =
                                 sell (2 dp) x m3
None was copied from the product's output. A case where the product
disagrees with the hand value would be marked ``xfail(strict=True)`` with a
one-line reason and reported as a suspected defect; the expected value is
never moved to match. Every value test runs on two inputs worked by hand
the same way.

No M or U case for F74 (it takes no input). No U case for F76 or F78
(counts and currency in the project's own unit). The BIM clash-tolerance
table was not taken in this batch: its values are stated only in product
code, with no source to hand-check them against.

Unit factors (exact by definition): 1 in = 25.4 mm; 1 ft = 0.3048 m, so
1 ft3 = 0.028316846592 m3.

All inputs are invented numbers; no project, person or document is named.
"""
from __future__ import annotations

import math

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", env
    return env["result"]


def _asks(name: str, params: dict, missing: str, withheld: tuple[str, ...]) -> dict:
    """The call must ask for ``missing`` and withhold every key in ``withheld``."""
    env = run_calculation(name, params)
    assert env.get("status") in ("needs_input", "error"), env
    text = " ".join(str(env.get(k) or "") for k in ("error", "question", "message", "note"))
    text += " " + " ".join(str(m) for m in (env.get("missing") or []))
    assert missing.lower() in text.lower(), (missing, env)
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


def _refuses(name: str, params: dict, withheld: tuple[str, ...]) -> dict:
    """A refusal: not success, no number in the withheld keys, no exception."""
    env = run_calculation(name, params)
    assert env.get("status") != "success", env
    result = env.get("result") or {}
    for key in withheld:
        assert result.get(key) is None, (key, env)
    return env


# ---------------------------------------------------------------- F74 guardrail
F74 = "guardrail_top_rail_height"


@pytest.mark.parametrize("key, inches, mm", [
    ("top_rail_height", 42, 1066.8),   # OSHA: 42 in; 42 x 25.4 = 1066.8 mm
    ("mid_rail_height", 21, 533.4),    # half of 42 in = 21 in; 21 x 25.4 = 533.4 mm
    ("tolerance", 3, 76.2),            # OSHA: +/- 3 in; 3 x 25.4 = 76.2 mm
])
def test_f74_t(key, inches, mm):
    r = _ok(F74, {})
    assert r[f"{key}_in"] == pytest.approx(inches, rel=_REL)
    # the mm figure is stated to the whole millimetre
    assert r[f"{key}_mm"] == pytest.approx(mm, abs=0.5)


@pytest.mark.parametrize("edge_in, inside", [
    (39, True),     # 42 - 3: lower edge of the OSHA band
    (45, True),     # 42 + 3: upper edge
    (38.9, False),  # below the band
    (45.1, False),  # above the band
])
def test_f74_b_tolerance_band(edge_in, inside):
    r = _ok(F74, {})
    lo = r["top_rail_height_in"] - r["tolerance_in"]
    hi = r["top_rail_height_in"] + r["tolerance_in"]
    assert (lo <= edge_in <= hi) is inside


# ---------------------------------------------------------------- F75 PT grout
F75 = "grout_pressure_calc"


@pytest.mark.parametrize("d, area, litres", [
    (100, 7853.98, 7.854),   # pi x 50^2 = 7853.98 mm2; x 1000 / 1e6 = 7.854 l/m
    (60, 2827.43, 2.827),    # pi x 30^2 = 2827.43 mm2; 2.827 l/m
])
def test_f75_t_duct(d, area, litres):
    r = _ok(F75, {"tendon_duct_diameter_mm": d})
    assert r["duct_area_mm2"] == pytest.approx(area, rel=_REL)
    assert r["grout_volume_l_m"] == pytest.approx(litres, rel=_REL)


@pytest.mark.parametrize("p, psi, kgf", [
    (0.5, 72.52, 5.10),    # 0.5 / 0.006894757 = 72.52 psi; 0.5 / 0.0980665 = 5.099
    (1.0, 145.04, 10.20),  # 1.0 / 0.006894757 = 145.04 psi; 1.0 / 0.0980665 = 10.197
])
def test_f75_t_pressure_conversion(p, psi, kgf):
    r = _ok(F75, {"tendon_duct_diameter_mm": 80, "required_pressure_n_mm2": p})
    assert r["pressure_n_mm2"] == pytest.approx(p, rel=_REL)
    assert r["pressure_psi"] == pytest.approx(psi, abs=0.06)       # stated to 0.1
    assert r["pressure_kg_cm2"] == pytest.approx(kgf, abs=0.06)    # stated to 0.1


@pytest.mark.parametrize("d, area", [
    ("2 in", 2026.83),   # 50.8 mm; pi x 25.4^2 = 2026.83 mm2
    ("4 in", 8107.32),   # 101.6 mm; pi x 50.8^2 = 8107.32 mm2
])
def test_f75_u(d, area):
    r = _ok(F75, {"tendon_duct_diameter_mm": d})
    assert r["duct_area_mm2"] == pytest.approx(area, rel=_REL)


@pytest.mark.parametrize("params", [
    {},
    {"required_pressure_n_mm2": 0.5},
])
def test_f75_m_diameter(params):
    _asks(F75, params, "duct diameter", ("duct_area_mm2", "grout_volume_l_m"))


@pytest.mark.parametrize("d", [0, -10])
def test_f75_b_no_duct_refused(d):
    _refuses(F75, {"tendon_duct_diameter_mm": d}, ("duct_area_mm2", "grout_volume_l_m"))


@pytest.mark.parametrize("d1, d2", [(40, 80), (50, 100)])
def test_f75_b_area_quadruples_when_diameter_doubles(d1, d2):
    a1 = _ok(F75, {"tendon_duct_diameter_mm": d1})["duct_area_mm2"]
    a2 = _ok(F75, {"tendon_duct_diameter_mm": d2})["duct_area_mm2"]
    assert a2 / a1 == pytest.approx(4.0, rel=_REL)   # (2d)^2 / d^2 = 4
    assert a1 == pytest.approx(math.pi * (d1 / 2) ** 2, rel=_REL)


# ---------------------------------------------------------------- F76 crane hire
F76 = "crane_cost_estimate"


@pytest.mark.parametrize("n, months, dry, op, rig, f, per, total, mob, demob, grand", [
    # (10000 + 4000 + 3000) x 1.2 = 20400; x 2 x 6 = 244800;
    # mob 2 x 10000 x 0.5 = 10000; demob 7500; grand 262300
    (2, 6, 10000, 4000, 3000, 1.2, 20400, 244800, 10000, 7500, 262300),
    # (8000 + 3000 + 2000) x 1.0 = 13000; x 1 x 12 = 156000;
    # mob 1 x 8000 x 0.5 = 4000; demob 3000; grand 163000
    (1, 12, 8000, 3000, 2000, 1.0, 13000, 156000, 4000, 3000, 163000),
])
def test_f76_t(n, months, dry, op, rig, f, per, total, mob, demob, grand):
    r = _ok(F76, {"num_cranes": n, "crane_capacity_tons": 50, "duration_months": months,
                  "dry_hire_rate_per_month": dry, "operator_rate_per_month": op,
                  "riggers_rate_per_month": rig, "remote_area_factor": f})
    assert r["per_crane_monthly"] == pytest.approx(per, rel=_REL)
    assert r["total_project_cost"] == pytest.approx(total, rel=_REL)
    assert r["mobilization"] == pytest.approx(mob, rel=_REL)
    assert r["demobilization"] == pytest.approx(demob, rel=_REL)
    assert r["grand_total"] == pytest.approx(grand, rel=_REL)


@pytest.mark.parametrize("given, missing", [
    ({"riggers_rate_per_month": 3000}, "operator"),
    ({"operator_rate_per_month": 4000}, "rigg"),
])
def test_f76_m_crew_rate(given, missing):
    params = {"num_cranes": 2, "crane_capacity_tons": 50, "duration_months": 6,
              "dry_hire_rate_per_month": 10000, **given}
    _asks(F76, params, missing, ("per_crane_monthly", "total_project_cost", "grand_total"))


@pytest.mark.parametrize("n, months, dry, rig, inc_rig, grand", [
    # operator excluded, no operator rate needed: per = 9000 + 1000 = 10000;
    # total 30000; mob 4500; demob 3375; grand 37875
    (1, 3, 9000, 1000, True, 37875),
    # both crews excluded: per = 5000; total 2 x 1 x 5000 = 10000;
    # mob 2 x 5000 x 0.5 = 5000; demob 3750; grand 18750
    (2, 1, 5000, None, False, 18750),
])
def test_f76_b_excluded_crew_needs_no_rate(n, months, dry, rig, inc_rig, grand):
    params = {"num_cranes": n, "crane_capacity_tons": 25, "duration_months": months,
              "dry_hire_rate_per_month": dry, "include_operator": False,
              "include_riggers": inc_rig}
    if rig is not None:
        params["riggers_rate_per_month"] = rig
    r = _ok(F76, params)
    assert r["operator_per_month"] == 0
    assert r["grand_total"] == pytest.approx(grand, rel=_REL)


@pytest.mark.parametrize("bad", [
    {"crane_capacity_tons": 0},
    {"crane_capacity_tons": -1},
    {"num_cranes": 0},
    {"num_cranes": -2},
])
def test_f76_b_refused(bad):
    params = {"num_cranes": 1, "crane_capacity_tons": 50, "duration_months": 6,
              "dry_hire_rate_per_month": 10000, "include_operator": False,
              "include_riggers": False, **bad}
    _refuses(F76, params, ("total_project_cost", "grand_total"))


# ---------------------------------------------------------------- F77 scan accuracy
F77 = "laser_scan_accuracy"


@pytest.mark.parametrize("rng, acc10, acc", [
    (50, 2, 10.0),   # 2 x 50 / 10 = 10.0 mm
    (25, 3, 7.5),    # 3 x 25 / 10 = 7.5 mm
])
def test_f77_t(rng, acc10, acc):
    r = _ok(F77, {"range_m": rng, "accuracy_at_10m_mm": acc10})
    assert r["estimated_accuracy_mm"] == pytest.approx(acc, rel=_REL)


@pytest.mark.parametrize("rng, metres, acc", [
    ("100 ft", 30.48, 6.096),    # 30.48 m; 2 x 30.48 / 10 = 6.096 mm
    ("200 ft", 60.96, 12.192),   # 60.96 m; 2 x 60.96 / 10 = 12.192 mm
])
def test_f77_u(rng, metres, acc):
    r = _ok(F77, {"range_m": rng, "accuracy_at_10m_mm": 2})
    assert r["range_m"] == pytest.approx(metres, rel=_REL)
    assert r["estimated_accuracy_mm"] == pytest.approx(acc, rel=_REL)   # stated to 0.01


@pytest.mark.parametrize("params", [{}, {"accuracy_at_10m_mm": 2}])
def test_f77_m_range(params):
    _asks(F77, params, "distance", ("estimated_accuracy_mm",))


@pytest.mark.parametrize("acc10", [2, 3.5])
def test_f77_b_at_10m_equals_stated_and_zero_range_is_zero(acc10):
    at10 = _ok(F77, {"range_m": 10, "accuracy_at_10m_mm": acc10})
    assert at10["estimated_accuracy_mm"] == pytest.approx(acc10, rel=_REL)   # x 10 / 10
    at0 = _ok(F77, {"range_m": 0, "accuracy_at_10m_mm": acc10})
    assert at0["estimated_accuracy_mm"] == 0


@pytest.mark.parametrize("rng", [-5, -0.1])
def test_f77_b_negative_range_refused(rng):
    _refuses(F77, {"range_m": rng}, ("estimated_accuracy_mm",))


# ---------------------------------------------------------------- F78 material
F78 = "material_consumption"


@pytest.mark.parametrize("waste, material, base", [
    ({"waste_percent": 5}, 26.25, 25.0),     # 100 / 4 = 25; x 1.05 = 26.25
    pytest.param({"waste_factor": 1.1}, 27.5, 25.0,   # 25 x 1.1 = 27.5
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "suspected defect: waste_factor is documented as a multiplier "
                     "(e.g. 1.05) but its input range is 0-1, so every real factor "
                     "is refused"))),
    pytest.param({"waste_factor": 1.05}, 26.25, 25.0,   # 25 x 1.05 = 26.25
                 marks=pytest.mark.xfail(strict=True, reason=(
                     "suspected defect: waste_factor 1.05 (the docstring's own "
                     "example) is refused by the 0-1 input range"))),
])
def test_f78_t_qty100(waste, material, base):
    r = _ok(F78, {"quantity_of_work": 100, "output_per_unit": 4, **waste})
    assert r["material_required"] == pytest.approx(material, rel=_REL)
    assert r["base_without_waste"] == pytest.approx(base, rel=_REL)


@pytest.mark.parametrize("qty, out, pct, material", [
    (240, 12, 10, 22.0),     # 240 / 12 = 20; x 1.10 = 22.0
    (90, 2.5, 2, 36.72),     # 90 / 2.5 = 36; x 1.02 = 36.72
])
def test_f78_t(qty, out, pct, material):
    r = _ok(F78, {"quantity_of_work": qty, "output_per_unit": out, "waste_percent": pct})
    assert r["material_required"] == pytest.approx(material, rel=_REL)


@pytest.mark.parametrize("params, missing", [
    ({"quantity_of_work": 100}, "output_per_unit"),
    ({"output_per_unit": 4}, "quantity_of_work"),
])
def test_f78_m(params, missing):
    _asks(F78, params, missing, ("material_required", "base_without_waste"))


@pytest.mark.parametrize("qty, out", [(0, 4), (0, 0.5)])
def test_f78_b_zero_work_needs_zero_material(qty, out):
    r = _ok(F78, {"quantity_of_work": qty, "output_per_unit": out, "waste_percent": 5})
    assert r["material_required"] == 0


@pytest.mark.parametrize("qty, out", [(100, 4), (60, 3)])
def test_f78_b_zero_waste_equals_base(qty, out):
    r = _ok(F78, {"quantity_of_work": qty, "output_per_unit": out, "waste_percent": 0})
    assert r["material_required"] == pytest.approx(qty / out, rel=_REL)


@pytest.mark.parametrize("out", [0, -1])
def test_f78_b_no_output_rate_refused(out):
    _refuses(F78, {"quantity_of_work": 100, "output_per_unit": out},
             ("material_required", "base_without_waste"))


# ---------------------------------------------------------------- F79 concrete build-up
F79 = "cost_buildup_concrete"

# Case A: cement 400 kg x 300/t = 120; aggregate 1.839 x 20 = 36.78;
#   water 0.16 x 5 = 0.80; microsilica 12 x 2 = 24; plasticizer 5 x 4 = 20
#   material = 201.58; x 1.03 = 207.6274; + 10 + 30 + 5 + 2 + 3 = 257.6274
#   sell = 257.6274 x 1.18 / 0.85 = 357.6474 -> 357.65; x 100 m3 = 35765
_A = {"cement_kg_m3": 400, "cement_price_per_t": 300, "aggregate_price_per_t": 20,
      "water_price_per_m3": 5, "microsilica_kg_m3": 12, "microsilica_price_per_kg": 2,
      "plasticizer_lit_m3": 5, "plasticizer_price_per_lit": 4, "plant_cost_per_m3": 10,
      "labour_cost_per_m3": 30, "erection_per_m3": 5, "power_per_m3": 2,
      "indirect_per_m3": 3, "indirect_pct": 0.18, "markup_pct": 0.15, "waste_pct": 0.03}
# Case B: cement 350 x 400/t = 140; aggregate 1.839 x 10 = 18.39;
#   water 0.16 x 10 = 1.60; microsilica 0 kg = 0; plasticizer 4 x 5 = 20
#   material = 179.99; x 1.05 = 188.9895; + 20 + 40 + 0 + 5 + 15 = 268.9895
#   sell = 268.9895 x 1.10 / 0.80 = 369.8606 -> 369.86; x 50 m3 = 18493
_B = {"cement_kg_m3": 350, "cement_price_per_t": 400, "aggregate_price_per_t": 10,
      "water_price_per_m3": 10, "microsilica_kg_m3": 0, "microsilica_price_per_kg": 3,
      "plasticizer_lit_m3": 4, "plasticizer_price_per_lit": 5, "plant_cost_per_m3": 20,
      "labour_cost_per_m3": 40, "erection_per_m3": 0, "power_per_m3": 5,
      "indirect_per_m3": 15, "indirect_pct": 0.10, "markup_pct": 0.20, "waste_pct": 0.05}


@pytest.mark.parametrize("rates, qty, material, direct, sell, total", [
    (_A, 100, 201.58, 257.63, 357.65, 35765),
    (_B, 50, 179.99, 268.99, 369.86, 18493),
])
def test_f79_t(rates, qty, material, direct, sell, total):
    r = _ok(F79, {"quantity_m3": qty, **rates})
    assert r["material_cost_per_m3"] == pytest.approx(material, rel=_REL)
    assert r["direct_cost_per_m3"] == pytest.approx(direct, rel=_REL)
    assert r["selling_price_per_m3"] == pytest.approx(sell, rel=_REL)
    assert r["total_project_value"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("rates, qty, total", [
    (_A, "3531.466672 ft3", 35765),   # 100 m3 / 0.028316846592 = 3531.466672 ft3
    (_B, "1765.733336 ft3", 18493),   # 50 m3 = 1765.733336 ft3
])
def test_f79_u(rates, qty, total):
    r = _ok(F79, {"quantity_m3": qty, **rates})
    assert r["total_project_value"] == pytest.approx(total, rel=_REL)


@pytest.mark.parametrize("drop, missing", [
    ("cement_price_per_t", "cement price"),
    ("labour_cost_per_m3", "labour cost"),
])
def test_f79_m(drop, missing):
    params = {"quantity_m3": 100, **{k: v for k, v in _A.items() if k != drop}}
    _asks(F79, params, missing,
          ("material_cost_per_m3", "direct_cost_per_m3", "selling_price_per_m3",
           "total_project_value"))


@pytest.mark.parametrize("rates, sell", [
    (_A, 304.00),   # markup 0: 257.6274 x 1.18 = 304.0003
    (_B, 295.89),   # markup 0: 268.9895 x 1.10 = 295.8885
])
def test_f79_b_markup_zero(rates, sell):
    r = _ok(F79, {"quantity_m3": 10, **{**rates, "markup_pct": 0}})
    assert r["selling_price_per_m3"] == pytest.approx(sell, rel=_REL)


@pytest.mark.parametrize("rates", [_A, _B])
def test_f79_b_zero_quantity_zero_total(rates):
    r = _ok(F79, {"quantity_m3": 0, **rates})
    assert r["total_project_value"] == 0


@pytest.mark.parametrize("qty", [-5, -0.5])
def test_f79_b_negative_quantity_refused(qty):
    _refuses(F79, {"quantity_m3": qty, **_A}, ("total_project_value",))
