# Formula Verification Table — additive library

One row per additive formula for the operator's one-by-one sign-off. Each row:
the exact inputs, the hand-derived oracle value, the governing clause, and the
oracle test that locks it. All values are reproduced in the test comments so you
can check them against the standard without running code.

Status legend: **GATED** = implemented + oracle test green + in the coverage
guard. Sign-off column is for the operator (`OK` / notes).

## Batch 1 — Structural steel (AISC 360-16 / EN 1993) + Loads (ASCE 7-16 / EN 1991/1998)

Module: `app/lib/construction_formulas_structural_steel.py`, `_loads.py`
Tests: `tests/test_formula_structural_steel.py`, `tests/test_formula_loads.py`

| Formula | Code | Inputs | Oracle | Clause | Status | Sign-off |
|---------|------|--------|--------|--------|--------|----------|
| steel_tension_capacity | AISC | Ag=3000, Ae=2550 mm², Fy=345, Fu=450 | rupture governs → **860.63 kN** (yield 931.5) | AISC 360-16 §D2 | GATED | |
| steel_tension_capacity | EC | A=3000, Anet=2550, fy=355, fu=490 | rupture → **899.64 kN** (yield 1065) | EN 1993-1-1 §6.2.3 | GATED | |
| bolt_shear_capacity | AISC | Ab=380 mm², Fnv=372, 1 plane | **106.02 kN** (0.75·372·380) | AISC 360-16 §J3.6 | GATED | |
| bolt_shear_capacity | EC | A=303 mm², fub=800, αv=0.6, 1 plane | **116.35 kN** (0.6·800·303/1.25) | EN 1993-1-8 Tbl 3.4 | GATED | |
| weld_capacity | AISC | leg=6, L=100 mm, FEXX=490 | throat 4.242 → **93.54 kN** | AISC 360-16 §J2.4 | GATED | |
| weld_capacity | EC | leg=6, L=100, fu=490, βw=0.9 | fvw,d 251.5 → **106.67 kN** | EN 1993-1-8 §4.5.3.3 | GATED | |
| wind_pressure | ASCE | V=50 m/s, Kz=1, Kzt=1, Kd=0.85 | **1302.6 Pa** (0.613·0.85·50²) | ASCE 7-16 §26.10 | GATED | |
| wind_pressure | EC | vb=50, ρ=1.25 | **1562.5 Pa** (0.5·1.25·50²) | EN 1991-1-4 §4.5 | GATED | |
| seismic_base_shear | ASCE | W=10000 kN, SDS=1.0, R=8, Ie=1 | Cs=0.125 → **1250 kN** | ASCE 7-16 §12.8.1 | GATED | |
| seismic_base_shear | EC | W=10000, ag/g=0.3, S=1.2, q=3.9, λ=0.85 | Cs=0.1962 → **1961.5 kN** | EN 1998-1 §4.3.3.2 | GATED | |
| live_load_reduction | ASCE | L0=4.79 kN/m², KLL=4, AT=40 m² | factor 0.611 → **2.928 kN/m²** | ASCE 7-16 §4.7.2 | GATED | |
| live_load_reduction | EC | L0=4.79, ψ0=0.7, A=40 | αA=0.75 → **3.593 kN/m²** | EN 1991-1-1 §6.3.1.2 | GATED | |

## Batch 2 — Quantities (geometry, code-agnostic)

Module: `app/lib/construction_formulas_quantities.py`
Tests: `tests/test_formula_quantities_earthwork.py`

| Formula | Inputs | Oracle | Basis | Status | Sign-off |
|---------|--------|--------|-------|--------|----------|
| concrete_volume (rect) | 10×5×0.3 m, 5% waste | net **15.0**, +waste **15.75** m³ | geometry | GATED | |
| concrete_volume (cyl) | D=0.6, H=10 m | **2.827 m³** (π·0.09·10) | geometry | GATED | |
| concrete_volume (trap) | top2/bot4/dep1.5/L10 | net **45.0**, +waste **47.25** m³ | geometry | GATED | |
| rebar_weight | d16, 12 m, ×50 | unit **1.578 kg/m** → **947.0 kg** | BS 8666 | GATED | |
| rebar_by_area | 20 m², 200 mm, d12 | 5 bars/m → **88.8 kg** | geometry | GATED | |

## Batch 3 — Earthwork / geotech (code-agnostic)

Module: `app/lib/construction_formulas_earthwork.py`
Tests: `tests/test_formula_quantities_earthwork.py`

| Formula | Inputs | Oracle | Basis | Status | Sign-off |
|---------|--------|--------|-------|--------|----------|
| excavation_volume | 20×10×3 m, 25% bulk | bank **600**, loose **750 m³** | geometry/bulking | GATED | |
| backfill_volume | excav 600, struct 200, 20% swell | void **400**, loose **480 m³** | geometry/swell | GATED | |
| cut_fill_balance | cut 5000, fill 3500 | surplus **1500** (export), haul **1875 m³** | mass balance | GATED | |
| compaction_control | field 1.90, MDD 2.00 | **95.0%** → PASS | ASTM D698/D1557 | GATED | |
| slope_fos_simple | φ=30°, β=20° | **1.586** (cohesionless); +c′ 5 kPa → **1.874** | infinite slope | GATED | |

## Batch 4 — Beam analysis (statics) + RC design (dual-code)

**Segregated deliberately:** demand (statics, code-agnostic) vs capacity (dual-code).
Modules: `_beam_analysis.py` (statics), `_structural_rc.py` (design).
Tests: `tests/test_formula_rc_beams.py`

| Formula | Kind | Inputs | Oracle | Basis | Status | Sign-off |
|---------|------|--------|--------|-------|--------|----------|
| beam_moment_simple | analysis | w=20 kN/m, L=6 m | **90 kN·m** (wL²/8) | statics | GATED | |
| beam_moment_point_load | analysis | P=50 kN, L=6 (central / a=2) | **75** (PL/4) / **66.67** (Pab/L) | statics | GATED | |
| beam_shear_simple | analysis | w=20, L=6 | **60 kN** (wL/2) | statics | GATED | |
| beam_moment_fixed_udl | analysis | w=20, L=6 | support **60**, mid **30 kN·m** | statics (fixed) | GATED | |
| rc_beam_moment_capacity | design·ACI | As1500, fy420, b300, d550, fc30 | a=82.4 → **288.5 kN·m** | ACI 318-19 §22.2 | GATED | |
| rc_beam_moment_capacity | design·EC | same | x=114 → **276.3 kN·m** | EN 1992-1-1 §6.1 | GATED | |
| rc_beam_shear_capacity | design·ACI | b300, d550, fc30 | **115.2 kN** | ACI 318-19 §22.5 | GATED | |
| rc_beam_shear_capacity | design·EC | +ρl=0.01 | k=1.603 → **98.6 kN** | EN 1992-1-1 §6.2.2 | GATED | |
| slab_thickness_min | design·ACI | L=6000, SS, fy420 | **300 mm** (L/20) | ACI 318-19 Tbl 7.3.1.1 | GATED | |
| slab_thickness_min | design·EC | L=6000, cantilever | **750 mm** (L/8) | EN 1992-1-1 §7.4.2 | GATED | |
| rebar_lap_length | design·ACI | d20, fy420, fc30 | ld=929 → lap **1208 mm** | ACI 318-19 §25.4.2 | GATED | |
| rebar_lap_length | design·EC | same | lb=600 → lap **900.8 mm** | EN 1992-1-1 §8.4/8.7 | GATED | |

Note: catalog `beam_moment_uniform` ≡ `beam_moment_simple` (both simply-supported
UDL); the distinct fixed-end case is `beam_moment_fixed_udl`. `column_axial_capacity`
partly overlaps existing `composite_column_design` — plain-RC/steel variant TBD.

## Batch 5 — QC + commercial + planning + safety

Modules: `_qc.py`, `_commercial.py`, `_planning.py`, `_safety.py`
Tests: `tests/test_formula_qc_commercial_safety.py`

| Formula | Inputs | Oracle | Basis | Status | Sign-off |
|---------|--------|--------|-------|--------|----------|
| concrete_cylinders | P=530 kN, d=150 | **30.0 MPa** (P/A) | ASTM C39 | GATED | |
| concrete_shrinkage | t=365 d | **711.75 µε** (365/400·780) | ACI 209R | GATED | |
| concrete_curing_time | 70% f28 | **6.91 days** | ACI 209/308 | GATED | |
| roi_calculator | gain 1.2M, cost 1.0M | **20%** | arithmetic | GATED | |
| unit_cost_total | 100 × 250 | **25,000** | BOQ line | GATED | |
| cost_per_area | 1.0M / 5000 | **200 /unit** | arithmetic | GATED | |
| productivity_rate | 500 / 40 h, crew 4 | **12.5/hr, 3.125/wkr-hr** | arithmetic | GATED | |
| critical_path_float | ES5 EF10 LS8 LF13 | **TF=3**, not critical | CPM | GATED | |
| scaffold_load_capacity | 10 m², medium | intended **24**, req **96 kN** (4:1) | OSHA 1926.451 | GATED | |
| fall_arrest_force | 100 kg, H1.8, d1.0 | **2.75 kN** (<8 kN) | OSHA 1926.502 | GATED | |
| crane_lift_capacity | chart50, ded3, load38 | net 47, **80.85%** → PASS | lift plan | GATED | |

## Batch 6 — Reference tables (cited lookups, `kind=reference_table` — NOT derived)

Module: `_reference_tables.py` · Tests: `tests/test_formula_reference_tables.py`

| Formula | Inputs | Oracle | Source (indicative) | Status | Sign-off |
|---------|--------|--------|---------------------|--------|----------|
| carbon_footprint_concrete | 100 m³ C30 | **32,000 kgCO2e** (×320) | ICE/EPD | GATED | |
| leed_points_estimate | 62 pts | **Gold** | LEED v4 BD+C | GATED | |
| bim_clash_tolerance | LOD 350 | **±12 mm** | BEP convention | GATED | |
| laser_scan_accuracy | 40 m, 2 mm@10 m | **8 mm** | scanner datasheet | GATED | |

Reference-table results are labelled `kind=reference_table` with a project/vendor
caveat in the note — they return cited values, not derivations.

## Batch 7 — Columns + masonry, WITH slenderness (dual-code)

Modules: `_columns.py`, `_masonry.py` · Tests: `tests/test_formula_columns_masonry.py`

| Formula | Code | Inputs | Oracle | Clause | Status | Sign-off |
|---------|------|--------|--------|--------|--------|----------|
| column_axial_capacity | ACI | Ag160k, Ast4000, fc30, fy420, lu3000, r115.47 | φPn **2942.2 kN**; λ **25.98** (slender); Pc **24,091 kN** | ACI 318-19 §22.4/§6.6.4 | GATED | |
| column_axial_capacity | ACI (magnifier) | +Pu2000, Cm1.0 | δns **1.124** | ACI §6.6.4.5.2 | GATED | |
| column_axial_capacity | EC | same | NRd **4580.9 kN**; λ 25.98 | EN 1992-1-1 §6.1/§5.8 | GATED | |
| masonry_wall_capacity | TMS | f'm10, An190k, h3000, t190 | R **0.8474** → Pa **402.5 kN** | TMS 402 §8.2/§8.3 | GATED | |
| masonry_wall_capacity | EC6 | same, γM2.7 | Φm **0.7288** → NRd **512.9 kN** | EN 1996-1-1 Annex G | GATED | |

Slenderness is treated explicitly: TMS via the [1−(h/140r)²] reduction; EC6 via
the Annex G Φm method (slenderness λc + eccentricity); the RC column reports the
short capacity + slenderness ratio + Euler Pc + (with an applied load) the ACI
moment magnifier.

## Complete — 39 net-new formulas GATED (registry 37 → 76)

Nothing deferred — the full drop-catalog gap-fill is built, each behind its oracle.
- Batch 5 — QC: concrete_cylinders, concrete_shrinkage, concrete_curing_time
- Batch 6 — Cost/PM/planning: roi_calculator, unit_cost_total, cost_per_sf, productivity_rate, critical_path_float
- Batch 7 — Safety (OSHA): scaffold_load_capacity, fall_arrest_force, crane_lift_capacity
- Batch 8 — Digital + reference-tables: prefab_module_weight, carbon_footprint_concrete, leed_points_estimate, bim_clash_tolerance, laser_scan_accuracy

Each batch: one module + oracle tests (dual-code where a code applies) + smoke
inputs, green through the full referee gate (`comprehensive_engine_test` 93/93 +
coverage guard + calc-intact) before merge.
