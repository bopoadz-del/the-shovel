# Additive Formula Library — scope (behind the referee gate)

Goal: implement the genuinely-absent textbook formulas from the drop catalog as a
**purely additive** library that CANNOT touch or regress the wired 93/93 engine,
and where **no formula ships without a hand-verified oracle from a named standard**.

## The non-negotiable: additive + gated

- **Physically separate.** New formulas live in NEW discipline modules
  (`app/lib/construction_formulas_<discipline>.py`), each exposing an
  `ADDITIONAL_CALCULATORS` dict merged into the registry via the existing
  `construction_formulas_additions.py` pattern (`_build_calculator_registry`
  already merges these). The existing 37 calculators and
  `comprehensive_engine_test` (93/93) are **never edited**.
- **The referee gate (already exists, extends automatically):**
  1. `comprehensive_engine_test` 93/93 stays green — additive-only guarantees it.
  2. Coverage guard: `test_smoke_inputs_cover_all_public_calculators` +
     `test_every_calculator_runs` force a `_SMOKE_INPUTS` entry for every
     registered calculator — a new formula with no smoke input **fails CI**.
  3. Per-formula **oracle test**: a hand-computed assertion against a cited
     standard, following the `guardrail_top_rail_height` template
     (`test_formula_additions.py`): multi-unit output, a `standard` field, a
     `note` step-trace, `pytest.approx` on the numbers.
  4. calc-intact eval (`scripts/rag_calc_intact_eval.py`) unchanged.
- **Anti-fabrication (hard rule):** every formula is implement → derive the
  oracle from a named source (clause/example) → unit-test → only then register.
  No bulk generation. A **pilot batch** ships first for you to sanity-check the
  values and the pattern; scale only after you bless it.

## Contract (matches the existing library)

Each formula = a public function returning a `@dataclass` (or dict) with:
- typed value fields with **units in the name** (`moment_capacity_kn_m`),
- a `standard` field naming the source clause,
- a `notes: List[str]` step-trace (the "show your working" the engine already does),
- values rounded; rates/allowables passed as **parameters** with indicative
  defaults (same "deterministic tool, not an oracle" rule as cost build-ups).

## Code basis (one decision to confirm)

Recommend **ACI 318 (concrete) + AISC 360 (steel) + ASCE 7 (loads)** — the
US-family the Saudi Building Code (SBC 301/304/306) is derived from, so it fits
the the client project/Gulf pilot. Eurocode (EC2/EC3/EC1) is the alternative. I'll default to
the ACI/AISC/ASCE family unless you say Eurocode; a few formulas will expose the
code factor as a parameter so the other family is a one-arg switch.

## The 53 names, triaged

**Already covered — do NOT reimplement (alias if the drop name is wanted):**
`earned_value_cv/sv/spi/cpi` → `calculate_evm`; `beam_deflection_simple` /
`beam_deflection_uniform` → `beam_deflection_ss_udl`; `soil_bearing_pressure` /
`footing_area` → `foundation_bearing_pressure`; `column_axial_capacity` →
overlaps `composite_column_design` (add a plain-RC/steel variant only if needed).

**Reference-table, not derived physics — implement as cited lookup tables,
labeled as such (don't fake a derivation):** `carbon_footprint_concrete`
(EPD kgCO2e factors x qty), `leed_points_estimate` (v4 credit thresholds),
`bim_clash_tolerance` (LOD/tolerance table), `laser_scan_accuracy` (scanner
grade table). These return standard values + source, flagged `kind="reference_table"`.

**Net-new derived formulas (the real work), by discipline:**

| Module | Formulas | Standard |
|--------|----------|----------|
| `_structural_steel` | steel_tension_capacity, bolt_shear_capacity, weld_capacity | AISC 360 |
| `_structural_rc` | beam_moment_simple, beam_moment_point_load, beam_moment_uniform, beam_shear_simple, slab_thickness_min, rebar_lap_length, masonry_wall_capacity | ACI 318 / TMS 402 |
| `_loads` | wind_pressure, seismic_base_shear, live_load_reduction | ASCE 7 |
| `_earthwork` | excavation_volume, backfill_volume, cut_fill_balance, compaction_control, slope_fos_simple | geometry / geotech |
| `_quantities` | concrete_volume (+ basic/cylinder/trapezoidal geometry), rebar_weight (+ basic), rebar_by_area | geometry / BS 8666 |
| `_qc` | concrete_cylinders, concrete_shrinkage, concrete_curing_time | ACI 214 / 209 |
| `_commercial` | roi_calculator, unit_cost_total, cost_per_sf, productivity_rate | arithmetic |
| `_planning` | critical_path_float | CPM |
| `_safety` | scaffold_load_capacity, fall_arrest_force, crane_lift_capacity | OSHA 1926 |
| `_digital` | prefab_module_weight | geometry x density |

~33 net-new derived + 4 reference-table + ~6 aliases-to-existing.

## Phasing (pilot first, per the "pilot-before-scale" rule)

- **Batch 1 (PILOT) — `_structural_steel` + `_loads` (6 formulas):**
  steel_tension_capacity, bolt_shear_capacity, weld_capacity, wind_pressure,
  seismic_base_shear, live_load_reduction. Each with an AISC/ASCE worked-example
  oracle. One PR behind the gate. **You sanity-check the numbers.** No further
  batch in the same PR.
- **Batches 2–N** after you bless the pilot pattern: `_structural_rc`,
  `_earthwork`, `_quantities`, `_qc`, `_commercial`+`_planning`, `_safety`,
  then the reference-table module. One discipline per PR, each green through the
  full gate before the next.

## Explicit non-goals

- Not merging the drop's divergent rewrites of the existing 37 (the 2026-07-14
  decision stands — it would clobber the 93/93 engine).
- Not a design-code compliance checker — these are single-formula calculators
  (capacity/demand), not full member design with all limit states.
- Not inventing precision for the reference-table items — they return cited
  standard values, honestly labeled.

## Decisions (made 2026-07-17)

1. **Code family: BOTH, parameterized.** Every structural/load formula takes a
   `code` parameter ("aci"/"eurocode"), ships both default factor sets, and gets
   an oracle test for EACH family. Doubles oracle work per formula; code-agnostic.
2. **Build all, review one-by-one.** Operator overrode pilot-first: build the full
   library, then check each formula. Delivery stays gated — every formula clears
   its oracle test + the coverage guard before merge — and a per-formula
   `docs/formula-verification-table.md` lists each formula, its clause, and both
   code oracle values for the operator's one-by-one sign-off. Shipped
   discipline-by-discipline (one PR per module, each green through the full gate),
   not one unreviewed dump.
