# Construction Knowledge Base

_Generated from `app/knowledge/construction_kb.json` (single source of truth) — version `2026.06.30-general-corpus`, 47 entries._


**General construction-engineering priors** — NOT code-of-practice design values. Every entry is kept GENERAL (not tied to a location or project). Values are site-experience references; **verify against your project specification and applicable standards** before use. Entries at credibility tier <=3 surface this warning automatically via the loader.


Four domains: `construction.buildings` · `construction.concrete` · `construction.roads` (incl. earthworks + geotech) · `construction.procurement`.


## construction.buildings  (20 entries)

| id | type | statement | tier | review |
|---|---|---|---|---|
| `columns.coupler_rule` | rule | Reinforcement-splicing rule for columns: mechanical couplers are required for bars larger  | 3 |  |
| `columns.high_strength` | threshold | Columns using concrete characteristic strength above 60 N/mm2 are treated as high-strength | 3 |  |
| `columns.value_engineering_grade_up` | decision_pattern | Value-engineering pattern observed on the tower: increasing column concrete grade from 60  | 3 |  |
| `diaphragm_wall.panel_sequence` | reference_design | Reference construction for a cast-in-situ diaphragm (slurry) wall, built as a 5-step panel | 3 |  |
| `enabling.tech_limits` | reference_design | Indicative technology envelopes used on high-rise enabling/superstructure works. Deep foun | 3 |  |
| `facade.sealant_joint_rules` | rule | Rules for facade weatherseal (movement-joint) sealant. Minimum sealant depth 6 mm; minimum | 3 |  |
| `foundation.dewatering_stop_floors` | rule | Construction dewatering must continue until the permanent structure's accumulated dead loa | 2 | yes |
| `foundation.shrinkage_strip_1200mm` | threshold | A 1.2 m (1200 mm) wide shrinkage/pour strip is left open at the raft-to-podium junction to | 3 |  |
| `foundation.uplift_fos` | formula | Factor of safety against hydrostatic/buoyant uplift = available counterweight (dead load + | 4 |  |
| `foundation.uplift_remediation_tension_piles` | decision_pattern | Decision pattern paired with foundation.uplift_fos. WHEN the uplift factor of safety (coun | 3 |  |
| `highrise.definition_contextual` | rule | There is no single universal floor-count that defines a 'high-rise'. The threshold is juri | 3 |  |
| `mep.commissioning_sequence` | checklist | MEP commissioning is sequenced and driven by activity-interface charts that map the depend | 3 |  |
| `posttension.five_day_cycle` | reference_design | Reference floor-cycle for a post-tensioned slab high-rise: a 5-day cycle per floor (form/r | 3 |  |
| `posttension.storage_checklist` | checklist | Storage/handling checklist for post-tensioning (PT) components before stressing. | 3 |  |
| `slabs.flat_vs_drophead` | decision_pattern | Decision pattern for high-rise floor slabs. A flat two-way plate gives the simplest soffit | 3 |  |
| `thermal.core_temp_limit` | threshold | Peak internal (core) temperature of a mass-concrete pour must be kept at or below 70 degC  | 3 |  |
| `thermal.equilibrium_time` | formula | Time for a mass concrete pour of half-thickness X (m) to reach thermal equilibrium with am | 3 |  |
| `thermal.gradient_limit` | threshold | The temperature differential between the concrete core and its surface (T_core - T_surface | 3 |  |
| `thermal.ice_cost_coeff` | reference_design | Decision aid for lowering fresh-concrete placing temperature: the choice between supplemen | 3 |  |
| `thermal.monitoring_protocol` | checklist | Thermocouple monitoring protocol for mass-concrete pours, used to enforce thermal.core_tem | 3 |  |

## construction.concrete  (3 entries)

| id | type | statement | tier | review |
|---|---|---|---|---|
| `concrete.grade_selection` | rule | Select compressive strength grade by element function. Core walls, columns and other prima | 3 |  |
| `rcc.compaction_95spd` | threshold | Roller-compacted concrete must be compacted to greater than 95% of Standard Proctor Densit | 3 |  |
| `rcc.mix_design` | reference_design | Reference RCC mix and placement envelope. Aggregate grading (percent passing): 38 mm = 100 | 3 |  |

## construction.procurement  (3 entries)

| id | type | statement | tier | review |
|---|---|---|---|---|
| `procurement.change_management` | workflow | Governs the change-management cycle from a Request for Modification through an internal Ba | 4 |  |
| `procurement.interim_payment_flow` | workflow | Governs the progress / interim payment cycle from a registered contractor payment request  | 4 |  |
| `procurement.tender_lifecycle` | workflow | Job Requisition -> Sole Source justification (optional) -> Request for Authority to Tender | 4 |  |

## construction.roads  (21 entries)

| id | type | statement | tier | review |
|---|---|---|---|---|
| `asphalt.bitumen_content` | reference_design | Reference bitumen content by asphalt course: base course 3-6%, binder course approximately | 3 |  |
| `asphalt.compaction_trial_30m` | threshold | Before full hot-mix asphalt production paving, lay a compaction trial section of at least  | 3 |  |
| `asphalt.plant_layout` | reference_design | Reference hot-mix asphalt batch plant layout, in process order: cold feed bins, dryer drum | 3 |  |
| `asphalt.rolling_rules` | checklist | Breakdown rolling checklist for hot-mix asphalt: keep breakdown-rolled layer thickness no  | 3 |  |
| `compaction.aashto_t180_ref` | reference_design | AASHTO T 180-93 (Modified Proctor) determines the laboratory maximum dry density and optim | 4 |  |
| `compaction.acceptance` | threshold | Minimum field compaction acceptance levels (percent of laboratory maximum dry density, Mod | 3 |  |
| `compaction.field_density` | formula | Field compaction percentage = (field dry density / laboratory maximum dry density) * 100,  | 4 |  |
| `dewatering.method_selection` | reference_design | Select a dewatering method by soil type, required water-table lowering, and permeability.  | 3 |  |
| `earthworks.compacted_material` | formula | Compacted (in-place) quantity = Loose quantity / Swelling factor, i.e. Compacted = E_loose | 4 |  |
| `earthworks.equipment_by_haul` | decision_pattern | Select earthmoving equipment by haul distance: haul <= 200 m, straight dozing (push with d | 3 |  |
| `earthworks.material_properties_table` | reference_design | Reference properties per material layer. Columns: A = Modified Proctor density (t/m3), B = | 3 |  |
| `earthworks.method_correction_lesson` | decision_pattern | Context-aware equipment selection by material. Holland loaders suit DUNE SAND ONLY -- thei | 3 |  |
| `earthworks.production_rates` | reference_design | Indicative production rates and specifications. Dozers: D8K 130-200 m3/hr, D9H 160-230 m3/ | 3 |  |
| `earthworks.swelling_factor` | formula | Swelling factor = (Proctor density A x Compaction factor C) / Loose density B. Reference t | 3 |  |
| `equipment.crane_rating_check` | reference_design | Always check the required lifted capacity per crane against the crane's rated-load (capaci | 3 |  |
| `ground_improvement.dynamic_compaction` | formula | Depth of ground improvement from dynamic (drop-weight) compaction: D = 0.5 * sqrt(M * H),  | 4 |  |
| `ground_improvement.taxonomy` | reference_design | Ground improvement techniques: (1) MSE / reinforced-earth walls (mechanically stabilized e | 3 |  |
| `roads.heavy_lift_feasibility` | formula | Required lifted capacity per crane for a multi-crane (tandem) beam erection = beam weight  | 4 |  |
| `roads.precast_ibeam_type4` | reference_design | Reference precast prestressed I-beam, Type-4 section: span 22.25 m, weight 66.6 T. Erectio | 3 |  |
| `roads.tunnel_cross_section` | reference_design | Reference cross-section for a cut-and-cover road underpass / tunnel. The combined mass-con | 3 |  |
| `stabilization.taxonomy` | rule | Stabilization methods classified by mechanism. (1) Mechanical / granular stabilization: bl | 3 |  |
