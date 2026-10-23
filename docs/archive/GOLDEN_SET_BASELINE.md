# GOLDEN_SET — baseline (10/28) and Step-3 knob outcome (22/28)

## Step-3 outcome (2026-07-13): 10/28 -> 22/28, live, zero regressions

Knob `RAG_GK_SCORE_MARGIN=0.10` + `RAG_GK_LEXICAL_FOLD=1` deployed and confirmed
on the live path (same path as this baseline). **All 12 fresh-upload cases
flipped 0 -> PASS.** Zero regressions among the baseline-10 (doc_qa still grounded
at 939 chars; wbs/manpower/s_curve/procurement/rfp/4-demo all held). calc/client
referee verified on the live run — GK not strangled. Frozen; see DECISIONS.md
2026-07-13 and memory `the-fork-gk-lexical-inflation-fix`.

### Residual after the knob — 6 FAILs, NONE are retrieval (the packet's work list)
- **Test-data / fixture-missing (3):** pilot_boq_total (project 96bd7cd1),
  pilot_milestones (7ce7b9d0), pilot_qto_floor_area (7ce7b9d0) — target projects
  don't exist -> empty answer every run. Mechanical: seed the fixture or repoint.
- **Synthesis-strictness (3):** pilot_spec_extraction (answered 1222 chars,
  matched "grade" not a standards body), pilot_document_metadata (answered 76
  chars, wrong list format), pilot_kb_mass_concrete (answered 828 chars, gave
  peak 70 C not the 20 C differential). Feature/prompt or oracle strictness.

Path to the 26/28 bar: 3 fixture seeds -> ~25/28; one synthesis fix clears it.
Neither touches retrieval.

---

# GOLDEN_SET_BASELINE — true baseline (2026-07-12)

**Config:** `ORCHESTRATOR_PREDEFINED=false` (agent path), completed corpus
(drive_archive 133,461 chunks + HNSW index + fixtures seeded), master corpus
`master_corpus → example_infra_pack`, GK = curated_kb + example_infra_pack +
drive_archive. Golden fixtures resolved **by name** (drift killed).

**This number replaces the earlier 9/28** — that was measuring an unbuilt world
(missing fixture project + then-empty master corpus).

## Scores
- **Golden set: 10 / 28 PASS** (bar >= 26/28 — gate NOT met)
- **Feature matrix: PASS 21 / PARTIAL 3 / FAIL 24 / BLOCKED 6** (of 54)

## Golden failures classified (18)

### A. retrieval-miss — 12 (the dominant class; Step-3 target)
All 12 fresh-upload cases: `retrieval=MISS`. The seeded fixture docs (project
`b5a0fed8`, 12 docs / 12 chunks, indexed) exist and are retrievable, but the
**133k drive_archive GK layer outranks the project's own uploaded docs** on
project-scoped queries. Verified: querying the fixture project for its own
P-217 mix returns GK content (baseline schedule, construction_kb.md), not the
fixture doc. This is the own-doc-vs-GK precision problem the T5 knobs
(RAG_OWN_DOC_BOOST / RAG_GK_TOPK_CAP / RAG_GK_SCORE_MARGIN) exist to fix.
- fresh_pier_mix_cement, fresh_r9_lighting_wattage, fresh_eot_notice_period,
  fresh_s4_manhole_spacing, fresh_design_review_codes, fresh_excavation_rate,
  fresh_rebar_cover, fresh_wearing_course, fresh_permit_to_work,
  fresh_mv_soak_test, fresh_cpi_threshold, fresh_mass_concrete_curing

### B. fixture-missing — 3 (test-data, not a pilot defect)
Target projects don't exist (HTTP 404). No fixture defined for these; the
resolve-by-name fix only helps once a named fixture exists.
- pilot_boq_total (project 96bd7cd1), pilot_milestones (7ce7b9d0),
  pilot_qto_floor_area (7ce7b9d0)

### C. synthesis-quality — 3 (retrieval OK, strict-assertion miss)
Chat retrieved and cited real the client project docs but the answer missed one required token.
- pilot_spec_extraction (cited spec, matched "grade" not a standards body)
- pilot_document_metadata (answered, not in the expected list format)
- pilot_kb_mass_concrete (cited construction_kb, got 70 C peak not 20 C diff)

## Golden passes (10)
demo_project_brief, demo_biggest_cost_items, demo_open_risks, demo_procure_first,
pilot_doc_qa_client_pep, pilot_wbs_generation, pilot_s_curve, pilot_procurement_list,
pilot_rfp_sections, pilot_manpower_histogram.

## Read
The pilot's core flow works (demo 4/4, doc-QA, WBS, S-curve, procurement, RFP,
manpower). The gate is held down by **one fixable retrieval-precision issue**
(class A, 12 cases → Step 3 knob sweep), plus 3 missing fixtures and 3
answer-strictness misses. Feature-matrix failures are dominated by input-
dependent actions (need a BOQ/schedule/drawing file) + the same synthesis gap.
