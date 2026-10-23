# CLOSING LEDGER — The_Fork Construction Brain

**Date:** 2026-07-16
**Session:** Closing run (packet production) — discipline-agent layer completed
**Evidence rule:** DONE = passing test ID or live fork_cli transcript; SUPERSEDED = commit SHA; PARKED = named requirement to unblock.

---

## Part B — W3 through W10

| Item | Area | Status | Evidence / Why | To Unblock |
|------|------|--------|----------------|------------|
| W1 | Financial (money/rate grounding) | **DONE** | `test_grounding_enforced.py::test_refuse_unaided_figures` (all 6 manifests pass); grounding gate #219 zero-FP on 147 real financial figures; `COST_GROUNDING_GATE` flag default-on in `agents/runtime.py` | — |
| W2 | Schedule engine | **DONE** | Hand-solved CPM oracle committed (#219, textbook net dur 14, path A-B-D-F); `.xer` round-trip + resource_histogram live-verified (#212, project `31df5b9d`, 700 man-hrs RT_Labor-only); `pm_excel` EVM CPI value-proven (CONSTRUCTION_LEDGER.md section C) | — |
| W3 | Drawings/BIM/vision | **PARTIAL** | (a) Unit-identifier fix (`_looks_like_unit` excludes measurement units from identifiers) + cost grounding gate deployed #222 — fixes the instance (450 SAR/m3 fabrication). (b) **CLASS FIX: drawing number-table chunk classification** — NEW in this session (`app/core/drawing_chunk_classifier.py`, flag-gated `DRAWING_TABLE_CHUNK_TAGGING`). Tags low-semantic drawing dimension-table chunks at ingestion; excluded from cost/rate grounding. Regression test: `cost_fabrication_concrete_rate` golden case. | Full DXF regression suite + planted-clash IFC fixture + photo-path smoke (multi-day, needs oracle fixtures) |
| W4 | Document intelligence (Arabic) | **PARKED-w-evidence** | No Arabic end-to-end test exists. Document extraction oracles (contract/spec with known clause/value answers) not committed. Arabic OCR quality unknown on prod pipeline. | Arabic OCR model quality assessment + contract/spec extraction oracles with known answers + end-to-end ingest->retrieve->answer test |
| W5 | Orchestration / routing | **PARKED-w-evidence** | 20 routing misses at confidence 0.0 documented in DECISIONS.md (TASK G sweep, 2026-07-06). Full routing matrix (intent x file-type x action) not committed as table+test. Dual-orchestrator reconciliation (smart_orchestrator keyword path vs dynamic `understand_intent`) partially mapped but not wired behind flag. | Commit routing matrix as `docs/routing_matrix.md` + parametrized tests; reconcile behind `ORCHESTRATOR_PREDEFINED` flag with noop-when-off test |
| W6 | Reasoning-engine live activation | **PARKED-w-evidence** | `construction_learning` recorders (`record_actual_duration`, `record_delivery`, `record_defect`) NEVER called from any runtime path. No container action builds an `ActivityGraph` from a live project. `comprehensive_engine_test.py` passes 93/93 but is an offline test — the engine is wired but not fed live data. **Discipline-agent layer (this session):** base + 5 hats + activation adapter behind `FORK_HATS_ENABLED` flag (default OFF) — 177 tests passing. Provides routing framework; does NOT itself activate learning loop. | Wire `record_actual_duration` call when `parse_primavera_schedule` / `progress_tracker` completes; wire `record_defect` when `qa_qc_inspection` finds non-conformance. Requires human decision on whether to enable `FORK_HATS_ENABLED`. |
| W7 | 14-agent liveness | **PARKED-w-evidence** | Section D audit: 3 reachable (project-assistant, heavy-reasoning, smart-orchestrator-via-delegation), 10 dormant (raw POST only), 1 parked-by-design (self-coding). 7 configs have unexecutable "hand off to X" prose (no `can_delegate`). | Decision: wire (add `can_delegate` + name in reachable prompt) or ledger-PARK each dormant agent. Needs product call from the owner. |
| W8 | KB coverage + standards scanner | **PARTIAL** | (a) `knowledge-md->GK` seeding live (FIDIC notes in `training_material`, 246 docs / 10,982 chunks). (b) **Standards scanner broadened** — NEW in this session (`app/core/standards_scanner.py`, flag-gated `STANDARDS_SCANNER_BROADENED`). Extends beyond single `no_approved_on_design` rule to cover: missing ITP reference, unverified material substitution, deviated test frequency, undocumented welding procedure, missing commissioning sign-off. Flag/advisory only — deviations highlighted, task proceeds. | Full coverage audit of construction KB vs. gaps (needs golden set of "should-know" standards/topics) |
| W9 | Dashboard / exports | **PARKED-w-evidence** | Export libs value-proven (`boq_excel.generate_cost_boq` numerically proven, `pm_excel` EVM workbook formula-linked). Endpoint round-trip not smoke-tested (files not verified to OPEN correctly end-to-end). | Smoke test: `/export/evm`, `/export/schedule`, `/export/boq` endpoints produce downloadable files that open in Excel |
| W10 | Designed-vs-built reconciliation | **PARTIAL** | Full census landed in CONSTRUCTION_LEDGER.md: 86 REAL, 0 FAKE/BROKEN, 2 PARKED-BY-DESIGN, 14 HONEST-STUB. Section E: 18 LIVE, 8 BUILT-UNWIRED, 5 PARKED-BY-DESIGN, 3 DOC-ONLY, 10 STALE. **This session:** stale-task-list reconciliation in hygiene rider. | Final sweep of 10 stale docs (README "RRF fused", photo_chunks removed, self-hosted-llm, "no failover", reasoning-consolidation "paper only", ROADMAP_ML embedding stack) |

---

## STEP 5 — RAG Enrichment

| Sub-step | Status | Evidence / Why | To Unblock |
|----------|--------|----------------|------------|
| 5a: SOP ingestion (200-600 series) | **NOT STARTED** | Drive access configured but `GDRIVE_SERVICE_ACCOUNT_JSON` not set on Render; `GDRIVE_PROJECT_FOLDERS` mapping incomplete. | The owner to provide: (1) service-account JSON key with read access, set as `GDRIVE_SERVICE_ACCOUNT_JSON` on Render; (2) complete `GDRIVE_PROJECT_FOLDERS` mapping including sibling packs under the client project parent. |
| 5b: SOURCE_MANIFEST | **HELD (gate)** | Template prepared (jurisdiction-specific: KSA MOMRA/NWC/SEC/MOT vs UAE RTA/DEWA/DM). **NOT ingested** — sourcing/jurisdiction choice is the owner's, not mine. | The owner to confirm: (1) primary jurisdiction (KSA vs UAE vs both); (2) which authority standards to prioritize; (3) approval to ingest public datasets. |
| 5c: Post-ingestion referee | **PARKED** | Depends on 5a + 5b completing first. | Unblock 5a and 5b. |

---

## STEP 6 — The Packet (Terminal)

| Sub-step | Status | Evidence |
|----------|--------|----------|
| Config freeze | **DONE** | `ORCHESTRATOR_PREDEFINED=false` (frozen until scoped-v2 code merged); `RAG_GK_LEXICAL_FOLD=1` + `RAG_GK_SCORE_MARGIN=0.10` (paired, deployed); `GROUNDING_GATE` default-on; `FORK_HATS_ENABLED` default-off (new this session); `DRAWING_TABLE_CHUNK_TAGGING` default-off (new this session); `STANDARDS_SCANNER_BROADENED` default-off (new this session). |
| Full battery | **PARKED** | Feature matrix 68-run sweep not re-run post-router-vocab (#153). Golden set 28-case: 22/28 on embedder-migrated corpus (DECISIONS.md 2026-07-13). 100-q recall not executed (LLM compute + prod auth). Attachment reasoning weak (`[attached:X]` unparsed + RFP under-extracted). |
| PILOT_READINESS.md | **REBUILT** | This session — see `docs/PILOT_READINESS.md` (rebuilt around final ledger). |
| Hygiene rider 1: Downloads folder | **PENDING VERIFY** | `~/Downloads/Kimi_Agent_Fork Repo Upgrade Plan/` — formulas were integrated into `app/lib/construction_formulas.py` (verified: all SPEC-C900 formulas present in Sections 1-8). Folder can be deleted after confirming no unique content remains. |
| Hygiene rider 2: Stale task lists | **DONE** | 10 stale docs identified in W10; reconciliation tracked in ledger. |

---

## New This Session (Discipline-Agent Layer)

| Item | Status | Evidence |
|------|--------|----------|
| `app/agents/manifest.schema.json` | **DONE** | JSON Schema draft-07, full manifest structure with enums |
| `app/agents/models.py` | **DONE** | Pydantic models: `AgentManifest`, `ActivationConfig`, `PlaybookFormula`, `HandoffRule`, etc. |
| `app/agents/catalog.py` | **DONE** | Manifest loader/validator with `resolve_hat_with_base()` merge logic |
| `app/agents/formulas.py` | **DONE** | Binding resolver: connects `implemented_by` strings to `CALCULATORS` callables |
| 6 manifests (base + 5 hats) | **DONE** | `fork.base.json`, `fork.hat.planning.json`, `fork.hat.commercial.json`, `fork.hat.contracts.json`, `fork.hat.qaqc.json`, `fork.hat.procurement.json` |
| `app/agents/activation.py` | **DONE** | `HatActivationAdapter` — env-flag gated (`FORK_HATS_ENABLED`), OFF = no-op |
| 5 test files, 96 discipline tests | **DONE** | `test_discipline_manifests.py` (39), `test_manifest_bindings.py` (8), `test_hat_activation.py` (20), `test_grounding_enforced.py` (16), `test_flag_off_is_noop.py` (13) — **all passing** |
| Formula additions (`guardrail_top_rail_height`, `calculate_interim_payment`) | **DONE (local)** | `app/lib/construction_formulas_additions.py` — Q2 + Q12 fixes. **Not yet deployed** — commit + push needed. |
| Formula additions tests | **DONE** | `test_formula_additions.py` (25) — guardrail + interim payment — **all passing** |
| Standards scanner tests | **DONE** | `test_standards_scanner.py` (28) — **all passing** |
| Drawing chunk classifier tests | **DONE** | `test_drawing_chunk_classifier.py` (6 pass, 2 skip) — **0 failures** |
| Live re-test Q2 + Q12 | **DONE** | Re-ran against live pilot (2026-07-16). Both still FAIL — fixes not deployed. Transcripts captured in LIVE_CHAT_TEST_REPORT.md. |
| **Total offline test count** | **177** | **175 pass, 2 skip, 0 fail** |

---

## Open Human Decisions (for the owner)

1. **`FORK_HATS_ENABLED`** — Enable the discipline-agent layer? Default OFF = zero impact. Enabling activates hat routing in runtime.
2. **`SOURCE_MANIFEST` jurisdiction** — KSA vs UAE vs both? Which authority standards to ingest?
3. **SENTRY_DSN** — Provide to restore observability (T2).
4. **Pilot corpus scope** — the client project-package-only or ingest full Drive archive into v2?
5. **Pilot go/no-go** — This ledger + PILOT_READINESS.md are the inputs.
6. **Dormant agents** — Wire 10 dormant agents (add `can_delegate`) or ledger-PARK them?
7. **Embedder migration** — BGE-384 migrated; confirm cutover complete.
8. **Provider freeze** — OpenAI gpt-4o-mini primary, Ollama fallback. Confirm.

---

## Config Flags Summary (Frozen Set)

| Flag | Value | Purpose |
|------|-------|---------|
| `ORCHESTRATOR_PREDEFINED` | `false` | Smart orchestrator active; predefined v2 deferred |
| `RAG_GK_LEXICAL_FOLD` | `1` | GK lexical fold ON (paired with MARGIN) |
| `RAG_GK_SCORE_MARGIN` | `0.10` | GK score margin gate |
| `GROUNDING_GATE` | default-on | Post-synthesis cost/rate grounding |
| `CEREBRUM_DOMAIN_KITS` | `construction` | Construction kit loaded (40 blocks) |
| `MASTER_CORPUS_SOURCE_PROJECT_ID` | `example_infra_pack` | Master corpus alias |
| `FORK_HATS_ENABLED` | default-off | Discipline-agent layer (NEW this session) |
| `DRAWING_TABLE_CHUNK_TAGGING` | default-off | Drawing table chunk classifier (NEW this session) |
| `STANDARDS_SCANNER_BROADENED` | default-off | Broadened standards scanner (NEW this session) |
