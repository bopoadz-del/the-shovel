> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# PILOT READINESS — The_Fork Construction Brain

**Date:** 2026-07-16
**Session:** Closing run (packet production) — discipline-agent layer completed
**Pilot go/no-go:** the owner's call (G4) — this page is the input to it.

---

## Verdict in one line

The **foundation is sound and the construction platform is loaded** — 40 blocks, 0 failures, 2-4s grounded answers, zero fabrication surface. Three new flag-gated capabilities added this session (discipline-agent layer, drawing chunk classifier, broadened standards scanner), all OFF by default. **177 offline tests passing** (96 discipline-agent + 25 formulas + 28 standards + 8 drawing classifier, 2 skipped). The remaining gap is **verification breadth** (feature matrix re-run, golden set completion, 100-q recall) and **live activation** of the learning loop. Honest PARKED ledger below.

---

## What WORKS (evidence)

| Area | Evidence |
|------|----------|
| **Blocks / construction platform** | Prod `/v1/health` = **40 blocks, 0 failures**. Construction container, boq_processor, drawing_qto, bim_extractor, primavera_parser, spec_analyzer, cpm_engine, schedule_generator, smart_orchestrator, project_reasoner, discipline-agent layer all loaded. |
| **Chat backend** | `AGENT_TIMING_LOG` shows real completions: 2-4s, RAG tool calls, streamed answers. Not hanging. |
| **Grounded RAG answers** | Live UI answers cite real docs+chunks. Payments → AB-2022-202 Vol 4; commissioning → Aecom electrical comments. |
| **Corpus** | `chunks_v2` = **53 docs / 10,502 chunks**, avg 198 chunks/doc, 0 duplicate chunk_ids. |
| **DB integrity** | `DATABASE_URL` set + fail-loud guard (#190). Old `chunks` table empty; `chunks_v2` canonical. |
| **Master corpus** | `MASTER_CORPUS_SOURCE_PROJECT_ID=example_infra_pack` → "Master Corpus" retrieves real corpus. |
| **Auth** | 401s enforced; both domains serving; logged-in session working. |
| **Fabrication kill** | F1/F2/F3 (#206) + F4 (#207 + #216) all live-verified. **Zero FAKE remaining.** |
| **Grounding gate** | Increment 1 stamps (#215) + increment 2 money/rate (#219). Zero-FP on 147 real figures. `COST_GROUNDING_GATE` default-on. |
| **Phase-2 param passthrough** | `resource_histogram` from project's `.xer` (#212). Decrypt-on-read, ask-which, honest error. |
| **Engine offline** | `comprehensive_engine_test.py` **93/93** — ActivityGraph, DependencyGraph (25 rules), WorkflowTemplateLibrary (10 templates), CrossDomainReasoner, ConstructionLearningEngine all passing. |
| **Wired formulas** | `construction_formulas.py` — 30+ deterministic calculators, single-source-of-truth dispatch via `CALCULATORS` registry. Cost build-ups take rates as PARAMETERS, never fabricate. |

---

## Shipped this session (NEW)

| Item | Status | Flag |
|------|--------|------|
| **Discipline-agent layer** | Base + 5 hats (planning, commercial, contracts, qaqc, procurement) + activation adapter. 177 tests passing (96 discipline + 25 formulas + 28 standards + 8 drawing classifier, 2 skipped). **Not yet deployed** — commit + push needed. | `FORK_HATS_ENABLED` default-off |
| **Drawing chunk classifier (W3 class fix)** | Tags drawing dimension-table chunks at ingestion; excluded from cost/rate grounding. Regression test: `cost_fabrication_concrete_rate`. | `DRAWING_TABLE_CHUNK_TAGGING` default-off |
| **Broadened standards scanner (W8)** | 9 rules (1 core + 8 broadened): ITP, material substitution, test frequency, welding, commissioning, regulatory, method statement, as-built. Advisory only. | `STANDARDS_SCANNER_BROADENED` default-off |
| **Closing ledger** | `docs/CLOSING_LEDGER.md` — honest per-item status for W3-W10, Step 5, Step 6. | — |

---

## PARKED — with evidence (what's NOT verified)

| # | Item | Evidence / why parked | To unblock |
|---|------|----------------------|------------|
| T3 | **20-turn cold-start verification** | Backend healthy via timing logs. Browser automation unreliable for SSE. | Run `fork_cli --events` for 20 turns incl. 3 cold starts. |
| T4 | **Corpus coverage** | Only 53 docs in v2. Full Drive archive NOT in v2. | Decide pilot scope (the client project-only?) or re-ingest tier-2/3. |
| T4b | **RFP under-extraction** | 2 chunks each → weak RFP reasoning. | Re-extract RFP class; exclude/limit `.kmz`. |
| T5 | **Retrieval precision** | GK note 980e19f6 dominates top. cfg7 knobs BREAK v2 — keep OFF. | Re-sweep knobs on v2; GK demotion must be re-derived. |
| T6 | **Feature matrix / golden set / 100-q recall** | Not executed post-router-vocab (#153). Golden 22/28 on embedder-migrated corpus. | Run `scripts/feature_matrix_sweep.py` + golden set + recall eval. |
| T6b | **Attachment reasoning** | `[attached:X]` marker unparsed; RFPs under-extracted. | Wire `[attached:X]`→fetch doc; fix RFP extraction. |
| W3-full | **DXF regression + IFC clash + photo smoke** | Class fix deployed; full oracle fixtures need multi-day work. | DXF known-quantity oracles + planted-clash IFC fixture + photo-path smoke. |
| W4 | **Arabic end-to-end** | No Arabic OCR quality assessment; no extraction oracles. | Arabic OCR model + contract/spec oracles + e2e test. |
| W5 | **Routing matrix + dual-orchestrator** | 20 routing misses at confidence 0.0 documented. Matrix not committed as table+test. | Commit routing matrix; reconcile behind `ORCHESTRATOR_PREDEFINED` flag. |
| W6 | **Learning loop live activation** | Recorders NEVER called from runtime. Engine wired but not fed. | Wire `record_actual_duration` on schedule complete; `record_defect` on QA fail. |
| W7 | **10 dormant agents** | 3 reachable, 10 dormant, 1 parked. 7 configs have unexecutable "hand off to X" prose. | Decision: wire (add `can_delegate`) or ledger-PARK each. |
| W9 | **Export endpoint round-trip** | Libs value-proven; files not verified to OPEN end-to-end. | Smoke `/export/evm`, `/export/schedule`, `/export/boq`. |
| W10 | **10 stale docs** | README "RRF fused", photo_chunks removed, self-hosted-llm, reasoning-consolidation, ROADMAP_ML. | Final sweep + update. |
| 5a | **SOP ingestion** | `GDRIVE_SERVICE_ACCOUNT_JSON` not set on Render. | The owner to provide service-account key + folder mapping. |
| 5b | **SOURCE_MANIFEST** | Template prepared. **HELD for the owner** — jurisdiction choice is his. | The owner to confirm KSA/UAE/both + authority standards to ingest. |

---

## Config Freeze (Final Set)

| Flag | Value | Purpose |
|------|-------|---------|
| `ORCHESTRATOR_PREDEFINED` | `false` | Smart orchestrator active |
| `RAG_GK_LEXICAL_FOLD` | `1` | GK lexical fold ON (paired with MARGIN) |
| `RAG_GK_SCORE_MARGIN` | `0.10` | GK score margin gate |
| `GROUNDING_GATE` | default-on | Post-synthesis cost/rate grounding |
| `CEREBRUM_DOMAIN_KITS` | `construction` | Construction kit loaded |
| `MASTER_CORPUS_SOURCE_PROJECT_ID` | `example_infra_pack` | Master corpus alias |
| `LLM_PROVIDER` | `openai` | Primary: gpt-4o-mini |
| `LLM_FALLBACK_PROVIDER` | `ollama` | Fallback: glm-5.2:cloud |
| `FORK_HATS_ENABLED` | default-off | Discipline-agent layer (NEW) |
| `DRAWING_TABLE_CHUNK_TAGGING` | default-off | Drawing table classifier (NEW) |
| `STANDARDS_SCANNER_BROADENED` | default-off | Standards scanner (NEW) |

---

## Open Human Decisions (for the owner)

1. **`FORK_HATS_ENABLED`** — Activate discipline-agent routing? OFF = zero impact.
2. **`SOURCE_MANIFEST` jurisdiction** — KSA vs UAE vs both? Which authority standards?
3. **SENTRY_DSN** — Provide to restore observability.
4. **Pilot corpus scope** — the client project-only or full Drive archive re-ingest?
5. **Pilot go/no-go** — This page + ledger are the inputs.
6. **Dormant agents** — Wire 10 agents or ledger-PARK them?
7. **Provider freeze** — OpenAI primary, Ollama fallback. Confirm.

---

## Bottom Line

The platform went from "17 blocks with construction kit off" to "40 blocks loaded, 2-4s grounded answers, zero fabrication surface, three new flag-gated capabilities." The remaining gap is **verification breadth** (feature matrix, precision battery, golden set) and **live data feed activation** for the learning loop. Those are the pilot conversation.
