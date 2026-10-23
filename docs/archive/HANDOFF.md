# HANDOFF — Construction Brain / Total-Functionality Program

State snapshot for the next session. Read alongside `CONSTRUCTION_LEDGER.md`
(the master), `PROGRESS.md`, `DECISIONS.md`. Newest state wins.

Last updated: 2026-07-14.

## Where we are

**Phase 1 (kill every fabrication): COMPLETE, deployed, live-accepted.** Zero FAKE.
F1/F2/F3 (#206), F4 (#207 + #216) all live-verified on prod.

**Phase 2 (param passthrough + grounding gate): COMPLETE.**
- Param resolution: `resource_histogram` schedule_file from the project's `.xer`
  (#212) — LIVE-through-brain verified (decrypt-on-read, ask-which, honest error).
- F4 location from project metadata (#216) — LIVE (real Riyadh weather).
- Grounding gate: increment 1 stamps (#215, live) + increment 2 money/rate
  (#219) — zero-FP verified on 147 real financial figures. FLAG-only, flag-gated
  `GROUNDING_GATE`.

**STEP 3 (residue): COMPLETE (#218).**
- U1 (upload 502): PARKED-with-evidence — DIAGNOSED transfer-bound, NOT
  buffer+encrypt (Fernet 25MB = 0.27s; prod scaling test linear ~2.3s/MB, all
  201, no cliff). Real fix = chunked/resumable upload (large feature). Typical
  office upstream uploads 25MB in ~20s → non-issue for normal clients/files.
- W1: `qa_inspection` + `progress_tracking` DELETED (redundant/dead);
  `track_progress` + `generate_construction_report` PARKED-BY-DESIGN (real, need
  multi-file/photo param resolution — the 2-file class deferred in 2b).
- W2: `digital_twin_sync` — honest `sync_status: prepared_not_pushed`.
- `historical_benchmark`: PARKED — the run prompt's ruling `[REWORD/RETIRE]` was
  left unfilled; needs the owner's actual decision.

**STEP 4 (Part B W1–W10): W1 + W2 done; W3–W10 PARKED-with-evidence.**
- W1 (financial): money/rate grounding gate (#219) + financial arithmetic already
  hand-tested (payment_certificate net_due=120000 etc.). Substantially done.
- W2 (schedule): hand-solved CPM oracle committed (#219); `.xer` round-trip +
  resource_histogram already live-verified; pm_excel EVM value-proven (ledger §C).
- W3–W10: see PARK list below.

## PARKED-with-evidence (the honest remaining work-list)

Each is real, non-trivial, and NOT claimed done. Ledger is the master.

- **W3 drawings/BIM/vision** — needs: a DXF regression suite (drawing_qto against
  known-quantity DXFs), a planted-clash IFC fixture (bim_clash_detection finds
  the N planted clashes), photo-path smoke. Blocks are REAL (ezdxf/ifcopenshell);
  the gap is committed oracle fixtures. Multi-day.
- **W4 document intelligence** — needs: contract/spec extraction oracles
  (document_engine against docs with known clause/value answers) + an Arabic
  end-to-end test. Multi-day.
- **W5 orchestration** — needs: the full routing matrix (every intent → expected
  action, as a committed table + test) + dual-orchestrator reconciliation
  (smart_orchestrator keyword path vs. dynamic understand_intent). Partially
  mapped (see last session's routing notes). Multi-day.
- **W6 reasoning-engine live activation** — the #166/#167 finding: the
  construction_learning recorders (record_actual_duration/delivery/defect/…) are
  NEVER called from any runtime path, and no container action builds an
  ActivityGraph from a live project. Wiring a real data feed (record actuals when
  a schedule/inspection completes) is the activation. Substantial.
- **W7 14-agent liveness** — audit each agent config: 3 reachable
  (project-assistant, heavy-reasoning, smart-orchestrator-via-delegation), ~10
  dormant, 1 parked (per §D). Needs a committed liveness table + a decision to
  wire or ledger-PARK each dormant agent.
- **W8 KB & prompts coverage** — knowledge-md → GK seeding is live (FIDIC notes);
  needs a coverage audit of what construction KB exists vs. gaps.
- **W9 dashboard/exports** — verify the export endpoints (schedule Excel, cost
  BOQ, EVM workbook) produce files that OPEN correctly end-to-end (the libs are
  value-proven; the endpoint round-trip needs a smoke).
- **W10 designed-vs-built reconciliation** — the ledger's "designed surface vs.
  what's wired" pass; largely captured in the ledger already, needs a final sweep.

## STEP 5 (RAG enrichment) — NOT STARTED
- 5a: ingest the owner's tier-2 SOP folders (200–600 series) from Drive → GK layer.
  No approval needed, but needs Drive access configured in this session.
- 5b: ⛔ GATE — build SOURCE_MANIFEST.md for public datasets, HOLD for the owner's
  approval before any public ingestion.
- 5c: post-ingestion referee (calc-intact + EOT + fresh-upload + golden), zero
  project-precision regression.

## STEP 6 (battery + PILOT_READINESS) — TERMINAL, not yet run
Config freeze, full battery (routing matrix, golden 28, 100-q recall),
PILOT_READINESS.md rebuilt around the final ledger. Run last.

## Merge / ops state
- Merge authority per R8 continues (green checks + passing gate = self-merge).
- Render workspace `tea-d2gv3pf5r7bs73fh82eg`, service `srv-d8hdc6ek1jcs739rq5sg`.
- `GROUNDING_GATE` env: default-on, instantly disableable.
- Hard gates that stop: billing, permanent deletion, provider ladder, pilot
  go/no-go, + the SOURCE_MANIFEST gate (5b).
- Throwaway test projects on prod: `31df5b9d`, `f54b2609`, `86d4fec5` (internal,
  not client-facing; archive if cleaning the list — never hard-delete rows).
