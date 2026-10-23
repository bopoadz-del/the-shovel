# CLOSING RUN — Handoff Prompt (execute exactly; do not improvise)

**You are finishing a MEASURE, not building a product.** No new architecture, no
new features. The exit condition is THE PACKET (Step 6), not a feeling that the
platform "seems done." Do not stop before the packet; do not invent work past it.

Read `HANDOFF.md` and `PILOT_READINESS.md` first — they are the source of truth for
Part B (W1–W10), Step 5, and Step 6. This prompt tells you how to close them without
breaking the repo or lying to the reader.

---

## 0. HARD REPO-SAFETY RULES (read twice; violating any is a failed run)

1. **Do NOT modify the wired engine.** `app/core/construction_knowledge.py` (ActivityGraph,
   DependencyGraph, WorkflowTemplateLibrary, CrossDomainReasoner, ConstructionLearningEngine)
   passes `tests/comprehensive_engine_test.py` 93/93. Extend via NEW files only. Re-run that
   test after every change; if it is not 93/93, you broke it — revert.
2. **Do NOT delete any project / corpus.** Deleting a project CASCADE-deletes its RAG chunks
   (`ON DELETE CASCADE`). NEVER delete: `projects_folder`, `training_material`,
   `master_corpus`, `example_infra_pack`, `curated_kb`, `drive_archive`, and eval project
   `bc812f36`. To clean a list, HIDE/gate the sidebar row — never delete the row. To prune
   specific docs use the existing `POST /v1/admin/corpus/delete-docs` (export bundle first).
3. **Flag-gate every behaviour change OFF by default** (mirror `ORCHESTRATOR_PREDEFINED`). Add a
   test proving flag-OFF is byte-for-byte the current pilot path. Never change live pilot
   behaviour to chase a "DONE".
4. **One logical change per PR.** Never bundle a risky change with docs. Every PR must be CI-green:
   `virgin` + `production-like` + `test-postgres` + diff-cover ≥50% on changed lines.
5. **Live-verify, don't trust unit tests alone.** "Wired + green tests != works live." Prove
   feature claims with a real call against the deployed pilot via `scripts/fork_cli.py` /
   `scripts/smoke.ps1`, and paste the transcript as evidence.
6. **No emojis** anywhere (code, comments, docs, commits, UI). **Provider routing:** agent LLM
   calls go through the configured chain (Groq → Ollama fallback); do NOT add DeepSeek / Anthropic /
   OpenAI. **Do NOT** rotate secrets or touch `.env` API keys.
7. **Scope of allowed CODE changes is CLOSED**: only (a) the drawing number-table chunk
   classification (W3), (b) broadening the standards-advisory scanner (W8), (c) test additions,
   (d) the packet docs, (e) the two hygiene riders. Everything else is VERIFY + REPORT + PARK.

## 1. EVIDENCE DISCIPLINE (the core of this run)

Every Part-B / Step item gets exactly one status, and status requires proof:

- **DONE** — cite a passing test id (`tests/…::test_…`) OR a live `fork_cli` transcript showing
  the real behaviour. No artifact = not DONE.
- **SUPERSEDED** — cite the commit SHA that made the item moot. "The platform feels done" or
  "we don't need it" does NOT qualify. No SHA = not SUPERSEDED.
- **PARKED (with evidence)** — legitimate and encouraged. State exactly what is missing and the
  named requirement to unblock it (e.g. "needs the Arabic OCR model" / "needs the owner's jurisdiction
  choice"). Relabeling a parked item as "optional" is NOT allowed — if it was in scope, it stays
  in the ledger as PARKED with its requirement.

If you cannot produce evidence for DONE or a SHA for SUPERSEDED, the honest answer is PARKED.
Producing an honest PARKED ledger is success. Producing a false DONE is a failed run.

## 2. STEP 1 — LEDGER RECONCILIATION FIRST

Before touching anything, walk `HANDOFF.md` and produce an honest per-item table for Part B
**W3–W10, Step 5, Step 6** with the status rules above. Output it as `docs/CLOSING_LEDGER.md`.
This ledger drives the rest of the run — you only execute items the ledger marks not-DONE.

## 3. STEP 2 — EXECUTE THE REMAINDER (per the standing Part-B prompt)

For each, the deliverable is a verified status in the ledger; code only where Rule 0.7 permits.

- **W3 (drawings/BIM/vision).** Verify drawing QTO / BIM extraction / vision against real oracles.
  **Fold in the drawing number-table chunk classification — this is NOT optional:** it is the class
  fix behind the cost-fabrication incident (a rate lifted from a drawing dimension table). Implement
  it as a chunk-classifier that tags drawing number-table cells so they cannot ground a cost/rate
  claim; add a regression test using the incident fixture. Flag-gated, engine untouched.
- **W4 (doc-intel depth + Arabic end-to-end — NOT optional).** The pilot market is bilingual. Add a
  real Arabic document end-to-end test (ingest → retrieve → grounded answer) and report the honest
  result. If Arabic extraction is weak, PARK with the named requirement — do not paper over it.
- **W5 (routing).** Full routing matrix (intent × file-type × action), route suites, file-type
  routing, and the dual-orchestrator reconciliation (smart_orchestrator vs the predefined
  orchestrator). VERIFY and document the matrix; only reconcile in code behind a flag with a
  noop-when-off test. Do not silently change routing on the pilot.
- **W6 (reasoning-engine live activation).** Confirm the wired engine is actually reachable on the
  live path and prove it with a `fork_cli` transcript. If activation requires a flag flip on the
  pilot, PARK it as an open human decision for the owner — do NOT flip it yourself.
- **W7 (14-agent liveness).** Prove each of the 14 agents responds on the live path (transcript per
  agent, or PARK the dead ones with the reason). Do not "fix" an agent by weakening a guard.
- **W8 (KB coverage + broaden the standards scanner).** Report KB coverage against the golden set.
  Broaden the standards-advisory scanner beyond the single `no_approved_on_design` rule (flag/advisory
  only, never blocking — deviations are highlighted, the task proceeds).
- **W9 (exports/dashboard).** Verify exports and the dashboard render real data; PARK gaps.
- **W10 (designed-vs-built).** Every document claim gets a ledger verdict (matches design / deviates /
  unverifiable-with-reason). Verify the mechanism on real docs; report honestly.

## 4. STEP 3 — STEP 5 STATUS

Confirm SOP ingestion (5a) actually ran (evidence: the docs present in the target corpus). Confirm
`SOURCE_MANIFEST` is built and **held (⛔) for the owner** with the corrected jurisdiction rule
(infra/authority standards are jurisdiction-specific — KSA MOMRA/NWC/SEC/MOT vs UAE RTA/DEWA/DM;
the sourcing/jurisdiction choice is the owner's, not yours). Do not ingest authority standards without
his decision.

## 5. STEP 4 — THE PACKET (the exit condition)

Produce, in one final PR:
- **Config freeze** — pin the orchestrator/config flags to their intended pilot values
  (reconcile `ORCHESTRATOR_PREDEFINED` freeze vs the deployed config; document the frozen set).
- **Full battery** — complete routing matrix, the golden set INCLUDING the new cost case
  (`cost_fabrication_concrete_rate`), and the 100-question recall run. Paste results.
- **`PILOT_READINESS.md` rebuilt around the final ledger** — per-action verdicts with evidence,
  an honest stub list with named requirements, the three headline tables, and the open human
  decisions (incl. the held `SOURCE_MANIFEST` jurisdiction call).
- **Hygiene riders:** delete the Downloads drop folder `~/Downloads/Kimi_Agent_Fork Repo Upgrade Plan/`
  (its formulas were already integrated — verify integration before deleting); reconcile the stale
  task lists.

## 6. STOP AT THE PACKET

Completion is the instruction. When `PILOT_READINESS.md` is rebuilt from the final ledger and the
battery is pasted, STOP. Do not invent work past it. Do not stop before it. Hand back the ledger,
the battery results, and the open human decisions.
