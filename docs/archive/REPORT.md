# Stop-point report - 2026-07-06 (post-merge session)

State of the pre-pilot engineering program at the operator's stop request.
Written mid-flight: one gate is still running (noted below). Companion
documents: PROGRESS.md (audit trail), DECISIONS.md (gates + dispositions),
RAG_AUDIT_V3.md (retrieval decision table), FEATURE_MATRIX.md (sweep v1).

## What is DONE and merged to main

| Item | Where | Evidence |
|---|---|---|
| CI revived (6h-zombie hang fixed, suite green, timeouts + cancel-in-progress + pytest-timeout tripwire) | PR #146 merged | run 28763386869 all-green; ~20 min/run vs 6h hangs |
| ZERO_CHUNK silent-indexing tripwire | PR #148 merged | grep-able in Render logs, 2 tests |
| GK contamination knobs (default-off) + RAG_AUDIT_V3 config grid | PR #149 merged | cfg7 recommended; calc-intact 3/3 under every config |
| TASK G feature sweep v1 (baseline 23/54) + runner + manifest + review_pack | PR #147 merged | 68 prompts through the orchestrator on prod |
| Dependabot: the 1 HIGH (vite) + 3 mediums fixed | PRs #67, #99 merged | remaining 7 alerts dispositioned in DECISIONS.md (defer, reasons given) |
| TASK 1 router vocabulary patch (additive, 9 pilot-critical actions) | PR #150 merged + DEPLOYED | 12/12 sweep-miss phrasings route; 90 tests on production-like profile; deployed live 07:19Z |
| TASK 2 EOT fix: RAG_GK_LEXICAL_FOLD (default-off) | PR #151 merged | all 4 acceptance gates passed locally: EOT rank 1 (GK out of top-10), fresh top-1 12/12, calc 3/3, recall 41/27 |
| TASK 4e golden-set gate (28 queries, 90% bar, runner + 11 contract tests) | PR #152 OPEN | built + oracle-fixed (EOT reject targets operative deadline only); NOT yet run |

## What is RUNNING right now

- **FEATURE_MATRIX_V2 sweep** (gate 1b for the vocabulary patch) against
  deployed prod: **62/68 runs done at stop time - routing PASS 56/62,
  execution PASS 39/62** (v1 full-run baseline: routing 48/68, exec ~39).
  Early segment ran 12/12 routing where v1 had 5 misses. Results land in
  .claude/worktrees/feature-matrix/feature_matrix_results.jsonl (kill-safe;
  resume with --start; v1 archived as feature_matrix_results_v1.jsonl).
  FEATURE_MATRIX_V2.md is generated from it with --report when complete.
- **Embedder selection agent (TASK 3a)**: benchmarking MiniLM-L6-v2 vs
  bge-small-en-v1.5 vs gte-small on a 20-question + 12-fresh-case harness
  with Render Starter resource-fit checks. Deliverable EMBEDDER_DECISION.md
  on branch feat/embedder-selection. In flight at stop time.

## What is NOT done (the remaining brief, in order)

1. **TASK 2 prod flip (held deliberately)**: the 4 env vars
   (RAG_GK_SCORE_MARGIN=0.15, RAG_OWN_DOC_BOOST=0.1, RAG_GK_TOPK_CAP=2,
   RAG_GK_LEXICAL_FOLD=1) are NOT yet set on prod - held so retrieval would
   not change mid-sweep. Flip after the V2 sweep completes, then verify the
   EOT case live through chat. Revert = unset the 4 vars.
2. **TASK 1 closeout**: FEATURE_MATRIX_V2.md from the finished sweep;
   verify targets (all pilot-critical route on both phrasings, PASS
   strictly > 23/54, zero regressions vs v1).
3. **TASK 3b-d**: migration plan doc, full ~142k-chunk re-index into a NEW
   namespace, RAG_AUDIT_V4 gate, cutover only if doc recall@5 >= 50%.
4. **TASK 4**: final sweep -> FEATURE_MATRIX_FINAL.md; smoke --runs 10;
   review_pack refresh for the 5 demo-critical features (default set: BOQ,
   schedule, S-curve, QTO, RFP); **golden-set gate RUN** (>=26/28 or
   not-pilot-ready); PILOT_READINESS.md + final report.

## Open decisions for the owner (unchanged)

- PR #152 merge (golden-set gate).
- K2 cutover / synthesis streaming: parked on the G2 quality verdict
  (K2_QUALITY_SAMPLES.md). Out of scope this session by the brief.
- 5c13510e BOQ total discrepancy: remembered SAR 62,236,109 vs live corpus
  citation of USD 29,207,138.5 - flagged in tests/golden_set.yaml; needs a
  human eye before that golden query gets a pinned number.
- DAA provider/data choice (pre-existing).

## Operational facts current at stop time

- Prod: healthy, main tip deployed (vocab patch live; fold flag present in
  code but OFF). Baseline smoke 3/3 PASS (recorded in PROGRESS.md); Groq
  429s still push runs to the Ollama-tunnel fallback regularly.
- Known latent bug (found during TASK 2, not fixed - out of scope): with
  sentence-transformers installed, get_embedder() can declare dim 384 while
  encoding 256 (backend/model-name mismatch). Matters for local eval only
  today; becomes relevant during TASK 3 work. Noted in PR #151.
- Local eval infrastructure: V3 harnesses + driver scripts are reusable;
  the local venv now has sentence-transformers (drift vs the V3 grid runs -
  shim documented in RAG_AUDIT_V3.md section 7).
- The Cloudflare Workers bot ("theshovel") still fails a build check on
  every PR - cosmetic, unhooked from merge decisions; the owner to disconnect
  or configure.
