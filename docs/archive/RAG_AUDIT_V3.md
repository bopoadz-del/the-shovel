# RAG Retrieval Config-Grid Evaluation (V3)

**Audited:** 2026-07-06
**Environment:** LOCAL instance only (uvicorn on 127.0.0.1:8123, branch `feat/gk-contamination-knobs`). Production was not touched; production defaults are unchanged.
**Baseline docs:** RAG_AUDIT.md (2026-06-23), RAG_AUDIT_V2.md (2026-07-05)
**Scope:** measure the three GK-contamination knobs shipped default-OFF in `app/core/rag/retriever.py` (`RAG_GK_SCORE_MARGIN`, `RAG_OWN_DOC_BOOST`, `RAG_GK_TOPK_CAP`) across 8 env-var configurations, with three harnesses per configuration. This is a decision table, not a deployment: the owner picks the config; nothing here changes any default.

---

## 1. Method

Each configuration is a fresh local server launch with the knob env vars set for
that run only (`ENV=development`, `DATA_DIR` pointed at the existing local data
directory). Per configuration, three harnesses run against `POST /v1/rag/search`:

1. **Recall** — `scripts/rag_recall_eval.py --sample 100 --seed 42 --delay 0.2`:
   the same seeded 100-question draw as RAG_AUDIT_V2 (from
   `training_scenarios_drive_archive_v2.jsonl`), measuring doc recall@5 and
   exact-chunk recall@5 against the master corpus.
2. **Fresh-upload wins** — `scripts/rag_fresh_upload_eval.py --query-only`:
   the 12 collision cases from RAG_AUDIT_V2 section 4, uploaded once into a
   local eval project (`d01db86a`); PASS = the uploaded note's chunk at rank 1
   of its project-scoped query.
3. **Calc-KB-intact** — `scripts/rag_calc_intact_eval.py` (new, committed):
   3 queries whose correct grounding is a curated GK note (mass-concrete
   thermal limits -> construction_kb.md; CESMM4 work classes ->
   boq_units_of_measurement.md; FIDIC 2017 cl. 20.2 time bar ->
   fidic_2017_administration.md), run twice each — with
   `intent="calculation"` (knobs must NOT apply; must stay 3/3 GK-top-1
   under every config) and with intent unset (informational: what a
   calc-shaped query classified as lookup would get).

### 1.1 Local-corpus caveats (read before comparing numbers to V2)

The local instance serves the operator machine's existing data, which differs
from production in three documented ways. **Relative comparisons across the 8
configs are the product of this audit; absolute numbers are NOT comparable to
RAG_AUDIT_V2's production numbers.**

* **Project id:** the master corpus lives locally under project id
  `drive_archive` (142,176 chunks — the same Drive corpus production serves as
  `projects_folder`). All master-corpus queries here are scoped to
  `drive_archive`; `rag_recall_eval.py` grew a `--project` flag for this
  (default unchanged: `projects_folder`).
* **Embedder:** the local chunk store is 256-dim `model2vec`
  (`minishlab/potion-base-8M` — the pre-July embedder; embedding blobs in the
  local DB are 1024 bytes = 256 float32). Production has moved to 384-dim
  `sentence-transformers/all-MiniLM-L6-v2`. Querying a 256-dim store with a
  384-dim model is impossible without re-indexing all 142k chunks (hours,
  and it would mutate the operator's local corpus), so this audit runs the
  local-consistent potion embedder. sentence-transformers is not installed
  in the local venv; `RAG_EMBEDDING_MODEL` was left unset, which resolves to
  the model2vec backend. Consequence: local absolute recall reflects the OLD
  embedder. The knobs act purely on the post-retrieval merge/rank layer, so
  the config-to-config deltas remain meaningful.
* **GK project:** locally the general-knowledge corpus is the default
  `training_material` project, seeded on boot from the branch's
  `docs/knowledge/*.md` (5 notes, 77 chunks). Production points GK at
  `curated_kb` — different id, same curated note content. The fresh-upload
  contamination mechanism (curated notes outranking uploads) reproduces
  locally with almost identical numbers (see baseline row), which is the
  evidence the local setup is representative.

### 1.2 Score-scale probe (margins 0.05 / 0.15 sanity check)

Top-10 baseline scores for probe queries against `drive_archive`:

* Raw fused scores are cosine-scale, ~0.45-0.75 for ordinary hits.
* `"bill of quantities"` — best GK 0.6745 vs best project 0.5842:
  **gap +0.09**. A 0.05 margin admits this GK chunk; 0.15 blocks it.
* `"vendor performance evaluation procedure"` — all top-10 are project
  chunks (0.68-0.75); GK does not reach the pool. Margins are no-ops here.
* `"extension of time claim notice period"` — GK occupies ALL top-10 slots at
  0.88-1.45: the GK lexical bonus (+0.25/term, cap +1.2) lifts curated notes
  far above every project chunk (best project chunk < 0.88). Gaps of +0.5 or
  more are beyond any sane margin — this regime is what the CAP and BOOST
  knobs exist for.

Verdict: 0.05 and 0.15 are correct magnitudes for the margin knob on this
score scale (meaningful project-vs-GK gaps observed: 0.02-0.15 without the
lexical bonus). **The grid was run exactly as specified — no substitution.**

## 2. The grid

Env combos per configuration (all other env identical across runs; server
killed and relaunched between configs):

| config | RAG_GK_SCORE_MARGIN | RAG_OWN_DOC_BOOST | RAG_GK_TOPK_CAP |
|---|---|---|---|
| cfg0 | — | — | — |
| cfg1 | 0.05 | — | — |
| cfg2 | 0.15 | — | — |
| cfg3 | — | — | 2 |
| cfg4 | 0.05 | 0.1 | — |
| cfg5 | 0.05 | — | 2 |
| cfg6 | 0.05 | 0.1 | 2 |
| cfg7 | 0.15 | 0.1 | 2 |

## 3. Results

| config | MARGIN / BOOST / CAP | doc recall@5 | chunk recall@5 | fresh top-1 /12 | fresh top-3 /12 | calc-intact | EOT case (fresh doc rank) |
|---|---|---|---|---|---|---|---|
| cfg0 (baseline) | - / - / - | 36/100 | 22/100 | 3 | 5 | 3/3 | absent from top-5 |
| cfg1 | 0.05 / - / - | 37/100 | 23/100 | 3 | 6 | 3/3 | absent from top-5 |
| cfg2 | 0.15 / - / - | 39/100 | 24/100 | 4 | 9 | 3/3 | absent from top-5 |
| cfg3 | - / - / 2 | 41/100 | 25/100 | 3 | 12 | 3/3 | rank 3 |
| cfg4 | 0.05 / 0.1 / - | 39/100 | 24/100 | 4 | 9 | 3/3 | absent from top-5 |
| cfg5 | 0.05 / - / 2 | 41/100 | 26/100 | 3 | 12 | 3/3 | rank 3 |
| cfg6 | 0.05 / 0.1 / 2 | 41/100 | 26/100 | 4 | 12 | 3/3 | rank 3 |
| cfg7 | 0.15 / 0.1 / 2 | 41/100 | 27/100 | **6** | 12 | 3/3 | rank 3 |

Notes:
* Local corpus + pre-July embedder (section 1.1) - compare configs to cfg0,
  not to RAG_AUDIT_V2's production absolutes.
* calc-intact = the 3 calculation-intent queries retrieve curated GK top-1
  under intent="calculation" (knobs bypassed). 3/3 under EVERY config - the
  intent exemption works.
* EOT case = the audit's sharpest collision (project doc says 21 days, GK's
  FIDIC note says 28). No config makes it top-1; the GK lexical bonus lifts
  the FIDIC note ~+0.5 above any margin. The cap is what pulls the project
  doc into view at rank 3.

## 4. Recommendation

cfg7 (`RAG_GK_SCORE_MARGIN=0.15`, `RAG_OWN_DOC_BOOST=0.1`,
`RAG_GK_TOPK_CAP=2`) is the recommended configuration:

* best or tied-best on every retrieval metric: doc recall@5 41% (vs 36%
  baseline), chunk recall@5 27% (vs 22%), fresh top-1 6/12 (double the
  baseline 3/12), fresh top-3 12/12 (vs 5/12);
* calc-intact 3/3 - the calculation/standards path is untouched by design
  (intent exemption), so the KB-driven features lose nothing;
* every gain is monotonic across the grid: the cap contributes most
  (recall +5, top-3 +7), the strong margin adds precision (top-1 +2 over
  the weak margin), the boost is a small but consistent tiebreaker.

If a single-knob conservative step is preferred, cfg3 (cap only) captures
most of the recall/top-3 gain with the smallest surface.

The EOT collision case stays GK-won at rank 1 under every config (project
doc reaches rank 3 with the cap). Fixing THAT outright requires either
demoting the GK lexical bonus for doc-lookup intents or the orchestrator
passing real intents into retrieval (the `intent` parameter is plumbed and
waiting) - both are post-pick follow-ups, not blockers.

**Production default remains UNCHANGED. The owner picks from this table (G4 gate).**

## 5. Embedder re-assessment

Doc recall@5 under the best config is 41% - still well below 50%.
The knobs fix GK contamination (ranking), not recall (the retriever
still misses the right document 6 times out of 10 on in-distribution
questions). Per the task rule, the embedder upgrade therefore ENTERS THE
PRE-PILOT LIST. Re-index plan sketch:

1. build the new index OFFLINE into a parallel table/store (chunks are
   already in SQLite; embed all ~142k chunks with the target model at
   ~1-2h on CPU, no downtime);
2. cut over behind a flag (RAG_EMBEDDING_MODEL + store path env), keep
   the old store for instant rollback;
3. re-run this V3 harness (all three evals) against the new index before
   flipping prod - the harnesses in scripts/ are the acceptance gate;
4. production is on all-MiniLM-L6-v2 already for QUERY embedding of new
   uploads - the debt is the 110k+ legacy drive_archive chunks on prod
   (and this machine's local copy), which still carry 256-dim potion
   vectors. Candidate targets: keep MiniLM-L6 (dim-consistent, cheapest)
   or evaluate BGE-small / GTE-small on the same harness.

## 6. Artifacts

* `evals/rag_audit_v3/cfg{0..7}.json` — committed per-config summaries
  (recall metrics, per-case fresh results, calc-intact status, EOT winner).
* `data/learning/rag_audit/v3_cfg{N}_{recall,fresh,calc}.json` — full
  artifacts with per-question detail (gitignored, operator machine only).
* `scripts/rag_calc_intact_eval.py` — new committed harness.
* `scripts/rag_recall_eval.py` — grew `--project`; `rag_fresh_upload_eval.py`
  grew `--out`. Defaults unchanged.

## 7. cfg7 + lexical fold (TASK 2, measured 2026-07-06)

Follow-up to the section-4 EOT residual: `RAG_GK_LEXICAL_FOLD` (new bool
flag, default OFF) folds the GK lexical bonus inside the H1 margin gate —
the margin comparison runs on each GK chunk's RAW fused score (lexical
bonus subtracted); survivors keep the bonus for ordering. Same intent
exemption as the H knobs (applies for intent None / DOC_LOOKUP; bypassed
for CALC_KB). Same local harnesses, same corpus, same seeded draw as the
grid above (server relaunched with the model2vec/256-dim embedder the
grid ran with — the venv has since gained sentence-transformers, which
was shimmed out of the server process to keep the environment identical).

| config | MARGIN / BOOST / CAP / FOLD | doc recall@5 | chunk recall@5 | fresh top-1 /12 | fresh top-3 /12 | calc-intact | EOT case (fresh doc rank) |
|---|---|---|---|---|---|---|---|
| cfg7 + fold | 0.15 / 0.1 / 2 / on | 41/100 | 27/100 | **12** | 12 | 3/3 | **rank 1** |

* The EOT case — the collision no grid config could win — now puts the
  user's own contract note at rank 1 (score 0.7512); the GK FIDIC note is
  absent from the top-10 entirely (its raw score fails the 0.15 margin;
  only its lexical bonus ever carried it past the bar).
* Fresh top-1 doubles again vs cfg7 (6 -> 12 of 12): every collision case
  where a bonused GK note edged out the upload now resolves to the upload.
* Recall and calc-intact are unchanged vs cfg7 (41/27, 3/3) — the fold
  touches only lexically-bonused GK chunks at the margin gate, so
  master-corpus recall and the calculation/standards path are untouched.

**This is the shipping config pending the prod flip:**
`RAG_GK_SCORE_MARGIN=0.15 RAG_OWN_DOC_BOOST=0.1 RAG_GK_TOPK_CAP=2
RAG_GK_LEXICAL_FOLD=1`. Production defaults remain unchanged until the
flip.
