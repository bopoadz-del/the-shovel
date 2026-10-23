# RAG Retrieval Re-Evaluation (V2)

**Audited:** 2026-07-05
**Environment:** production `https://the-fork.onrender.com`, commit `7cbd96d`
**Baseline:** RAG_AUDIT.md, 2026-06-23 (commit `221ac51`) — doc recall@5 = 31 %, chunk recall@5 = 20 % on a 100-question sample
**Scope:** re-measure retrieval recall@5 against the same corpus and question set family; add a new "fresh upload wins" eval class. Measure-only: no embedder, ranking-weight, or GK-bonus changes were made.

---

## 1. What changed in the retrieval stack since the baseline

Between 2026-06-23 and this audit the stack gained (all previously shipped, none altered by this audit):

* embedder moved from 256-dim `model2vec`/potion to `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — the baseline audit's recommendation #1
* hybrid BM25 + vector retrieval with reciprocal-rank fusion
* per-item 5-field RAG chunks for BOQ items (2026-07-01)
* GK re-pointed from the polluted `training_material` blend to the curated KB (2026-07-04)

Scores in this report are RRF-fused ranks, not cosine similarities — **top-1 "score" values are not comparable to the baseline's ~0.69 cosine numbers.**

## 2. Method

Identical query surface to the baseline: `POST /v1/rag/search` with `project_id=projects_folder` (the master corpus backing `master_corpus`), `k=5`. Ground truth is the `drive_archive:<doc_id>:<chunk_index>` source recorded per training question.

Two measurements:

1. **Paired 30-question re-run** — the exact 30 queries stored in the June-23 artifact `sample_retrieval_recall_projects_folder.json`, re-executed verbatim. Same questions, same ground truth: the delta is attributable to the retrieval stack, not sampling noise.
2. **Seeded 100-question sample** — 100 questions drawn (seed 42) from `training_scenarios_drive_archive_v2.jsonl`, the same canonical in-distribution set the baseline sampled from. Not the identical 100 questions (the baseline's sample was not persisted), so treat deltas here as indicative.

Runner: `scripts/rag_recall_eval.py` (new, committed with this audit; reusable as the recall@K gate the baseline recommended). Queries are paced 3 s apart — an unpaced first attempt saturated the single-worker box until Render's 5 s health check failed and restarted the instance (see section 6).

## 3. Results — recall@5 vs baseline

### 3.1 Paired 30-question re-run (same questions as June-23)

| metric | June-23 | 2026-07-05 | delta |
|---|---|---|---|
| doc recall@5 | 8/30 = 27 % | 9/30 = 30 % | +3 pts |
| exact-chunk recall@5 | 4/30 = 13 % | 8/30 = 27 % | **+14 pts (2×)** |

### 3.2 Seeded 100-question sample

| metric | June-23 (n=100, unpersisted sample) | 2026-07-05 (n=100, seed 42) |
|---|---|---|
| doc recall@5 | 31 % | 29 % |
| exact-chunk recall@5 | 20 % | 24 % |

### 3.3 What actually moved (paired transitions, n=30)

Doc recall churned rather than improved: 3 kept, 6 gained, 5 lost. The gains and
losses are not random:

* **Gained** — identifier-bearing drawing queries. 4 of the 6 doc-level gains are
  "what datum and coordinate system are used for drawing IP-..." style questions;
  the BM25 leg + identifier-aware boost now match drawing numbers the old
  embedder collapsed.
* **Lost** — pure-semantic procedure/contract lookups (Rapid Award Process
  PRC-963A, vendor performance score, HSE Observation Report, commencement
  date). These previously surfaced on cosine similarity and now lose rank in
  the fused ordering.

Exact-chunk recall genuinely improved (2 kept, 6 gained, 2 lost): when the
right document is found at all, the right chunk inside it is found far more
often than in June.

**Net read:** the stack is markedly better at exact-reference retrieval and
slightly worse at fuzzy procedure lookup; overall recall is still low (~30 %),
so the baseline audit's verdict — do not trust RAG for fine-grained document
lookup without verification — stands.

## 4. Results — fresh-upload-wins (new eval class)

Setup: a dedicated eval project (`bc812f36`, "RAG Audit V2 - Fresh Upload
Eval") received 12 small .txt notes whose topics deliberately collide with
GK/master-corpus content (lighting wattage, FIDIC notice periods,
design-review status codes, BOQ rates, curing, HSE, EVM thresholds) but whose
values are project-unique. PASS = a chunk from the uploaded note ranks top-1
for the project-scoped natural-language query. Runner:
`scripts/rag_fresh_upload_eval.py`.

| metric | result |
|---|---|
| fresh doc at rank 1 | **3 / 12** |
| fresh doc in top 3 | 4 / 12 |
| fresh doc absent from top 5 entirely | 5 / 12 |

Re-queried after an extra multi-minute settle: identical — this is **ranking,
not indexing latency**. The losing docs ARE indexed: querying case 1's note
text near-verbatim returns it at rank 1 (score 2.84). The natural-language
question loses.

**What wins instead:** the curated GK knowledge-base notes. Two GK documents
(the construction fact-table note and the CESMM4/POMI BOQ guide) occupy 25 of
the 36 top-3 slots across the 12 cases; the FIDIC clause notes take most of
the rest. Mechanism (from `retrieve_with_filter`'s design, not conjecture):
GK candidates are merged with project candidates and re-ranked purely by
score — the active project wins only ties. The GK notes are dense,
keyword-rich tables that score highly against almost any construction query,
so they systematically outrank a short freshly uploaded project note.

**Consequence for the pilot:** a user who uploads a document and asks about
its content will, most of the time, get an answer grounded in the general KB
rather than their own document — including cases where the two disagree (the
EOT notice-period case: project says 21 days, GK's FIDIC notes say 28).

This audit is measure-only (ranking weights and the GK merge are explicitly
out of scope), so no fix was applied. Candidate directions for a follow-up
decision: score-margin gating on GK chunks (GK only when the project's best
chunk is weak), a fresh-document recency boost, or capping GK chunks per
result set.

## 5. Verdict

| dimension | June-23 | now |
|---|---|---|
| doc recall@5 (master corpus) | 31 % | 29 % — flat |
| exact-chunk recall@5 | 20 % | 24-27 % — improved |
| identifier-bearing queries (drawing codes) | weak | clearly improved (BM25 + identifier boost) |
| semantic procedure lookups | weak | slightly worse (fusion re-ranking) |
| fresh-upload-wins | not measured | **3/12 — the headline problem** |

Retrieval headline recall is essentially unchanged at ~30 % — still not
production-trusted for precise document lookup. The stack's character
changed: exact references improved, fuzzy lookups regressed slightly. The
new measurement exposes the sharpest issue: **project-scoped retrieval is
dominated by the curated GK corpus, so fresh project uploads lose to general
knowledge in their own project.** That is the highest-leverage retrieval
problem to fix next, ahead of further embedder work.

## 6. Operational findings (incidental)

* **Retrieval bursts can kill the box.** ~17 rapid consecutive `/v1/rag/search` calls over the 110k-chunk corpus starved the single-worker instance until the platform health check timed out; Render restarted it (event `server_failed` / `HTTP health check failed (timed out after 5 seconds)`, 20:20:01Z). Recovery was automatic in ~25 s. Any client that fires unthrottled searches can reproduce this. Mitigations to consider (out of scope for this audit): a second uvicorn worker, request queuing, or rate limiting on the search route.
* Boot-time warning seen on restart: `init_db: spurious master-corpus row cleanup failed` ending in a Sentry-related `RecursionError` (doc_index.init_db -> _ensure_db -> init_db loop). Non-fatal but worth a look.

## 7. Artifacts

* `data/learning/rag_audit/v2_baseline30_rerun.json` — paired 30-question re-run, per-question detail
* `data/learning/rag_audit/v2_sample100_seed42_0-50.json`, `_50-100.json` — seeded 100-question sample
* `data/learning/rag_audit/v2_fresh_upload_wins.json` — fresh-upload-wins cases
* `scripts/rag_recall_eval.py`, `scripts/rag_fresh_upload_eval.py` — reusable runners

`data/` is gitignored, so the JSON artifacts live on the operator machine only
(same as the June-23 audit's). Unlike June, every measurement here is
re-runnable from the repo alone: the 30-question set re-derives from the June
artifact when present, the 100-question sample is a seeded draw
(`--sample 100 --seed 42`), and the fresh-upload cases are hardcoded in the
runner (re-query with `--query-only --project bc812f36`).
