# Retrieval recall floor: measured diagnosis + reranker verdict (2026-08-02)

Decision record for the KNOWN_LIMITATIONS §1 recall-floor investigation.
Everything below is measured against LIVE production (drive_archive,
BGE-384), not assumed. Sample: seed-42, `scripts/rag_recall_eval.py` ground
truth (`training_scenarios_drive_archive_v2.jsonl`).

## Headline numbers

| measurement | value |
|---|---|
| doc-recall@5, live, n=100 | **41%** (chunk 36%) |
| doc-recall@50, live, n=60 | **53%** (chunk 52%) |
| cross-encoder rerank of live top-50 -> top-5 (offline) | **43%** — WORSE than the 47% baseline on the same n=60 |
| ground-truth docs absent from the live corpus registry | **9/60 (15%)** — eval ceiling is 85%, not 100% |

Proof artifacts: `data/learning/rag_audit/rerank_proof.json` and
`chunk_gap_probe.txt` (verbatim-phrase probe transcript).

## What the floor actually decomposes into (n=60, never-found@50 = 28)

1. **Corpus gap — 9 queries (15% of sample).** The target document is not in
   the live registry at all (2,737 docs paged from
   `/v1/projects/drive_archive/documents`). Consistent with the 2026-07-12
   re-encode that stalled at 95% (6,485 chunks / a coherent the client project block never
   re-ingested) plus eval ground truth authored against the June corpus.
   NO retrieval change can answer these. Fix = corpus reconciliation against
   the Drive originals + finishing the backfill (owner-sequenced; the PG
   tier bump that gated it is already done).
2. **Vocabulary mismatch — 11 queries.** The doc's chunks ARE in the store
   (a verbatim phrase from the ground-truth chunk retrieves them), but the
   natural-language question never surfaces them at k=50. Real headroom, but
   the lever is query/first-stage work (query expansion, domain embedder
   fine-tune), not ranking.
3. **Near-duplicate sheet ambiguity — 7 queries.** Even the verbatim
   title-block phrase cannot isolate the target because hundreds of drawing
   sheets share the same text ("Example Ring Road (South) — ORIENTATION —
   DATUM"...). Irreducible by any text ranking; would need metadata-aware
   retrieval (sheet-number extraction at query time; `identifier_search`
   already covers part of this).
4. **Ranked 6th–50th — the remainder.** The only slice a second-stage
   reranker can touch, and the measured verdict on that follows.

## Reranker verdict: measured NEGATIVE — do not enable with the default model

`cross-encoder/ms-marco-MiniLM-L-6-v2` reranking live top-50 into top-5:
doc-recall **47% -> 43%**, chunk-recall **43% -> 38%** (n=60). MS-MARCO
training (web passages) transfers badly to OCR-flavoured construction chunks
(BOQ tables, drawing title blocks, template boilerplate) — it reorders
confidently and wrongly. This is why the flag ships **OFF** and why
`RAG_RERANKER=1` with the default model is a documented footgun.

Also measured as a dead lever: **re-embedding**. The potion-256 ->
bge-small-384 migration (2026-07-12) left recall@5 at ~41%. A third embedder
migration would spend a production-corpus rewrite on a lever that measurably
does not move.

## What ships (dormant, repo precedent: layered RAG)

Flag-gated, tested, model-agnostic second-stage infrastructure — so a
DOMAIN-suitable cross-encoder can be trialled later by env change alone:

- `app/core/rag/reranker.py`: `enabled()` / `candidate_depth(k)` /
  `rerank(query, chunks, k)`. Lazy singleton, HF_HOME cache, every failure
  degrades to the original cosine order.
- Retriever hook: with `RAG_RERANKER=1` the final-selection loop collects
  `RAG_RERANK_CANDIDATES` (default 50) through the SAME gates (noise filter,
  revision suppression, GK cap) and the cross-encoder picks the top-k.
  Gates run first — a suppressed chunk can never be resurrected.
- Flag OFF (shipped default) = byte-identical pipeline, zero boot memory.

| env | default | note |
|---|---|---|
| `RERANK_ENABLED` | off | canonical flag; keep OFF until a domain-suitable model is validated |
| `RAG_RERANKER` | off | alias; ignored when `RERANK_ENABLED` is set |
| `RAG_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | measured NEGATIVE on drive_archive — replace before enabling |
| `RAG_RERANK_CANDIDATES` | 50 | hybrid top-50; never below k |

Enable-later checklist: (1) pick a candidate model sized for the box,
(2) re-run `scratchpad`-style offline proof against live top-50 FIRST,
(3) enable only on a measured lift, (4) watch 512Mi web-box memory
(bge-small is already resident) and chat latency, (5) re-run
`rag_recall_eval.py --sample 100 --seed 42` live to confirm.

## The honest priority order for recall now

1. **Corpus completeness** (owner): reconcile registry vs Drive originals;
   finish the 6,485-chunk backfill. Worth up to ~15pts of the eval gap and —
   unlike everything else here — it is missing CLIENT DATA, not a metric.
2. Query-side work (vocabulary mismatch): post-pilot investigation.
3. Metadata-aware sheet retrieval (sibling ambiguity): post-pilot.
4. Domain reranker via the shipped dormant hook: only after 1–2, with proof.
