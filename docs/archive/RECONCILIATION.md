# Phase-1 Corpus Reconciliation (v2 namespace) — 2026-07-12

Assembled from live prod Postgres (`the-fork-db`, `chunks_v2`, BGE-384, namespace v2).

## Headline
The v2 corpus is **well-formed and the client project-scoped**, not the full Drive archive. Per-document chunking is healthy; the audit's "<2 chunks/file" alarm was a wrong-denominator artifact (6,178 *expected Drive files* vs 53 *actually-ingested docs*).

## Counts (canonical store = `chunks_v2`)
| project | docs | chunks |
|---|---|---|
| example_infra_pack | 39 | 9,857 |
| curated_kb (GK) | 8 | 605 |
| example_estate_2 | 6 | 40 |
| **total** | **53** | **10,502** |

- Chunks/doc: **min 1, avg 198, max 1,936**. Large contracts/specs are fully chunked (AB-2023-101 Vol 3 = 1,323 + 1,198; Accelerated Work Programme = 1,936; PMC contract = 776; demolition contract = 750). **No truncation of large PDFs.**
- **Duplicate chunk_ids: 0.**

## Completeness (4a)
- **the client project package (the pilot core): ingested** — 39 docs / 9,857 chunks, incl. the AB-2022-202 demolition contract, AB-2023-101 Vol 1-3, PMC/PSA contracts, drawings, programmes.
- **Coverage gap:** only 53 docs are in v2. The broader Drive archive (a prior `drive_archive` of ~2,997 docs / ~139,949 chunks per session memory) is **NOT** in this DB — there is no `drive_archive` table. Individual non-the client project project shells (Ha Long Xanh, etc.) are near-empty (UI: "still being indexed"). **If the pilot requires the full multi-project Drive corpus, that ingestion has not run into v2.** PARKED — needs a scoped re-ingest decision (pilot may be the client project-only by design).

## Chunk-math sanity (4b)
Healthy per-doc. Low-chunk docs are legitimately low-text: `.pptx` presentations (1 chunk), graphical drawing PDFs (3-4). **Two real quality flags:**
1. **Uploaded RFPs under-extracted** — `Example Client - Request for Proposals.docx`, `RFP Appendix B.xlsx`, `RFP_Kenya_200MW_DataCenter.docx` each produced only **2 chunks**. A multi-page RFP yielding 2 chunks = the docx/xlsx text extractor grabbed little. **This is the likely cause of weak RFP/attachment reasoning.** PARKED-with-evidence → re-extract this class (targeted).
2. **`.kmz` over-chunked** — `PMCM Status Tracking.kmz` = 810 chunks (map XML coordinate noise polluting retrieval). PARKED → exclude `.kmz` or extract its metadata only.
3. Duplicate *documents* (not chunks): the client RFP was uploaded 2-3×. Doc-level dedup opportunity; chunk_ids remain unique.

## Duplicate-chunk check (4c)
`SELECT COUNT(*) FROM (SELECT chunk_id FROM chunks_v2 GROUP BY chunk_id HAVING COUNT(*)>1)` = **0**. No fleet-overlap duplication in v2.

## Database canonicalization (4e)
- **Canonical: `chunks_v2`** (BGE-384, namespace v2) — 10,502 rows.
- Legacy `chunks` table = **0 rows** (retired in place, safe).
- The "empty SQLite" incident: the app silently falls back to `sqlite:///{DATA_DIR}/the_fork.db` when `DATABASE_URL` is unset. **`DATABASE_URL` is now SET on prod** and must stay set. **Follow-up safeguard (recommended, not yet shipped): fail-loud at startup if `ENV=production` and `DATABASE_URL` is unset**, so the SQLite fallback can never silently strand the corpus again.

## Tier 2/3 (4f)
Not separately tracked in v2. Only tier-1 the client project package + GK notes present. Tier 2/3 Drive folders: NOT ingested into v2 (see Coverage gap). PARKED.
