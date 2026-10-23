# Fleet Corpus Forensics — 2026-07-12

**Question:** the 5-shard / drive_archive ingestion wrote ~1,500–2,997 files (~140k chunks); v2 canonical has only 53 docs / 10,502 chunks. Where did the writes go, and are they recoverable?

## Where the writes are
- **the-fork-db (canonical Postgres):** only two chunk stores — `chunks` = **0 rows**, `chunks_v2` = 10,502 (53 docs, 100% BGE-384). **No `drive_archive`, no shard tables, no orphaned namespace.**
- The `drive_archive` corpus was **`project_id='drive_archive'` in the OLD `chunks` table**, migrated from a local SQLite via `/v1/admin/corpus/bulk-insert` (admin.py:912, audit_training_data.py:73). The "clean rebuild" **wiped `chunks` (now 0 rows)**.
- **The source SURVIVES locally** (on this machine, `The_Fork/data/`):
  - `data/rag/vectors.db` — chunks total 143,472, **drive_archive = 139,949**
  - `data/the_fork.db` — chunks total 142,218, **drive_archive = 142,176** (+ chunks_v2 21,672)
  - `data/the_fork_cleaned.db` — drive_archive = 142,176 (+ chunks_v2 10,931)
- Other Postgres instances: `cerebrum-db` (Feb, old), `the-fork-db-drill-20260612` (drill copy) — both **firewalled** (ipAllowList excludes this IP; only the-fork-db allows 51.39.1.13). Not queried; the local SQLite already answers the question.

## Embedding identity — the verdict driver
- **drive_archive embeddings = 256-dim** (1024 bytes / 4 = 256 float32) → **potion/model2vec (old embedder).**
- **v2 canonical = BGE-384.** The `chunks` table has no `embedding_model`/`embedding_dim` columns (old schema).
- **256 ≠ 384 → INCOMPATIBLE.** A namespace attach would inject dimension-mismatched vectors into v2 and corrupt retrieval (the exact failure the embedding-identity guard exists to prevent).

## VERDICT
- **Namespace attach / migrate: WRITE-OFF.** The drive_archive *embeddings* are the wrong embedder (256-dim potion). No legitimate attach — identity does not match. **No bulk side-door.**
- **The TEXT is fully recoverable.** ~140k already-chunked `drive_archive` rows (chunk_id, doc_id, text) survive in local SQLite. **Re-ingestion = re-embed that existing text with BGE-384 → chunks_v2.** No re-parsing/re-download needed (text is done) — same shape as the batch-14 re-embed: local BGE encode → stream to Postgres, resumable.
- **Cost:** ~140k chunks to re-embed locally (~minutes on 12-thread CPU) + streamed INSERTs. Cheap. But it triples the corpus (10k → ~150k) and mixes the broad Drive archive into retrieval — which will interact with the T5 GK/precision work.

## Recommendation (the owner's scope call — do NOT auto-start)
1. **If the pilot is the client project-scoped:** leave drive_archive out; the 53-doc the client project corpus is the pilot.
2. **If the pilot needs the full Drive corpus:** re-embed the local `drive_archive` text (BGE-384) into a **dedicated project** (not merged into client), and settle T5 precision first (150k mixed chunks will worsen GK/ranking contamination).
**Report-before-re-ingestion honored: paused here for the scope decision.**
