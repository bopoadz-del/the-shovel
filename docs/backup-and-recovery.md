# Backup and Recovery

This document covers backup coverage and the recovery procedure for the
production platform at `the-fork-jn3t.onrender.com`. Owners: the operator
running the live deploy. Audience: anyone who needs to restore service
after data loss.

## What lives where

| Asset | Storage | Backed up? | Recovery target |
|---|---|---|---|
| Projects, documents, conversations, messages, agent_facts | Postgres (`dpg-d8m22mcm0tmc73b04elg-a`) | YES — Render daily logical backups (7-day retention) | RTO < 30 min, RPO 24h |
| RAG chunks + embeddings (~142k rows, drive_archive corpus) | Postgres `chunks` table | YES — same daily backups | RTO < 30 min, RPO 24h |
| Uploaded source documents (PDF/DOCX/XLSX) | Render persistent disk `dsk-d8hdc6mk1jcs739rq69g`, mount `/app/data/uploads` | YES — Render daily disk snapshots (7-day retention) | RTO ~ 1h, RPO 24h |
| `.secret_key` (JWT signing fallback) | Render disk `/app/data/.secret_key` AND `SECRET_KEY` env var | Env var is durable; disk fallback also snapshotted daily | n/a if env set, RTO < 1h if disk-only |
| `audit.log` (compliance trail) | Render disk `/app/data/audit.log` | YES — daily snapshot | RPO 24h |
| `rag_audit.jsonl` (RAG explainability) | Render disk `/app/data/logs/rag_audit.jsonl` | YES — daily snapshot | RPO 24h |
| Google Drive OAuth refresh tokens | Render disk `/app/data/google_drive_token_<user>.json` | YES — daily snapshot | RPO 24h (worst case: each user re-OAuths) |
| `evidence_vault.json` | Render disk `/app/data/evidence_vault.json` (per `.env.example`) | YES — daily snapshot | RPO 24h |
| `learning_engine.json` (hydration scheduler state) | Render disk `/app/data/learning_engine.json` | YES — daily snapshot | RPO 24h |
| Vector store fallback (SQLite) | Used only when `DATABASE_URL` unset (dev only) | N/A on prod | n/a |

## Current snapshot state — verified live

Listed via `GET https://api.render.com/v1/disks/dsk-d8hdc6mk1jcs739rq69g/snapshots`:

- 7 daily disk snapshots present at audit time (2026-06-15 → 2026-06-21)
- Snapshot cadence: daily, automatic, included in the disk plan
- Retention: 7 days rolling (oldest expires as newest lands)
- Triggered: Render-managed, no app-side cron required

Listed via `GET https://api.render.com/v1/postgres/dpg-d8m22mcm0tmc73b04elg-a`:

- Plan: `basic_256mb`, Postgres 16, Oregon region
- Disk size: 15 GB (currently ~445 MB used after the drive_archive
  migration — 14.5 GB / 97 % headroom)
- Backup: daily logical backups, 7-day retention (Render basic-tier
  default; PITR is Pro+ only — known limitation, see "Upgrade path")

## Recovery procedure — disk snapshot

When to use: corrupted volume, accidental file deletion, single-file
recovery of a snapshotted upload.

1. List available snapshots:

   ```bash
   curl -H "Authorization: Bearer ${RENDER_API_KEY}" \
     "https://api.render.com/v1/disks/dsk-d8hdc6mk1jcs739rq69g/snapshots" \
     | python -m json.tool
   ```

   Each entry has `createdAt` and `snapshotKey`. Snapshots are listed
   newest first.

2. Restore via the Render dashboard:
   `https://dashboard.render.com/web/srv-d8hdc6ek1jcs739rq5sg`
   → `Disks` tab → select disk `the-fork-data` → `Snapshots` → click
   the desired date → `Restore`.

   Render restores by attaching the snapshot as a new disk and swapping
   it in. The service redeploys automatically; uptime impact is the
   time of a normal deploy cycle (~3-5 min).

3. After restore, verify:

   ```bash
   curl -H "Authorization: Bearer ${ADMIN_KEY}" \
     "https://the-fork-jn3t.onrender.com/v1/admin/debug/pilot-preflight" \
     | python -m json.tool
   ```

   Check `row_counts` matches expectation, and verify the LIVE vector
   store — not the retired legacy table:

   - `active_chunk_table` is the namespaced store the retriever reads
     (`chunks_v2` unless `RAG_VECTOR_NAMESPACE` says otherwise).
   - `row_counts[active_chunk_table]` is non-zero. The legacy `chunks`
     table has been empty since the v2 migration, so a zero there is
     expected and is NOT evidence of a bad restore.
   - `embedding_dim_ok` is `true`. `false` means the restored store's
     vector width disagrees with the configured embedder — stop and
     resolve before serving, because mixed widths corrupt retrieval.
     `null` means the embedder was not warm enough to compare (issue any
     retrieval request, then re-check); treat `null` as unverified, not
     as pass.

   Sample a known document via `/v1/admin/debug/doc-extract` to confirm
   files are readable.

## Recovery procedure — Postgres backup

When to use: corrupted database, accidental DELETE/DROP, schema
migration rollback.

1. List backups:
   `https://dashboard.render.com/d/dpg-d8m22mcm0tmc73b04elg-a/backups`
   → daily entries with timestamps.

2. Restore creates a NEW Postgres instance from the backup. Render does
   NOT overwrite in place. Workflow:
   - Click `Restore` on the chosen backup
   - Render provisions a new instance, e.g. `dpg-NEW-id`
   - Copy the new internal/external connection string
   - Update `DATABASE_URL` env var on the web service
     (`srv-d8hdc6ek1jcs739rq5sg`)
   - Trigger a redeploy
   - Verify with `/v1/admin/debug/pilot-preflight`
   - When confirmed working, delete the OLD instance to stop billing

3. After restore, if disk and Postgres are out of sync (Postgres rolled
   back to T-1 but disk is at T-0), the safest move is to restore BOTH
   to the same timestamp — uploaded files reference `documents.id`
   rows, so a doc row missing from a restored Postgres while the file
   still exists on disk creates an orphan.

## Upgrade path (when daily/7-day stops being enough)

| Trigger | Action |
|---|---|
| Need point-in-time recovery (sub-day RPO) | Bump Postgres plan to Pro+ — enables PITR. Approx cost: jumps from $7/mo to ~$95/mo at the cheapest Pro tier. |
| Need longer retention | Schedule a daily `pg_dump` to S3/R2 via a Render cron service. The S3 bucket retention policy then controls duration. |
| Need off-Render redundancy | Schedule a daily upload sync of `/app/data/uploads` to Cloudflare R2 (no egress fees) via boto3 + a Render cron. Doc corpus restoreable even if all of Render is unreachable. |

The above are deferred until justified by an actual SLA requirement.
For the current pilot, daily snapshots + daily backups are sufficient.

## What this document does NOT cover

- Application-state-only restores ("revert THIS conversation"): manual
  via Postgres SQL on the running instance, not via this procedure.
- Frontend asset recovery: assets are built from the repo at deploy
  time; Git is the source of truth, no separate backup needed.
- Secret rotation: see the security runbook (separate doc, TBD).
