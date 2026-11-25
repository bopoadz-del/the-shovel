# Postgres restore drill — 2026-11-15

The first three runs of the Neon restore drill. They ran through
`.github/workflows/restore-drill.yml` (`workflow_dispatch`), which runs
`scripts/restore_drill.py --create --minutes-back 60`. The previous record,
`docs/RESTORE_DRILL_20260805.md`, was a Render point-in-time recovery and
could not compare rows.

## What the drill does (as it ran on 2026-11-15)

1. Creates a scratch Neon branch of the live database at a restore point
   60 minutes back, with a read-only endpoint, and times how long the
   endpoint takes to serve.
2. Opens the restore branch and live, each in a READ ONLY transaction, and
   compares row counts for the tables in `TABLES`, documents per ingest
   status, and the chunk rows of the serving chunk table.
3. Deletes the scratch branch and prints a verdict. A table that cannot be
   counted on the restore branch (`unreadable`) or is empty there while
   live has rows (`empty_on_restore`) fails the drill.

It writes to neither database and prints no connection string, row or
row text.

## Runs

| run | created (UTC) | commit | result | cause |
|---|---|---|---|---|
| 43709753474 | 11:36Z | 2eaa3f6b | failure, exit 2 | `RESTORE_DRILL refused=neon_http_400` — the Neon API refused a call; the line did not say which one |
| 43721443808 | 13:54Z | 52f001d3 | failure, exit 2 | `RESTORE_DRILL refused=neon_http_400:GET:/projects` — the unscoped project list is refused for an organisation-scoped key |
| 43732275921 | 15:52Z | dcb38a2e | **success** | verdict ok |

Fixes between the runs, both on `scripts/restore_drill.py`:

- #341 (bf6dea49): a refused Neon call names its route, never ids or body.
  This is what turned the first failure's bare `neon_http_400` into the
  second failure's `GET:/projects`.
- #353 (bf0de87c): list projects per organisation when the unscoped list
  is refused.

Both failures stopped on a refused Neon API call; neither reached the
row comparison.

## Measured result (run 43732275921)

| checkpoint | result |
|---|---|
| Restore point | 2026-11-15T14:55:03Z |
| Restore endpoint serving (`ready_s`) | **6.5 s** |
| Comparison done (`compared_s`) | 17.8 s |
| Scratch branch deleted (`deleted_s`) | 18.1 s |
| Verdict | `ok`, `unreadable=-`, `empty_on_restore=-` |

| table | restore | live | live − restore |
|---|---|---|---|
| users | 48 | 48 | 0 |
| projects | 56 | 56 | 0 |
| documents | 6971 | 6971 | 0 |
| conversations | 13851 | 13851 | 0 |
| messages | 31857 | 31857 | 0 |
| project_facts | 35 | 35 | 0 |
| hydration_runs | 103 | 103 | 0 |
| ingestion_jobs | 44 | 44 | 0 |
| audit_events | 0 | 0 | 0 |
| chunk table | 42530 | 42534 | 4 |

Documents with status `EXTRACT_FAILED`: 947 on both sides.

The 4 extra chunk rows on live are rows written after the restore point;
the drill reports that delta and does not judge it.

## What this drill did NOT verify — stated plainly

- **That the app can serve from the restored branch.** The comparison
  counts rows. `count(*)` succeeds on a table whose columns no longer match
  the code, so `ok` here means the rows are there, not that the app can
  read them. `ready_s` is the Neon endpoint answering, not the service.
  Draft PR #571 adds a load of each of the app's own models on both sides;
  it was not merged when this record was written.
- **Recency.** Nothing schedules the drill; this run is the last proof
  until it is dispatched again.
- **Uploaded files.** The drill covers Postgres only.
