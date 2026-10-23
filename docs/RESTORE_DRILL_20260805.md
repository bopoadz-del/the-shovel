# Postgres restore drill — 2026-08-05

First rehearsal of the recovery procedure in `docs/backup-and-recovery.md`.
Before this drill, the RTO figures in that doc were estimates; nothing had
ever actually been restored.

## What was done

1. Point-in-time recovery of the live database
   (`dpg-d8m22mcm0tmc73b04elg-a`, basic_256mb, PITR window since
   2026-07-31) requested via `POST /v1/postgres/{id}/recovery` with a
   restore point one hour in the past.
2. Render provisioned a NEW instance — the live database is never touched
   by a restore: `dpg-d9p7d7oae00c73ec91lg-a`, database `thefork_0nyj`.

## Measured results

| checkpoint | result |
|---|---|
| Recovery request accepted | immediately (single API call) |
| Recovered instance status `available` | **≤ 8 minutes** after the request (created 23:32:47Z; already `available` at first poll 23:41Z) |
| Postgres engine on the copy answering | confirmed — the engine responds on 5432 (its TLS-required rejection of a non-SSL probe is a *response from Postgres*, proving the recovered cluster booted) |
| Plan of the recovered copy | **pro_4gb** — Render recovers onto a larger plan than basic_256mb; the copy bills hourly until deleted |

## What this drill did NOT verify — stated plainly

Row-level integrity of the recovered data (count comparison against live)
was not performed in this drill: the operator's ISP silently blackholes
outbound Postgres traffic on port 5432 (TCP connects, protocol data is
dropped — verified with a raw socket test), so no external client on this
network can query either instance directly. The `available` status plus a
responding Postgres engine is strong but indirect evidence.

To close that gap on the next drill (either works):
- run `psql "<recovered external connection string>" -c "SELECT count(*) FROM chunks"` from any network that passes 5432 (mobile hotspot suffices), or
- use the query tab on the recovered instance's dashboard page.

## Cleanup

The recovered copy must be deleted after the drill — it is a pro_4gb
instance billing hourly. Deletion is via
`DELETE /v1/postgres/dpg-d9p7d7oae00c73ec91lg-a` or the instance's
dashboard → Settings → Delete. The drill scripts hard-refuse to touch the
live instance id.

## Lessons folded back

1. **Restore-to-new works and is fast** — the ≤8-minute figure beats the
   RTO < 30 min target in `backup-and-recovery.md`.
2. **The recovered copy is not the same plan** as the source. Budget-wise
   this is fine for a drill measured in minutes, but a real cutover keeps
   paying pro_4gb until the plan is changed or the old instance replaced.
3. **The operator network cannot reach Render Postgres externally at all**
   (ISP 5432 filtering). Any runbook step that says "psql from your
   machine" silently fails on this network — every DB-touching recovery
   step must go through the dashboard, Render-side execution, or another
   network. Added to the runbook considerations above.
