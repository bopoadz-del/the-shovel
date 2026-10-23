# Data Governance

How The Fork handles client documents — Roadmap V2 · Epic 6.

This document describes actual, implemented behaviour. It is kept honest: it
does not claim controls that are not in the code.

## What is stored, and where

| Data | Location | Notes |
|------|----------|-------|
| Uploaded documents | `DATA_DIR` (default `./data`) | Original files, UUID-prefixed |
| App persistence (users, projects, RAG, etc.) | `DATABASE_URL` when set (PostgreSQL + pgvector), else `DATA_DIR/the_fork.db` (SQLite) | Unified schema via Alembic migration 0001 |
| Custom document types | `DATA_DIR/custom_document_types.json` | |
| Audit log | `DATA_DIR/audit.log` | Append-only JSONL |

When `DATABASE_URL` points at managed Postgres, relational data leaves the app
host; uploaded document **files** still live under `DATA_DIR` unless you mount
object storage separately. There is no third-party data processor for chat
inference beyond the LLM providers you configure.

## Encryption at rest

Uploaded document files in `DATA_DIR` can be encrypted at rest with symmetric
encryption (Fernet — AES-128-CBC with an HMAC-SHA256 authentication tag).

- **Opt-in via `DATA_ENCRYPTION_KEY`.** Set this env var to a valid Fernet key
  to enable encryption. Generate one with:
  `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`
- **When the key is set**, every newly uploaded document (`POST /v1/upload`,
  `POST /v1/projects/{id}/documents`, `POST /ingest`, `POST /ingest-via-block`)
  is written to disk as ciphertext. Reads (OCR, PDF text extraction, the
  document engine) transparently decrypt to a short-lived temp file that is
  removed immediately after processing.
- **When the key is unset**, encryption is off and files are stored in the
  clear — this is the default and matches the platform's original behaviour.
- **Backward compatible.** Files written before encryption was enabled stay
  readable: the reader detects whether a file on disk is actually a Fernet
  token and passes legacy plaintext files through untouched. Enabling the key
  does **not** retroactively encrypt or break existing files.
- **Scope.** Encryption covers document *files*. The SQLite metadata DB
  (`projects.db`), `audit.log`, and `custom_document_types.json` are not
  encrypted by this feature.
- **Key custody.** The key is held only in the `DATA_ENCRYPTION_KEY`
  environment variable. If it is lost, encrypted documents cannot be
  recovered. Store and rotate it outside the application.

## Retention

- Default: documents are kept **indefinitely**.
- Set `DATA_RETENTION_DAYS=<n>` to enable a retention window.
- `POST /v1/governance/purge` deletes documents (records **and** files) older
  than that window. Run it on a schedule (cron) to enforce retention.
- `GET /v1/governance` reports the current data directory and retention setting.

## Deletion on request

- `DELETE /v1/projects/{id}` removes the project, its document records, its
  project memory, **and the document files on disk** — it returns the count of
  files purged.
- `DELETE /v1/projects/{id}/documents/{doc_id}` removes one document (record +
  file).
- `DELETE /v1/projects/{id}/memory/{key}` forgets a single stored fact.

## Audit trail

Every create/upload/delete/purge is appended to `DATA_DIR/audit.log` with a UTC
timestamp. Read it per project via `GET /v1/projects/{id}/audit`. The log is
append-only — entries are never rewritten.

## Confidentiality

- All `/v1/*` endpoints require an API key (`Authorization: Bearer <key>`).
- The `$CEREBRUM_DEV_KEY` development key works **only** in development/test
  environments — it is rejected when `ENV` is production.
- Project memory is scoped to a single project; facts never leak across
  projects.

## Known limitations (not yet implemented)

- No per-user access control beyond the API key (no row-level ownership).
- On the Render free tier there is no persistent disk — data placed in a temp
  directory does not survive a redeploy. Use a persistent volume for real data.
