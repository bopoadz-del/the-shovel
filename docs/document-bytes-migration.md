# Document bytes — existing rows

No schema migration. `documents.file_path` is still the read key. Rows are
not rewritten by this change.

## What already works

- A row whose `file_path` still points at a real file is read from that
  path, including files written before this change and files that are not
  under the current store root. The on-disk format is unchanged
  (`app/core/file_crypto.py`: plaintext, a single Fernet token, or the
  framed `FORK-ENC-STREAM-1` stream).
- New uploads are streamed into `DOCUMENT_STORE_ROOT` (or `DATA_DIR` when
  that is unset) through the same format. The row's `file_path` is the new
  path.
- Google Drive rows are unchanged. The ingest still does not keep a copy
  (`docs/INGEST_EXCLUSION_RULE.md`). A stale `file_path` on a Drive row is
  not a lost upload and is not labeled "File missing, re-upload".

## Rows whose bytes are gone

A user upload (no Drive id, or `metadata.provenance = user_upload`) whose
file is confirmed absent shows **File missing, re-upload** on the document
panel. That is not "extraction failed".

- The row stays. `chunk_count` and any chunks already stored stay.
- The document does not count as indexed (`searchable` is false) until the
  bytes are back, even when those chunks are still there.
- Permission errors, network errors, decryption errors, and a store mount
  that is not actually mounted are not reported as a missing file.
- The ingest worker marks that outcome **failed**. It does not mark it
  completed.
- Re-upload is `POST /v1/projects/{project_id}/documents/{document_id}/bytes`
  by anyone who can already add a document to the project. The same row id
  is kept. Drive-sourced rows are refused.

`ingest_status` is left as it was. Moving a lost file onto a new ledger
status would put it in the set that deletes chunks. The panel and
`searchable` flag are what hide it from "indexed".

## Moving bytes that are still on disk

If a host volume still has the old files:

1. Mount the durable volume (see `deploy/aws/README.md`).
2. Copy the existing tree onto that mount, preserving filenames. Paths
   stored on the rows must keep resolving, or the copy must land at the
   same paths the rows already have. A row whose path does not match the
   new location is a lost file until someone re-uploads it.
3. Set `DOCUMENT_STORE_ROOT` to the mount and `DOCUMENT_BYTE_STORE=efs`.
4. Boot once and confirm the log says the mount was verified.
5. Only then set `REQUIRE_DURABLE_DOCUMENT_STORE=1`. Setting it before the
   mount exists makes the process refuse to start.

There is nothing to run in the database.
