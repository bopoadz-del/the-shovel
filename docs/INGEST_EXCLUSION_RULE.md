# Ingest rules — text formats only, and no archive

**Read this before touching any ingest path.** Three owner rules, all binding:

1. **Text formats only** (below): video, CAD drawings, photos/images, map
   files and every format not in `TEXT_BEARING_EXTS` are never ingested,
   downloaded or retried.
2. **No archive** ([below](#no-archive)): the platform keeps no copy of an
   original file. Google Drive is the source of truth for admin-added
   project documents; the extracted chunks are what the RAG needs.
3. **Only the admin path adds to the knowledge base**
   ([below](#who-adds-to-the-knowledge-base)): a user's upload is not indexed
   into the project RAG.

Both follow [`RAG_GAPS_REVIEW_2026-09-12.md` §E](RAG_GAPS_REVIEW_2026-09-12.md).

The knowledge base is text-only by design. From
[`RAG_GAPS_REVIEW_2026-09-12.md` §E](RAG_GAPS_REVIEW_2026-09-12.md):

> **Excluded BY DESIGN — never ingested, not gaps:** CAD (`.dwg/.dxf`),
> images (`.jpg/.png`), Google Earth (`.kmz/.kml`), video, GIS internals,
> fonts. Only text formats produce retrievable chunks.

Photos are not corpus material either
([`PHOTO_RAG_STATUS.md`](PHOTO_RAG_STATUS.md)): a photo attached in chat is
question context (`POST /v1/chat/analyze-photo`), never a document.

## The single declaration

`app/core/text_extractors.py` -- the **extractor registry**. A format is
ingestible only when a working text extractor is registered for it; each
extractor declares its formats with `@reads(...)`, and
`ingest_status.TEXT_BEARING_EXTS` (with `is_ingestible()`) is built from the
registry, never written by hand. To support a format, register an extractor.
A test reads a real sample of every registered format into a chunk.

| entry point | how it applies the rule |
|---|---|
| Drive ingest (`scripts/p1b_ingest_drive_server.py`) | non-ingestible files are filtered at discovery: never assigned or downloaded; counted in `accounting.unsupported_by_format` |
| Indexer (`doc_index`) | dispatches through the registry; a file that yields no text is stamped ZERO_CHUNK, never recorded as indexed |
| Project document upload (`POST /v1/projects/{id}/documents`) | 415 with the rule's message |
| `/upload` with a `project_id` | 415 before anything is written |
| Drive walker + Drive import (`app/routers/drive.py`) | the registry's formats only; import returns 415 |
| CDE ingest (`app/core/cde/ingest.py`) | refused with the rule's message |

**Compressed files** (zip, rar and every other compressed-folder format) are
never ingested, opened or unpacked. No extractor reads one; an upload of one
(either route, with or without a project) answers 415 "Compressed files are
not supported. Unzip it and upload the documents." (`app/core/compressed.py`).

Session/sandbox uploads (no project) and chat photos keep their own
acceptance: they are context for one conversation, never persisted into the
knowledge base. Redline runs on PDF drawings only.

## No archive

The platform does not archive original files (owner ruling, 2026-10-04).

| what | where it lives |
|---|---|
| an admin-added project document's original | Google Drive — the ledger row's `drive_file_id` |
| what the RAG retrieves | the extracted chunks (Neon `chunks_v2`) |
| a user's own upload | the data volume, as uploaded (it is the user's file, not a Drive copy) |

- The Drive ingest writes each download to **one transient copy** on the
  ingest task's own disk, indexes it and deletes it. Nothing is written to
  the shared data volume and nothing is uploaded anywhere. The ledger row's
  `file_path` is therefore stale by design.
- Opening a cited source (preview, raw view, download) resolves the bytes
  from the row's `drive_file_id` (`projects.materialize_document_file`).
  A row with no Drive id and no local file answers a clear 404.
- There is no object store. The Cloudflare R2 archive (`app/core/r2_storage.py`)
  was removed: it never held a file (every upload was denied and no
  document row ever carried an `r2_object_key`), and it was the single
  largest memory cost of an ingest — a whole-file Fernet encrypt plus the
  upload buffer, ~7.7x the file size, now 1x.
- Originals are read only when a person asks: an admin runs the ingest, or a
  user opens a cited document. Nothing in the app fetches originals by itself.

## Who adds to the knowledge base

One rule, `app/core/privileges.caller_may_add_to_project_rag` (admin only),
decides every path into a project's RAG (owner ruling, 2026-10-04):

| path | admin | anyone else |
|---|---|---|
| Drive ingest (ECS RunTask) | adds | — (operator-run) |
| `POST /v1/projects/{id}/documents`, `/upload` with a project | stored and indexed | stored, **not indexed** (`indexing.status = not_indexed`) |
| Drive folder index, Drive import, Aconex sync / event ingest | adds | 403 |
| Aconex via `/v1/execute` | rows cached and indexed | rows cached, not indexed |
| Cost / priced BOQ export with `ingest` | download + added | download only |
| Nightly hydration | reads conversations only — never documents | same |

The user layer of the RAG is written only when the user explicitly asks
through the LLM; no upload writes it.

## Why this page exists

2026-10-04: a Drive ingest run downloaded, encrypted and registered 8 drone
videos (2 GB), hundreds of DWG drawings and photos into a client project.
None of it could ever be retrieved, and the multi-hundred-MB files
OOM-killed the ingest repeatedly while effort went into engineering around
them. The first question about any file that breaks an ingest is
**"should this file be here at all?"**
