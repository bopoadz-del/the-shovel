> **SUPERSEDED (2026-09-28).** Historical snapshot. Not a source of open items or current status. Current state lives in the fleet standing-order loop and the live measurements.

# Master Folder Ingestion Status Report

## Decision (2026-07-08)

**Local ingestion STOPPED after batch 0.**

The local run was a **pipeline-validation exercise only**. The vectors it wrote landed in a **laptop-local database** and are explicitly **throwaway** — they do NOT ship to prod. The bulk-insert ban stands; prod vectors must be produced by the platform's own pipeline running where the prod DB lives.

What the local run taught us (from the first ~20 files):

| Observation | Rate / Note |
|-------------|-------------|
| Unsupported Google Workspace files (`.gdoc`, `.gsheet`, etc.) | Expected; skip cleanly |
| `.rar` archives unreadable from Drive mount | Expected on Windows; skip cleanly |
| `ZERO_CHUNK` (empty extraction / failed OCR / bad PDF) | 1 of first ~15 real PDFs — will need itemizing |
| Errors (non-zero exit) | 1 of first ~15 real PDFs — font/PyMuPDF issues |
| Per-file time | 30–90s dominated by OCR + embedding on local CPU |

**Conclusion:** local laptop ingestion is too slow and produces non-shippable data. Move to server-side.

---

## Correct Process: Server-Side Ingestion on Render

**Target:** Run the same ingestion pipeline as a job on Render, writing into the prod vector store namespace.

**Why this is the right path:**
- Files fetched via the platform's Google Drive OAuth path (datacenter-to-datacenter, not through a laptop mount).
- Embedding and vector writes happen next to the prod DB.
- Batched, resumable, per-file logs, `ZERO_CHUNK` watched, provenance stamped — same rules as before.
- Instance can be temporarily scaled up for the ingestion window if the 2 GB worker cannot hold `bge-small` + OCR concurrently.

**Open question for the owner:** Can the Render web service instance (2 GB) hold the embedder + OCR in memory, or should it be scaled up for the ingestion window?

---

## Priority Manifest (Required Before Server Run)

Do NOT ingest the Master Folder as one undifferentiated 7,221-file blob.

| Tier | Priority | Content | Gate Dependency |
|------|----------|---------|-----------------|
| **Tier 1** | Must have first | `the client project` + folders backing the golden set and fixtures | Phase 2 battery gates on this |
| **Tier 2** | Pilot-relevant | Other active project folders (commercial, design, QA, safety, procurement, RFP-related) | Phase 2 battery may include spot checks |
| **Tier 3** | Background | Everything else — reference material, old revisions, archives | Drains in background; not pilot-blocking |

**Next action:** publish the concrete priority manifest (folder → tier → expected file count) and get the owner's approval before starting the server-side run.

---

## Per-Folder Triage Tally

Running tally must be kept per folder:

| Folder | Succeeded | Zero-chunk | Unsupported | Error | Skipped too large | Notes |
|--------|-----------|------------|-------------|-------|-------------------|-------|
| `Master Folder/the client project/Contract Docs/Contractor/Contract docs SIGNED` | 4 | 1 | 0 | 1 | 0 | Font/OCR issues observed |
| `Master Folder/.archivetemp...` | 0 | 0 | 1 | 0 | 0 | `.gdoc` placeholder |
| `Master Folder/Owner_CV PM.gdoc` | 0 | 0 | 1 | 0 | 0 | `.gdoc` placeholder |
| `Master Folder/Copy of CAD.rar` | 0 | 0 | 1 | 0 | 0 | `.rar` unsupported on Windows mount |

This tally will be expanded as the server-side run progresses.

---

## Extractors Verified Locally (transferable to server)

| Format | Status | Evidence |
|--------|--------|----------|
| `.pptx` | ✅ Working | `NOC Tracker...` → 1,503 chars |
| `.kmz` | ✅ Working | `6. Metro Box.kmz` → 6,589 chars |
| `.zip` | ✅ Working (small/medium) | `OneDrive_1_6-23-2022.zip` → 84,011 chars |
| `.rar` | ⚠️ Unsupported without `unrar` binary | Skip or add server-side unrar |
| `.msg` | ✅ Working | `240829 email from JM...` → 1,167 chars |
| `.doc` | ✅ Working | `Owner_CV.doc` → 11,445 chars |

Unsupported formats (`.dwg`, `.plt`, `.nwd`, etc.) must be itemized in the reconciliation report, never silently skipped.

---

## Guards in Place

| Guard | Setting | Purpose |
|-------|---------|---------|
| `PDF_MAX_SIZE_MB` | 100 | Skip PDFs > 100 MB |
| `PDF_OCR_MAX_SIZE_MB` | 25 | Disable OCR for PDFs > 25 MB, use text layer only |
| `P1B_MAX_FILE_SIZE_MB` | 100 | Skip copying files > 100 MB from Drive |
| Archive file size | 50 MB | Skip archives > 50 MB compressed |
| Archive total bytes | 50 MB | Limit uncompressed total per archive |
| Archive member count | 100 | Limit files extracted per archive |
| Archive image members | skipped | Avoid per-photo OCR/YOLO inside ZIPs |

---

## Issues Log

1. First local run died at ~124/7222 — OOM on large contract PDF.
2. 500-file batch killed by session close at file 22.
3. Batch runner killed silently on 446 MB drawing PDFs.
4. Batch runner killed again on 36 MB scanned PDF OCR.
5. Local 1,000-file batch reduced embedder reload but still laptop-bound.
6. **Local run halted** — vectors are laptop-local and non-shippable.
7. Server runs 37159882e871 (316/1361), 315/1362 and 323/1380 stopped with no
   traceback, no tally line and no final report. See below.

## Silent Server Exits (runs 315/1362, 323/1380, 316/1361)

`scripts/p1b_ingest_drive_server.py` has no code path that produces that
signature: every early `return` prints `ERROR: ...` and sits above the
`run_id=` line, and an exception prints a traceback. A run that emits the run
line and then nothing was **ended from outside the interpreter or crashed
below it**, and the build recorded nothing about which.

Two causes fit the evidence equally well and were not distinguishable:

| Cause | Why it fits | Why it is not proven |
|-------|-------------|----------------------|
| cgroup OOM `SIGKILL` | Three deaths cluster at the same position in a size-sorted queue (files of 4.7–4.9 MB, from the point where `--resume` leaves only the larger tail). `app/core/extract_isolated` measured ONE 4.4 MB PDF taking the live box 790 MB → 3.9 GB of 4 GB before a SIGKILL, and `P1B_PARALLELISM=2` runs two of those at once with `P1B_MAX_FILE_SIZE_MB` defaulting to 0. | No SIGKILL leaves a log line, and nothing read the cgroup `oom_kill` counter. |
| Container/shell lifecycle `SIGTERM`/`SIGHUP` | Per-file time is roughly constant, so a time-based kill also lands at a similar file count. A Render Shell run that is not a session leader is HUPed on shell close; a deploy of the attached web service recycles the container. | No handler was installed, so neither signal was ever named. |

Both are now recorded rather than inferred — see the ingest lifecycle notes in
`AGENTS.md`. The next run names the signal (`exit_reason=signal:SIGTERM`), or
leaves a sidecar heartbeat whose post-mortem says whether the cgroup
`oom_kill` counter moved and whether the host was replaced.

---

## Server-Side Assets Created

| Asset | Path | Purpose |
|-------|------|---------|
| Priority manifest builder | `scripts/build_priority_manifest.py` | Generates `manifests/p1b_priority_manifest.json` from existing Drive manifest + known folder list. |
| Priority manifest | `manifests/p1b_priority_manifest.json` | Tier-1/2/3 folder assignment; the client project subfolder counts populated; tier 2/3 folder IDs need the owner. |
| Server ingestion job | `scripts/p1b_ingest_drive_server.py` | Runs on Render; walks Drive via `app.core.gdrive_service`, downloads, indexes, keeps per-folder tally. |

## Render Instance Assessment

| Component | Approximate RAM | Note |
|-----------|-----------------|------|
| FastAPI app baseline | ~200–400 MB | Already running on prod |
| `bge-small-en-v1.5` embedder | ~130 MB | Loaded once per process |
| Tesseract OCR + page images | ~100–500 MB | Spiky; large/dense PDFs hit the high end |
| OS + other overhead | ~300–500 MB | Render 2 GB instance |
| **Headroom** | **tight** | Consecutive large scanned PDFs risk OOM |

**Recommendation:** Scale the Render instance to **4 GB** for the ingestion window. This is the correct spend: it avoids laptop-days and keeps the job stable. Scale back to 2 GB after tier 1 + tier 2 are done.

## Open Blockers for the owner

1. **GDRIVE_SERVICE_ACCOUNT_JSON** on Render is broken / missing. The server-side job cannot mint Drive tokens until this is replaced.
2. **Tier 2/3 folder IDs** are missing from `manifests/p1b_priority_manifest.json`. Provide the Drive folder IDs for `construction-3-001`, `200-Project Controls Procedures`, `300-Delivery Management Procedures`, `400-Construction Management Procedures`, `500-Design Management Procedures`, `600-Procurement & Contracts`, and the scanned-files folders.
3. **Render instance scaling** — confirm 4 GB for ingestion, or accept OOM risk on 2 GB.

## Next Steps

1. The owner replaces `GDRIVE_SERVICE_ACCOUNT_JSON` and supplies tier 2/3 folder IDs.
2. Scale Render instance to 4 GB.
3. Run tier 1 on Render: `.venv/Scripts/python scripts/p1b_ingest_drive_server.py --tier 1 --resume`.
4. Monitor `manifests/p1b_server_ingestion_report.json`; itemize every gap.
5. Gate Phase 2 battery on tier 1 + tier 2 completion.
6. Let tier 3 drain in the background; scale back to 2 GB.
