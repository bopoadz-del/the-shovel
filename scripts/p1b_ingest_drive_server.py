#!/usr/bin/env python3
"""P1b — server-side Drive ingestion for the clean rebuild.

READ FIRST -- THE RAG TAKES TEXT FORMATS ONLY. Video, images, CAD
(.dwg/.dxf), Google Earth (.kmz/.kml), GIS internals and fonts are excluded
BY DESIGN (docs/RAG_GAPS_REVIEW_2026-09-12.md section E). The single
declaration is ``ingest_status.TEXT_BEARING_EXTS``: a file not in it is
filtered at discovery -- never assigned, downloaded, stored or archived, only
counted in ``accounting.unsupported_by_format``. When a file breaks a run,
ask "should it be here at all?" first. See docs/INGEST_EXCLUSION_RULE.md.

NO ARCHIVE. The platform keeps no copy of an original file. Google Drive is
the source of truth for admin-added project documents; the extracted chunks
are everything the RAG needs. Each download is written to ONE transient copy
on this task's own disk, indexed, and deleted -- never stored on the data
volume, never uploaded anywhere. A cited source opens from the ledger row's
``drive_file_id``. See docs/INGEST_EXCLUSION_RULE.md ("No archive") and
docs/RAG_GAPS_REVIEW_2026-09-12.md section E.

Runs as one ECS RunTask (or any server with the prod DB attached). Files are
fetched via the platform's Google Drive service-account path and indexed
through the normal doc_index pipeline into the configured vector namespace.

Differences from the local p1b script:
- Source is Google Drive API, not a laptop Drive mount.
- Folder selection is driven by manifests/p1b_priority_manifest.json tiers.
- Per-folder tally is kept and flushed to the report.
- Provenance metadata includes Drive file id, source tier, and ingestion run id.

Required env vars on the server:
  RAG_EMBEDDING_MODEL, RAG_VECTOR_NAMESPACE, DATABASE_URL, DATA_DIR,
  GDRIVE_SERVICE_ACCOUNT_JSON, GDRIVE_PROJECT_FOLDERS (optional)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Set defaults BEFORE importing app modules so they pick them up.
os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")

# Files that cannot be downloaded as bytes from Drive or are not indexable.
_UNSUPPORTED_MIMES = {
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.google-apps.drawing",
    "application/vnd.google-apps.folder",
}

# Which formats are ingestible is decided by the extractor registry alone
# (ingest_status.is_ingestible <- app/core/text_extractors.py); geospatial
# containers, compressed folders and Google-native shortcuts have no
# extractor, so they are excluded and counted at discovery.

# Sidecar heartbeat. Lives next to the report because it answers the question
# the report cannot when the run is killed: where was it, and what was the box
# doing, at the last moment it was alive. Default stays under manifests/ so
# accounting evidence under DATA_DIR is not mixed with a per-file heartbeat.
DEFAULT_RUN_STATE = "manifests/p1b_ingest_run_state.json"

# Set in the child by --supervise so it cannot supervise itself recursively.
SUPERVISED_ENV = "P1B_SUPERVISED"

# Every stderr line carries the run id. Exactly one of the 26 lines used to
# (the opening banner), so a Web Shell scrollback holding several runs could
# not be split back into runs: a progress line from a dead run reads exactly
# like one from the live worker.
_RUN_ID = "-"


def set_run_id(run_id: str) -> None:
    """Bind the run id every subsequent :func:`log` line is attributed to."""
    global _RUN_ID
    _RUN_ID = run_id


def current_run_id() -> str:
    return _RUN_ID


def log(message: str) -> None:
    """Emit one run-attributable line on stderr.

    Unbuffered on purpose: a diagnostic still sitting in a stdio buffer when
    SIGKILL arrives never existed. Every line carries ``run_id=`` so a log
    stream that holds several runs can be split back into runs.
    """
    print(f"[p1b-server] run_id={_RUN_ID} {message}", file=sys.stderr, flush=True)


def resolve_evidence_dir() -> Path:
    """Directory for durable run evidence.

    ``manifests/`` lives on the container image layer, which Render throws
    away on every deploy and restart — and a disk-backed service like
    ``the-fork`` restarts on every deploy by construction. ``DATA_DIR`` is the
    mounted disk (``/app/data`` in production), so evidence written there
    outlives the process, the shell session, and the deploy.
    """
    raw = os.getenv("P1B_EVIDENCE_DIR") or os.getenv("DATA_DIR") or "."
    return Path(raw)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write ``payload`` to ``path`` via a temp file + :func:`os.replace`.

    A plain ``open(path, "w")`` truncates before it writes, so a SIGKILL
    part-way through (the documented OOM / session-close failure mode) leaves
    unparseable JSON exactly where the reader expects either the previous
    report or the new one. ``os.replace`` is atomic on POSIX, so a reader sees
    one or the other and never a half-file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


class RunLockUnavailable(RuntimeError):
    """Another live ingest run already holds this (tier, project) lock."""


def advisory_lock_keys(tier: int, project_id: str) -> Tuple[int, int]:
    """Deterministic signed-int32 ``(classid, objid)`` for a pg advisory lock.

    Postgres' two-argument advisory-lock form takes two ``int4`` values, so
    the sha256 digest is truncated to four bytes and mapped into the signed
    range. sha256, not :func:`hash` — Python's string hash is per-process
    randomized and would hand two workers different keys for the same folder.
    """

    def _int32(data: bytes) -> int:
        value = int.from_bytes(hashlib.sha256(data).digest()[:4], "big")
        return value - (1 << 32) if value >= (1 << 31) else value

    return _int32(b"p1b_ingest_drive_server"), _int32(f"{tier}:{project_id}".encode())


@contextlib.contextmanager
def run_advisory_lock(tier: int, project_id: str) -> Iterator[str]:
    """Hold a session-level Postgres advisory lock for ``(tier, project_id)``.

    Two concurrent ``--tier 1 --resume`` runs each snapshot ``already_indexed``
    at their own start time and then both call ``projects.add_document``, which
    mints a fresh uuid with no uniqueness on ``drive_file_id`` — so the same
    Drive file lands twice, with two sets of chunks. Nothing in the schema
    prevents that. This lock does.

    Yields ``"acquired"``, or ``"unsupported"`` on a non-Postgres dialect
    (SQLite dev/test has no advisory locks, so the guard degrades to a logged
    no-op instead of blocking local work). Raises :class:`RunLockUnavailable`
    when another live session holds it.

    The lock is session-scoped and released explicitly on exit. A run that is
    SIGKILLed cannot wedge the next one either: Postgres drops the lock when
    it notices the dead connection.
    """
    from sqlalchemy import text

    from app.core import db as db_mod

    engine = db_mod.get_engine()
    if engine.dialect.name != "postgresql":
        log(
            f"WARN: run lock unavailable on dialect {engine.dialect.name!r}; "
            "concurrent-run protection is OFF"
        )
        yield "unsupported"
        return

    key1, key2 = advisory_lock_keys(tier, project_id)
    # AUTOCOMMIT so the held connection sits 'idle' rather than
    # 'idle in transaction' for the hours this run may take.
    conn = engine.connect().execution_options(isolation_level="AUTOCOMMIT")
    try:
        acquired = bool(
            conn.execute(
                text("SELECT pg_try_advisory_lock(:k1, :k2)"),
                {"k1": key1, "k2": key2},
            ).scalar()
        )
        if not acquired:
            raise RunLockUnavailable(
                f"another ingest run holds the lock for tier={tier} "
                f"project_id={project_id} (keys {key1},{key2})"
            )
        log(f"run lock acquired tier={tier} project_id={project_id}")
        try:
            yield "acquired"
        finally:
            conn.execute(
                text("SELECT pg_advisory_unlock(:k1, :k2)"),
                {"k1": key1, "k2": key2},
            )
            log(f"run lock released tier={tier} project_id={project_id}")
    finally:
        conn.close()


def future_result_or_error(
    fut: Future,
    file_meta: Dict[str, Any] | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Drain a worker future without letting the exception escape.

    #477 caught exceptions inside _ingest_file, but ``fut.result()`` in the
    main thread was still bare. A leaked worker exception there aborts
    the whole TIER-1 run (de6a06b7542c died at 323/1380 after 'ingest
    continues' plus a leftover traceback).
    """
    rel = ""
    if file_meta:
        rel = file_meta.get("_drive_path") or file_meta.get("name") or ""
    try:
        rel_out, result = fut.result()
        return (rel_out or rel), result
    except Exception as exc:  # noqa: BLE001 — one file must not kill the run
        log(
            f"worker future raised {type(exc).__name__}: {exc}; "
            f"continuing ingest for {rel!r}"
        )
        return rel or "?", {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _is_geodatabase_internal(path: str) -> bool:
    """Esri geodatabase folders contain non-indexable internal files (gdb,
    timestamps, a00000001.gdbtable, etc.) that Drive reports as tiny
    application/octet-stream blobs. Skip any file inside a ``*.gdb`` folder.
    """
    return any(part.lower().endswith(".gdb") for part in Path(path).parts[:-1])


def tally_key_for_index_result(result: Dict[str, Any]) -> str:
    """Run-tally bucket for one ``index_document`` result.

    A recoverable CAD skip (``.dwg`` / ``.dxf``, no parser) is
    ``skipped_recoverable``, not ``zero_chunk``. ``zero_chunk`` stays the
    bucket for a supported file that extracted nothing.
    """
    if result.get("skip_reason") == "skipped_recoverable":
        return "skipped_recoverable"
    if result.get("status") == "error":
        err = result.get("error")
        if err == "ZERO_CHUNK":
            return "zero_chunk"
        if err in ("SKIPPED_TOO_LARGE", "SKIPPED_MEMORY_BUDGET"):
            return "skipped_too_large"
        if err == "SKIPPED_TOO_SMALL":
            return "skipped_too_small"
        if err == "SKIPPED_UNSUPPORTED":
            return "skipped_unsupported"
        if err == "SKIPPED_EMPTY":
            return "skipped_empty"
        if err == "DUPLICATE_SHA":
            return "duplicate_sha"
        if isinstance(err, str) and err.startswith("DOWNLOAD_FAILED"):
            return "download_failed"
        if isinstance(err, str) and err.startswith("QUARANTINED_"):
            return "quarantined"
        return "errors"
    if result.get("rag_indexed", 0) == 0:
        return "zero_chunk"
    return "succeeded"


def _ingest_status_from_index_result(
    result: Dict[str, Any], existing_doc: Dict[str, Any],
) -> str:
    """Classify the in-place retry outcome for the now_indexed tally."""
    from app.core import ingest_status as ist

    stamped = result.get("ingest_status")
    if stamped in ist.ALL_STATUSES:
        return str(stamped)
    n = int(result.get("rag_indexed") or result.get("total_chunks") or 0)
    ext = ist.document_extension(existing_doc) or ".docx"
    return ist.classify(chunk_count=n, extension=ext).status


def _safe_stored_name(original: str) -> str:
    """Filesystem-safe stored name; preserves extension."""
    base = Path(original).stem
    ext = Path(original).suffix
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:80]
    return f"{safe}{ext}"


def _transient_copy(raw_bytes: bytes, stored_name: str) -> Path:
    """Write the one copy the extractors read, on this task's ephemeral disk.

    Plaintext, because it is never stored: it lives for one file's indexing
    and is then deleted (``_discard_transient_copy``). The decrypted-temp
    prefix makes the plaintext sweeper reap it if the process dies first.
    """
    import tempfile

    from app.core import file_crypto

    fd, path = tempfile.mkstemp(prefix=file_crypto.DECRYPTED_TEMP_PREFIX, suffix="_" + stored_name)
    with os.fdopen(fd, "wb") as fh:
        fh.write(raw_bytes)
    return Path(path)


def _discard_transient_copy(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        log(f"could not delete transient copy {Path(path).name}: {exc}")


def _drive_reports_empty(file_meta: Dict[str, Any]) -> bool:
    """True when Drive states the file's size and it is zero bytes."""
    raw = str(file_meta.get("size", "")).strip()
    return raw.isdigit() and int(raw) == 0


def _ingest_file(
    file_meta: Dict[str, Any],
    project_id: str,
    data_dir: Path,
    run_id: str,
    gdrive_service: Any,
    existing_doc: Dict[str, Any] | None = None,
    reingest_of: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Download one Drive file and index it. Returns (drive_path, result).

    ``existing_doc`` — a document row already registered for this Drive file
    (a previous pass added the row but indexing produced zero chunks, e.g.
    the NUL-byte DataError). The file is re-downloaded to the same
    deterministic path and re-indexed into the SAME document row instead of
    creating a duplicate.
    """
    from app.core import doc_index, projects as projects_mod

    rel = file_meta.get("_drive_path") or file_meta.get("name", "")
    mime = file_meta.get("mimeType", "")

    # Skip Google-native, non-ingestible, and geodatabase internal files.
    from app.core import ingest_status as _ist_gate

    if (
        mime in _UNSUPPORTED_MIMES
        or not _ist_gate.is_ingestible(rel)
        or _is_geodatabase_internal(rel)
    ):
        return rel, {
            "status": "error",
            "error": "SKIPPED_UNSUPPORTED",
            "mimeType": mime,
        }

    # Size guard before download. Default OFF (0) — Tier-1 the client project contract
    # volumes (signed PSA, drawing packs) exceed 500 MB and were still
    # SKIPPED_TOO_LARGE after the 100→500 bump. Set P1B_MAX_FILE_SIZE_MB
    # to a positive number to re-enable a cap.
    size = int(file_meta.get("size") or 0)
    max_size = int(os.getenv("P1B_MAX_FILE_SIZE_MB", "0")) * 1024 * 1024
    if max_size > 0 and size > max_size:
        return rel, {
            "status": "error",
            "error": "SKIPPED_TOO_LARGE",
            "size_mb": round(size / (1024 * 1024), 1),
        }

    # Minimum useful size guard. Files below this threshold cannot produce
    # indexable content (empty placeholders, geodatabase internals, etc.).
    # Default OFF (0); set P1B_MIN_FILE_SIZE_BYTES to skip them cleanly.
    min_size = int(os.getenv("P1B_MIN_FILE_SIZE_BYTES", "0"))
    if min_size > 0 and size < min_size:
        return rel, {
            "status": "error",
            "error": "SKIPPED_TOO_SMALL",
            "size_bytes": size,
        }

    # Memory admission BEFORE download (see extract_isolated.admission_refusal):
    # a file whose store-and-index demand cannot fit is recorded on the ledger
    # as a closed EXTRACT_FAILED instead of OOM-killing the run.
    from app.core import extract_isolated
    from app.core import ingest_status as ist
    from app.core.ingest_reconcile import source_content_token

    refusal = extract_isolated.admission_refusal(
        size, child_budget_mb=doc_index._file_budget()[0],
    )
    if refusal:
        token = source_content_token(file_meta)
        if existing_doc is not None:
            doc_id = existing_doc["id"]
            key_source = existing_doc.get("content_sha256") or existing_doc.get("drive_md5") or token
        else:
            doc_id = projects_mod.add_document(
                project_id=project_id,
                original_name=Path(rel).name,
                size=size,
                metadata={
                    "drive_file_id": file_meta["id"],
                    "drive_path": rel,
                    "source": "p1b_server_drive_reingestion",
                    "ingestion_run_id": run_id,
                    "mimeType": mime,
                },
                drive_md5=token,
            )["id"]
            key_source = token
        projects_mod.record_ingest_attempt(doc_id, ist.attempt_key(key_source))
        doc_index._stamp_index_ledger(
            doc_id, Path(rel).name, 0, extract_failed=True, reason_override=refusal,
        )
        log(f"MEMORY_ADMISSION refused {rel!r} size={size}: {refusal}")
        return rel, {
            "status": "error",
            "error": "SKIPPED_MEMORY_BUDGET",
            "reason": refusal,
            "size_mb": round(size / (1024 * 1024), 1),
        }

    # Download from Drive.
    raw_bytes, dl_err = gdrive_service.download_file_bytes(file_meta["id"])
    if dl_err:
        return rel, {
            "status": "error",
            "error": f"DOWNLOAD_FAILED: {dl_err}",
        }

    if not raw_bytes:
        return rel, {
            "status": "error",
            "error": "SKIPPED_EMPTY",
        }

    content_sha = hashlib.sha256(raw_bytes).hexdigest()
    existing_by_sha = projects_mod.find_document_by_sha(project_id, content_sha)
    reingest_of = projects_mod.scoped_reingest_of(
        reingest_of, content_sha, existing_by_sha,
    )
    if existing_by_sha and not reingest_of and existing_doc is None:
        from app.core.ingest_reconcile import source_content_token

        projects_mod.record_source_alias(
            existing_by_sha["id"], file_meta["id"], source_content_token(file_meta),
        )
        return rel, {
            "status": "error",
            "error": "DUPLICATE_SHA",
            "existing_id": existing_by_sha["id"],
            "content_sha256": content_sha,
        }
    stored_name = f"{hashlib.sha256(rel.encode()).hexdigest()[:8]}_{_safe_stored_name(Path(rel).name)}"
    # NO ARCHIVE (owner ruling, docs/INGEST_EXCLUSION_RULE.md): the platform
    # keeps no copy of an original. Google Drive is the source of truth and the
    # chunks are what the RAG needs. The extractors read one TRANSIENT copy on
    # this task's own ephemeral disk (never the shared data volume), deleted
    # as soon as the file is indexed. It carries the decrypted-temp prefix, so
    # the plaintext sweeper reaps it if the process dies first.
    dest = _transient_copy(raw_bytes, stored_name)

    # Release the payload BEFORE indexing. Extraction is the memory peak on
    # this box (app/core/extract_isolated measured one 4.4 MB PDF taking the
    # live instance from 790 MB to 3.9 GB of 4 GB); every reader from here on
    # works off the path.
    downloaded_bytes = len(raw_bytes)
    del raw_bytes

    from app.core.ingest_reconcile import source_content_token

    drive_token = source_content_token(file_meta)
    common_meta = {
        "drive_file_id": file_meta["id"],
        "drive_path": rel,
        "source": "p1b_server_drive_reingestion",
        "ingestion_run_id": run_id,
        "mimeType": mime,
        "content_sha256": content_sha,
    }
    if drive_token:
        common_meta["drive_md5"] = drive_token

    if existing_doc is not None and not reingest_of:
        # In-place retry: the row exists from a prior pass (zero-chunk, or a
        # TEXT_SPARSE .docx stamped by an older extractor). Re-index the SAME
        # document id so chunks are replaced. No new row, no supersede.
        projects_mod.update_document_metadata(existing_doc["id"], common_meta)
        # Index THIS download: the row's old file_path is a deleted (or stale)
        # copy from an earlier pass.
        projects_mod.set_document_file_path(existing_doc["id"], str(dest))
        if drive_token:
            projects_mod.set_document_drive_md5(existing_doc["id"], drive_token)
        result = doc_index.index_document(
            project_id, existing_doc["id"],
        )
        result["reindexed_existing_doc"] = True
        if existing_doc.get("_stale_extractor_retry"):
            result["stale_extractor_retry"] = True
            result["ingest_status"] = _ingest_status_from_index_result(
                result, existing_doc,
            )
        _discard_transient_copy(dest)
        log(
            f"VERIFICATION doc_id={existing_doc['id']} "
            f"drive_file_id={file_meta['id']} drive_path={rel!r} "
            f"mime={mime!r} drive_size={size} downloaded_bytes={downloaded_bytes} "
            f"rag_indexed={result.get('rag_indexed', 0)} "
            f"index_status={result.get('status')} "
            f"index_error={result.get('error')}"
        )
        return rel, result

    try:
        doc = projects_mod.add_document(
            project_id=project_id,
            original_name=Path(rel).name,
            stored_as=stored_name,
            file_path=str(dest),
            size=size,
            content_sha256=content_sha,
            metadata=common_meta,
            reingest_of=reingest_of,
            drive_md5=drive_token,
        )
    except projects_mod.DuplicateContentError as exc:
        _discard_transient_copy(dest)
        projects_mod.record_source_alias(exc.existing_id, file_meta["id"], drive_token)
        return rel, {
            "status": "error",
            "error": "DUPLICATE_SHA",
            "existing_id": exc.existing_id,
            "content_sha256": content_sha,
        }
    result = doc_index.index_document(
        project_id, doc["id"], stamp_as_indexed=bool(reingest_of),
    )
    if reingest_of:
        result["reingest_of"] = reingest_of
        result["superseded"] = True
    _discard_transient_copy(dest)
    log(
        f"VERIFICATION doc_id={doc['id']} "
        f"drive_file_id={file_meta['id']} drive_path={rel!r} "
        f"mime={mime!r} drive_size={size} downloaded_bytes={downloaded_bytes} "
        f"rag_indexed={result.get('rag_indexed', 0)} "
        f"index_status={result.get('status')} "
        f"index_error={result.get('error')}"
    )
    return rel, result


def _load_priority_manifest(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Priority manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Server-side Drive ingestion by priority tier")
    ap.add_argument("--priority-manifest", default="manifests/p1b_priority_manifest.json",
                    help="Path to the priority manifest")
    ap.add_argument("--tier", type=int, default=1,
                    help="Ingest only this tier (1, 2, or 3)")
    ap.add_argument(
        "--output",
        default=None,
        help=(
            "Report path. Defaults to <P1B_EVIDENCE_DIR|DATA_DIR>/"
            "p1b_server_ingestion_report.json so evidence lands on the mounted "
            "disk rather than the throwaway image layer under manifests/."
        ),
    )
    ap.add_argument("--resume", action="store_true",
                    help=("Accepted for compatibility. Resume is ALWAYS on for "
                          "live runs — see the chunk-count resume check in main()."))
    ap.add_argument(
        "--folder-id",
        action="append",
        default=None,
        dest="folder_ids",
        metavar="ID",
        help=(
            "Restrict this run to these Drive folder id(s). Repeat the flag or "
            "pass comma-separated ids. Walks that folder instead of the full "
            "tier parent; documents still use the matching parent project's "
            "row (project_id / folder_name from the tier entry). If the id is "
            "not a parent folder_id in this tier, it is treated as a subfolder "
            "of the unique parent that has a folder_id. Omit to walk every "
            "folder in the tier (default --tier N --resume behaviour)."
        ),
    )
    ap.add_argument("--keep-alive", action="store_true",
                    help="After the ingestion pass completes, sleep forever so a background worker stays live")
    ap.add_argument("--total-shards", type=int, default=None,
                    help="Total worker shards (env: INGEST_TOTAL_SHARDS)")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="This worker's shard index (env: INGEST_SHARD_INDEX)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — walk + shard, no DB/index writes (env: INGEST_DRY_RUN)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N files after shard filter (env: INGEST_LIMIT)")
    ap.add_argument("--offset", type=int, default=None,
                    help="Skip first N files after shard filter (env: INGEST_OFFSET)")
    ap.add_argument(
        "--run-state",
        default=None,
        help=(
            "Sidecar heartbeat file (env: P1B_RUN_STATE, default "
            f"{DEFAULT_RUN_STATE}). Updated after every file so a SIGKILLed run "
            "can still be post-mortemed on the next start."
        ),
    )
    ap.add_argument(
        "--supervise",
        action="store_true",
        help=(
            "Run the ingest as a child process and report HOW it died. Only a "
            "parent can see a SIGKILL: run 37159882e871 vanished at 316/1361 "
            "with no traceback and nothing recorded which signal ended it."
        ),
    )
    ap.add_argument(
        "--reingest",
        default=None,
        metavar="OLD_ID",
        help=(
            "Allow a file whose sha256 already exists by creating a NEW "
            "documents row and hiding OLD_ID (retrieval_visible=false, "
            "superseded_by=new). Without this flag a duplicate sha is "
            "refused. Hard delete is never used."
        ),
    )
    return ap


def parse_folder_ids(raw: List[str] | None) -> List[str]:
    """Flatten repeatable and comma-separated ``--folder-id`` values."""
    if not raw:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        for part in str(item).split(","):
            fid = part.strip()
            if not fid or fid in seen:
                continue
            seen.add(fid)
            out.append(fid)
    return out


def folder_entries_for_run(
    tier_folders: List[Dict[str, Any]],
    requested_ids: List[str],
    *,
    tier: int,
) -> List[Dict[str, Any]]:
    """Return the folder entries this run should walk.

    With no ``requested_ids``, the tier list is unchanged (full parent walk).
    With ids, each id that matches a parent ``folder_id`` walks that parent.
    An id that is not a parent is a subfolder walk of the unique parent that
    has a ``folder_id`` — same ``project_id`` / ``folder_name``, different
    walk root. Raises ``ValueError`` when an id matches no parent and there
    is no unique parent to attach it to.
    """
    if not requested_ids:
        return list(tier_folders)

    parents_by_id: Dict[str, Dict[str, Any]] = {}
    parents_with_id: List[Dict[str, Any]] = []
    for entry in tier_folders:
        fid = (entry.get("folder_id") or "").strip() or None
        if not fid:
            continue
        parents_by_id[fid] = entry
        parents_with_id.append(entry)

    walks: List[Dict[str, Any]] = []
    unmatched: List[str] = []
    for fid in requested_ids:
        if fid in parents_by_id:
            spec = dict(parents_by_id[fid])
            spec["walk_folder_id"] = fid
            walks.append(spec)
            continue
        if len(parents_with_id) == 1:
            spec = dict(parents_with_id[0])
            spec["walk_folder_id"] = fid
            walks.append(spec)
            continue
        unmatched.append(fid)

    if unmatched:
        raise ValueError(
            f"--folder-id not in tier {tier} parent folders and no unique "
            f"parent to treat as a subfolder walk: {', '.join(unmatched)}"
        )
    if not walks:
        raise ValueError(
            f"--folder-id did not match any folder in tier {tier}"
        )
    return walks


def run_state_path(args: argparse.Namespace, *, dry_run: bool | None = None) -> Path:
    """Where this run heartbeats. Dry runs get their own file so a plan-only
    pass cannot overwrite the evidence left by a live run that died."""
    raw = args.run_state or os.getenv("P1B_RUN_STATE") or DEFAULT_RUN_STATE
    path = Path(raw)
    if dry_run is None:
        dry_run = bool(getattr(args, "dry_run", False))
    if dry_run:
        return path.with_name(path.stem + "_dry" + path.suffix)
    return path


def supervised_child_argv(argv: List[str]) -> List[str]:
    """The same command, re-executed unbuffered, without ``--supervise``."""
    return [sys.executable, "-u"] + [a for a in argv if a != "--supervise"]


def run_under_supervisor(args: argparse.Namespace, argv: List[str] | None = None) -> int:
    """Re-exec this script as a child so its death has a witness.

    SIGKILL runs no handler, writes no traceback and flushes no buffer, so the
    killed process itself can never explain what happened — every one of the
    three silent TIER-1 deaths looked identical from inside. The parent sees
    the wait status and the cgroup oom_kill counter, which is the difference
    between "worker PID 829 disappeared" and "killed by SIGKILL (9), oom_kill
    0 -> 1".
    """
    from app.core import ingest_lifecycle as lifecycle

    child = supervised_child_argv(list(sys.argv if argv is None else argv))
    env = dict(os.environ)
    env[SUPERVISED_ENV] = "1"
    return lifecycle.run_supervised(
        child,
        env=env,
        state_path=run_state_path(args),
        forward=lifecycle.stop_signals(),
    )


def run_knowledge_phase(*, dry_run: bool, shard_index: int = 0) -> Dict[str, int]:
    """Index pending admin-knowledge documents; return their tally.

    Idempotent: only rows still pending are touched, so a run after a
    complete one indexes nothing and writes nothing. Shards other than 0
    skip the phase. A failure here is logged and counted, never fatal to the
    Drive phase that follows.
    """
    tally = {"pending": 0, "indexed": 0, "failed": 0}
    if shard_index != 0:
        return tally
    try:
        from app.core.knowledge_ingest import process_pending_knowledge_documents

        tally.update(process_pending_knowledge_documents(dry_run=dry_run, log=log))
    except Exception as exc:  # noqa: BLE001 — the Drive phase still runs
        log(f"ERROR: knowledge phase failed: {type(exc).__name__}: {exc}")
        tally["failed"] += 1
    log("KNOWLEDGESUMMARY " + json.dumps(tally, sort_keys=True, separators=(",", ":")))
    return tally


def main() -> int:
    from app.core import ingest_lifecycle as lifecycle
    from app.core import ingest_sharding as sharding

    args = build_parser().parse_args()

    if args.supervise and not os.getenv(SUPERVISED_ENV):
        return run_under_supervisor(args)

    try:
        total_shards, shard_index = sharding.resolve_shard_env(
            total_shards=args.total_shards,
            shard_index=args.shard_index,
        )
    except ValueError as exc:
        log(f"ERROR: {exc}")
        return 1

    dry_run = sharding.resolve_dry_run(args.dry_run)
    limit, offset = sharding.resolve_limit_offset(limit=args.limit, offset=args.offset)

    from app.core import gdrive_service, projects as projects_mod
    from app.core.rag import embeddings as _emb, vector_store as _vs

    if not dry_run:
        _emb.reset_embedder_cache()
        _vs.reset_store_cache()

    # The run id exists before the first log line, knowledge phase included.
    run_id = uuid.uuid4().hex[:12]
    set_run_id(run_id)

    # Admin-added reference works first. They need no Drive, so this phase
    # runs whether or not Drive is configured; one shard does it, so
    # concurrent shards never index the same book twice.
    knowledge_tally = run_knowledge_phase(dry_run=dry_run, shard_index=shard_index)

    if not gdrive_service.is_configured():
        log("ERROR: GDRIVE_SERVICE_ACCOUNT_JSON is not set.")
        return 1

    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    if not dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)

    priority_manifest = _load_priority_manifest(Path(args.priority_manifest))
    tier_key = str(args.tier)
    tier = priority_manifest.get("tiers", {}).get(tier_key)
    if not tier:
        log(f"ERROR: tier {args.tier} not found in priority manifest.")
        return 1

    requested_ids = parse_folder_ids(args.folder_ids)
    if args.folder_ids is not None and not requested_ids:
        log("ERROR: --folder-id was given but no folder id was provided.")
        return 1
    try:
        folder_entries = folder_entries_for_run(
            tier["folders"], requested_ids, tier=args.tier,
        )
    except ValueError as exc:
        log(f"ERROR: {exc}")
        return 1

    t0_global = time.monotonic()

    evidence_dir = resolve_evidence_dir()
    report_path = (
        Path(args.output) if args.output
        else evidence_dir / "p1b_server_ingestion_report.json"
    )
    # Legacy location, kept so existing tooling that reads
    # manifests/ingest_shard_<i>_of_<n>.json keeps working. The evidence-dir
    # copy is the durable one.
    legacy_shard_path = sharding.shard_manifest_path(shard_index, total_shards)
    durable_shard_path = evidence_dir / legacy_shard_path.name

    log(
        f"tier={args.tier} folders={len(folder_entries)} "
        f"shard_index={shard_index} total_shards={total_shards} dry_run={dry_run}"
        + (f" folder_ids={requested_ids}" if requested_ids else "")
    )
    log(
        f"evidence report={report_path} shard_manifest={durable_shard_path} "
        f"legacy_shard_manifest={legacy_shard_path}"
    )

    # Track per-folder tally.
    folder_tallies: Dict[str, Dict[str, Any]] = {}
    global_tally = {
        "succeeded": 0,
        "zero_chunk": 0,
        "skipped_too_large": 0,
        "skipped_too_small": 0,
        "skipped_unsupported": 0,
        "skipped_empty": 0,
        "duplicate_sha": 0,
        "download_failed": 0,
        "errors": 0,
        "quarantined": 0,
        "skipped_recoverable": 0,
        "already_indexed": 0,
        "stale_extractor_open": 0,
        "retried": 0,
        "now_indexed": 0,
    }
    results: List[Dict[str, Any]] = []

    # Explicit accounting, separate from `global_tally`.
    #
    # `global_tally` cannot answer "did this run finish". It mixes two
    # universes — `skipped_unsupported` counts DISCOVERY-side files that were
    # never assigned to anyone — so summing it yields `discovered`, a number
    # that looks like a legitimate total and is not one. It also has no
    # attempted/assigned pair, and `assigned` used to exist only in a stderr
    # line that no artifact recorded.
    #
    # THE COMPLETION PREDICATE IS `attempted == batch`, NOTHING ELSE.
    # In particular `already_indexed + assigned == supported` is an identity:
    # with total_shards == 1, filter_by_shard returns its input unchanged, so
    # that equation is forced at discovery time, before a single file is
    # downloaded, and holds just as well for a run that dies on file 1.
    accounting: Dict[str, int] = {
        "discovered": 0,
        "supported": 0,
        "unsupported": 0,
        "already_indexed": 0,
        "assigned": 0,   # this shard's work set
        "batch": 0,      # what this invocation queued, after offset/limit
        "attempted": 0,  # completed futures accounted for in the tally
        "outstanding": 0,
        "walk_errors": 0,
        "skipped_no_folder_id": 0,
        "stale_extractor_open": 0,
        "retried": 0,
        "now_indexed": 0,
        # Pending admin-knowledge documents this run found / indexed / could
        # not index (run_knowledge_phase). A second run finds none pending.
        "knowledge_pending": knowledge_tally["pending"],
        "knowledge_indexed": knowledge_tally["indexed"],
        "knowledge_failed": knowledge_tally["failed"],
    }
    folder_accounting: List[Dict[str, Any]] = []
    walk_error_messages: List[str] = []
    skipped_no_folder_id_folders: List[Dict[str, Any]] = []
    # Advisory locks are entered per folder and released by the lifecycle
    # flush. A process that dies without unwinding cannot wedge the next run
    # either: the locks are session-scoped, so Postgres drops them with the
    # connection.
    run_locks = contextlib.ExitStack()

    def _bump_folder(folder_name: str, key: str) -> None:
        folder_tallies.setdefault(folder_name, {
            "succeeded": 0, "zero_chunk": 0, "skipped_too_large": 0,
            "skipped_too_small": 0, "skipped_unsupported": 0,
            "skipped_empty": 0, "duplicate_sha": 0, "download_failed": 0, "errors": 0,
            "quarantined": 0, "skipped_recoverable": 0,
        })[key] += 1

    def _sync_outstanding() -> None:
        accounting["outstanding"] = accounting["batch"] - accounting["attempted"]

    def build_report(exit_reason: str, run_complete: bool) -> Dict[str, Any]:
        _sync_outstanding()
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "tier": args.tier,
            "shard_index": shard_index,
            "total_shards": total_shards,
            "embedder": os.environ["RAG_EMBEDDING_MODEL"],
            "namespace": os.environ["RAG_VECTOR_NAMESPACE"],
            "exit_reason": exit_reason,
            # Both names are load-bearing: #481's silent-exit tests read
            # ``complete``; #482's accounting tests read ``run_complete``.
            # They are the same boolean — completion is attempted == batch.
            "complete": run_complete,
            "run_complete": run_complete,
            "elapsed_sec": round(time.monotonic() - t0_global, 3),
            "accounting": dict(accounting),
            "walk_error_messages": list(walk_error_messages),
            "skipped_no_folder_id_folders": list(skipped_no_folder_id_folders),
            "folder_accounting": [dict(entry) for entry in folder_accounting],
            "global_tally": global_tally,
            "folder_tallies": folder_tallies,
            "results": results,
        }

    def _write_partial_report(
        exit_reason: str = "running", run_complete: bool = False,
    ) -> None:
        atomic_write_json(report_path, build_report(exit_reason, run_complete))

    def build_run_summary(exit_reason: str, run_complete: bool) -> Dict[str, Any]:
        """The payload of the terminal RUNSUMMARY line.

        Bounded on purpose (no `results`) so it stays one greppable line in
        Render's log stream — which, for a background worker, is the only
        completion authority that survives the container.
        """
        _sync_outstanding()
        return {
            "run_id": run_id,
            "tier": args.tier,
            "shard_index": shard_index,
            "total_shards": total_shards,
            "dry_run": dry_run,
            "exit_reason": exit_reason,
            "complete": run_complete,
            "run_complete": run_complete,
            "accounting": dict(accounting),
            "global_tally": dict(global_tally),
            "report_path": str(report_path),
            "shard_manifest_path": str(durable_shard_path),
            "elapsed_sec": round(time.monotonic() - t0_global, 3),
        }

    def _emit_run_summary(exit_reason: str, run_complete: bool) -> None:
        log("RUNSUMMARY " + json.dumps(
            build_run_summary(exit_reason, run_complete),
            separators=(",", ":"), ensure_ascii=False, sort_keys=True,
        ))

    def build_shard_manifest(exit_reason: str) -> Dict[str, Any]:
        return sharding.empty_shard_manifest(
            shard_index=shard_index,
            total_shards=total_shards,
            extra={
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "tier": args.tier,
                "embedder": os.environ["RAG_EMBEDDING_MODEL"],
                "namespace": os.environ["RAG_VECTOR_NAMESPACE"],
                # `processed` is global_tally["succeeded"], NOT the number of
                # files this run worked through. Kept because existing readers
                # parse it; `attempted` is the one to compare against
                # `assigned`. Reading `processed` as "files handled" makes a
                # finished run look like it left the failures outstanding.
                "processed": global_tally["succeeded"],
                "attempted": accounting["attempted"],
                "skipped": global_tally["skipped_too_large"] + global_tally["skipped_too_small"]
                + global_tally["skipped_empty"] + global_tally.get("duplicate_sha", 0),
                "failed": global_tally["errors"] + global_tally["zero_chunk"]
                + global_tally["download_failed"],
                "unsupported": global_tally["skipped_unsupported"],
                "already_indexed": global_tally["already_indexed"],
                "exit_reason": exit_reason,
                "complete": True,
                "run_complete": True,
                "accounting": dict(accounting),
                "folder_accounting": [dict(entry) for entry in folder_accounting],
                "walk_error_messages": list(walk_error_messages),
                "skipped_no_folder_id_folders": list(skipped_no_folder_id_folders),
                "durations_sec": {"_total": round(time.monotonic() - t0_global, 3)},
                "global_tally": global_tally,
                "folder_tallies": folder_tallies,
                "results": results,
            },
        )

    def _flush_final(reason: str, complete: bool) -> None:
        """The run's last words — report, optional shard, tally, RUNSUMMARY.

        Called exactly once by ``RunLifecycle`` on every exit this process can
        observe (clean finish, stop signal after drain, exception, SystemExit,
        atexit). SIGKILL still cannot reach here; the sidecar heartbeat is
        what names that death.

        The shard manifest stays TERMINAL evidence: written only when every
        queued file is accounted for (``complete`` / ``run_complete``). Its
        absence is the on-disk discriminator between "the folder loop ran to
        the end" and "the process died".
        """
        _sync_outstanding()
        elapsed_global = time.monotonic() - t0_global
        if not dry_run:
            _write_partial_report(reason, complete)
            if complete:
                manifest = build_shard_manifest(reason)
                atomic_write_json(durable_shard_path, manifest)
                atomic_write_json(legacy_shard_path, manifest)
                log(f"report written to {report_path}")
                log(f"shard manifest → {durable_shard_path} (+ {legacy_shard_path})")
            else:
                log(
                    f"ABORTED report written to {report_path} "
                    f"(exit_reason={reason}, "
                    f"attempted={accounting['attempted']}/"
                    f"batch={accounting['batch']}); "
                    f"terminal shard manifest NOT written"
                )
        # Prefix the message with ``[p1b-server] `` so the #481 silent-exit
        # substring ``[p1b-server] tier 1:`` still matches inside a
        # run-attributed ``log()`` line.
        log(
            f"[p1b-server] tier {args.tier}: {global_tally['succeeded']} succeeded, "
            f"{global_tally['zero_chunk']} zero-chunk, "
            f"{global_tally['skipped_recoverable']} skipped-recoverable, "
            f"{global_tally['skipped_too_large']} too-large, "
            f"{global_tally['skipped_too_small']} too-small, "
            f"{global_tally['skipped_empty']} empty, "
            f"{global_tally['download_failed']} download-failed, "
            f"{global_tally['errors']} errors "
            f"quarantined={global_tally['quarantined']} "
            f"| attempted={accounting['attempted']}/batch={accounting['batch']} "
            f"assigned={accounting['assigned']} "
            f"already_indexed={global_tally['already_indexed']} "
            f"stale_extractor_open={global_tally['stale_extractor_open']} "
            f"retried={global_tally['retried']} "
            f"now_indexed={global_tally['now_indexed']} "
            f"unsupported={global_tally['skipped_unsupported']} "
            f"walk_errors={accounting['walk_errors']} "
            f"skipped_no_folder_id={accounting['skipped_no_folder_id']} "
            f"{elapsed_global:.1f}s shard={shard_index}/{total_shards} "
            f"dry_run={dry_run} exit_reason={reason} complete={complete}"
        )
        if not complete:
            log(
                f"RUN INCOMPLETE exit_reason={reason} — the tally "
                f"above counts only the files that finished. Re-run with "
                f"--resume to continue; nothing here marks the tier done."
            )
        _emit_run_summary(reason, complete)
        try:
            run_locks.close()
        except Exception as exc:  # noqa: BLE001 — release is best-effort
            log(
                f"WARN: releasing the run lock raised {type(exc).__name__}: {exc}; "
                "the Postgres session ends with this process, which releases it"
            )

    run = lifecycle.RunLifecycle(
        run_id=run_id,
        label=f"p1b-server tier={args.tier} shard={shard_index}/{total_shards}",
        state_path=run_state_path(args, dry_run=dry_run),
        flush=_flush_final,
        log=log,
        stall_after_s=float(os.getenv("P1B_STALL_WARN_S", "900")),
        heartbeat_every=max(1, int(os.getenv("P1B_HEARTBEAT_EVERY", "10"))),
    )
    run.start(
        context={
            "tier": args.tier,
            "output": str(report_path),
            "folders": len(folder_entries),
            "folder_ids": requested_ids,
            "shard_index": shard_index,
            "total_shards": total_shards,
            "dry_run": dry_run,
            "limit": limit,
            "offset": offset,
            "parallelism": int(os.getenv("P1B_PARALLELISM", "2")),
            "supervised": bool(os.getenv(SUPERVISED_ENV)),
            "argv": sys.argv[1:],
        },
    )

    if dry_run:
        # Fresh dry-run manifest each invocation (multi-folder appends below).
        dry_manifest = sharding.shard_manifest_path(shard_index, total_shards)
        if dry_manifest.exists():
            dry_manifest.unlink()

    for folder_entry in folder_entries:
        if run.should_stop():
            log(
                f"not starting further folders: "
                f"stop requested ({run.stop_reason()})"
            )
            break
        parent_folder_id = folder_entry.get("folder_id")
        folder_id = folder_entry.get("walk_folder_id") or parent_folder_id
        folder_name = folder_entry.get("folder_name") or folder_entry.get("project_id")
        project_id_for_folder = folder_entry.get("project_id")

        if not folder_id:
            accounting["skipped_no_folder_id"] += 1
            skipped_no_folder_id_folders.append({
                "folder": folder_name, "project_id": project_id_for_folder,
            })
            log(
                f"SKIP {folder_name}: no folder_id (needs the owner to set) — "
                "this folder contributes 0 to the tier tally, so run_complete "
                "does NOT mean the tier is covered"
            )
            continue

        if parent_folder_id and folder_id != parent_folder_id:
            log(
                f"walking {folder_name} subfolder ({folder_id}) "
                f"parent={parent_folder_id} ..."
            )
        else:
            log(f"walking {folder_name} ({folder_id}) ...")
        files, walk_errors = gdrive_service.walk_folder(folder_id, max_depth=12, page_size=200)
        if walk_errors:
            # A partial walk silently shrinks `discovered` and `supported`
            # together, so the identity re-balances against a smaller
            # denominator and the run still exits clean. Record the errors so
            # a reader can tell a complete walk from a truncated one.
            accounting["walk_errors"] += len(walk_errors)
            for err in walk_errors:
                walk_error_messages.append(f"{folder_name}: {err}")
                log(f"WALK ERROR: {err}")

        # Ensure ONE project row for this folder BEFORE sharding / resume.
        # projects.name is not UNIQUE — parallel shard workers that each
        # create_project(folder_name) would mint separate random ids and
        # split documents/chunks across projects. Prefer the manifest's
        # stable project_id; fall back to a name-derived id.
        already_indexed: set[str] = set()
        retry_doc_by_fid: Dict[str, Dict[str, Any]] = {}
        folder_stale_open = 0
        project_id: str | None = None
        if not dry_run:
            preferred_id = (project_id_for_folder or "").strip() or None
            project, created = projects_mod.get_or_create_project(
                folder_name,
                project_id=preferred_id,
                origin="user_drive_import",
            )
            project_id = project["id"]
            log(
                f"{'created' if created else 'using'} project "
                f"{project_id} for {folder_name} "
                f"(shard={shard_index}/{total_shards})"
            )

            try:
                run_locks.enter_context(run_advisory_lock(args.tier, project_id))
            except RunLockUnavailable as exc:
                log(f"ERROR: {exc}; refusing to start a second concurrent run")
                run.finish("aborted_run_lock_unavailable", complete=False)
                return 1
            except Exception as exc:  # noqa: BLE001 — surfaced as a hard abort
                # Fail closed for the same reason the resume check does. An
                # unhandled raise here would abort with a traceback and no
                # flushed evidence, which is the failure shape being removed.
                log(
                    f"ERROR: could not take the run lock "
                    f"({type(exc).__name__}: {exc}); aborting rather than "
                    f"ingesting without concurrent-run protection"
                )
                run.finish("aborted_run_lock_error", complete=False)
                return 1

            # Resume is ALWAYS on for live runs. A document row alone is
            # NOT proof of success — skip only files whose doc actually
            # has chunks AND is not an open stale-extractor .docx.
            # Zero-chunk docs and TEXT_SPARSE .docx stamped by an older
            # extractor are re-indexed in place (same row, same id).
            #
            # FAIL CLOSED. This used to fall back to row-presence resume on
            # any exception, which counts ZERO_CHUNK documents as indexed: one
            # transient Neon blip moved 40 known-broken docs from the retry
            # cohort into `already_indexed`, shrank `assigned` to match, and
            # the run then exited 0 with a clean tally and a shard manifest.
            # The identity `already_indexed + assigned == supported` held the
            # whole time. An unusable resume check must stop the run, not
            # quietly redefine what "already indexed" means.
            # Same fail-closed for stale .docx: if status / extractor_version
            # cannot be read, do not silently count the row as already_indexed.
            try:
                chunk_counts: Dict[str, int] = _vs.get_store().count_by_doc(project_id)
            except Exception as exc:  # noqa: BLE001 — surfaced as a hard abort below
                log(
                    f"ERROR: chunk-count resume check failed for {folder_name} "
                    f"({type(exc).__name__}: {exc}); aborting rather than "
                    f"degrading to row-presence resume, which would count "
                    f"zero-chunk documents as already indexed"
                )
                run.finish("aborted_chunk_count_unavailable", complete=False)
                return 1

            def _doc_is_unsupported(doc: Dict[str, Any]) -> bool:
                meta = doc.get("metadata") or {}
                path = meta.get("drive_path") or ""
                mime = meta.get("mimeType", "")
                from app.core import ingest_status as _ist_gate

                return (
                    mime in _UNSUPPORTED_MIMES
                    or not _ist_gate.is_ingestible(path)
                    or _is_geodatabase_internal(path)
                )

            from app.core import doc_index as _doc_index
            from app.core import ingest_status as ist
            from app.core.ingest_reconcile import should_skip_resume, source_content_token

            drive_by_id = {str(fm.get("id") or ""): fm for fm in files if fm.get("id")}

            folder_stale_open = 0
            for doc in projects_mod.list_documents(project_id):
                # Duplicate-content source files recorded against this row:
                # done while their content token is unchanged.
                for alias_fid, alias_token in ((doc.get("metadata") or {}).get("source_aliases") or {}).items():
                    current_token = source_content_token(drive_by_id.get(alias_fid))
                    if alias_fid in drive_by_id and (
                        not alias_token or not current_token or alias_token == current_token
                    ):
                        already_indexed.add(alias_fid)
                fid = (doc.get("metadata") or {}).get("drive_file_id")
                if not fid:
                    continue
                chunks = chunk_counts.get(doc["id"], 0)
                # Settled rows are done at ANY chunk count: a terminal skip
                # (jpg:terminal, empty_file:terminal) or a RECOVERABLE one
                # this build has no parser for is not work, and re-running
                # the indexer over it only rewrites the index (2026-10-03).
                if should_skip_resume(
                    doc, chunks, drive_by_id.get(fid),
                    parseable_exts=_doc_index._SUPPORTED_EXTS,
                ):
                    already_indexed.add(fid)
                    continue
                if chunks > 0:
                    # Open due to stale/missing extractor on .docx, or
                    # unreadable status/version (fail closed). Leave the
                    # file assigned so ~401 re-indexes the same row.
                    if ist.docx_stale_extractor_open(
                        doc.get("ingest_status"),
                        extension=ist.document_extension(doc),
                        extractor_version=doc.get("extractor_version"),
                    ) or (
                        ist.document_extension(doc) == ".docx"
                        and (
                            "ingest_status" not in doc
                            or "extractor_version" not in doc
                        )
                    ):
                        folder_stale_open += 1
                        retry_doc = dict(doc)
                        retry_doc["_stale_extractor_retry"] = True
                        if fid not in retry_doc_by_fid:
                            retry_doc_by_fid[fid] = retry_doc
                    elif fid not in retry_doc_by_fid:
                        retry_doc_by_fid[fid] = doc
                elif _doc_is_unsupported(doc):
                    # Legacy zero-chunk row for a file we now know is
                    # unsupported (e.g. geodatabase internals). Don't retry.
                    continue
                elif fid not in retry_doc_by_fid:
                    retry_doc_by_fid[fid] = doc
            if folder_stale_open:
                accounting["stale_extractor_open"] += folder_stale_open
                global_tally["stale_extractor_open"] += folder_stale_open

        # Drop unsupported before sharding so every worker sees the same
        # supported universe and sha256 assignment stays partition-complete.
        def _is_unsupported(fm: Dict[str, Any]) -> bool:
            from app.core import ingest_status as ist

            path = fm.get("_drive_path") or fm.get("name") or ""
            mime = fm.get("mimeType", "")
            return (
                mime in _UNSUPPORTED_MIMES
                or _is_geodatabase_internal(path)
                # The RAG ingests text formats only (TEXT_BEARING_EXTS, see
                # the module docstring): anything else is never assigned,
                # downloaded, stored or archived -- only counted.
                or not ist.is_ingestible(path)
                # A file Drive reports as zero bytes has nothing to ingest.
                # It used to be downloaded, found empty and dropped with no
                # record -- so every resume downloaded it again.
                or _drive_reports_empty(fm)
            )

        unsupported_files = [f for f in files if _is_unsupported(f)]
        supported_files = [f for f in files if not _is_unsupported(f)]
        for fm in unsupported_files:
            from app.core import ingest_status as _ist

            path = fm.get("_drive_path") or fm.get("name") or ""
            fmt = Path(path).suffix.lower() or "(none)"
            if _ist.is_ingestible(path) and _drive_reports_empty(fm):
                fmt = "(empty)"  # a text format, but nothing in it
            by_format = accounting.setdefault("unsupported_by_format", {})
            by_format[fmt] = by_format.get(fmt, 0) + 1
        filtered_files = [f for f in supported_files if f["id"] not in already_indexed]
        skipped_already = len(supported_files) - len(filtered_files)

        def _drive_identity(fm: Dict[str, Any]) -> str:
            return sharding.file_identity(
                drive_file_id=fm.get("id"),
                source_path=fm.get("_drive_path") or fm.get("name"),
            )

        def _drive_path(fm: Dict[str, Any]) -> str:
            return fm.get("_drive_path") or fm.get("name") or fm.get("id") or ""

        shard_files = sharding.filter_by_shard(
            filtered_files,
            identity_fn=_drive_identity,
            shard_index=shard_index,
            total_shards=total_shards,
        )
        # Sort before offset/limit so a limited batch picks the intended
        # end of the size distribution, while the shard partition itself
        # stays stable. Default smallest-first; set P1B_LARGEST_FIRST=1
        # to process largest files first (useful for quickly reaching real
        # documents during testing). Files with no reported size (0) go to
        # the end regardless.
        largest_first = os.getenv("P1B_LARGEST_FIRST", "").strip().lower() in {"1", "true", "yes"}

        def _sort_key(f: Dict[str, Any]) -> Tuple[bool, int]:
            size = int(f.get("size") or 0)
            # size==0 always sorts last; otherwise largest-first uses
            # negative size so the reverse flag is not needed.
            return (size == 0, -size if largest_first else size)

        shard_files.sort(key=_sort_key)
        # Offset/limit AFTER shard filter so assignment stays stable.
        batch_files = sharding.apply_offset_limit(
            shard_files, offset=offset, limit=limit,
        )
        log(
            f"{folder_name}: discovered={len(files)} "
            f"supported={len(supported_files)} unsupported={len(unsupported_files)} "
            f"already_indexed={skipped_already} "
            f"shard={shard_index}/{total_shards} assigned={len(shard_files)} "
            f"batch={len(batch_files)} "
            f"({len(retry_doc_by_fid)} in-place retries, "
            f"{folder_stale_open} stale-extractor-open)"
        )
        # Recorded BEFORE any file is processed, so a run that dies mid-folder
        # still leaves the denominator behind. `assigned` used to live only in
        # the line above, which meant a killed run left nothing to compare
        # progress against.
        folder_record: Dict[str, Any] = {
            "folder": folder_name,
            "folder_id": folder_id,
            "parent_folder_id": parent_folder_id,
            "project_id": project_id,
            "discovered": len(files),
            "supported": len(supported_files),
            "unsupported": len(unsupported_files),
            "already_indexed": skipped_already,
            "assigned": len(shard_files),
            "batch": len(batch_files),
            "attempted": 0,
            "zero_chunk_retries": len(retry_doc_by_fid),
            "stale_extractor_open": folder_stale_open,
            "retried": 0,
            "now_indexed": 0,
            "walk_errors": len(walk_errors),
        }
        folder_accounting.append(folder_record)
        accounting["discovered"] += len(files)
        accounting["supported"] += len(supported_files)
        accounting["unsupported"] += len(unsupported_files)
        accounting["already_indexed"] += skipped_already
        accounting["assigned"] += len(shard_files)
        accounting["batch"] += len(batch_files)

        global_tally["skipped_unsupported"] += len(unsupported_files)
        global_tally["already_indexed"] += skipped_already
        if skipped_already:
            log(f"skipped_already_indexed={skipped_already} for {folder_name}")

        if dry_run:
            report = sharding.dry_run_report(
                total_discovered=len(files),
                supported_items=filtered_files,
                unsupported_count=len(unsupported_files),
                shard_index=shard_index,
                total_shards=total_shards,
                identity_fn=_drive_identity,
                path_fn=_drive_path,
                already_indexed_count=skipped_already,
            )
            report.update({
                "folder": folder_name,
                "folder_id": folder_id,
                "run_id": run_id,
                "tier": args.tier,
                "offset": offset,
                "limit": limit,
                "batch_after_offset_limit": len(batch_files),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            })
            out_path = sharding.shard_manifest_path(shard_index, total_shards)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Merge multi-folder dry-runs into one manifest when possible.
            existing_report: Dict[str, Any] = {}
            if out_path.exists():
                try:
                    existing_report = json.loads(out_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    existing_report = {}
            folders = list(existing_report.get("folders") or [])
            folders.append(report)
            merged = {
                "dry_run": True,
                "db_writes": False,
                "shard_index": shard_index,
                "total_shards": total_shards,
                "run_id": run_id,
                "tier": args.tier,
                "folders": folders,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            atomic_write_json(out_path, merged)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            log(f"dry-run manifest → {out_path}")
            continue

        assert project_id is not None  # set above for live runs

        # Use the sharded (+offset/limit) batch for live work.
        filtered_files = batch_files

        t0_folder = time.monotonic()

        def _tally_result(rel: str, result: Dict[str, Any]) -> None:
            results.append({"folder": folder_name, "path": rel, "result": result})
            # One increment per completed future, whatever the outcome. This
            # is the number that gets compared against `batch`; the tally
            # buckets below cannot be summed for it because
            # `skipped_unsupported` also carries discovery-side files.
            accounting["attempted"] += 1
            folder_record["attempted"] += 1
            bucket = tally_key_for_index_result(result)
            global_tally[bucket] += 1
            _bump_folder(folder_name, bucket)
            if result.get("stale_extractor_retry"):
                accounting["retried"] += 1
                global_tally["retried"] += 1
                folder_record["retried"] = folder_record.get("retried", 0) + 1
                from app.core import ingest_status as ist
                ended = result.get("ingest_status") or ""
                if ended == ist.INDEXED:
                    accounting["now_indexed"] += 1
                    global_tally["now_indexed"] += 1
                    folder_record["now_indexed"] = (
                        folder_record.get("now_indexed", 0) + 1
                    )

        def _process_one(file_meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
            rel = file_meta.get("_drive_path") or file_meta.get("name", "")
            try:
                _, result = _ingest_file(
                    file_meta, project_id, data_dir, run_id, gdrive_service,
                    existing_doc=retry_doc_by_fid.get(file_meta["id"]),
                    reingest_of=getattr(args, "reingest", None),
                )
            except Exception as exc:  # noqa: BLE001 — one file must not kill the run
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            finally:
                # The next file's memory admission must see real headroom,
                # not this file's freed-but-retained heap.
                lifecycle.release_freed_memory()
            return rel, result

        # Parallel ingestion: downloads are network-bound and extraction
        # releases the GIL in fitz/torch. Keep only ``pool_size`` futures
        # in flight — submitting all 6k at once made the 4 GB worker OOM
        # after the first few SKIPPED_TOO_LARGE completions (the other
        # workers were already downloading multi-hundred-MB PDFs).
        # Ordering is already applied before offset/limit via P1B_LARGEST_FIRST.
        sort_label = "largest-first" if largest_first else "smallest-first"
        pool_size = max(1, int(os.getenv("P1B_PARALLELISM", "2")))
        # RLock, not Lock: only the main thread ever takes it (workers never
        # touch the tally), and the signal handler runs on that same thread —
        # a plain Lock would self-deadlock if the signal landed while the
        # main loop held it.
        tally_lock = threading.RLock()
        done_count = 0
        stop_announced = False
        log(f"parallel ingestion with {pool_size} workers "
            f"({sort_label}, {len(filtered_files)} files)")
        with ThreadPoolExecutor(max_workers=pool_size) as pool:
            pending: Dict[Future, Dict[str, Any]] = {}
            file_iter = iter(filtered_files)

            def _submit_next() -> bool:
                fm = next(file_iter, None)
                if fm is None:
                    return False
                pending[pool.submit(_process_one, fm)] = fm
                return True

            for _ in range(pool_size):
                if not _submit_next():
                    break
            while pending:
                done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    fm = pending.pop(fut, None)
                    rel, result = future_result_or_error(fut, fm)
                    with tally_lock:
                        done_count += 1
                        idx = done_count
                        _tally_result(rel, result)
                        status = result.get("error") or result.get("status", "ok")
                        log(
                            f"{folder_name} {idx}/{len(filtered_files)} "
                            f"[{status}] {rel}"
                        )
                        if idx % 10 == 0:
                            _write_partial_report()
                    # Heartbeat outside the tally lock: this also writes the
                    # sidecar, and the signal handler runs on this same thread.
                    run.note_progress(
                        idx,
                        len(filtered_files),
                        folder=folder_name,
                        in_flight=len(pending),
                    )
                    # A stop request stops SUBMITTING but still drains what is
                    # already running, so the tally covers every file actually
                    # attempted. At most ``pool_size`` files are in flight.
                    if not run.should_stop():
                        _submit_next()
                    elif not stop_announced:
                        stop_announced = True
                        log(
                            f"stop requested ({run.stop_reason()}) — "
                            f"draining {len(pending)} in-flight file(s), "
                            f"submitting no more"
                        )

        elapsed_folder = time.monotonic() - t0_folder
        log(f"{folder_name} done in {elapsed_folder:.1f}s")
        if run.should_stop():
            log(
                f"not starting further folders: "
                f"stop requested ({run.stop_reason()})"
            )
            break

    _sync_outstanding()
    # Exactly one final flush, here or from the atexit/excepthook fallback.
    # Completion is still ``attempted == batch``; a stop signal is never
    # complete even if the drain happened to empty the queue.
    if run.should_stop():
        run.finish(f"signal:{run.stop_reason()}", complete=False)
        exit_code = lifecycle.signal_exit_code(
            run.stop_flag.signum or getattr(signal, "SIGTERM", 15)
        )
    else:
        fully_accounted = dry_run or accounting["outstanding"] == 0
        from app.core import ingest_status as ist
        stale_complete = ist.stale_extractor_run_complete(
            stale_extractor_open=accounting["stale_extractor_open"],
            retried=accounting["retried"],
            otherwise_complete=fully_accounted,
        )
        if not fully_accounted:
            log(
                f"ERROR: {accounting['outstanding']} queued file(s) were never "
                f"accounted for; refusing to write a terminal shard manifest"
            )
            run.finish("incomplete_accounting", complete=False)
            exit_code = 1
        elif not stale_complete:
            log(
                f"ERROR: stale_extractor_open={accounting['stale_extractor_open']} "
                f"but retried=0; refusing to mark the run complete — this is "
                f"the gate that would have caught runs that skipped sparse "
                f".docx extracted before {ist.EXTRACTOR_VERSION}"
            )
            run.finish("stale_extractor_not_retried", complete=False)
            exit_code = 1
        else:
            run.finish(lifecycle.PHASE_COMPLETED, complete=True)
            exit_code = 0

    # ``--keep-alive`` held a finished container open so the always-on ECS
    # service would not restart it. A run-once task must exit, so the
    # deployment can switch it off with P1B_KEEP_ALIVE=0.
    keep_alive = args.keep_alive and os.getenv("P1B_KEEP_ALIVE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )
    if keep_alive and not dry_run and exit_code == 0:
        log("pass complete; keeping container alive for log inspection.")
        while True:
            time.sleep(3600)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
