#!/usr/bin/env python3
"""Gap-fill Drive imports per configured project mapping.

Walks each Drive folder in GDRIVE_PROJECT_FOLDERS, compares the files found
against local documents whose metadata records a drive_file_id, and imports
any missing files. Dry-run by default; pass --execute to write.

Exit code: 0 (even when zero_chunk errors are reported; check the manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from typing import Any, Dict, List, Set, Tuple

from sqlalchemy import select, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("reconcile_drive_delta")


def _local_docs_by_drive_file_id(project_id: str) -> Dict[str, Dict[str, Any]]:
    """Return {drive_file_id: {id, original_name, metadata}} for a project.

    Uses the Postgres JSONB operator when available; otherwise loads rows and
    filters in Python.
    """
    from app.core.db import SessionLocal, get_engine
    from app.core.models import Document

    engine = get_engine()
    rows: List[Dict[str, Any]] = []
    if engine.dialect.name == "postgresql":
        with SessionLocal() as session:
            result = session.execute(
                text(
                    "SELECT id, original_name, metadata, drive_md5, "
                    "ingest_status, chunk_count FROM documents "
                    "WHERE project_id = :pid AND metadata ? 'drive_file_id'"
                ),
                {"pid": project_id},
            ).mappings()
            rows = [dict(r) for r in result]
    else:
        with SessionLocal() as session:
            docs = session.scalars(
                select(Document).where(Document.project_id == project_id)
            ).all()
        rows = [
            {
                "id": d.id,
                "original_name": d.original_name,
                "metadata": d.metadata_,
                "drive_md5": getattr(d, "drive_md5", None),
                "ingest_status": getattr(d, "ingest_status", None),
                "chunk_count": int(getattr(d, "chunk_count", 0) or 0),
            }
            for d in docs
            if d.metadata_ and d.metadata_.get("drive_file_id")
        ]

    return {
        (r["metadata"] or {}).get("drive_file_id"): r
        for r in rows
        if (r["metadata"] or {}).get("drive_file_id")
    }


def _import_missing_file(
    project_id: str,
    f_meta: Dict[str, Any],
    execute: bool,
) -> Tuple[bool, bool, str]:
    """Download, write, register and index one Drive file.

    Returns (imported, zero_chunk_error, message).
    """
    from app.core import doc_index, file_crypto, gdrive_service, projects
    from app.core.ingest_reconcile import source_content_token

    fid = f_meta["id"]
    name = (f_meta.get("name") or "").strip() or fid
    drive_path = f_meta.get("_drive_path", "")

    if not execute:
        return True, False, f"would import {name} ({fid})"

    blob, dl_err = gdrive_service.download_file_bytes(fid)
    if blob is None:
        return False, False, f"download failed for {name}: {dl_err}"

    content_sha = hashlib.sha256(blob).hexdigest()
    existing = projects.find_document_by_sha(project_id, content_sha)
    if existing is not None:
        return False, False, (
            f"DUPLICATE_SHA {name} ({fid}) existing_id={existing['id']}"
        )
    data_dir = os.getenv("DATA_DIR", "data")
    os.makedirs(data_dir, exist_ok=True)
    stored_as = f"{hashlib.sha256(fid.encode()).hexdigest()[:8]}_{name}"
    stored_as = os.path.basename(stored_as.replace("\\", "/"))
    filepath = os.path.join(data_dir, stored_as)
    file_crypto.write_document(filepath, blob)

    doc = projects.add_document(
        project_id=project_id,
        original_name=name,
        stored_as=stored_as,
        file_path=filepath,
        size=len(blob),
        content_sha256=content_sha,
        metadata={
            "drive_file_id": fid,
            "drive_path": drive_path,
            "source": "reconcile_drive_delta",
        },
        drive_md5=source_content_token(f_meta),
    )

    idx_result = doc_index.index_document(project_id, doc["id"])
    if idx_result.get("status") == "error":
        return True, True, (
            f"imported {name} ({fid}) as {doc['id']} but ZERO_CHUNK: "
            f"{idx_result.get('banner', '')}"
        )

    return True, False, (
        f"imported {name} ({fid}) as {doc['id']} "
        f"({idx_result.get('total_chunks', 0)} chunks)"
    )


def _reconcile_project(
    project_id: str,
    folder_id: str,
    execute: bool,
) -> Dict[str, Any]:
    from app.core import doc_index, gdrive_service, projects
    from app.core.ingest_reconcile import (
        apply_tombstones,
        plan_reconcile,
        resume_source_changed,
        source_content_token,
    )

    logger.info("Reconciling project=%s folder=%s", project_id, folder_id)
    local_before = _local_docs_by_drive_file_id(project_id)
    local_docs_before = len(local_before)

    files, walk_errors = gdrive_service.walk_folder(folder_id)
    if walk_errors:
        for err in walk_errors:
            logger.warning("walk error: %s", err)

    seen_ids: Set[str] = set()
    imported = 0
    reindexed = 0
    zero_chunk_errors = 0

    for f_meta in files:
        fid = f_meta.get("id")
        if not fid:
            continue
        seen_ids.add(fid)
        if not gdrive_service.is_downloadable(f_meta):
            continue
        existing = local_before.get(fid)
        if existing is not None:
            if resume_source_changed(existing, f_meta):
                if execute:
                    token = source_content_token(f_meta)
                    if token:
                        projects.set_document_drive_md5(existing["id"], token)
                    idx = doc_index.index_document(project_id, existing["id"])
                    if idx.get("status") == "error":
                        zero_chunk_errors += 1
                    else:
                        reindexed += 1
                    logger.info(
                        "source_changed reindex %s (%s) as %s",
                        existing.get("original_name"), fid, existing["id"],
                    )
                else:
                    reindexed += 1
                    logger.info(
                        "would reindex %s (%s) source_changed",
                        existing.get("original_name"), fid,
                    )
            else:
                logger.debug("Skipping already-local file %s", fid)
            continue
        ok, zero_chunk, msg = _import_missing_file(project_id, f_meta, execute)
        if ok:
            imported += 1
        if zero_chunk:
            zero_chunk_errors += 1
        logger.info(msg)

    plan = plan_reconcile(
        local_docs=list(local_before.values()),
        drive_files=files,
        walk_complete=not walk_errors,
    )
    tombstoned = apply_tombstones(plan.to_tombstone, execute=execute)
    if plan.skipped_tombstones_reason:
        logger.warning(
            "tombstones skipped: %s (walk incomplete)",
            plan.skipped_tombstones_reason,
        )
    elif plan.to_tombstone:
        logger.info(
            "%s %s gone-from-Drive row(s)",
            "tombstoned" if execute else "would tombstone",
            len(plan.to_tombstone),
        )

    # Refresh count after import.
    local_after = _local_docs_by_drive_file_id(project_id) if execute else local_before
    local_docs_after = len(local_after)

    return {
        "project_id": project_id,
        "folder": folder_id,
        "drive_files_seen": len(seen_ids),
        "local_docs_before": local_docs_before,
        "local_docs_after": local_docs_after,
        "imported": imported,
        "reindexed": reindexed,
        "tombstoned": tombstoned if execute else len(plan.to_tombstone),
        "zero_chunk_errors": zero_chunk_errors,
        "walk_errors": walk_errors,
        "index_count": plan.index_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gap-fill Drive files missing from local documents."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually download, write and index missing files (default is dry-run).",
    )
    parser.add_argument(
        "--project",
        dest="projects",
        action="append",
        help="Restrict to one or more project_ids (can be repeated).",
    )
    args = parser.parse_args()

    from app.core import gdrive_service

    mapping = gdrive_service.parse_project_folder_map()
    if not mapping:
        logger.error("GDRIVE_PROJECT_FOLDERS is empty or malformed")
        sys.exit(1)

    projects_to_run = args.projects or list(mapping.keys())
    manifest: List[Dict[str, Any]] = []

    for project_id in projects_to_run:
        folder_id = mapping.get(project_id)
        if not folder_id:
            logger.warning("No Drive folder configured for %s; skipping", project_id)
            continue
        row = _reconcile_project(project_id, folder_id, execute=args.execute)
        manifest.append(row)

    # Print manifest table.
    print("\n" + "=" * 100)
    header = (
        f"{'project_id':<30} {'folder':<25} {'seen':>6} "
        f"{'before':>7} {'after':>7} {'imported':>9} {'zero_chunk':>11}"
    )
    print(header)
    print("-" * len(header))
    for row in manifest:
        print(
            f"{row['project_id']:<30} {row['folder']:<25} "
            f"{row['drive_files_seen']:>6} {row['local_docs_before']:>7} "
            f"{row['local_docs_after']:>7} {row['imported']:>9} "
            f"{row['zero_chunk_errors']:>11}"
        )
    print("=" * 100)

    total_imported = sum(r["imported"] for r in manifest)
    total_zero_chunk = sum(r["zero_chunk_errors"] for r in manifest)
    print(f"\nTotal imported: {total_imported}, zero-chunk errors: {total_zero_chunk}")

    if total_zero_chunk > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
