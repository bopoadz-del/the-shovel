#!/usr/bin/env python3
"""P1b — controlled re-ingestion of one local Drive folder through the normal pipeline.

Creates a project, adds every file under the local folder as a document, runs
``doc_index.index_document`` for each, and verifies chunks land in the v2
namespace. Unsupported files (e.g. .mp4) are recorded as skipped, not silently
lost. Logs every failure with the file path so Phase 1c reconciliation is
auditable.

Horizontal scale (temporary identical workers):
  --total-shards / INGEST_TOTAL_SHARDS
  --shard-index  / INGEST_SHARD_INDEX
  --dry-run      / INGEST_DRY_RUN
  --limit        / INGEST_LIMIT   (applied AFTER shard filter)
  --offset       / INGEST_OFFSET  (applied AFTER shard filter)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Set the approved embedder and namespace BEFORE any module imports.
os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_UNSUPPORTED_EXTS = {".gdoc", ".gsheet", ".gslides", ".gdraw", ".rar"}


def _safe_stored_name(original: str) -> str:
    """Filesystem-safe stored name; preserves extension."""
    base = Path(original).stem
    ext = Path(original).suffix
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in base)[:80]
    return f"{safe}{ext}"


def _rel_path(src_path: Path, drive_root: Path) -> str:
    return str(src_path.relative_to(drive_root)).replace("\\", "/")


def _is_unsupported(src_path: Path) -> bool:
    return src_path.suffix.lower() in _UNSUPPORTED_EXTS


def _ingest_file(
    src_path: Path,
    project_id: str,
    data_dir: Path,
    drive_root: Path,
    reingest_of: str | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """Copy one file into the project and index it. Returns (relative_path, result)."""
    from app.core import doc_index, projects as projects_mod

    rel = _rel_path(src_path, drive_root)

    if _is_unsupported(src_path):
        return rel, {
            "status": "error",
            "error": "SKIPPED_UNSUPPORTED",
            "reason": "Google Workspace native or unsupported archive placeholder",
        }

    try:
        size = src_path.stat().st_size
    except OSError as exc:
        return rel, {
            "status": "error",
            "error": "SKIPPED_UNREADABLE",
            "reason": f"Cannot stat file on Drive mount: {exc}",
        }

    # Skip multi-hundred-MB files before copying — the doc_index PDF guard also
    # rejects them, but copying them from the Drive mount is itself a timeout/OOM
    # risk on the 2 GB worker.
    max_size = int(os.getenv("P1B_MAX_FILE_SIZE_MB", "100")) * 1024 * 1024
    if max_size > 0 and size > max_size:
        return rel, {
            "status": "error",
            "error": "SKIPPED_TOO_LARGE",
            "size_mb": round(size / (1024 * 1024), 1),
        }

    try:
        content_sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    except OSError as exc:
        return rel, {
            "status": "error",
            "error": "SKIPPED_UNREADABLE",
            "reason": f"Cannot read file on Drive mount: {exc}",
        }

    existing_by_sha = projects_mod.find_document_by_sha(project_id, content_sha)
    reingest_of = projects_mod.scoped_reingest_of(
        reingest_of, content_sha, existing_by_sha,
    )
    if existing_by_sha and not reingest_of:
        return rel, {
            "status": "error",
            "error": "DUPLICATE_SHA",
            "existing_id": existing_by_sha["id"],
            "content_sha256": content_sha,
        }

    stored_name = (
        f"{hashlib.sha256(str(src_path).encode()).hexdigest()[:8]}_"
        f"{_safe_stored_name(src_path.name)}"
    )
    dest = data_dir / stored_name
    shutil.copy2(src_path, dest)

    try:
        doc = projects_mod.add_document(
            project_id=project_id,
            original_name=src_path.name,
            stored_as=stored_name,
            file_path=str(dest),
            size=size,
            content_sha256=content_sha,
            metadata={"local_drive_path": rel, "source": "p1b_local_reingestion"},
            reingest_of=reingest_of,
        )
    except projects_mod.DuplicateContentError as exc:
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
    return rel, result


def main() -> int:
    from app.core import ingest_sharding as sharding

    ap = argparse.ArgumentParser(description="Ingest one local Drive folder into v2")
    ap.add_argument("--folder", default="construction-3-001",
                    help="Folder name under the Drive root")
    ap.add_argument("--project-name", default=None,
                    help="Platform project name (default: folder name)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Ingest at most N files after shard filter (0 = all)")
    ap.add_argument("--offset", type=int, default=None,
                    help="Skip the first N files after shard filter")
    ap.add_argument("--output", default="manifests/p1b_ingestion_report.json",
                    help="Report path")
    ap.add_argument("--resume", action="store_true",
                    help="Skip files already indexed in the project (by local_drive_path)")
    ap.add_argument("--total-shards", type=int, default=None,
                    help="Total worker shards (env: INGEST_TOTAL_SHARDS)")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="This worker's shard index (env: INGEST_SHARD_INDEX)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan only — no DB writes (env: INGEST_DRY_RUN)")
    ap.add_argument("--drive-root", default=os.environ.get("FORK_DRIVE_ROOT", "/data/example"),
                    help="Local Drive mount root")
    ap.add_argument(
        "--reingest",
        default=None,
        metavar="OLD_ID",
        help=(
            "Create a new row for a sha256 that already exists and hide "
            "OLD_ID (retrieval_visible=false). Without this flag a "
            "duplicate sha is refused."
        ),
    )
    args = ap.parse_args()

    try:
        total_shards, shard_index = sharding.resolve_shard_env(
            total_shards=args.total_shards,
            shard_index=args.shard_index,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    dry_run = sharding.resolve_dry_run(args.dry_run)
    limit, offset = sharding.resolve_limit_offset(limit=args.limit, offset=args.offset)

    print(
        f"[p1b] shard identity: shard_index={shard_index} total_shards={total_shards} "
        f"dry_run={dry_run}",
        file=sys.stderr,
    )

    drive_root = Path(args.drive_root)
    folder_path = drive_root / args.folder
    if not folder_path.exists():
        print(f"ERROR: folder not found: {folder_path}", file=sys.stderr)
        return 1

    files = sorted(p for p in folder_path.rglob("*") if p.is_file())
    total_discovered = len(files)

    unsupported = [p for p in files if _is_unsupported(p)]
    supported = [p for p in files if not _is_unsupported(p)]

    def _identity(p: Path) -> str:
        return sharding.file_identity(source_path=_rel_path(p, drive_root))

    def _path(p: Path) -> str:
        return _rel_path(p, drive_root)

    # Already-indexed skip (chunks present). In dry-run we still report counts
    # but never write. Resume / default skip both use path identity.
    already_indexed: Set[str] = set()
    project_id: str | None = None
    project_name = args.project_name or args.folder

    if not dry_run:
        from app.core import projects as projects_mod
        from app.core.rag import embeddings as _emb, vector_store as _vs

        _emb.reset_embedder_cache()
        _vs.reset_store_cache()

        data_dir = Path(os.getenv("DATA_DIR", "./data"))
        data_dir.mkdir(parents=True, exist_ok=True)

        # Atomic get-or-create BEFORE shard filtering so parallel workers
        # share one project id (projects.name is not UNIQUE).
        project, created = projects_mod.get_or_create_project(
            project_name,
            origin="user_drive_import",
        )
        project_id = project["id"]
        print(
            f"[p1b] {'created' if created else 'using'} project {project_id} "
            f"for {project_name} (shard={shard_index}/{total_shards})",
            file=sys.stderr,
        )

        # Idempotency: skip when the doc already has chunks. If chunk counts
        # are unavailable, --resume falls back to path-row presence.
        chunk_counts: Dict[str, int] = {}
        counts_available = True
        try:
            chunk_counts = _vs.get_store().count_by_doc(project_id)
        except Exception as exc:  # noqa: BLE001
            counts_available = False
            print(
                f"[p1b] WARN: chunk-count check failed ({type(exc).__name__}: {exc}); "
                f"falling back to path-row resume={args.resume}",
                file=sys.stderr,
            )

        from app.core import ingest_status as ist

        for doc in projects_mod.list_documents(project_id):
            rel = (doc.get("metadata") or {}).get("local_drive_path")
            if not rel:
                continue
            rel_n = rel.replace("\\", "/")
            if counts_available:
                chunks = chunk_counts.get(doc["id"], 0)
                if chunks > 0 and ist.resume_is_already_indexed(doc, chunks):
                    already_indexed.add(rel_n)
            elif args.resume:
                already_indexed.add(rel_n)

        print(
            f"[p1b] {len(already_indexed)} files already indexed (skipped_already_indexed)",
            file=sys.stderr,
        )
    else:
        data_dir = Path(os.getenv("DATA_DIR", "./data"))

    work_supported = [
        p for p in supported
        if _path(p) not in already_indexed
    ]
    skipped_already = len(supported) - len(work_supported)

    # Shard AFTER unsupported filter and already-indexed filter.
    shard_files = sharding.filter_by_shard(
        work_supported,
        identity_fn=_identity,
        shard_index=shard_index,
        total_shards=total_shards,
    )
    # Offset/limit AFTER shard so assignment is stable across workers.
    batch_files = sharding.apply_offset_limit(
        shard_files, offset=offset, limit=limit,
    )

    if dry_run:
        report = sharding.dry_run_report(
            total_discovered=total_discovered,
            supported_items=work_supported,
            unsupported_count=len(unsupported),
            shard_index=shard_index,
            total_shards=total_shards,
            identity_fn=_identity,
            path_fn=_path,
            already_indexed_count=skipped_already,
        )
        report.update({
            "folder": args.folder,
            "folder_path": str(folder_path),
            "offset": offset,
            "limit": limit,
            "batch_after_offset_limit": len(batch_files),
            "embedder": os.environ["RAG_EMBEDDING_MODEL"],
            "namespace": os.environ["RAG_VECTOR_NAMESPACE"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        out_path = sharding.shard_manifest_path(shard_index, total_shards)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        # Also echo human summary.
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"[p1b] dry-run manifest written to {out_path}", file=sys.stderr)
        print(
            f"[p1b] dry-run: discovered={total_discovered} supported={len(supported)} "
            f"unsupported={len(unsupported)} already_indexed={skipped_already} "
            f"shard_assigned={report['files_assigned_to_this_shard']} "
            f"batch={len(batch_files)} per_shard={report['per_shard_counts']}",
            file=sys.stderr,
        )
        return 0

    # Live ingestion path.
    from app.core import projects as projects_mod  # noqa: F401 — used via _ingest_file

    successes = 0
    zero_chunk = 0
    skipped_too_large = 0
    skipped_unsupported = len(unsupported)
    skipped_unreadable = 0
    skipped_already_indexed = skipped_already
    errors: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    durations: Dict[str, float] = {}
    ext_ok: Counter[str] = Counter()

    manifest = sharding.empty_shard_manifest(
        shard_index=shard_index,
        total_shards=total_shards,
        extra={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "project_name": project_name,
            "folder": args.folder,
            "folder_path": str(folder_path),
            "embedder": os.environ["RAG_EMBEDDING_MODEL"],
            "namespace": os.environ["RAG_VECTOR_NAMESPACE"],
            "total_files": total_discovered,
            "supported_files": len(supported),
            "batch_offset": offset,
            "batch_limit": limit,
            "batch_files": len(batch_files),
            "already_indexed": skipped_already_indexed,
            "unsupported": skipped_unsupported,
        },
    )

    def _write_manifest() -> None:
        manifest.update({
            "processed": successes,
            "skipped": skipped_too_large + skipped_unreadable,
            "failed": len(errors) + zero_chunk,
            "unsupported": skipped_unsupported,
            "already_indexed": skipped_already_indexed,
            "durations_sec": durations,
            "results": results,
            "successes": successes,
            "zero_chunk": zero_chunk,
            "skipped_too_large": skipped_too_large,
            "skipped_unreadable": skipped_unreadable,
            "errors": errors,
            "extension_ok": dict(ext_ok),
        })
        shard_out = sharding.shard_manifest_path(shard_index, total_shards)
        shard_out.parent.mkdir(parents=True, exist_ok=True)
        with open(shard_out, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)

    t0 = time.monotonic()
    for idx, src_path in enumerate(batch_files, start=1):
        rel = _path(src_path)
        print(f"[p1b] {idx}/{len(batch_files)} {rel}", file=sys.stderr)
        t_file = time.monotonic()
        try:
            _, result = _ingest_file(
                src_path, project_id, data_dir, drive_root,
                reingest_of=args.reingest,
            )
            results.append({"path": rel, "identity": _identity(src_path), "result": result})
            if result.get("status") == "error":
                if result.get("error") == "ZERO_CHUNK":
                    zero_chunk += 1
                elif result.get("error") == "SKIPPED_TOO_LARGE":
                    skipped_too_large += 1
                elif result.get("error") == "SKIPPED_UNSUPPORTED":
                    skipped_unsupported += 1
                elif result.get("error") == "SKIPPED_UNREADABLE":
                    skipped_unreadable += 1
                elif result.get("error") == "DUPLICATE_SHA":
                    errors.append({
                        "path": rel,
                        "error": "DUPLICATE_SHA",
                        "existing_id": result.get("existing_id"),
                    })
                else:
                    errors.append({"path": rel, "error": result.get("error")})
            elif result.get("rag_indexed", 0) == 0:
                errors.append({"path": rel, "error": "no RAG chunks indexed"})
            else:
                successes += 1
                ext_ok[src_path.suffix.lower() or "(none)"] += 1
        except Exception as exc:
            errors.append({"path": rel, "error": f"{type(exc).__name__}: {exc}"})
        durations[rel] = round(time.monotonic() - t_file, 3)
        if idx % 10 == 0:
            _write_manifest()

    elapsed = time.monotonic() - t0
    durations["_total"] = round(elapsed, 3)
    _write_manifest()
    print(
        f"[p1b] shard {shard_index}/{total_shards} report → "
        f"{sharding.shard_manifest_path(shard_index, total_shards)}",
        file=sys.stderr,
    )
    print(
        f"[p1b] shard {shard_index}/{total_shards}: "
        f"{successes} succeeded, {zero_chunk} zero-chunk, {skipped_too_large} too-large, "
        f"{skipped_unsupported} unsupported, {skipped_already_indexed} already_indexed, "
        f"{skipped_unreadable} unreadable, {len(errors)} errors, {elapsed:.1f}s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
