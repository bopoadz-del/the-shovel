#!/usr/bin/env python3
"""Platform-native RAG backfill for existing zero-chunk documents.

Reads a manifest of document IDs and re-indexes each one through the
platform's own doc_index pipeline (fitz text layer + OCR + chunking +
embedding). Intended to run on Render where tesseract is installed.

Usage (inside Render worker container, from /app):
    python scripts/rag_backfill_platform_index.py \
        --manifest rag_backfill_indexable_candidates.json \
        --project-id example_infra_pack \
        --batch-index 1

Required env:
    DATABASE_URL, DATA_DIR,
    GDRIVE_SERVICE_ACCOUNT_JSON (Google Drive is the only source of an original),
    RAG_EMBEDDING_MODEL, RAG_VECTOR_NAMESPACE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")

from app.core import doc_index, file_crypto, gdrive_service, projects as projects_mod


def _safe_stored_name(original: str) -> str:
    base = Path(original).stem
    ext = Path(original).suffix
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in base)[:80]
    return f"{safe}{ext}"






def _index_one(
    doc_id: str,
    project_id: str,
    data_dir: Path,
    candidate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "doc_id": doc_id,
        "project_id": project_id,
        "status": "pending",
        "error": None,
        "rag_indexed": 0,
        "chunk_count": 0,
        "extracted_text_len": 0,
    }

    doc = projects_mod.get_document(doc_id)
    if doc is None:
        result["status"] = "ERROR"
        result["error"] = "document row not found"
        return result

    original_name = doc.get("original_name") or (candidate or {}).get("original_name", "unknown")
    result["original_name"] = original_name

    # Google Drive is the only source of an original (no archive).
    drive_id = projects_mod.extract_document_source_pointers(doc)["drive_file_id"]
    if not drive_id:
        result["status"] = "ERROR"
        result["error"] = "no drive_file_id on this document"
        return result

    raw_bytes, err = gdrive_service.download_file_bytes(drive_id)
    if err or raw_bytes is None:
        result["status"] = "ERROR"
        result["error"] = err or "download returned no bytes"
        return result

    content_sha = hashlib.sha256(raw_bytes).hexdigest()
    stored_name = _safe_stored_name(original_name)
    dest = data_dir / f"{content_sha[:16]}_{stored_name}"
    file_crypto.write_document(str(dest), raw_bytes)
    del raw_bytes

    # Point the row at this download so index_document reads it.
    projects_mod.set_document_file_path(doc_id, str(dest))

    # Index through platform pipeline.
    idx_result = doc_index.index_document(project_id, doc_id)
    result["rag_indexed"] = int(idx_result.get("rag_indexed", 0) or idx_result.get("indexed", 0))
    result["status"] = "OK" if result["rag_indexed"] > 0 else "ZERO_CHUNK"
    if idx_result.get("error"):
        result["error"] = str(idx_result["error"])

    # Introspection for reporting.
    try:
        text, meta = doc_index._extract_with_meta(str(dest), original_name)
        result["extracted_text_len"] = len(text)
        result["chunk_count"] = len(doc_index.chunk_text(text))
        if meta.get("ocr_low_quality"):
            result["ocr_low_quality"] = True
        if meta.get("ocr_truncated"):
            result["ocr_truncated"] = True
    except Exception as exc:  # noqa: BLE001
        result["introspection_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        # The platform keeps no copy of an original: Drive is the source.
        dest.unlink(missing_ok=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Platform-native RAG backfill indexer")
    parser.add_argument("--manifest", required=True, help="Path to candidates+batches JSON")
    parser.add_argument("--batch-index", type=int, required=True, help="1-based batch index")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--resume", action="store_true", help="Skip docs already rag_indexed > 0")
    args = parser.parse_args()

    if not gdrive_service.is_configured():
        print("WARNING: GDRIVE_SERVICE_ACCOUNT_JSON not set; Drive-sourced docs will fail", file=sys.stderr)

    with open(args.manifest, "r", encoding="utf-8") as f:
        data = json.load(f)
    batches = data["batches"]

    if args.batch_index < 1 or args.batch_index > len(batches):
        print(f"Invalid batch index {args.batch_index}; {len(batches)} batches available", file=sys.stderr)
        return 1

    batch = batches[args.batch_index - 1]
    print(f"[platform-index] batch {args.batch_index}/{len(batches)}: {len(batch)} docs, "
          f"{sum(c.get('size', 0) for c in batch)/1024/1024:.1f} MB")

    data_dir = Path(args.data_dir) if args.data_dir else Path(os.getenv("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for c in batch:
        doc_id = c["doc_id"]
        print(f"[platform-index] processing {doc_id} {c.get('original_name', '')[:60]}")

        if args.resume:
            doc = projects_mod.get_document(doc_id)
            if doc and (doc.get("metadata") or {}).get("rag_indexed", 0) > 0:
                print(f"[platform-index] {doc_id}: already indexed, skipping")
                results.append({"doc_id": doc_id, "status": "SKIPPED_ALREADY_INDEXED"})
                continue

        result = _index_one(doc_id, args.project_id, data_dir, candidate=c)
        results.append(result)
        print(f"[platform-index] {doc_id}: status={result['status']} rag_indexed={result['rag_indexed']} "
              f"chunks={result['chunk_count']} text_len={result['extracted_text_len']}")

    report_path = data_dir / f"platform_index_batch_{args.batch_index}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "project_id": args.project_id,
            "batch_index": args.batch_index,
            "manifest": args.manifest,
            "results": results,
            "summary": {
                "total": len(results),
                "ok": sum(1 for r in results if r.get("status") == "OK"),
                "zero_chunk": sum(1 for r in results if r.get("status") == "ZERO_CHUNK"),
                "error": sum(1 for r in results if r.get("status") == "ERROR"),
                "skipped": sum(1 for r in results if "SKIPPED" in str(r.get("status", ""))),
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"[platform-index] report written to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
