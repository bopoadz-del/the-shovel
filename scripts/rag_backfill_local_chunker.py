#!/usr/bin/env python3
"""Local chunk generator for RAG backfill.

Reads a batch JSON, extracts text from local files using the platform's own
doc_index pipeline, chunks, and writes a JSONL of chunks. No DB, no object store, no
embeddings — just chunks.

Usage:
    .venv/Scripts/python scripts/rag_backfill_local_chunker.py \
        --batch rag_backfill_batch_5.json \
        --project-id example_infra_pack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core import doc_index  # noqa: E402


def _read_local_file(candidate: Dict[str, Any]) -> Optional[bytes]:
    local_path = candidate.get("local_path")
    if local_path and Path(local_path).exists():
        return Path(local_path).read_bytes()
    return None


def _write_jsonl(chunks: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def process_file(candidate: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    original_name = candidate["original_name"]
    doc_id = candidate["doc_id"]

    raw_bytes = _read_local_file(candidate)
    if raw_bytes is None:
        return {
            "doc_id": doc_id,
            "status": "ERROR",
            "original_name": original_name,
            "error": "local file not found",
        }

    suffix = Path(original_name).suffix or ".bin"
    local_path = work_dir / f"{doc_id}{suffix}"
    local_path.write_bytes(raw_bytes)

    try:
        text, meta = doc_index._extract_with_meta(str(local_path), original_name)
        chunks = doc_index.chunk_text(text, words_per_chunk=500)
        if not chunks:
            return {
                "doc_id": doc_id,
                "status": "ZERO_CHUNK",
                "original_name": original_name,
                "error": "no chunks produced",
                "extracted_text_len": len(text),
            }
        return {
            "doc_id": doc_id,
            "status": "OK",
            "original_name": original_name,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "extracted_text_len": len(text),
            "ocr_low_quality": meta.get("ocr_low_quality", False),
            "ocr_truncated": meta.get("ocr_truncated", False),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "doc_id": doc_id,
            "status": "ERROR",
            "original_name": original_name,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Local chunk generator")
    parser.add_argument("--batch", required=True, help="Path to batch JSON")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.batch, "r", encoding="utf-8") as f:
        batch_data = json.load(f)
    files = batch_data["files"]
    batch_id = batch_data["batch_id"]

    print(f"[chunker] batch {batch_id}: {len(files)} files, "
          f"{sum(c.get('size', 0) for c in files)/1024/1024:.1f} MB")

    out_dir = Path(args.out_dir) if args.out_dir else Path("rag_backfill_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(tempfile.mkdtemp(prefix="rag_chunk_"))
    all_chunks: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []

    for c in files:
        print(f"[chunker] {c['doc_id']} {c['original_name'][:55]}")
        result = process_file(c, work_dir)
        results.append(result)

        if result.get("status") == "OK":
            for i, txt in enumerate(result["chunks"]):
                all_chunks.append({
                    "project_id": args.project_id,
                    "doc_id": c["doc_id"],
                    "original_name": c["original_name"],
                    "chunk_index": i,
                    "text": txt,
                    "extracted_text_len": result.get("extracted_text_len"),
                    "ocr_low_quality": result.get("ocr_low_quality", False),
                    "ocr_truncated": result.get("ocr_truncated", False),
                })

    # Write per-doc chunk JSONLs
    for c in files:
        doc_chunks = [ch for ch in all_chunks if ch["doc_id"] == c["doc_id"]]
        if doc_chunks:
            _write_jsonl(doc_chunks, out_dir / f"{c['doc_id']}_chunks.jsonl")

    # Write combined batch JSONL
    batch_jsonl = out_dir / f"batch_{batch_id}_chunks.jsonl"
    _write_jsonl(all_chunks, batch_jsonl)

    # Write report
    report = {
        "batch_id": batch_id,
        "project_id": args.project_id,
        "total_files": len(files),
        "ok": sum(1 for r in results if r.get("status") == "OK"),
        "zero_chunk": sum(1 for r in results if r.get("status") == "ZERO_CHUNK"),
        "error": sum(1 for r in results if r.get("status") == "ERROR"),
        "total_chunks": len(all_chunks),
        "results": results,
    }
    report_path = out_dir / f"batch_{batch_id}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[chunker] OK: {report['ok']}/{len(files)}")
    print(f"[chunker] zero_chunk: {report['zero_chunk']}")
    print(f"[chunker] error: {report['error']}")
    print(f"[chunker] total chunks: {len(all_chunks)}")
    print(f"[chunker] output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
