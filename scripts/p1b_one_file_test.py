#!/usr/bin/env python3
"""One-file end-to-end Drive ingestion test.

Usage (inside the Render worker container, from /app):
    python scripts/p1b_one_file_test.py <drive_file_id> <project_id> [drive_path]

Reports every stage so a human can verify:
    Drive -> document row -> parsed text -> chunks -> embeddings -> rag_indexed
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Defaults for the Render worker environment.
os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")

from app.core import doc_index, file_crypto, gdrive_service, projects as projects_mod


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/p1b_one_file_test.py <drive_file_id> <project_id> [drive_path]", file=sys.stderr)
        return 2

    drive_file_id = sys.argv[1]
    project_id = sys.argv[2]
    drive_path = sys.argv[3] if len(sys.argv) > 3 else drive_file_id

    print(f"[one-file-test] drive_file_id={drive_file_id} project_id={project_id} path={drive_path!r}")

    if not gdrive_service.is_configured():
        print("[one-file-test] FAIL: GDRIVE_SERVICE_ACCOUNT_JSON not configured", file=sys.stderr)
        return 1

    # 1. Fetch Drive metadata via httpx (googleapiclient is not in the worker image).
    try:
        import httpx
        token = gdrive_service._mint_access_token()
        if not token:
            raise RuntimeError("service account token unavailable")
        resp = httpx.get(
            f"https://www.googleapis.com/drive/v3/files/{drive_file_id}",
            headers={"Authorization": f"Bearer {token}"},
            params={"fields": "id,name,mimeType,size,modifiedTime"},
            timeout=30,
        )
        resp.raise_for_status()
        meta = resp.json()
        file_name = meta.get("name", "unknown")
        mime = meta.get("mimeType", "application/octet-stream")
        drive_size = int(meta.get("size") or 0)
        print(f"[one-file-test] KNOWN file_name={file_name} mime={mime} drive_size={drive_size}")
    except Exception as exc:
        print(f"[one-file-test] FAIL: Drive metadata lookup failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    # 2. Download raw bytes.
    raw_bytes, dl_err = gdrive_service.download_file_bytes(drive_file_id)
    if dl_err or not raw_bytes:
        print(f"[one-file-test] FAIL: download error={dl_err} bytes={len(raw_bytes) if raw_bytes else 0}", file=sys.stderr)
        return 1
    downloaded_bytes = len(raw_bytes)
    content_sha = hashlib.sha256(raw_bytes).hexdigest()
    print(f"[one-file-test] downloaded_bytes={downloaded_bytes} sha256={content_sha[:16]}...")

    # 3. Write local temp copy (inside DATA_DIR).
    data_dir = Path(os.getenv("DATA_DIR", "./data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in Path(file_name).stem)[:60]
    ext = Path(file_name).suffix
    stored_name = f"{hashlib.sha256(drive_path.encode()).hexdigest()[:8]}_{safe_name}{ext}"
    dest = data_dir / stored_name
    file_crypto.write_document(str(dest), raw_bytes)
    print(f"[one-file-test] local_temp={dest}")

    # 5. Create document row.
    common_meta = {
        "drive_file_id": drive_file_id,
        "drive_path": drive_path,
        "source": "p1b_one_file_test",
        "mimeType": mime,
        "content_sha256": content_sha,
    }

    doc = projects_mod.add_document(
        project_id=project_id,
        original_name=file_name,
        stored_as=stored_name,
        file_path=str(dest),
        size=drive_size,
        content_sha256=content_sha,
        metadata=common_meta,
    )
    doc_id = doc["id"]
    print(f"[one-file-test] document_id={doc_id}")

    # 6. Index.
    result = doc_index.index_document(project_id, doc_id)
    print(f"[one-file-test] index_status={result.get('status')} index_error={result.get('error')} rag_indexed={result.get('rag_indexed', 0)}")

    # 7. Cleanup local temp.
    Path(dest).unlink(missing_ok=True)  # no archive: originals stay in Drive

    # 8. Best-effort introspection of parsed text / chunk count.
    try:
        from app.document_engine.main import process_document
        proc = process_document(str(dest), original_filename=file_name)
        text_len = len(proc.get("text", ""))
        chunks = proc.get("chunks", [])
        print(f"[one-file-test] extracted_text_length={text_len} chunk_count={len(chunks)}")
    except Exception as exc:
        print(f"[one-file-test] introspection skipped: {type(exc).__name__}: {exc}")
        text_len = None
        chunks = []

    # Final structured report.
    report = {
        "known": {
            "file_name": file_name,
            "drive_id": drive_file_id,
            "drive_path": drive_path,
            "drive_size": drive_size,
            "mime_type": mime,
            "command": " ".join(sys.argv),
            "document_id": doc_id,
            "downloaded_bytes": downloaded_bytes,
            "extracted_text_length": text_len,
            "chunk_count": len(chunks),
            "embedding_insert_count": result.get("rag_indexed", 0),
            "final_rag_indexed": result.get("rag_indexed", 0),
        },
        "verdict": "PASS" if result.get("rag_indexed", 0) > 0 else "FAIL",
        "failed_stage": None if result.get("rag_indexed", 0) > 0 else (result.get("error") or "index_document returned rag_indexed=0"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

