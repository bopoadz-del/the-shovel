#!/usr/bin/env python3
"""Migrate the local drive_archive SQLite corpus back to Render Postgres.

Source: data/the_fork.db (drive_archive project, ~142k chunks).
Destination: /v1/admin/corpus/bulk-insert on prod (or FORK_BASE_URL).

The migration manifest (C:\tmp\migration_manifest.json) maps each doc_id to a
destination project_id. We send projects + documents first, then chunks in
batches so the HTTP payloads stay small and retryable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "the_fork.db"
MANIFEST_PATH = Path("C:/tmp/migration_manifest.json")
BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")
API_KEY = os.getenv("FORK_API_KEY") or os.getenv("CEREBRUM_MASTER_KEY")
CHUNK_BATCH_SIZE = int(os.getenv("CHUNK_BATCH_SIZE", "2000"))


def api_post(path: str, payload: dict) -> dict:
    url = f"{BASE}{path}"
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"HTTP {e.code}: {text}") from e


def load_manifest() -> dict[str, dict[str, Any]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    doc_map = {}
    for doc in manifest.get("documents", []):
        doc_map[doc["doc_id"]] = doc
    return doc_map


def build_projects_and_documents(doc_map: dict) -> tuple[list[dict], list[dict]]:
    projects_by_id: dict[str, dict] = {}
    documents: list[dict] = []

    # Project display names by dest_project_id
    names = {
        "projects_folder": "Master Corpus",
        "training_material": "Training Material",
        "unclassified": "Unclassified Drive Archive",
    }

    for doc_id, doc in doc_map.items():
        pid = doc["dest_project_id"]
        if pid not in projects_by_id:
            projects_by_id[pid] = {
                "id": pid,
                "name": names.get(pid, pid),
                "user_id": "system",
                "status": "active",
            }
        documents.append({
            "id": doc_id,
            "project_id": pid,
            "original_name": doc.get("original_name") or doc_id,
            "doc_type": "document",
            "doc_role": "other",
            "size": 0,
            "file_path": doc.get("original_path"),
        })

    return list(projects_by_id.values()), documents


def iter_chunk_batches(doc_map: dict):
    """Yield lists of up to CHUNK_BATCH_SIZE chunk rows from the local DB."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "SELECT chunk_id, doc_id, chunk_index, text, embedding, created_at "
        "FROM chunks WHERE project_id = 'drive_archive'"
    )

    batch: list[dict] = []
    for row in cur:
        chunk_id, doc_id, chunk_index, text, emb_blob, created_at = row
        doc = doc_map.get(doc_id)
        if doc is None:
            # Should not happen for drive_archive; skip defensively.
            continue
        emb = np.frombuffer(emb_blob, dtype=np.float32)
        batch.append({
            "chunk_id": chunk_id,
            "project_id": doc["dest_project_id"],
            "doc_id": doc_id,
            "chunk_index": chunk_index,
            "text": text,
            "embedding": emb.tolist(),
            "created_at": created_at,
        })
        if len(batch) >= CHUNK_BATCH_SIZE:
            yield batch
            batch = []
    if batch:
        yield batch
    conn.close()


def main() -> int:
    print("[migrate] starting", flush=True)
    if not DB_PATH.exists():
        print(f"Local DB not found: {DB_PATH}", file=sys.stderr)
        return 2
    if not MANIFEST_PATH.exists():
        print(f"Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2
    if not API_KEY:
        print("Set FORK_API_KEY or CEREBRUM_MASTER_KEY", file=sys.stderr)
        return 2

    print("[migrate] loading manifest...", flush=True)
    doc_map = load_manifest()
    print(f"Loaded manifest: {len(doc_map)} documents", flush=True)

    projects, documents = build_projects_and_documents(doc_map)
    print(f"Projects to create: {[p['id'] for p in projects]}", flush=True)
    print(f"Documents to create: {len(documents)}", flush=True)

    # 1. Projects + documents
    print("\n[1/2] Inserting projects + documents...", flush=True)
    t0 = time.time()
    resp = api_post("/v1/admin/corpus/bulk-insert", {
        "projects": projects,
        "documents": documents,
        "chunks": [],
    })
    print(f"  done in {time.time()-t0:.1f}s: {resp['counts']}", flush=True)

    # 2. Chunks in batches
    print(f"\n[2/2] Inserting chunks in batches of {CHUNK_BATCH_SIZE}...", flush=True)
    total_inserted = 0
    total_seen = 0
    batch_num = 0
    t0_total = time.time()
    for batch in iter_chunk_batches(doc_map):
        batch_num += 1
        resp = api_post("/v1/admin/corpus/bulk-insert", {
            "projects": [],
            "documents": [],
            "chunks": batch,
        })
        counts = resp["counts"]
        total_inserted += counts["chunks"]
        total_seen += counts["chunks_seen"]
        if batch_num == 1 or batch_num % 5 == 0:
            print(f"  batch {batch_num}: +{counts['chunks']} chunks "
                  f"(total inserted {total_inserted}, seen {total_seen})", flush=True)

    print(f"\nMigration complete in {time.time()-t0_total:.1f}s", flush=True)
    print(f"Total chunks inserted: {total_inserted}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
