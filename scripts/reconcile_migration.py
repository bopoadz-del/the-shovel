#!/usr/bin/env python3
"""Migration reconciliation reporter.

Read-only verification against prod (or any Fork deployment). Prints a table
comparing document/chunk counts from the database, the project-detail API, and
the admin corpus collections — then flags mismatches and reports chunks stored
under the wrong project_id.

Usage:
  export FORK_API_KEY=...
  python scripts/reconcile_migration.py

Environment:
  FORK_BASE_URL  target deployment (default: https://the-fork-jn3t.onrender.com)
  FORK_API_KEY   admin CEREBRUM_MASTER_KEY
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")

# Projects involved in the drive_archive migration + the GK identity.
WATCHED_PROJECTS = {
    "master_corpus",
    "projects_folder",
    "training_material",
    "unclassified",
    "curated_kb",
}


def _auth_header(base: str) -> dict:
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        email, pw = os.getenv("FORK_EMAIL"), os.getenv("FORK_PASSWORD")
        if not (email and pw):
            sys.exit("No credentials: set FORK_API_KEY / FORK_TOKEN / FORK_EMAIL+FORK_PASSWORD")
        r = httpx.post(f"{base}/v1/users/login", json={"email": email, "password": pw}, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def _project_list(client: httpx.Client) -> List[Dict[str, Any]]:
    r = client.get("/v1/projects")
    r.raise_for_status()
    return r.json().get("projects", []) or []


def _collections(client: httpx.Client) -> Dict[str, Dict[str, Any]]:
    r = client.get("/v1/admin/corpus/collections?folder_breakdown=false")
    r.raise_for_status()
    return {c["project_id"]: c for c in r.json().get("collections", [])}


def _project_detail(client: httpx.Client, project_id: str) -> Optional[Dict[str, Any]]:
    r = client.get(f"/v1/projects/{project_id}")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def _reconcile_report(client: httpx.Client) -> Dict[str, Any]:
    r = client.post("/v1/admin/corpus/reconcile?sample_limit=0")
    r.raise_for_status()
    return r.json()


def main() -> int:
    base = os.getenv("FORK_BASE_URL", DEFAULT_BASE)
    headers = _auth_header(base)

    with httpx.Client(base_url=base, headers=headers, timeout=120) as client:
        projects = _project_list(client)
        cols = _collections(client)
        report = _reconcile_report(client)

    by_id = {p["id"]: p for p in projects}

    # Build rows for watched projects that exist in either the project list
    # or the collections table.
    rows: List[Dict[str, Any]] = []
    all_ids = WATCHED_PROJECTS | set(by_id) | set(cols)
    for pid in sorted(all_ids):
        if pid not in by_id and pid not in cols:
            continue
        proj = by_id.get(pid) or {}
        col = cols.get(pid) or {}
        detail = None
        if pid in by_id:
            with httpx.Client(base_url=base, headers=headers, timeout=60) as client:
                detail = _project_detail(client, pid)

        rows.append({
            "project_id": pid,
            "name": proj.get("name") or col.get("project_id") or pid,
            "docs_db": col.get("documents", 0),
            "chunks_db": col.get("chunks", 0),
            "api_chunk_count": detail.get("chunk_count") if detail else None,
            "admin_chunks": col.get("chunks", 0),
        })

    # Print reconciliation table.
    print("| project_id | name | docs (DB) | chunks (DB) | API chunk_count | admin chunks | flag |")
    print("|---|---|---:|---:|---:|---:|---|")
    mismatches = []
    for r in rows:
        flag = ""
        api = r["api_chunk_count"]
        db = r["chunks_db"]
        if api is None:
            flag = "api_missing"
        elif api != db:
            flag = f"mismatch (api={api}, db={db})"
        if r["project_id"] in (report.get("summary") or {}):
            flag += "; misplaced_chunks" if flag else "misplaced_chunks"
        if not flag:
            flag = "ok"
        if "misplaced" in flag or "mismatch" in flag:
            mismatches.append((r["project_id"], flag))
        print(
            f"| {r['project_id']} | {r['name']} | {r['docs_db']} | "
            f"{r['chunks_db']} | {api if api is not None else 'N/A'} | "
            f"{r['admin_chunks']} | {flag} |"
        )

    # Reconcile endpoint summary.
    print()
    print(f"Reconcile report: {report['mismatches_total']} mismatched chunks, "
          f"{report['dangling_total']} dangling chunks, dry_run={report['dry_run']}")
    summary = report.get("summary") or {}
    if summary:
        print("| project_id | mismatched_to | mismatched_from |")
        print("|---|---|---|")
        for pid, info in sorted(summary.items()):
            to_ = info.get("mismatched_to", {})
            from_ = info.get("mismatched_from", {})
            print(f"| {pid} | {to_ or '-'} | {from_ or '-'} |")
    else:
        print("No misplaced chunks detected.")

    if mismatches:
        print()
        print("Flagged projects:")
        for pid, reason in mismatches:
            print(f"  - {pid}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
