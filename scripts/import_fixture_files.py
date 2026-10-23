#!/usr/bin/env python
"""Import known Drive files into fixture projects and verify chunks."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")

FIXTURES = {
    "FIXTURE — BOQ": {
        "files": [
            {"file_id": "1ExampleDriveFileId001xxxxxxxxxxx", "name": "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf"},
        ]
    },
    "FIXTURE — Programme+Drawings": {
        "files": [
            {"file_id": "1ExampleFolderId003xxxxxxxxxxxxxx", "name": "Annexure 2 - Baseline Program XER.xer"},
            {"file_id": "1ExampleDriveFileId002xxxxxxxxxxx", "name": "AVE conditional IFC DWGs - xyz.pdf"},
        ]
    },
}


def _auth_header(base: str) -> dict:
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        email, pw = os.getenv("FORK_EMAIL"), os.getenv("FORK_PASSWORD")
        if not (email and pw):
            sys.exit("No credentials: set FORK_TOKEN / FORK_API_KEY / FORK_EMAIL+FORK_PASSWORD")
        r = httpx.post(f"{base}/v1/users/login", json={"email": email, "password": pw}, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
        print(f"[auth] logged in as {email}", file=sys.stderr)
    return {"Authorization": f"Bearer {tok}"}


def _resolve_project(client: httpx.Client, name: str) -> Optional[Dict[str, Any]]:
    r = client.get("/v1/projects")
    r.raise_for_status()
    for p in r.json().get("projects", []) or []:
        if p.get("name") == name:
            return p
    return None


def _create_project(client: httpx.Client, name: str) -> Dict[str, Any]:
    r = client.post("/v1/projects", json={"name": name})
    r.raise_for_status()
    return r.json()


def _get_or_create_project(client: httpx.Client, name: str) -> Tuple[Dict[str, Any], bool]:
    existing = _resolve_project(client, name)
    if existing:
        return existing, False
    proj = _create_project(client, name)
    print(f"[create] {name} -> {proj['id']}")
    return proj, True


def _import_drive_file(client: httpx.Client, project_id: str, file_id: str, name: str) -> Dict[str, Any]:
    r = client.post(
        f"/v1/projects/{project_id}/drive/import",
        json={"file_id": file_id, "name": name},
        timeout=300,
    )
    r.raise_for_status()
    return r.json()


def _list_docs(client: httpx.Client, project_id: str) -> List[Dict[str, Any]]:
    r = client.get(f"/v1/projects/{project_id}/documents", params={"limit": 200})
    r.raise_for_status()
    return r.json().get("documents", []) or []


def _verify_chunks(client: httpx.Client, project_id: str, max_wait: int = 300, poll_interval: int = 5) -> Tuple[int, int, List[str]]:
    waited = 0
    while waited <= max_wait:
        docs = _list_docs(client, project_id)
        if docs and all("chunk_count" in d for d in docs):
            zero_chunk = [d["id"] for d in docs if (d.get("chunk_count") or 0) == 0]
            total_chunks = sum((d.get("chunk_count") or 0) for d in docs)
            if not zero_chunk:
                return len(docs), total_chunks, []
            print(f"  [wait] {len(zero_chunk)} docs still zero-chunk, {waited}s elapsed")
        time.sleep(poll_interval)
        waited += poll_interval
    docs = _list_docs(client, project_id)
    zero_chunk = [d["id"] for d in docs if (d.get("chunk_count") or 0) == 0]
    total_chunks = sum((d.get("chunk_count") or 0) for d in docs)
    return len(docs), total_chunks, zero_chunk


def main() -> int:
    base = os.getenv("FORK_BASE_URL", DEFAULT_BASE)
    headers = _auth_header(base)
    client = httpx.Client(base_url=base, headers=headers, timeout=60)

    results = []
    for name, spec in FIXTURES.items():
        print(f"\n[{name}]")
        proj, created = _get_or_create_project(client, name)
        project_id = proj["id"]
        if not created:
            print(f"[skip-create] {name} already exists as {project_id}")

        existing = {d["original_name"] for d in _list_docs(client, project_id)}
        for fspec in spec["files"]:
            fname = fspec["name"]
            if fname in existing:
                print(f"[skip] {fname}")
                continue
            print(f"[import] {fname} ...")
            try:
                res = _import_drive_file(client, project_id, fspec["file_id"], fname)
                doc = res.get("document", {})
                print(f"[imported] {fname} -> doc {doc.get('id')}  type={doc.get('doc_type')}  role={doc.get('doc_role')}")
            except httpx.HTTPError as exc:
                print(f"[error] {fname}: {exc}")

        print(f"[wait] indexing {name} ...")
        docs, chunks, zero = _verify_chunks(client, project_id)
        results.append({"name": name, "project_id": project_id, "docs": docs, "chunks": chunks, "zero": zero})

    print("\n" + "=" * 80)
    print(f"{'fixture':<32} {'project_id':<12} {'docs':>6} {'chunks':>8} {'status':<15}")
    print("-" * 80)
    failed = False
    for r in results:
        status = "OK" if not r["zero"] else "ZERO_CHUNK"
        print(f"{r['name']:<32} {r['project_id']:<12} {r['docs']:>6} {r['chunks']:>8} {status:<15}")
        if r["zero"]:
            failed = True
            print(f"  ZERO_CHUNK docs: {', '.join(r['zero'])}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
