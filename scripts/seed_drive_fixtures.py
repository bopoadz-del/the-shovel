#!/usr/bin/env python
"""One-off Drive-import seeder for BOQ and Programme+Drawings fixtures.

Searches the connected Google Drive for files by name, creates the fixture
projects if missing, and imports matches through the normal
POST /v1/projects/{id}/drive/import endpoint. Only imports allowed extensions;
DWG is allowed by the API but will not produce chunks (BIM block rejects it),
so the seeder skips DWG unless no other drawing candidate exists.

Verification: every fixture must contain at least one expected candidate per
candidate group (existing or imported), and every doc that RAG can index must
reach chunk_count > 0. Schedule (.xer) and CAD (.dxf/.dwg) files register as
project documents but are never text-indexed (app/core/doc_index.py supports
text/PDF/Office/image extensions only) — they are consumed by their parser
blocks (primavera_parser, drawing_qto) instead, so they are excluded from the
chunk gate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")

sys.path.insert(0, str(ROOT / "scripts"))

BOQ_FIXTURE_NAME = "FIXTURE — BOQ"
PROGRAMME_DRAWINGS_FIXTURE_NAME = "FIXTURE — Programme+Drawings"

# Search candidates in priority order.
BOQ_CANDIDATES = [
    "the client project_Sewer_WasteWater_Network_BOQ_verified.xlsx",
    "QZ-ENG-310-0000-EXC-BOQ-CA-000007-B_Bill of Quantities (Priced).pdf",
    "QZ-ENG-310 Priced BOQ - General Summary (verified total).txt",
]
PROGRAMME_CANDIDATES = [
    "Annexure 2 - Baseline Program XER.xer",
    "project_programme.xer",
]
DRAWING_CANDIDATES = [
    "AVE conditional IFC DWGs - xyz.pdf",
    "ground_floor_plan.dxf",
]

# Accepted project documents that are handled by parser blocks rather than the
# RAG text indexer — they legitimately stay at chunk_count == 0 forever, so the
# chunk gate must not wait on them (see app/core/doc_index.py _SUPPORTED_EXTS).
NO_CHUNK_EXTENSIONS = (".xer", ".mpp", ".dxf", ".dwg", ".ifc", ".rvt")


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


def _search_drive(client: httpx.Client, q: str) -> List[Dict[str, Any]]:
    r = client.get("/v1/drive/files", params={"q": q})
    r.raise_for_status()
    return r.json().get("files", []) or []


def _import_drive_file(client: httpx.Client, project_id: str, file_id: str, name: str) -> Dict[str, Any]:
    r = client.post(
        f"/v1/projects/{project_id}/drive/import",
        json={"file_id": file_id, "name": name},
    )
    r.raise_for_status()
    return r.json()


def _list_docs(client: httpx.Client, project_id: str) -> List[Dict[str, Any]]:
    r = client.get(f"/v1/projects/{project_id}/documents", params={"limit": 200})
    r.raise_for_status()
    return r.json().get("documents", []) or []


def _is_chunkable(doc: Dict[str, Any]) -> bool:
    """True when the doc's type is text-indexed by RAG (so chunks are expected)."""
    name = doc.get("original_name") or doc.get("name") or ""
    return Path(name).suffix.lower() not in NO_CHUNK_EXTENSIONS


def _verify_chunks(client: httpx.Client, project_id: str, max_wait: int = 180, poll_interval: int = 5) -> Tuple[int, int, List[str]]:
    """Wait until every RAG-indexable doc in the project has chunks.

    Docs with NO_CHUNK_EXTENSIONS (e.g. .xer schedules consumed by
    primavera_parser) are excluded from the gate — they never chunk by design.
    """
    waited = 0
    while waited <= max_wait:
        docs = _list_docs(client, project_id)
        chunkable = [d for d in docs if _is_chunkable(d)]
        if docs and all("chunk_count" in d for d in chunkable):
            zero_chunk = [d["id"] for d in chunkable if (d.get("chunk_count") or 0) == 0]
            total_chunks = sum((d.get("chunk_count") or 0) for d in docs)
            if not zero_chunk:
                return len(docs), total_chunks, []
        time.sleep(poll_interval)
        waited += poll_interval
    docs = _list_docs(client, project_id)
    zero_chunk = [d["id"] for d in docs if _is_chunkable(d) and (d.get("chunk_count") or 0) == 0]
    total_chunks = sum((d.get("chunk_count") or 0) for d in docs)
    return len(docs), total_chunks, zero_chunk


def _find_and_import(client: httpx.Client, project_id: str, candidates: List[str], skip_extensions: Tuple[str, ...] = ()) -> List[Dict[str, Any]]:
    imported = []
    existing = {d["original_name"] for d in _list_docs(client, project_id)}
    for name in candidates:
        if name in existing:
            print(f"  [skip] {name} already in project")
            continue
        ext = Path(name).suffix.lower()
        if ext in skip_extensions:
            print(f"  [skip] {name} extension {ext} is skipped")
            continue
        print(f"  [search] {name}")
        hits = _search_drive(client, name)
        # Prefer exact name match.
        hit = next((h for h in hits if h.get("name") == name), hits[0] if hits else None)
        if not hit:
            print(f"  [not found] {name}")
            continue
        try:
            res = _import_drive_file(client, project_id, hit["id"], hit["name"])
            doc = res.get("document", {})
            print(f"  [imported] {hit['name']} ({hit['id']}) -> doc {doc.get('id')}")
            imported.append({"name": hit["name"], "file_id": hit["id"], "doc_id": doc.get("id")})
        except httpx.HTTPError as exc:
            print(f"  [error] importing {name}: {exc}")
    return imported


def _missing_groups(client: httpx.Client, project_id: str, groups: List[Tuple[str, List[str]]]) -> List[str]:
    """Return labels of candidate groups with no document present in the project.

    Guards against Drive search misses and import errors that are only logged:
    each fixture must end up with at least one expected document per group,
    otherwise the acceptance battery would run against an empty fixture.
    """
    present = {d.get("original_name") for d in _list_docs(client, project_id)}
    return [label for label, candidates in groups if not present.intersection(candidates)]


def main() -> int:
    base = os.getenv("FORK_BASE_URL", DEFAULT_BASE)
    headers = _auth_header(base)
    client = httpx.Client(base_url=base, headers=headers, timeout=120)

    results = []

    # BOQ fixture
    print(f"\n[BOQ] {BOQ_FIXTURE_NAME}")
    boq_proj, _ = _get_or_create_project(client, BOQ_FIXTURE_NAME)
    boq_imported = _find_and_import(client, boq_proj["id"], BOQ_CANDIDATES)
    print(f"[wait] indexing {BOQ_FIXTURE_NAME} ...")
    boq_docs, boq_chunks, boq_zero = _verify_chunks(client, boq_proj["id"])
    results.append({
        "name": BOQ_FIXTURE_NAME,
        "project_id": boq_proj["id"],
        "imported": boq_imported,
        "docs": boq_docs,
        "chunks": boq_chunks,
        "zero_chunk_docs": boq_zero,
        "missing_groups": _missing_groups(client, boq_proj["id"], [("boq", BOQ_CANDIDATES)]),
    })

    # Programme+Drawings fixture — skip .dwg because it registers but does not chunk.
    print(f"\n[Programme+Drawings] {PROGRAMME_DRAWINGS_FIXTURE_NAME}")
    pd_proj, _ = _get_or_create_project(client, PROGRAMME_DRAWINGS_FIXTURE_NAME)
    pd_imported = _find_and_import(
        client, pd_proj["id"],
        PROGRAMME_CANDIDATES + DRAWING_CANDIDATES,
        skip_extensions=(".dwg",),
    )
    print(f"[wait] indexing {PROGRAMME_DRAWINGS_FIXTURE_NAME} ...")
    pd_docs, pd_chunks, pd_zero = _verify_chunks(client, pd_proj["id"])
    results.append({
        "name": PROGRAMME_DRAWINGS_FIXTURE_NAME,
        "project_id": pd_proj["id"],
        "imported": pd_imported,
        "docs": pd_docs,
        "chunks": pd_chunks,
        "zero_chunk_docs": pd_zero,
        "missing_groups": _missing_groups(
            client, pd_proj["id"],
            [("programme", PROGRAMME_CANDIDATES), ("drawing", DRAWING_CANDIDATES)],
        ),
    })

    print("\n" + "=" * 80)
    print(f"{'fixture':<32} {'project_id':<12} {'docs':>6} {'chunks':>8} {'status':<15}")
    print("-" * 80)
    failed = False
    for r in results:
        problems = []
        if r["zero_chunk_docs"]:
            problems.append("ZERO_CHUNK")
        if r["missing_groups"]:
            problems.append("MISSING_FIXTURE")
        status = "OK" if not problems else "+".join(problems)
        print(f"{r['name']:<32} {r['project_id']:<12} {r['docs']:>6} {r['chunks']:>8} {status:<15}")
        if r["zero_chunk_docs"]:
            failed = True
            print(f"  ZERO_CHUNK docs: {', '.join(r['zero_chunk_docs'])}")
        if r["missing_groups"]:
            failed = True
            print(f"  MISSING fixture groups (no candidate present): {', '.join(r['missing_groups'])}")
        for imp in r["imported"]:
            print(f"  - {imp['name']} -> {imp['doc_id']}")
    print("=" * 80)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
