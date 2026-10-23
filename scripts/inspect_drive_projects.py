#!/usr/bin/env python
"""Inspect approved/Drive-linked projects and emit a re-import manifest.

Usage:
  export FORK_API_KEY=...
  python scripts/inspect_drive_projects.py                  # damage snapshot
  python scripts/inspect_drive_projects.py --discover <parent_drive_folder_id>

The --discover mode lists child folders of the given Drive parent, matches them
against the Drive-linked project names in the recovery manifest, and prints a
proposed GDRIVE_PROJECT_FOLDERS value. It is a read-only proposal: the operator
must confirm the mapping before it goes into Render.
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BASE = os.getenv("FORK_BASE_URL", "https://the-fork-jn3t.onrender.com")

# Drive-linked projects from docs/recovery/re_import_manifest.md.
# Each entry has the platform project_id and one or more folder-name aliases to
# match against. The anchor pair example_infra_pack is validated on discovery.
RECOVERY_PROJECTS: List[Dict[str, Any]] = [
    # master_corpus is a virtual alias backed by projects_folder;
    # the physical hydration target is projects_folder.
    {"project_id": "master_corpus", "names": ["Master Corpus"], "no_map": True, "note": "alias -> projects_folder"},
    {"project_id": "projects_folder", "names": ["Projects Folder"]},
    {"project_id": "example_estate", "names": ["Ha Long Xanh"]},
    {"project_id": "example_estate_2", "names": ["Ha Long Xanh"]},
    {"project_id": "example_infra_pack", "names": ["the client project"]},
    {"project_id": "5c13510e", "names": ["the client project Bills of Quantities"]},
]

ANCHOR_PROJECT = os.environ.get("FORK_ANCHOR_PROJECT", "example_infra_pack")
ANCHOR_FOLDER_ID = os.environ.get("FORK_ANCHOR_FOLDER_ID", "1ExampleFolderId001xxxxxxxxxxxxxx")


def _auth_header(base: str) -> dict:
    tok = os.getenv("FORK_TOKEN") or os.getenv("FORK_API_KEY")
    if not tok:
        email, pw = os.getenv("FORK_EMAIL"), os.getenv("FORK_PASSWORD")
        if not (email and pw):
            sys.exit("No credentials: set FORK_TOKEN / FORK_API_KEY / FORK_EMAIL+FORK_PASSWORD")
        r = httpx.post(f"{base}/v1/users/login", json={"email": email, "password": pw}, timeout=30)
        r.raise_for_status()
        tok = r.json()["token"]
    return {"Authorization": f"Bearer {tok}"}


def _folder_map() -> Dict[str, str]:
    raw = (os.getenv("GDRIVE_PROJECT_FOLDERS") or "").strip()
    out: Dict[str, str] = {}
    if not raw:
        return out
    for chunk in raw.split(","):
        if ":" not in chunk:
            continue
        pid, fid = chunk.split(":", 1)
        out[pid.strip()] = fid.strip()
    return out


def _inspect_project(client: httpx.Client, proj: Dict[str, Any]) -> Dict[str, Any]:
    pid = proj["id"]
    docs: List[Dict[str, Any]] = []
    try:
        # Project detail enriches chunk_count when doc_count is small enough.
        r = client.get(f"/v1/projects/{pid}")
        r.raise_for_status()
        detail = r.json()
        docs = detail.get("documents", []) or []
        total = detail.get("document_count", len(docs))
    except httpx.HTTPError as exc:
        print(f"[warn] could not inspect {pid}: {exc}", file=sys.stderr)
        total = proj.get("document_count", 0)

    files_present = sum(1 for d in docs if d.get("file_path") and d.get("size", 0) > 0)
    # chunk_count is present only when detail enrichment ran.
    chunk_counts = [d.get("chunk_count") for d in docs if "chunk_count" in d]
    zero_chunk = sum(1 for c in chunk_counts if (c or 0) == 0)
    total_chunks = sum(c or 0 for c in chunk_counts)
    return {
        "project_id": pid,
        "name": proj.get("name", ""),
        "origin": proj.get("origin", ""),
        "docs": total,
        "files_present": files_present,
        "zero_chunk_docs": zero_chunk if chunk_counts else None,
        "total_chunks": total_chunks if chunk_counts else None,
    }


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _best_folder_match(
    project_names: List[str],
    folders: List[Dict[str, str]],
    used: set,
) -> Tuple[Optional[Dict[str, str]], str, float]:
    """Return (folder, confidence_label, ratio) for the best available match."""
    best: Optional[Dict[str, str]] = None
    best_ratio = 0.0
    best_confidence = "unmatched"
    for folder in folders:
        fid = folder["id"]
        if fid in used:
            continue
        fname = folder["name"]
        norm_folder = _normalise(fname)
        for pname in project_names:
            norm_project = _normalise(pname)
            if norm_project == norm_folder:
                ratio = 1.0
                confidence = "exact"
            else:
                ratio = difflib.SequenceMatcher(None, norm_project, norm_folder).ratio()
                confidence = "fuzzy" if ratio >= 0.7 else "unmatched"
            if ratio > best_ratio:
                best_ratio = ratio
                best = folder
                best_confidence = confidence
    return best, best_confidence, best_ratio


def _discover_folders(parent_folder_id: str) -> List[Dict[str, str]]:
    from app.core import gdrive_service

    if not gdrive_service.is_configured():
        sys.exit(
            "Google Drive service account is not configured.\n"
            "Set GDRIVE_SERVICE_ACCOUNT_JSON (file path or inline JSON) and try again."
        )

    files, err = gdrive_service.list_folder_files(parent_folder_id, page_size=200)
    if err:
        sys.exit(f"Drive list failed: {err}")

    folder_mime = "application/vnd.google-apps.folder"
    return [
        {"id": f["id"], "name": f.get("name", "")}
        for f in files
        if f.get("mimeType") == folder_mime
    ]


def _run_discovery(parent_folder_id: str) -> int:
    folders = _discover_folders(parent_folder_id)
    if not folders:
        print(f"No child folders found under Drive parent {parent_folder_id}.")
        return 0

    used: set = set()
    matches: List[Dict[str, Any]] = []

    for project in RECOVERY_PROJECTS:
        folder, confidence, ratio = _best_folder_match(project["names"], folders, used)
        if folder:
            used.add(folder["id"])
        matches.append({
            "project_id": project["project_id"],
            "names": project["names"],
            "folder": folder,
            "confidence": confidence,
            "ratio": ratio,
        })

    # Proposed env string (only confident matches for projects that need a map).
    proposed_pairs = [
        f"{m['project_id']}:{m['folder']['id']}"
        for m in matches
        if m["folder"] and m["confidence"] in ("exact", "fuzzy") and not m.get("no_map")
    ]
    proposed_value = ",".join(proposed_pairs)

    print(f"## Proposed GDRIVE_PROJECT_FOLDERS (from Drive parent {parent_folder_id})")
    print()
    print("```text")
    print(proposed_value)
    print("```")
    print()
    print("| project_id | project name | matched folder | folder id | confidence | ratio | note |")
    print("|---|---|---|---|---|---|---|")
    for m in matches:
        pname = m["names"][0]
        note = m.get("note", "")
        if m["folder"]:
            print(
                f"| {m['project_id']} | {pname} | {m['folder']['name']} | "
                f"{m['folder']['id']} | {m['confidence']} | {m['ratio']:.2f} | {note} |"
            )
        else:
            print(f"| {m['project_id']} | {pname} | — | — | unmatched | — | {note} |")

    unmatched_folders = [f for f in folders if f["id"] not in used]
    if unmatched_folders:
        print()
        print("## Unmatched Drive folders")
        print()
        print("| folder name | folder id |")
        print("|---|---|")
        for f in unmatched_folders:
            print(f"| {f['name']} | {f['id']} |")

    # Anchor validation.
    anchor_match = next(
        (m for m in matches if m["project_id"] == ANCHOR_PROJECT),
        None,
    )
    print()
    if anchor_match and anchor_match.get("folder") and anchor_match["folder"]["id"] == ANCHOR_FOLDER_ID:
        print(f"✓ Anchor validated: {ANCHOR_PROJECT} = {ANCHOR_FOLDER_ID}")
    else:
        print(
            f"⚠ ANCHOR MISMATCH: expected {ANCHOR_PROJECT} = {ANCHOR_FOLDER_ID}, "
            f"but discovery proposed "
            f"{anchor_match['folder']['id'] if anchor_match and anchor_match.get('folder') else 'none'}. "
            f"Review the parent folder before accepting the mapping."
        )

    print()
    print("This is a proposed mapping only. Confirm it before setting GDRIVE_PROJECT_FOLDERS on Render.")
    return 0


def _run_inspect() -> int:
    base = os.getenv("FORK_BASE_URL", DEFAULT_BASE)
    headers = _auth_header(base)
    folder_map = _folder_map()

    with httpx.Client(base_url=base, headers=headers, timeout=60) as client:
        r = client.get("/v1/projects")
        r.raise_for_status()
        projects = r.json().get("projects", []) or []

        # Focus on Drive-linked / admin-approved / master corpus projects.
        candidates = [
            p for p in projects
            if p.get("origin") in ("admin_drive_approved",)
            or p.get("is_master_corpus")
            or p.get("id") in ("projects_folder", "training_material")
        ]

        print("| project_id | name | origin | docs | files_present | zero_chunk_docs | total_chunks | folder_id |")
        print("|---|---|---|---|---|---|---|---|")
        for p in candidates:
            info = _inspect_project(client, p)
            folder_id = folder_map.get(info["project_id"], "")
            print(
                f"| {info['project_id']} | {info['name']} | {info['origin']} | "
                f"{info['docs']} | {info['files_present']} | "
                f"{info['zero_chunk_docs'] if info['zero_chunk_docs'] is not None else 'N/A'} | "
                f"{info['total_chunks'] if info['total_chunks'] is not None else 'N/A'} | {folder_id} |"
            )

    print("\nMissing folder_id entries need to be provided in GDRIVE_PROJECT_FOLDERS or manually.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect Drive-linked projects and propose folder mappings")
    ap.add_argument("--discover", metavar="PARENT_FOLDER_ID",
                    help="List child folders of the given Drive folder and propose a GDRIVE_PROJECT_FOLDERS mapping")
    args = ap.parse_args()

    if args.discover:
        return _run_discovery(args.discover)
    return _run_inspect()


if __name__ == "__main__":
    sys.exit(main())
