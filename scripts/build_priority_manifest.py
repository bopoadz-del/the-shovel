#!/usr/bin/env python3
"""Build the prioritized ingestion manifest for server-side Phase 1b.

Reads the existing Drive manifest (p1a_drive_manifest_client.json) and the
known top-level Drive folders, then assigns each folder to a tier:

  Tier 1: the client project + golden-set/fixture-backing folders
  Tier 2: Other pilot-relevant project/procedure folders
  Tier 3: Archives, scanned reference material, everything else

Outputs manifests/p1b_priority_manifest.json.

This script does NOT perform ingestion; it only produces the manifest that
the server-side job will consume.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = REPO_ROOT / "manifests"

# Known top-level Drive folders and their intended tier.
# Folder IDs must be supplied when the service-account credential is fixed.
KNOWN_FOLDERS: list[dict] = [
    # Tier 1 — pilot project + golden set/fixtures
    {"project_id": "example_infra_pack", "folder_name": "the client project", "tier": 1, "folder_id": "1ExampleFolderId001xxxxxxxxxxxxxx"},

    # Tier 2 — procedures and other pilot-relevant folders
    {"project_id": "sop_project_controls", "folder_name": "200-Project Controls Procedures", "tier": 2, "folder_id": None},
    {"project_id": "sop_delivery_mgmt", "folder_name": "300-Delivery Management Procedures", "tier": 2, "folder_id": None},
    {"project_id": "sop_construction_mgmt", "folder_name": "400-Construction Management Procedures", "tier": 2, "folder_id": None},
    {"project_id": "sop_design_mgmt", "folder_name": "500-Design Management Procedures", "tier": 2, "folder_id": None},
    {"project_id": "sop_procurement_contracts", "folder_name": "600-Procurement & Contracts", "tier": 2, "folder_id": None},

    # Tier 3 — scanned reference archives
    {"project_id": "scanned_high_rise", "folder_name": "Scaned Files - High Rise Building", "tier": 3, "folder_id": None},
    {"project_id": "scanned_road_works", "folder_name": "Scaned Files - Road Works", "tier": 3, "folder_id": None},
    {"project_id": "scanned_concrete_problems", "folder_name": "Scaned Files -Concrete Problems", "tier": 3, "folder_id": None},
]


def _load_client_subfolders() -> list[dict]:
    """Return the client project top-level subfolder counts from the existing manifest."""
    path = MANIFEST_DIR / "p1a_drive_manifest_client.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    for folder in data.get("folders", []):
        if folder.get("project_id") != "example_infra_pack":
            continue
        counts: Counter[str] = Counter()
        for f in folder.get("files", []):
            p = f.get("path", "")
            top = p.split("/")[0] if "/" in p else "(root)"
            counts[top] += 1
        return [
            {
                "subfolder": name,
                "file_count": count,
                "tier": 1,
                "note": "the client project — tier 1 by default",
            }
            for name, count in counts.most_common()
        ]
    return []


def main() -> int:
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": "manifests/p1a_drive_manifest_client.json",
        "tiers": {
            "1": {
                "description": "the client project + golden-set/fixture-backing folders",
                "gate_dependency": "Phase 2 battery gates on this tier",
                "folders": [],
            },
            "2": {
                "description": "Other pilot-relevant project folders and company procedures",
                "gate_dependency": "Phase 2 battery may include spot checks",
                "folders": [],
            },
            "3": {
                "description": "Reference archives and everything else",
                "gate_dependency": "Background drain; not pilot-blocking",
                "folders": [],
            },
        },
    }

    for entry in KNOWN_FOLDERS:
        tier = str(entry.pop("tier"))
        if entry["project_id"] == "example_infra_pack":
            entry["subfolders"] = _load_client_subfolders()
            entry["file_count"] = sum(s["file_count"] for s in entry["subfolders"])
        else:
            entry["file_count"] = None  # unknown until Drive is walked
        manifest["tiers"][tier]["folders"].append(entry)

    total_tier_1 = manifest["tiers"]["1"]["folders"][0].get("file_count") or 0
    manifest["summary"] = {
        "tier_1_file_count": total_tier_1,
        "tier_2_file_count": None,
        "tier_3_file_count": None,
        "total_known": total_tier_1,
        "note": "Tier 2/3 counts require folder IDs and a fresh Drive walk once the service-account credential is restored.",
    }

    out_path = MANIFEST_DIR / "p1b_priority_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[priority] manifest written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
