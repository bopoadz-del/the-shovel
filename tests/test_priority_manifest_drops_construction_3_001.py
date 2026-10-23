"""The priority ingest manifest must not name construction-3-001.

The platform project is being removed. These checks read the committed
manifest and the builder that regenerates it. They do not open a database
or call Drive.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "p1b_priority_manifest.json"
BUILDER = ROOT / "scripts" / "build_priority_manifest.py"

_FORBIDDEN = ("construction_3_001", "construction-3-001", "1ee147a4")

# Every other KNOWN_FOLDERS project_id, in builder order.
_REMAINING = (
    "example_infra_pack",
    "sop_project_controls",
    "sop_delivery_mgmt",
    "sop_construction_mgmt",
    "sop_design_mgmt",
    "sop_procurement_contracts",
    "scanned_high_rise",
    "scanned_road_works",
    "scanned_concrete_problems",
)


def _folder_project_ids(manifest: dict) -> list[str]:
    ids: list[str] = []
    for tier_key in ("1", "2", "3"):
        tier = manifest["tiers"][tier_key]
        assert isinstance(tier["folders"], list)
        for folder in tier["folders"]:
            ids.append(folder["project_id"])
    return ids


def test_priority_manifest_omits_construction_3_001_and_keeps_other_rows():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["tiers"]) == {"1", "2", "3"}
    blob = json.dumps(manifest)
    for token in _FORBIDDEN:
        assert token not in blob
    assert _folder_project_ids(manifest) == list(_REMAINING)
    tier1 = manifest["tiers"]["1"]["folders"]
    assert len(tier1) == 1
    assert tier1[0]["project_id"] == "example_infra_pack"
    assert tier1[0]["folder_id"]


def test_build_priority_manifest_source_omits_construction_3_001():
    src = BUILDER.read_text(encoding="utf-8")
    for token in _FORBIDDEN:
        assert token not in src
