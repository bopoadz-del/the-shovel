"""Unit tests for p1b server-ingest folder scoping.

No live Drive, no prod DB — argparse + the folder-loop filter only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.p1b_ingest_drive_server import (
    build_parser,
    folder_entries_for_run,
    parse_folder_ids,
)

CLIENT_PARENT_ID = "1ExampleFolderId001xxxxxxxxxxxxxx"
MISC_FOLDER_ID = "1ExampleFolderId004xxxxxxxxxxxxxx"

TIER1_FOLDERS = [
    {
        "project_id": "example_infra_pack",
        "folder_name": "the client project",
        "folder_id": CLIENT_PARENT_ID,
    },
    {
        "project_id": "construction_3_001",
        "folder_name": "construction-3-001",
        "folder_id": None,
    },
]


def test_parser_accepts_folder_id():
    args = build_parser().parse_args(
        ["--tier", "1", "--resume", "--folder-id", MISC_FOLDER_ID]
    )
    assert args.tier == 1
    assert args.resume is True
    assert args.folder_ids == [MISC_FOLDER_ID]


def test_parser_accepts_repeatable_and_comma_separated_folder_ids():
    args = build_parser().parse_args(
        ["--folder-id", "aaa,bbb", "--folder-id", "ccc"]
    )
    assert parse_folder_ids(args.folder_ids) == ["aaa", "bbb", "ccc"]


def test_parser_accepts_reingest_old_id():
    args = build_parser().parse_args(["--reingest", "5a1e0001"])
    assert args.reingest == "5a1e0001"


def test_parser_omitting_folder_id_leaves_default_unset():
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert args.folder_ids is None
    assert parse_folder_ids(args.folder_ids) == []


def test_folder_loop_default_is_full_tier():
    walks = folder_entries_for_run(TIER1_FOLDERS, [], tier=1)
    assert walks == TIER1_FOLDERS
    assert all("walk_folder_id" not in entry for entry in walks)


def test_folder_loop_filters_to_parent_folder_id():
    walks = folder_entries_for_run(TIER1_FOLDERS, [CLIENT_PARENT_ID], tier=1)
    assert len(walks) == 1
    assert walks[0]["walk_folder_id"] == CLIENT_PARENT_ID
    assert walks[0]["project_id"] == "example_infra_pack"
    assert walks[0]["folder_name"] == "the client project"


def test_folder_loop_filters_subfolder_to_parent_project():
    """Misc is not a tier parent; walk it, keep the client project row."""
    walks = folder_entries_for_run(TIER1_FOLDERS, [MISC_FOLDER_ID], tier=1)
    assert len(walks) == 1
    assert walks[0]["walk_folder_id"] == MISC_FOLDER_ID
    assert walks[0]["folder_id"] == CLIENT_PARENT_ID
    assert walks[0]["project_id"] == "example_infra_pack"
    assert walks[0]["folder_name"] == "the client project"


def test_folder_loop_unmatched_without_unique_parent_errors():
    tier2 = [
        {"project_id": "sop_a", "folder_name": "A", "folder_id": None},
        {"project_id": "sop_b", "folder_name": "B", "folder_id": None},
    ]
    with pytest.raises(ValueError, match="no unique parent"):
        folder_entries_for_run(tier2, ["not-a-parent"], tier=2)


def test_real_tier1_misc_keeps_client_project_row():
    manifest = json.loads(
        Path("manifests/p1b_priority_manifest.json").read_text(encoding="utf-8")
    )
    folders = manifest["tiers"]["1"]["folders"]
    walks = folder_entries_for_run(folders, [MISC_FOLDER_ID], tier=1)
    assert len(walks) == 1
    assert walks[0]["walk_folder_id"] == MISC_FOLDER_ID
    assert walks[0]["project_id"] == "example_infra_pack"
    assert walks[0]["folder_name"] == "the client project"
