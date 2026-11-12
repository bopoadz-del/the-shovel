"""Unit tests for p1b server-ingest folder scoping.

No live Drive, no prod DB — argparse + the folder-loop filter only.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.p1b_ingest_drive_server import (
    apply_env_fallbacks,
    build_parser,
    folder_entries_for_run,
    missing_project_for_override,
    override_project_id,
    parse_folder_ids,
    parse_reextract_exts,
    reextract_requested,
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


@pytest.mark.parametrize("raw", ["xlsx,xls,xlsm,csv", "csv,xlsx"])
def test_parser_accepts_reextract_exts(raw):
    args = build_parser().parse_args(["--reextract-exts", raw])
    assert args.reextract_exts == raw


def test_parser_reextract_exts_defaults_empty():
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert not args.reextract_exts


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("", frozenset()),
        (None, frozenset()),
        ("xlsx", frozenset({".xlsx"})),
        ("XLSX,.csv", frozenset({".xlsx", ".csv"})),
        ("xls, xlsm, csv ", frozenset({".xls", ".xlsm", ".csv"})),
        (".XLS,.XLSX", frozenset({".xls", ".xlsx"})),
    ],
)
def test_parse_reextract_exts_normalizes(raw, expected):
    assert parse_reextract_exts(raw) == expected


@pytest.mark.parametrize(
    "original_name, reextract_exts, expected",
    [
        ("priced_boq.xlsx", frozenset({".xlsx"}), True),
        ("rates.xlsm", frozenset({".xlsm"}), True),
        ("priced_boq.xlsx", frozenset({".xls"}), False),
        ("drawing.pdf", frozenset({".xlsx", ".csv"}), False),
        ("priced_boq.xlsx", frozenset(), False),
    ],
)
def test_reextract_requested_two_input(original_name, reextract_exts, expected):
    doc = {"original_name": original_name}
    assert reextract_requested(doc, reextract_exts) is expected


@pytest.mark.parametrize(
    "project_id",
    ["retarget_a", "retarget_b"],
)
def test_parser_accepts_project_id_override(project_id):
    args = build_parser().parse_args(["--tier", "1", "--project-id", project_id])
    assert args.project_id == project_id


def test_parser_omitting_project_id_leaves_default_unset():
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert args.project_id is None


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


@pytest.mark.parametrize(
    "project_id",
    ["retarget_a", "retarget_b"],
)
def test_project_id_override_applies_to_every_walk_entry(project_id):
    entries = override_project_id(TIER1_FOLDERS, project_id)
    assert [entry["project_id"] for entry in entries] == [project_id, project_id]
    # Non-project fields are preserved.
    assert entries[0]["folder_name"] == "the client project"
    assert entries[0]["folder_id"] == CLIENT_PARENT_ID
    assert entries[1]["folder_id"] is None


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_project_id_override_blank_is_noop(blank):
    entries = override_project_id(TIER1_FOLDERS, blank)
    assert [entry["project_id"] for entry in entries] == [
        "example_infra_pack",
        "construction_3_001",
    ]


@pytest.mark.parametrize(
    "project_id, requested_ids, expected_walk_folder_id",
    [
        ("retarget_a", [MISC_FOLDER_ID], MISC_FOLDER_ID),
        ("retarget_b", [CLIENT_PARENT_ID], CLIENT_PARENT_ID),
    ],
)
def test_project_id_override_preserves_walk(
    project_id, requested_ids, expected_walk_folder_id
):
    walks = folder_entries_for_run(TIER1_FOLDERS, requested_ids, tier=1)
    retargeted = override_project_id(walks, project_id)
    assert len(retargeted) == 1
    assert retargeted[0]["walk_folder_id"] == expected_walk_folder_id
    assert retargeted[0]["folder_id"] == CLIENT_PARENT_ID
    assert retargeted[0]["project_id"] == project_id


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_missing_project_override_blank_is_none(blank):
    consulted = []

    def get_project(project_id):
        consulted.append(project_id)
        return {"id": project_id}

    assert missing_project_for_override(blank, get_project=get_project) is None
    assert consulted == []  # a blank override never consults the DB


@pytest.mark.parametrize(
    "project_id",
    ["retarget_a", "retarget_b"],
)
def test_missing_project_override_existing_project_is_none(project_id):
    rows = {"retarget_a": {"id": "retarget_a"}, "retarget_b": {"id": "retarget_b"}}
    assert missing_project_for_override(project_id, get_project=rows.get) is None


@pytest.mark.parametrize(
    "project_id",
    ["missing_a", "missing_b"],
)
def test_missing_project_override_missing_project_is_returned(project_id):
    rows = {"existing_other": {"id": "existing_other"}}
    assert missing_project_for_override(project_id, get_project=rows.get) == project_id


@pytest.mark.parametrize(
    "env_value, expected",
    [("envfolder_a,envfolder_b", ["envfolder_a", "envfolder_b"]), ("envfolder_c", ["envfolder_c"])],
)
def test_env_folder_ids_fallback_applies_when_flag_absent(monkeypatch, env_value, expected):
    monkeypatch.setenv("P1B_FOLDER_IDS", env_value)
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert args.folder_ids is None
    apply_env_fallbacks(args)
    assert parse_folder_ids(args.folder_ids) == expected


@pytest.mark.parametrize("env_value", ["envfolder_a,envfolder_b", "envfolder_c"])
def test_env_folder_ids_fallback_does_not_override_a_flag(monkeypatch, env_value):
    monkeypatch.setenv("P1B_FOLDER_IDS", env_value)
    args = build_parser().parse_args(["--folder-id", "flag_folder"])
    apply_env_fallbacks(args)
    assert parse_folder_ids(args.folder_ids) == ["flag_folder"]


@pytest.mark.parametrize("env_value", ["retarget_a", "retarget_b"])
def test_env_project_id_fallback_applies_when_flag_absent(monkeypatch, env_value):
    monkeypatch.setenv("P1B_PROJECT_ID", env_value)
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert args.project_id is None
    apply_env_fallbacks(args)
    assert args.project_id == env_value


@pytest.mark.parametrize("env_value", ["retarget_a", "retarget_b"])
def test_env_project_id_fallback_does_not_override_a_flag(monkeypatch, env_value):
    monkeypatch.setenv("P1B_PROJECT_ID", env_value)
    args = build_parser().parse_args(["--project-id", "flag_project"])
    apply_env_fallbacks(args)
    assert args.project_id == "flag_project"


@pytest.mark.parametrize(
    "env_value, expected",
    [
        ("xlsx,CSV", frozenset({".xlsx", ".csv"})),
        (".docx", frozenset({".docx"})),
    ],
)
def test_env_reextract_exts_fallback_applies_when_flag_absent(monkeypatch, env_value, expected):
    monkeypatch.setenv("P1B_REEXTRACT_EXTS", env_value)
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    assert args.reextract_exts is None
    apply_env_fallbacks(args)
    assert parse_reextract_exts(args.reextract_exts) == expected


@pytest.mark.parametrize("env_value", ["xlsx,csv", ".pdf"])
def test_env_reextract_exts_fallback_does_not_override_a_flag(monkeypatch, env_value):
    monkeypatch.setenv("P1B_REEXTRACT_EXTS", env_value)
    args = build_parser().parse_args(["--reextract-exts", "xlsm"])
    apply_env_fallbacks(args)
    assert parse_reextract_exts(args.reextract_exts) == frozenset({".xlsm"})


@pytest.mark.parametrize("env_value", ["", "   "])
def test_env_reextract_exts_blank_env_is_a_no_op(monkeypatch, env_value):
    monkeypatch.setenv("P1B_REEXTRACT_EXTS", env_value)
    args = build_parser().parse_args(["--tier", "1", "--resume"])
    apply_env_fallbacks(args)
    assert parse_reextract_exts(args.reextract_exts) == frozenset()
