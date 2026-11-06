"""Drive re-link for the link-missing rows (E-2 of the corpus order).

The rows are NOT ghosts: their files are in Drive, and the missing piece is
``metadata.drive_file_id``. These tests pin the mechanism, not any row:

* exactly one basename match relinks, and the row's ingest ledger reopens
  (UNVERIFIED, reason NULL);
* several matches with a known size pick the size-consistent one;
* several matches with no decidable size are left UNTOUCHED and ambiguous
  (the guard -- never guessed; parametrized);
* zero matches are left UNTOUCHED and missing;
* a second run changes nothing (idempotence);
* a row that already has a drive_file_id is never overwritten.
"""

from __future__ import annotations

import pytest

from app.core import projects as projects_mod
from app.core.drive_relink import (
    basename_of,
    relink_shell_rows,
    resolve_row,
)
from app.core.projects import relink_document_drive_id
from app.core.ingest_status import EXTRACT_FAILED, UNVERIFIED


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DATA_DIR + fresh projects DB for each test."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    return tmp_path


def _index(*names: str):
    index = {}
    for name in names:
        key = name.strip().lower()
        index.setdefault(key, []).append({
            "id": f"id-{key}",
            "name": name,
            "size": 100,
            "path": f"folder/{name}",
        })
    return index


def _row(source_path: str, size_bytes: int = 0) -> dict:
    return {"id": "row-1", "source_path": source_path, "size_bytes": size_bytes}


# --- the decision table ------------------------------------------------------


@pytest.mark.parametrize(
    "source_path",
    [r"G:\Archive Root\Sub\file.pdf", "/Archive Root/Sub/file.pdf"],
)
def test_exactly_one_basename_match_relinks(source_path):
    outcome, hit = resolve_row(_row(source_path), _index("file.pdf"))
    assert outcome == "relink"
    assert hit["id"] == "id-file.pdf"


@pytest.mark.parametrize("size_bytes", [100, 200])
def test_several_matches_pick_the_size_consistent_one(size_bytes):
    index = {}
    for name, size in (("plan.pdf", 100), ("plan.pdf", 200)):
        index.setdefault(name, []).append({
            "id": f"id-{size}", "name": name, "size": size, "path": f"a/{name}",
        })
    outcome, hit = resolve_row(
        _row(r"G:\plan.pdf", size_bytes=size_bytes), index,
    )
    assert outcome == "relink"
    assert hit["size"] == size_bytes


@pytest.mark.parametrize(
    "row",
    [
        _row(r"G:\plan.pdf", size_bytes=0),
        _row(r"G:\plan.pdf", size_bytes=150),
    ],
)
def test_undecidable_matches_stay_untouched_and_ambiguous(row):
    index = {}
    for size in (100, 200):
        index.setdefault("plan.pdf", []).append({
            "id": f"id-{size}", "name": "plan.pdf", "size": size, "path": "plan.pdf",
        })
    outcome, hit = resolve_row(row, index)
    assert outcome == "ambiguous"
    assert hit is None


@pytest.mark.parametrize(
    "source_path",
    [r"G:\Archive Root\gone.pdf", r"G:\Archive Root\other\gone.pdf"],
)
def test_zero_matches_stay_untouched_and_missing(source_path):
    outcome, hit = resolve_row(_row(source_path), _index("present.pdf"))
    assert outcome == "missing"
    assert hit is None


@pytest.mark.parametrize(
    "path,expected",
    [
        (r"G:\Archive Root\file.pdf", "file.pdf"),
        ("/Archive Root/file.pdf", "file.pdf"),
    ],
)
def test_basename_is_platform_neutral(path, expected):
    assert basename_of(path) == expected


# --- the index build ---------------------------------------------------------


@pytest.mark.parametrize(
    "entries",
    [
        [{"id": "d1", "name": "Spec.pdf", "mimeType": "application/pdf", "size": "11", "_drive_path": "a/Spec.pdf"}],
        [{"id": "d2", "name": "Plan.pdf", "mimeType": "application/pdf", "size": "22", "_drive_path": "b/Plan.pdf"},
         {"id": "d3", "name": "sub", "mimeType": "application/vnd.google-apps.folder", "size": "0", "_drive_path": "sub"}],
    ],
)
def test_build_drive_index_indexes_basenames_and_skips_folders(monkeypatch, entries):
    from app.core import drive_relink as dr
    monkeypatch.setattr(
        "app.core.gdrive_service.walk_folder",
        lambda root: (entries, []),
    )
    index = dr.build_drive_index("root")
    for entry in entries:
        if entry["mimeType"] == dr.FOLDER_MIME:
            assert entry["name"].lower() not in index
        else:
            assert entry["name"].lower() in index


@pytest.mark.parametrize("blank_name", ["", "   "])
def test_build_drive_index_skips_blank_names(monkeypatch, blank_name):
    from app.core import drive_relink as dr
    monkeypatch.setattr(
        "app.core.gdrive_service.walk_folder",
        lambda root: ([{"id": "d1", "name": blank_name, "mimeType": "application/pdf",
                        "size": "1", "_drive_path": "x"}], []),
    )
    index = dr.build_drive_index("root")
    assert "" not in index


@pytest.mark.parametrize("has_drive_id", [True, False])
def test_rows_without_drive_id_skips_linked_rows(fresh_db, has_drive_id):
    from app.core import drive_relink as dr
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    meta = {"source_path": r"G:\Archive Root.pdf"}
    if has_drive_id:
        meta["drive_file_id"] = "already-linked"
    projects_mod.add_document(pid, "a.pdf", file_path="a.pdf", size=0, metadata=meta)
    rows = dr.rows_without_drive_id()
    assert (len(rows) == 0) if has_drive_id else (len(rows) == 1)


# --- the write side ----------------------------------------------------------


@pytest.mark.parametrize("name", ["one.pdf", "two.pdf"])
def test_relinking_reopens_the_row_with_the_drive_id(fresh_db, name):
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(
        pid, name, file_path=name, size=0,
        metadata={"source_path": rf"G:\Archive Root\{name}"},
    )
    updated = relink_document_drive_id(doc["id"], "drive-file-9")
    assert updated is True
    row = projects_mod.get_document(doc["id"])
    assert (row.get("metadata") or {}).get("drive_file_id") == "drive-file-9"
    assert row["ingest_status"] == UNVERIFIED
    assert row.get("ingest_status_reason") is None
    assert row.get("extractor_version") is None


@pytest.mark.parametrize("first_id", ["drive-file-1", "drive-file-a"])
def test_running_the_relink_twice_changes_nothing_the_second_time(fresh_db, first_id):
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(
        pid, "plan.pdf", file_path="plan.pdf", size=0,
        metadata={"source_path": r"G:\Archive Root\plan.pdf"},
    )
    assert relink_document_drive_id(doc["id"], first_id) is True
    assert relink_document_drive_id(doc["id"], "drive-file-2") is False
    row = projects_mod.get_document(doc["id"])
    assert (row.get("metadata") or {}).get("drive_file_id") == first_id


@pytest.mark.parametrize("existing_id", ["drive-file-7", "other-id"])
def test_a_row_with_a_drive_file_id_is_never_overwritten(fresh_db, existing_id):
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    doc = projects_mod.add_document(
        pid, "plan.pdf", file_path="plan.pdf", size=0,
        metadata={
            "source_path": r"G:\Archive Root\plan.pdf",
            "drive_file_id": existing_id,
        },
    )
    projects_mod.stamp_document_index(
        doc["id"], chunk_count=0, ingest_status=EXTRACT_FAILED,
        ingest_status_reason="attempts_exhausted@old:1:terminal",
    )
    updated = relink_document_drive_id(doc["id"], "new-id")
    assert updated is False
    row = projects_mod.get_document(doc["id"])
    assert (row.get("metadata") or {}).get("drive_file_id") == existing_id
    assert row["ingest_status"] == EXTRACT_FAILED


@pytest.mark.parametrize("apply", [False, True])
def test_the_runner_counts_relinked_ambiguous_and_missing(monkeypatch, fresh_db, apply):
    pid = projects_mod.create_project("demo_proj", "demo_owner")["id"]
    projects_mod.add_document(
        pid, "solo.pdf", file_path="solo.pdf", size=0,
        metadata={"source_path": r"G:\Archive Root\solo.pdf"},
    )
    projects_mod.add_document(
        pid, "dupe.pdf", file_path="dupe.pdf", size=0,
        metadata={"source_path": r"G:\Archive Root\dupe.pdf"},
    )
    projects_mod.add_document(
        pid, "gone.pdf", file_path="gone.pdf", size=0,
        metadata={"source_path": r"G:\Archive Root\gone.pdf"},
    )
    monkeypatch.setattr(
        "app.core.drive_relink.build_drive_index",
        lambda root: {
            "solo.pdf": [{"id": "d1", "name": "solo.pdf", "size": 10, "path": "solo.pdf"}],
            "dupe.pdf": [
                {"id": "d2", "name": "dupe.pdf", "size": 10, "path": "dupe.pdf"},
                {"id": "d3", "name": "dupe.pdf", "size": 20, "path": "dupe.pdf"},
            ],
        },
    )
    out = relink_shell_rows("root", apply=apply)
    assert out["relinked"] == 1
    assert out["ambiguous"] == 1
    assert out["missing"] == 1
    assert out["rows"] == 3
    if apply:
        solo = next(
            d for d in projects_mod.list_documents(pid)
            if d["original_name"] == "solo.pdf"
        )
        assert (solo.get("metadata") or {}).get("drive_file_id") == "d1"
