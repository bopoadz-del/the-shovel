"""RQ-3: document type from the type table, document role from a folder table.

Every rule gets two inputs that land on different sides of it. Tables are
built by each test (env), so nothing depends on a real project or folder.
"""

import json
import re

import pytest

from app.core import doc_typing
from app.core.doc_typing import (
    DEFAULT_DOC_TYPE,
    FOLDER_ROLES,
    folder_role_table,
    role_for_path,
    source_path,
    type_for,
)
from app.core.projects import ROLE_BASELINE, ROLE_OTHER, resolve_doc_role


@pytest.fixture(autouse=True)
def _isolated_tables(tmp_path, monkeypatch):
    # Custom document types live under DATA_DIR; keep them out of the test.
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DOC_ROLE_FOLDERS", raising=False)
    monkeypatch.delenv("DOC_ROLE_FOLDERS_FILE", raising=False)


def _table(monkeypatch, mapping):
    monkeypatch.setenv("DOC_ROLE_FOLDERS", json.dumps(mapping))


# ── doc_type from the type table ───────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("Bill of Quantities rev 2.pdf", "boq"),
    ("Bill No 3 - Earthworks.xlsx", "boq"),
    ("Minutes of Meeting 14.docx", "minutes"),
    ("Weekly report 12.pdf", "report"),
    ("Letter to Engineer 041.pdf", "correspondence"),
    ("Outline spec.docx", "specification"),
    ("Payment certificate template.xlsx", "template"),
    ("Payment certificate 07.xlsx", DEFAULT_DOC_TYPE),
])
def test_type_for_reads_the_table(name, expected):
    assert type_for(name) == expected


@pytest.mark.parametrize("name,expected", [
    ("Site Instruction 12.pdf", DEFAULT_DOC_TYPE),
    ("IMG_2041.jpg", "photo"),
])
def test_site_in_a_name_is_not_a_photo(name, expected):
    assert type_for(name) == expected


@pytest.mark.parametrize("content,expected", [
    ("Dear Sir, we refer to your instruction. Yours faithfully", "correspondence"),
    ("Item 1.2 concrete grade C40 placed in bay 4", DEFAULT_DOC_TYPE),
])
def test_type_for_reads_content_when_given(content, expected):
    assert type_for("scan_0042.pdf", content) == expected


# ── folder table ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("role,kept", [("tender", True), ("not_a_role", False)])
def test_table_drops_roles_outside_the_vocabulary(monkeypatch, role, kept):
    _table(monkeypatch, {"Folder A": role})
    assert ("folder a" in folder_role_table()) is kept


@pytest.mark.parametrize("raw,expected", [
    ('{"Folder A": "award"}', {"folder a": "award"}),
    ("{not json", {}),
])
def test_inline_table_must_be_json(monkeypatch, raw, expected):
    monkeypatch.setenv("DOC_ROLE_FOLDERS", raw)
    assert folder_role_table() == expected


@pytest.mark.parametrize("body,expected", [
    ("folder_roles:\n  Folder B: design\n", {"folder b": "design"}),
    ("other_key:\n  Folder B: design\n", {}),
])
def test_table_file_override(tmp_path, monkeypatch, body, expected):
    f = tmp_path / "roles.yaml"
    f.write_text(body, encoding="utf-8")
    monkeypatch.setenv("DOC_ROLE_FOLDERS_FILE", str(f))
    assert folder_role_table() == expected


def test_shipped_table_only_names_known_roles():
    # The default config file is loaded and every entry survives validation.
    raw = doc_typing._raw_table()
    assert raw, "config/doc_role_folders.yaml should ship a table"
    assert len(folder_role_table()) == len(raw)


def test_vocabulary_matches_the_database_constraint():
    from app.core.models import Document
    check = next(a for a in Document.__table_args__
                 if getattr(a, "name", "") == "ck_documents_doc_role")
    allowed = set(re.findall(r"'(\w+)'", str(check.sqltext)))
    assert FOLDER_ROLES <= allowed


# ── role_for_path ──────────────────────────────────────────────────────────

_T = {"folder a": "contract", "folder b": "draft"}


@pytest.mark.parametrize("path,expected", [
    ("Root/Folder A/sub/x.pdf", "contract"),
    ("Root/Other/sub/x.pdf", None),
])
def test_role_from_an_enclosing_folder(path, expected):
    assert role_for_path(path, _T) == expected


@pytest.mark.parametrize("path,expected", [
    ("Folder A/Folder B/x.pdf", "draft"),
    ("Folder B/Folder A/x.pdf", "contract"),
])
def test_nearest_folder_wins(path, expected):
    assert role_for_path(path, _T) == expected


@pytest.mark.parametrize("path,expected", [
    ("Root/folder a/x.pdf", "contract"),
    ("Root/  FOLDER   A /x.pdf", "contract"),
])
def test_folder_match_ignores_case_and_spacing(path, expected):
    assert role_for_path(path, _T) == expected


@pytest.mark.parametrize("path,expected", [
    ("Folder A", None),
    ("Folder A/x.pdf", "contract"),
])
def test_file_name_is_not_a_folder(path, expected):
    assert role_for_path(path, _T) == expected


@pytest.mark.parametrize("path,expected", [
    ("Root\\Folder A\\x.pdf", "contract"),
    ("Root\\Folder C\\x.pdf", None),
])
def test_backslash_paths(path, expected):
    assert role_for_path(path, _T) == expected


@pytest.mark.parametrize("table,expected", [({}, None), (_T, "contract")])
def test_empty_table_gives_no_role(table, expected):
    assert role_for_path("Folder A/x.pdf", table) == expected


@pytest.mark.parametrize("meta,expected", [
    ({"drive_path": "A/b.pdf", "source_path": "C/d.pdf"}, "A/b.pdf"),
    ({"drive_path": "  ", "source_path": "C/d.pdf"}, "C/d.pdf"),
    (None, ""),
])
def test_source_path_from_metadata(meta, expected):
    assert source_path(meta) == expected


# ── resolve_doc_role (what add_document stores) ────────────────────────────

@pytest.mark.parametrize("role,expected", [
    ("weekly_report", "weekly_report"),
    ("other", "contract"),
])
def test_explicit_role_wins_but_other_is_not_explicit(monkeypatch, role, expected):
    _table(monkeypatch, {"Folder A": "contract"})
    assert resolve_doc_role("x.pdf", {"drive_path": "Folder A/x.pdf"}, role) == expected


@pytest.mark.parametrize("name,expected", [
    ("baseline programme.xer", ROLE_BASELINE),
    ("programme.xer", "contract"),
])
def test_readiness_name_role_before_folder(monkeypatch, name, expected):
    _table(monkeypatch, {"Folder A": "contract"})
    assert resolve_doc_role(name, {"drive_path": f"Folder A/{name}"}) == expected


@pytest.mark.parametrize("meta,expected", [
    ({"drive_path": "Folder A/x.pdf"}, "award"),
    ({"drive_path": "Elsewhere/x.pdf"}, ROLE_OTHER),
])
def test_folder_role_or_other(monkeypatch, meta, expected):
    _table(monkeypatch, {"Folder A": "award"})
    assert resolve_doc_role("x.pdf", meta) == expected
