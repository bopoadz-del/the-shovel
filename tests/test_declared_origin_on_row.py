"""A knowledge copy's own "Generated from `<table>`" line reaches the search row.

D3 follow-up: ``doc_typing.declared_origin`` read the declaration, but nothing
stored it, so a search row for a generated copy looked like any other
document. The seed now stores the declaration on the document (and heals an
unchanged, already-seeded copy), and the row reads it back. Every input is
built here; no knowledge file on disk is read.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core import doc_index, knowledge_seed
from app.core.doc_typing import declared_origin


@pytest.mark.parametrize(
    "line, expected",
    [
        ("Generated from `lib/table_a.json` by a script.", "lib/table_a.json"),
        ("_Generated from `lib/table_b.json` (single source)._", "lib/table_b.json"),
        ("**Generated from `lib/table_c.json`**", "lib/table_c.json"),
    ],
)
def test_a_declaration_line_is_read_with_or_without_emphasis(line, expected):
    text = f"# Title\n\n{line}\n\n| a | b |\n"
    assert declared_origin(text) == {"kind": "encoded_table", "source": expected}


@pytest.mark.parametrize(
    "text",
    [
        "# Title\n\nThe rates were generated from `a survey` in 2024.\n",
        "# Title\n\nSee `lib/table.json` for the values.\n",
    ],
)
def test_a_mention_inside_prose_is_not_a_declaration(text):
    assert declared_origin(text) == {"kind": "document", "source": None}


@pytest.mark.parametrize(
    "raw, expected",
    [
        (b"# T\n\n_Generated from `x/rates.json` by gen._\n",
         {"declared_origin": {"kind": "encoded_table", "source": "x/rates.json"}}),
        (b"# T\n\nA document that speaks for itself.\n", {}),
    ],
)
def test_seed_metadata_is_the_files_own_declaration(raw, expected):
    assert knowledge_seed._declared_origin_metadata(raw) == expected


class _Store:
    def __init__(self):
        self.writes: list[tuple[str, dict]] = []

    def update_document_metadata(self, doc_id, metadata):
        self.writes.append((doc_id, metadata))


_META = {"declared_origin": {"kind": "encoded_table", "source": "x/rates.json"}}


@pytest.mark.parametrize(
    "stored, meta, expected_writes",
    [
        ({}, _META, [("d1", _META)]),  # seeded before the declaration was stored
        ({"declared_origin": {"kind": "encoded_table", "source": "old.json"}}, _META, [("d1", _META)]),
        (dict(_META), _META, []),  # already in line: no write
        ({}, {}, []),  # the file declares nothing: no write
        (dict(_META), {}, []),  # the file no longer declares: never written as empty
    ],
)
def test_an_unchanged_seeded_copy_is_healed_only_when_out_of_line(stored, meta, expected_writes):
    store = _Store()
    knowledge_seed._heal_declared_origin(store, {"id": "d1", "metadata": stored}, meta)
    assert store.writes == expected_writes


class _SeedStore:
    """Just enough of ``app.core.projects`` for one ``seed_knowledge`` pass."""

    def __init__(self, existing_sha: str | None, existing_meta: dict):
        self.existing_sha = existing_sha
        self.existing_meta = existing_meta
        self.added: list[dict] = []
        self.writes: list[tuple[str, dict]] = []

    def get_project(self, pid):
        return {"origin": "admin_drive_approved", "is_approved": True}

    def general_knowledge_project_ids(self):
        return []

    def find_document_by_sha(self, pid, sha):
        if sha == self.existing_sha:
            return {"id": "old", "metadata": self.existing_meta}
        return None

    def list_documents(self, pid, all_sessions=True):
        return []

    def add_document(self, pid, name, stored_as, filepath, size, content_sha256=None, metadata=None):
        self.added.append({"name": name, "metadata": metadata})
        return {"id": f"new-{len(self.added)}"}

    def update_document_metadata(self, doc_id, metadata):
        self.writes.append((doc_id, metadata))


def _seed(monkeypatch, tmp_path, files: dict[str, bytes], already_seeded: str | None, stored: dict):
    import hashlib

    from app.core import file_crypto, projects, users

    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    for name, raw in files.items():
        (kdir / name).write_bytes(raw)
    sha = None
    if already_seeded:
        sha = hashlib.sha256(files[already_seeded] + b"::chunker=markdown").hexdigest()
    store = _SeedStore(sha, stored)
    for attr in ("get_project", "general_knowledge_project_ids", "find_document_by_sha",
                 "list_documents", "add_document", "update_document_metadata"):
        monkeypatch.setattr(projects, attr, getattr(store, attr))
    monkeypatch.setattr(knowledge_seed, "KNOWLEDGE_DIR", str(kdir))
    monkeypatch.setattr(knowledge_seed, "_gk_project_id", lambda: "gk-test")
    monkeypatch.setattr(users, "ensure_user_exists", lambda *a, **k: None)
    monkeypatch.setattr(file_crypto, "write_document", lambda path, raw: None)
    monkeypatch.setattr(doc_index, "index_document", lambda *a, **k: None)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    knowledge_seed.seed_knowledge()
    return store


_TABLE_COPY = b"# Rates\n\n_Generated from `x/rates.json` by gen._\n\n| a | b |\n"
_PLAIN_DOC = b"# Notes\n\nA document that speaks for itself.\n"


_UNITS_COPY = b"# Units\n\nGenerated from `y/units.json` by gen.\n"
_UNITS_META = {"declared_origin": {"kind": "encoded_table", "source": "y/units.json"}}


@pytest.mark.parametrize(
    "files, expected",
    [
        ({"a.md": _TABLE_COPY, "b.md": _PLAIN_DOC}, {"a.md": _META, "b.md": None}),
        ({"c.md": _PLAIN_DOC, "d.md": _UNITS_COPY}, {"c.md": None, "d.md": _UNITS_META}),
    ],
)
def test_seed_stores_the_declaration_on_a_new_copy_only(monkeypatch, tmp_path, files, expected):
    store = _seed(monkeypatch, tmp_path, files, None, {})
    assert {d["name"]: d["metadata"] for d in store.added} == expected
    assert store.writes == []


@pytest.mark.parametrize(
    "stored, expected_writes",
    [({}, [("old", _META)]), (dict(_META), [])],
)
def test_seed_heals_an_unchanged_copy_without_reindexing(monkeypatch, tmp_path, stored, expected_writes):
    store = _seed(monkeypatch, tmp_path, {"a.md": _TABLE_COPY}, "a.md", stored)
    assert store.added == []
    assert store.writes == expected_writes


def _row(monkeypatch, metadata):
    doc = {"doc_type": "reference", "doc_role": "other"}
    if metadata is not None:
        doc["metadata"] = metadata
    monkeypatch.setattr(doc_index._projects, "get_document", lambda doc_id: doc)
    return doc_index._row_document_fields("d1", [SimpleNamespace(page=None)])


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"declared_origin": {"kind": "encoded_table", "source": "x/rates.json"}},
         {"kind": "encoded_table", "source": "x/rates.json"}),
        ({"declared_origin": {"kind": "encoded_table", "source": "y/units.json"}, "other": 1},
         {"kind": "encoded_table", "source": "y/units.json"}),
    ],
)
def test_the_row_carries_a_stored_declaration(monkeypatch, metadata, expected):
    row = _row(monkeypatch, metadata)
    assert row["declared_origin"] == expected
    assert row["doc_type"] == "reference"


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"declared_origin": {"kind": "document", "source": None}},
        {"declared_origin": {"kind": "document", "source": "x/rates.json"}},
        {"declared_origin": {"kind": "encoded_table", "source": "  "}},
        {"declared_origin": "x/rates.json"},
    ],
)
def test_the_row_says_nothing_without_a_usable_declaration(monkeypatch, metadata):
    assert "declared_origin" not in _row(monkeypatch, metadata)
