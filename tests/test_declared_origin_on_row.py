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
    ],
)
def test_an_unchanged_seeded_copy_is_healed_only_when_out_of_line(stored, meta, expected_writes):
    store = _Store()
    knowledge_seed._heal_declared_origin(store, {"id": "d1", "metadata": stored}, meta)
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
        {"declared_origin": {"kind": "encoded_table", "source": "  "}},
        {"declared_origin": "x/rates.json"},
    ],
)
def test_the_row_says_nothing_without_a_usable_declaration(monkeypatch, metadata):
    assert "declared_origin" not in _row(monkeypatch, metadata)
