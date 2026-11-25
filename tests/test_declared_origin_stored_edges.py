"""Edge cases for the stored declared origin (``doc_typing.with_declared_origin``
and ``doc_typing.stored_origin``).

Every input is invented here, two per decision: a copy that is not the
caller's mapping, a partial stored origin, a wrong-typed row, a stored kind
other than the encoded table, extra or non-text stored fields, and empty or
missing text.
"""
from __future__ import annotations

import pytest

from app.core import doc_typing

KEY = doc_typing.ORIGIN_METADATA_KEY
_SILENT = ["# Field notes\n\nMeasured on site.\n", "Plain prose with no origin line.\n"]


@pytest.mark.parametrize("text,given", [(_SILENT[0], {"k": 1}), (_SILENT[1], {"drive_path": "p/q.md", "n": [2]})])
def test_a_silent_document_returns_a_copy_not_the_callers_mapping(text, given):
    before = dict(given)
    kept = doc_typing.with_declared_origin(given, text)
    assert kept == before and kept is not given
    kept["added"] = True
    assert given == before


@pytest.mark.parametrize("text", ["", None])
def test_empty_or_missing_text_stores_nothing(text):
    assert doc_typing.with_declared_origin(None, text) is None
    assert doc_typing.with_declared_origin({"k": 3}, text) == {"k": 3}


@pytest.mark.parametrize("text", ["Generated from ``\n", "Generated from `\nx.json`\n"])
def test_an_empty_or_broken_origin_line_stores_nothing(text):
    assert doc_typing.with_declared_origin({"k": 4}, text) == {"k": 4}


@pytest.mark.parametrize("metadata", [["not", "a", "mapping"], "drive_path=a/b.md"])
def test_wrong_typed_metadata_with_a_declaring_text_gets_only_the_origin(metadata):
    out = doc_typing.with_declared_origin(metadata, "Generated from `cfg/t.json`\n")
    assert out == {KEY: {"kind": "encoded_table", "source": "cfg/t.json"}}


@pytest.mark.parametrize("stored", [{KEY: {"source": "app/x.json"}}, {KEY: {"kind": "", "source": "cfg/y.yaml"}}])
def test_a_stored_origin_without_a_kind_speaks_for_itself(stored):
    assert doc_typing.stored_origin(stored) == {"kind": "document", "source": None}


@pytest.mark.parametrize("row", ['{"declared_origin": {"kind": "encoded_table", "source": "a.json"}}',
                                 [("declared_origin", {"kind": "encoded_table", "source": "b.json"})]])
def test_a_row_whose_metadata_is_not_a_mapping_speaks_for_itself(row):
    assert doc_typing.stored_origin(row) == {"kind": "document", "source": None}


@pytest.mark.parametrize("kind,source", [("survey_table", "data/s.csv"), ("encoded_table", "lib/r.json")])
def test_the_stored_kind_reads_back_as_stored(kind, source):
    assert doc_typing.stored_origin({KEY: {"kind": kind, "source": source}}) == {"kind": kind, "source": source}


@pytest.mark.parametrize("value,expected", [
    ({"kind": "encoded_table", "source": "t.json", "extra": 1}, {"kind": "encoded_table", "source": "t.json"}),
    ({"kind": "encoded_table", "source": 42}, {"kind": "encoded_table", "source": "42"}),
])
def test_the_read_back_has_only_kind_and_text_source(value, expected):
    assert doc_typing.stored_origin({KEY: value}) == expected
