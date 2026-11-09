"""Tests for the three retrieval-action adapters (item 2)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.rag.evidence_actions import (
    _lexical_retrieve,
    _rows_for_item,
    _semantic_retrieve,
    chunks_to_text,
    lexical_action,
    semantic_action,
    structural_action,
)


def _chunk(text):
    return SimpleNamespace(text=text)


@pytest.mark.parametrize("query,chunks,want", [
    ("bulk excavation", ["Bulk excavation", "in all classes"],
     "Bulk excavation\nin all classes"),
    ("blinding", ["75 mm blinding"], "75 mm blinding"),
])
def test_semantic_action_extracts_chunk_text(query, chunks, want):
    def fake_retrieve(q, project_id, k):
        assert q == query
        return ([_chunk(c) for c in chunks], 0)

    action = semantic_action(fake_retrieve, "p1")
    assert action(query) == want


@pytest.mark.parametrize("query,want", [
    ("blinding", "75 mm blinding"),
    ("excavation", "Bulk excavation"),
])
def test_lexical_action_extracts_chunk_text(query, want):
    def fake_bm25(project_id, q, k):
        return [_chunk(want)]

    action = lexical_action(fake_bm25, "p1")
    assert action(query) == want


@pytest.mark.parametrize("query,want", [
    ("bulk excavation", "Bulk excavation"),
    ("blinding", "75 mm blinding"),
])
def test_semantic_retrieve_helper(query, want):
    def fake_retrieve(q, project_id, k):
        return ([_chunk(want)], 0)

    assert _semantic_retrieve(fake_retrieve, "p1", 5, query) == want


@pytest.mark.parametrize("query,want", [
    ("blinding", "75 mm blinding"),
    ("excavation", "Bulk excavation"),
])
def test_lexical_retrieve_helper(query, want):
    def fake_bm25(project_id, q, k):
        return [_chunk(want)]

    assert _lexical_retrieve(fake_bm25, "p1", 50, query) == want


@pytest.mark.parametrize("term,want_sheet", [
    ("excavation", "Bill No.1"),
    ("walls", "Bill No.4"),
])
def test_rows_for_item_helper(term, want_sheet):
    sheets = {
        "Bill No.1": [
            ["code", "description", "unit"],
            ["A1", "Bulk excavation in all classes", "m3"],
        ],
        "Bill No.4": [
            ["B1", "C40/50 shear walls", "m3"],
        ],
    }
    text = _rows_for_item(sheets, {"Bill No.1": sheets["Bill No.1"][0]}, term)
    assert want_sheet in text


def test_structural_action_returns_matching_rows_across_sheets():
    sheets = {
        "Bill No.1": [
            ["code", "description", "unit"],
            ["A1", "Bulk excavation in all classes", "m3"],
        ],
        "Bill No.4": [
            ["B1", "C40/50 shear walls", "m3"],
        ],
    }
    action = structural_action(sheets, headers={"Bill No.1": sheets["Bill No.1"][0]})
    text = action("excavation")
    assert "Bill No.1" in text
    assert "Bulk excavation" in text
    assert "shear walls" not in text


@pytest.mark.parametrize("term", ["plaster", "ceilings"])
def test_structural_action_empty_when_nothing_matches(term):
    action = structural_action({"Bill No.1": [["A1", "Bulk excavation"]]})
    assert action(term) == ""


@pytest.mark.parametrize("chunks,want", [
    ([_chunk("a"), _chunk(""), _chunk("  ")], "a"),
    ([SimpleNamespace(text=None), "plain"], "plain"),
])
def test_chunks_to_text_skips_empty_and_handles_non_chunks(chunks, want):
    assert chunks_to_text(chunks) == want
