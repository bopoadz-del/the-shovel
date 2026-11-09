"""Tests for the three retrieval-action adapters (item 2)."""
from __future__ import annotations

from types import SimpleNamespace

from app.core.rag.evidence_actions import (
    chunks_to_text,
    lexical_action,
    semantic_action,
    structural_action,
)


def _chunk(text):
    return SimpleNamespace(text=text)


def test_semantic_action_extracts_chunk_text():
    def fake_retrieve(query, project_id, k):
        assert query == "bulk excavation"
        return ([_chunk("Bulk excavation"), _chunk("in all classes")], 0)

    action = semantic_action(fake_retrieve, "p1")
    assert action("bulk excavation") == "Bulk excavation\nin all classes"


def test_lexical_action_extracts_chunk_text():
    def fake_bm25(project_id, query, k):
        return [_chunk("75 mm blinding")]

    action = lexical_action(fake_bm25, "p1")
    assert action("blinding") == "75 mm blinding"


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


def test_structural_action_empty_when_nothing_matches():
    action = structural_action({"Bill No.1": [["A1", "Bulk excavation"]]})
    assert action("plaster") == ""


def test_chunks_to_text_skips_empty_and_handles_non_chunks():
    assert chunks_to_text([_chunk("a"), _chunk(""), _chunk("  ")]) == "a"
    assert chunks_to_text([SimpleNamespace(text=None), "plain"]) == "plain"
