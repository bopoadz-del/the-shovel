"""Tests for the scenario JSONL validation pipeline.

The validator runs AFTER generation: drops empty/short, dedupes by
embedding cosine >= 0.85, flags suspicious noun-overlap.
"""
from __future__ import annotations

import pytest


def test_validate_drops_empty_and_short():
    from scripts.generate_training_scenarios import _validate_scenarios
    rows = [
        {"instruction": "Q1", "response": "fully formed answer about CPM"},
        {"instruction": "Q2", "response": ""},
        {"instruction": "Q3", "response": "too short"},  # < 30 chars
        {"instruction": "",   "response": "no question"},
    ]
    kept, report = _validate_scenarios(rows)
    assert len(kept) == 1
    assert kept[0]["instruction"] == "Q1"
    assert report["dropped_empty"] >= 2
    assert report["dropped_short"] >= 1


def test_validate_dedupes_by_response_cosine(monkeypatch):
    """Two near-identical responses should collapse to one. Use the
    fake embedder so the test is deterministic."""
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from scripts.generate_training_scenarios import _validate_scenarios
    rows = [
        {"instruction": "Q1", "response": "Concrete cover is 30mm per ACI 318 for slab moderate exposure"},
        {"instruction": "Q2", "response": "Concrete cover is 30mm per ACI 318 for slab moderate exposure"},
        {"instruction": "Q3", "response": "Saudi switchgear lead time is 28 weeks from Europe"},
    ]
    kept, report = _validate_scenarios(rows)
    assert len(kept) == 2
    assert report["dropped_duplicates"] == 1
