"""Tests for the driver Sources panel display name (app/agents/driver/sources.py).

A user's own document is cited by the name its owner gave it (original_name /
filename), never a content-derived title. A curated library file keeps its title.
"""
from __future__ import annotations

import json

import pytest

import app.agents.driver.sources as sources


@pytest.mark.parametrize(
    "filename",
    [
        "S1_contract_data.md",
        "S2_demolition_boq.md",
    ],
)
def test_display_name_prefers_owner_filename(filename):
    # A content-derived title must not shadow the owner-given filename.
    hit = {"title": "Contract 4da5c2", "filename": filename}
    assert sources._display_name(hit, "own") == filename


@pytest.mark.parametrize(
    "title, filename",
    [
        ("Design values metric", "design_values_metric.md"),
        ("Formwork notes", "formwork_notes.md"),
    ],
)
def test_display_name_uses_title_for_library(title, filename):
    # Curated library files are titled by content, not their storage key.
    hit = {"title": title, "filename": filename}
    assert sources._display_name(hit, "general_knowledge") == title


def _trail(hits):
    return [
        {
            "role": "tool",
            "name": "search_project_documents",
            "content": json.dumps({"result": {"results": hits}}),
        }
    ]


@pytest.mark.parametrize(
    "filename",
    [
        "S1_contract_data.md",
        "S2_demolition_boq.md",
    ],
)
def test_panel_cites_owner_filename_not_title(filename):
    hit = {
        "document_id": "d1",
        "title": "Contract 4da5c2",
        "filename": filename,
        "snippet": "The Time for Completion is 365 days.",
        "score": 0.9,
        "origin": "own",
    }
    rows = sources.panel(_trail([hit]), [])
    assert len(rows) == 1
    assert rows[0]["doc_name"] == filename


@pytest.mark.parametrize(
    "title, filename",
    [
        ("Design values metric", "design_values_metric.md"),
        ("Formwork notes", "formwork_notes.md"),
    ],
)
def test_panel_library_keeps_title(title, filename):
    hit = {
        "document_id": "g1",
        "title": title,
        "filename": filename,
        "snippet": "The design value for the metric.",
        "score": 0.9,
        "origin": "general_knowledge",
    }
    rows = sources.panel(_trail([hit]), [])
    assert len(rows) == 1
    assert rows[0]["doc_name"] == title
