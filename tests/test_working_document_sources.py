"""An answer that uses a working document lists that document, not leftovers.

The file is read for the turn (its text is in the tool result) and is not
a retrieved chunk. Sources must name it by the name the result carries,
and must not fill the panel with retrieved chunks the answer did not use.
An answer that did not use a working document keeps the retrieved list.
"""
from __future__ import annotations

import json

from app.agents.runtime import _build_sources_from_audit

_NOTE = (
    "The slab pour is scheduled for level three at forty two cubic metres "
    "after the curing inspection."
)
_ANSWER = (
    "Your note says the slab pour is scheduled for level three at forty "
    "two cubic metres."
)
_KB = (
    "Delay damages accrue daily once the time for completion has passed "
    "and the engineer has certified the overrun."
)


def _audit() -> dict:
    return {
        "chunks": [
            {"doc_id": "kb-1", "chunk_index": 0, "score": 0.93, "layer": "general_knowledge", "text": _KB},
            {"doc_id": "kb-2", "chunk_index": 2, "score": 0.88, "layer": "general_knowledge", "text": _KB},
            {"doc_id": "kb-3", "chunk_index": 4, "score": 0.81, "layer": "general_knowledge", "text": _KB},
            {"doc_id": "kb-4", "chunk_index": 9, "score": 0.40, "layer": "general_knowledge", "text": _KB},
        ],
    }


def _names(monkeypatch) -> None:
    known = {
        "kb-1": "guide-a.pdf",
        "kb-2": "guide-b.pdf",
        "kb-3": "guide-c.pdf",
        "kb-4": "guide-d.pdf",
    }
    monkeypatch.setattr(
        "app.core.projects.get_document",
        lambda did: {"original_name": known.get(did, "")},
    )


def _read(filename: str = "pour-note.txt", text: str = _NOTE, doc_id: str = "upload-1") -> dict:
    return {
        "name": "fetch_document",
        "ok": True,
        "predispatched": True,
        "result": {"filename": filename, "document_id": doc_id, "content": text},
    }


def test_quoted_working_document_is_listed_and_unused_chunks_are_not(monkeypatch):
    _names(monkeypatch)
    out = _build_sources_from_audit(_audit(), _ANSWER, tool_results=[_read()])
    shown = [s.get("doc_name") for s in out]
    assert shown == ["pour-note.txt"]
    assert out[0].get("doc_id") == "upload-1"
    assert "guide-a.pdf" not in shown
    assert "guide-b.pdf" not in shown
    assert "guide-c.pdf" not in shown


def test_unused_working_document_does_not_replace_retrieved_sources(monkeypatch):
    _names(monkeypatch)
    out = _build_sources_from_audit(
        _audit(),
        "The overrun is certified and damages then accrue.",
        tool_results=[_read()],
    )
    shown = [s.get("doc_name") for s in out]
    assert "pour-note.txt" not in shown
    assert shown == ["guide-a.pdf", "guide-b.pdf", "guide-c.pdf"]


def test_quoted_working_document_keeps_a_chunk_the_answer_also_cited(monkeypatch):
    _names(monkeypatch)
    answer = _ANSWER + " See [source: guide-a.pdf, chunk 0]."
    out = _build_sources_from_audit(_audit(), answer, tool_results=[_read()])
    shown = [s.get("doc_name") for s in out]
    assert "pour-note.txt" in shown
    assert "guide-a.pdf" in shown
    assert "guide-b.pdf" not in shown
    assert "guide-c.pdf" not in shown


def test_no_working_document_still_lists_the_top_retrieved_chunks(monkeypatch):
    _names(monkeypatch)
    out = _build_sources_from_audit(_audit(), "answer with no citations")
    assert [s.get("doc_name") for s in out] == ["guide-a.pdf", "guide-b.pdf", "guide-c.pdf"]


def test_driver_panel_lists_the_quoted_file_not_unused_hits():
    from app.agents.driver import sources

    trail = [
        {
            "role": "tool",
            "name": "search_general_knowledge",
            "content": json.dumps({
                "ok": True,
                "result": {"results": [
                    {"document_id": "g1", "filename": "guide-a.pdf", "score": 0.91,
                     "origin": "general_knowledge", "snippet": _KB},
                    {"document_id": "g2", "filename": "guide-b.pdf", "score": 0.84,
                     "origin": "general_knowledge", "snippet": _KB},
                    {"document_id": "g3", "filename": "guide-c.pdf", "score": 0.77,
                     "origin": "general_knowledge", "snippet": _KB},
                ]},
            }),
        },
        {
            "role": "tool",
            "name": "fetch_document",
            "content": json.dumps({"ok": True, "result": {
                "filename": "pour-note.txt", "document_id": "upload-1", "content": _NOTE,
            }}),
        },
    ]
    out = sources.panel(trail, [], _ANSWER)
    shown = [s.get("doc_name") for s in out]
    assert "pour-note.txt" in shown
    assert "guide-a.pdf" not in shown
    assert "guide-b.pdf" not in shown
    assert "guide-c.pdf" not in shown
