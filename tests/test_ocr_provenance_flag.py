"""Item 7 — OCR provenance flag on scanned-PDF chunks.

When the PDF extractor OCR's some pages (``ocr_page_numbers``) and stores
per-chunk source pages (``chunk_pages``), the indexer marks those chunks
``ocr=True`` so citations can carry the flag. This file covers the mapping
rule, the Chunk column hydration, the index-project→RAG thread, and the
Sources panel forwarding.
"""

import importlib
import json
from types import SimpleNamespace

import pytest


# ────────────────────────────────────────────────────────────────────────────
# _ocr_flags_for: the per-chunk mapping rule
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "ocr_pages,chunk_pages,expected",
    [
        ([1], [1, 2], [True, False]),
        ([2], [1, 2], [False, True]),
        ([1, 2], [1, 2], [True, True]),
    ],
)
def test_ocr_flags_for_maps_pages_to_chunks(ocr_pages, chunk_pages, expected):
    from app.core import doc_index
    meta = {"ocr_page_numbers": ocr_pages, "chunk_pages": chunk_pages}
    assert doc_index._ocr_flags_for(["a", "b"], meta) == expected


@pytest.mark.parametrize(
    "meta",
    [
        {},                                             # no ocr pages at all
        {"ocr_page_numbers": [1]},                      # no chunk_pages
        {"ocr_page_numbers": [], "chunk_pages": [1]},   # empty ocr pages
    ],
)
def test_ocr_flags_for_returns_none_when_pages_misalign(meta):
    from app.core import doc_index
    assert doc_index._ocr_flags_for(["a"], meta) is None


def test_ocr_flags_for_returns_none_on_length_mismatch():
    from app.core import doc_index
    meta = {"ocr_page_numbers": [1], "chunk_pages": [1]}
    assert doc_index._ocr_flags_for(["a", "b"], meta) is None


# ────────────────────────────────────────────────────────────────────────────
# Chunk.ocr: default + row hydration
# ────────────────────────────────────────────────────────────────────────────

def test_chunk_ocr_defaults_false():
    from app.core.rag.vector_store import Chunk
    assert Chunk(chunk_id="1", project_id="p", doc_id="d",
                 chunk_index=0, text="t").ocr is False


@pytest.mark.parametrize("stored_ocr", [True, False])
def test_row_as_chunk_hydrates_ocr(stored_ocr):
    from app.core.rag.vector_store import VectorStore
    row = SimpleNamespace(
        chunk_id="c1", project_id="p", doc_id="d", chunk_index=0,
        text="text", knowledge_layer=None, authority=None, page=1,
        ocr=stored_ocr,
    )
    chunk = VectorStore._row_as_chunk(row)
    assert chunk.ocr is stored_ocr


# ────────────────────────────────────────────────────────────────────────────
# index_project threads ocr_flags into RAG
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ocr_pages,expected", [
    ([1], [True, False]),
    ([2], [False, True]),
])
def test_index_project_passes_ocr_flags_to_rag(tmp_path, monkeypatch, ocr_pages, expected):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from app.core import doc_index
    importlib.reload(doc_index)
    from app.core.rag import retriever as retriever_mod

    captured = []

    def fake_index_chunks(project_id, doc_id, chunks, *, pages=None, ocr_flags=None):
        captured.append((doc_id, list(ocr_flags) if ocr_flags else None))
        return len(chunks)

    monkeypatch.setattr(retriever_mod, "index_chunks", fake_index_chunks)
    monkeypatch.setattr(
        doc_index._projects, "list_documents",
        lambda pid: [{
            "id": "doc-1",
            "original_name": "scan.pdf",
            "uploaded_at": "t",
            "size": 13,
            "file_path": str(tmp_path / "scan.pdf"),
        }],
    )
    monkeypatch.setattr(
        doc_index, "_extract_with_meta",
        lambda fp, fn, *a, **k: ("scanned text", {"ocr_page_numbers": ocr_pages}),
    )
    monkeypatch.setattr(
        doc_index, "_text_chunks_with_pages",
        lambda text, meta, chunker, fn: (["chunk a", "chunk b"], [1, 2]),
    )
    monkeypatch.setattr(doc_index, "_boq_chunks_for_document", lambda *a, **k: [])
    monkeypatch.setattr(doc_index, "_drawing_chunks_for_document", lambda *a, **k: [])

    doc_index.index_project("pid")

    assert captured, "index_chunks was never called"
    doc_id, flags = captured[0]
    assert doc_id == "doc-1"
    assert flags[:2] == expected


# ────────────────────────────────────────────────────────────────────────────
# Sources panel forwards the flag
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ocr_flag", [True, False])
def test_sources_panel_forwards_ocr_flag(ocr_flag):
    from app.agents.driver import sources as sources_mod
    trail = [{
        "role": "tool",
        "name": "search_project_documents",
        "content": json.dumps({"result": {"results": [{
            "document_id": "doc-1",
            "filename": "scan.pdf",
            "snippet": "scanned text",
            "score": 0.9,
            "origin": "own",
            "page": 1,
            "ocr": ocr_flag,
        }]}}),
    }]
    rows = sources_mod.panel(trail, [], answer="", calculated=False)
    assert rows[0]["ocr"] is ocr_flag


# ─────────────────────────────────────────────────────────────────────────────
# make_rag_chunk_class: the legacy namespace (empty string) reuses RagChunk.
# drop_rule removes the namespace guard and would build a dynamic class for
# ``chunks_``, so ``is RagChunk`` fails and kills the mutant.
# ─────────────────────────────────────────────────────────────────────────────

def test_make_rag_chunk_class_empty_namespace_returns_base():
    from app.core.models import RagChunk, make_rag_chunk_class
    assert make_rag_chunk_class("", 8, "fake") is RagChunk
