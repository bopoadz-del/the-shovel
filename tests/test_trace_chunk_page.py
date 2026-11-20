"""The retrieval trace says which page each chunk starts on.

A long volume (hundreds of pages, thousands of chunks) shows up in the
trace as one document id with many chunk ordinals. Without the page, a
trace cannot say whether the chunk left out by the handed-set budget was
the page a question needed, and the only way to find out was a database
read. The chunk already carries its stored page; the trace now prints it
for handed and not-handed chunks alike. Never text.

Every test builds its own chunks. No real document, project, question or
value is named.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag.inject import _emit_retrieval_trace, _trace_chunk
from app.core.rag.vector_store import Chunk


def _chunk(doc: str, idx: int, page) -> Chunk:
    c = Chunk(chunk_id=f"p:{doc}:{idx}", project_id="p", doc_id=doc,
              chunk_index=idx, text="invented body", score=0.5)
    c.source_name = f"invented-{doc}.pdf"
    c.page = page
    return c


@pytest.mark.parametrize("page", [7, 412])
def test_trace_chunk_reports_the_stored_page(page):
    assert _trace_chunk(1, _chunk("d", 0, page))["page"] == page


@pytest.mark.parametrize("page", [None, 0])
def test_a_chunk_with_no_page_traces_none(page):
    """A non-PDF chunk, or one indexed before pages were recorded."""
    assert _trace_chunk(1, _chunk("d", 0, page))["page"] is None


@pytest.mark.parametrize("handed_page,left_page", [(3, 9), (250, 31)])
def test_handed_and_left_out_chunks_both_carry_their_page(
        monkeypatch, caplog, handed_page, left_page):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    handed = _chunk("a", 0, handed_page)
    left = _chunk("a", 5, left_page)
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=None,
            conversation_id=None, agent_name="t", retrieved=[handed, left],
            handed=[handed], threshold_fired=False, identifier_miss=False,
            top_score=0.5,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    payload = json.loads(line.split(" ", 1)[1])
    assert [c["page"] for c in payload["chunks"]] == [handed_page]
    assert [c["page"] for c in payload["not_handed"]] == [left_page]
