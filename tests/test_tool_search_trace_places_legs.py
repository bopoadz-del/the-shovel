"""The tool-path trace line places a search leg by leg, like pre_injection.

A search the model runs itself (``path="tool"``) wrote only its handed and
collapsed chunk ids. A chunk it never handed could not be placed: was it in
the vector leg at rank 12, the lexical leg at rank 3, or in neither? The
pre_injection line answers that with ``fused_candidates`` and ``lexical``;
the tool line now carries both, read from the tool search's OWN retrieval
(never the turn's earlier pre-injection record), plus the conversation id.
Each leg's head (``leg_top``) names its chunks by id, so a miss can be placed.

Every input is constructed here: invented ids, texts and scores. The real
``VectorStore.search``, ``_rrf_combine`` and retriever run; only the two legs
and the side lookups are stubbed (the S-R7 fixture's helpers).
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core import doc_index
from app.core.rag import fusion_trace as ft
from app.core.rag import vector_store as vs
from app.core.rag.inject import emit_search_trace
from app.core.rag.vector_store import Chunk
from tests.test_fused_rank_is_chunk_score import (
    PID,
    QUERY,
    _chunk,
    _install,
    _vector_leg,
)


@pytest.fixture
def turn():
    token = vs.start_turn_memo()
    try:
        yield
    finally:
        vs.end_turn_memo(token)


def _legs(n_both: int):
    """A vector leg of 30 hits; the lexical leg's rank 1 is a chunk only it
    found (``x``), followed by the first ``n_both`` vector hits."""
    semantic = _vector_leg(30)
    x = _chunk("x", 0.09, kind="lexical", n=101)
    agree = [Chunk(**{**c.__dict__, "score": round(0.08 - i * 0.001, 6)})
             for i, c in enumerate(semantic[:n_both])]
    return semantic, [x] + agree


def _line(caplog) -> dict:
    lines = [
        json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records
        if r.name == "app.rag.retrieval_trace"
        and r.getMessage().startswith("rag_retrieval_trace ")
    ]
    assert len(lines) == 1, lines
    return lines[0]


@pytest.mark.parametrize("n_both, conv", [(6, "ws-a-1"), (11, "ws-b-2")])
def test_the_tool_line_carries_its_own_search_leg_by_leg(
        monkeypatch, tmp_path, turn, caplog, n_both, conv):
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    # An earlier retrieval in the same turn (pre-injection) left its record;
    # the tool line must not report that one.
    vs.record_fused_candidates(PID, {"n_candidates": 999})

    _rows, trace = doc_index._search_project_documents_traced(PID, QUERY, top_k=2)
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        emit_search_trace(path="tool", project_id=PID, requested_top_k=2,
                          trace=trace, conversation_id=conv)
    line = _line(caplog)

    assert line["path"] == "tool"
    assert line["conversation_id"] == conv
    assert line["lexical"] == {PID: "hit"}
    assert line["degraded"] is False
    fc = line["fused_candidates"]
    assert fc["n_candidates"] == 31
    assert fc["n_both_legs"] == n_both
    assert fc["n_lexical_only"] == 1
    assert fc["best_lexical_only"]["leg_rank"] == 1
    # Each leg's head by id: the lexical leg's rank-1 chunk is named.
    lex_top = fc["leg_top"]["lexical"]
    assert lex_top[0]["chunk_id"] == "x" and lex_top[0]["leg_rank"] == 1
    vec_ids = [e["chunk_id"] for e in fc["leg_top"]["vector"]]
    assert vec_ids == [c.chunk_id for c in semantic[:ft.LEG_TOP_TRACED]]
    # Handed position is stated against the tool's own handed chunks.
    handed = {c["chunk_id"] for c in line["chunks"]}
    for e in lex_top + fc["leg_top"]["vector"]:
        assert (e["handed_position"] is not None) == (e["chunk_id"] in handed)
    # Ids and ranks only.
    assert QUERY not in json.dumps(line)


@pytest.mark.parametrize("query", ["zz-one", "zz-two"])
def test_outside_a_turn_the_tool_line_states_no_fusion(
        monkeypatch, tmp_path, caplog, query):
    """No turn memo (a search outside a chat turn): nothing was kept, and the
    line says so rather than borrowing another retrieval's record."""
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(4)
    _install(monkeypatch, tmp_path, semantic, lexical)
    _rows, trace = doc_index._search_project_documents_traced(PID, query, top_k=2)
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        emit_search_trace(path="tool", project_id=PID, requested_top_k=2, trace=trace)
    line = _line(caplog)
    assert line["fused_candidates"] is None
    assert line["lexical"] == {} and line["degraded"] is False
    assert line["conversation_id"] is None


class _T:
    def __init__(self, cid, doc):
        self.chunk_id = cid
        self.doc_id = doc
        self.text = "t"


@pytest.mark.parametrize("named_doc", ["d1", "d2"])
def test_a_chunk_of_a_document_whose_id_is_its_name_is_digested(named_doc):
    """The parent-row fallback stores ``original_name = id``; a chunk id of
    such a document would carry a file name, so it is printed as a digest.
    Every other chunk keeps its id."""
    order = [_T("d1:0", "d1"), _T("d2:0", "d2")]
    summary = ft.leg_summary(order, {"d1:0": 1, "d2:0": 2}, {}, 10)
    rec = ft.Follow(summary).finish(order, order, order)
    names = {"d1": "d1-stored-name", "d2": "d2-stored-name", named_doc: named_doc}
    view = ft.public_view(rec, order, names)
    by_rank = {e["leg_rank"]: e for e in view["leg_top"]["vector"]}
    for rank, c in ((1, order[0]), (2, order[1])):
        e = by_rank[rank]
        assert "doc_id" not in e
        if c.doc_id == named_doc:
            assert "chunk_id" not in e
            assert e["chunk_digest"] == ft.doc_digest(c.chunk_id)
        else:
            assert e["chunk_id"] == c.chunk_id and "chunk_digest" not in e
