"""The tool-path trace line places a search leg by leg, like pre_injection.

A search the model runs itself (``path="tool"``) wrote only its handed and
collapsed chunk ids. A chunk it never handed could not be placed: was it in
the vector leg at rank 12, the lexical leg at rank 3, or in neither? The
pre_injection line answers that with ``fused_candidates`` and ``lexical``;
the tool line now carries both, read from the tool search's OWN retrieval
(never the turn's earlier pre-injection record), through the same public
view: ranks, handed positions and ``leg_chunk_digests``, no chunk id. It
also carries the conversation id. The trace adds no database read, and it
fails open: the search's results never change because of it.

Every input is constructed here: invented ids, texts and scores. The real
``VectorStore.search``, ``_rrf_combine``, retriever and tool handler run;
only the two legs and the side lookups are stubbed (the S-R7 fixture's
helpers).
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from app.agents.base.tools import handle_search_project_documents
from app.agents.core.tool_registry import ToolCall
from app.core import doc_index
from app.core.rag import fusion_trace as ft
from app.core.rag import inject
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


def _lines(caplog) -> list[dict]:
    return [
        json.loads(r.getMessage().split(" ", 1)[1]) for r in caplog.records
        if r.name == "app.rag.retrieval_trace"
        and r.getMessage().startswith("rag_retrieval_trace ")
    ]


def _line(caplog) -> dict:
    lines = _lines(caplog)
    assert len(lines) == 1, lines
    return lines[0]


def _no_name_read(monkeypatch) -> list:
    """Record every document-name lookup; the tool path must make none."""
    from app.core import projects

    calls: list = []

    def _names(ids):
        calls.append(list(ids))
        return {}

    monkeypatch.setattr(projects, "document_names", _names)
    return calls


@pytest.mark.parametrize("n_both, conv", [(6, "ws-a-1"), (11, "ws-b-2")])
def test_the_tool_line_carries_its_own_search_leg_by_leg(
        monkeypatch, tmp_path, turn, caplog, n_both, conv):
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    name_reads = _no_name_read(monkeypatch)
    # An earlier retrieval in the same turn (pre-injection) left its record;
    # the tool line must not report that one.
    vs.record_fused_candidates(PID, {"n_candidates": 999})

    _rows, trace = doc_index._search_project_documents_traced(PID, QUERY, top_k=2)
    # The retrieval's own name prefetch (on main, part of the search) is not
    # the trace's; count only what writing the line reads.
    name_reads.clear()
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
    # Each leg placed by rank digest (#463's field), position = leg rank.
    digests = fc["leg_chunk_digests"]
    assert digests["vector"] == [ft.id_digest(c.chunk_id) for c in semantic]
    assert digests["lexical"] == [ft.id_digest(c.chunk_id) for c in lexical]
    # Handed position is stated against the tool's own handed chunks.
    handed = [c["chunk_id"] for c in line["chunks"]]
    lex_top = fc["leg_top"]["lexical"]
    assert lex_top[0]["leg_rank"] == 1
    want = handed.index("x") + 1 if "x" in handed else None
    assert lex_top[0]["handed_position"] == want
    # No raw chunk id and no raw document id anywhere in the fused view.
    fc_text = json.dumps(fc)
    assert "chunk_id" not in fc_text
    for c in semantic + lexical:
        assert f'"{c.chunk_id}"' not in fc_text
        assert f'"{c.doc_id}"' not in fc_text
    # No database read added: writing the line made no name lookup.
    assert name_reads == []
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


# ── Through the real tool handler ───────────────────────────────────────────


def _call(query: str, conv: str | None) -> ToolCall:
    return ToolCall(
        agent=None, name="search_project_documents",
        args={"query": query, "top_k": 3}, project_id=PID, conversation_id=conv,
    )


def _run_handler(query: str, conv: str | None) -> dict:
    token = vs.start_turn_memo()
    try:
        return asyncio.run(handle_search_project_documents(_call(query, conv)))
    finally:
        vs.end_turn_memo(token)


def _bytes(out: dict) -> str:
    return json.dumps(out, sort_keys=True, default=str)


@pytest.mark.parametrize("n_both, query, conv", [
    (6, QUERY, "ws-p1-1700000000000"),
    (11, "temporary site access gravel barrier", "ws-p2-1700000000001"),
])
def test_the_conversation_id_reaches_the_line_through_the_tool_handler(
        monkeypatch, tmp_path, caplog, n_both, query, conv):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        out = _run_handler(query, conv)
    assert out["ok"] is True
    tool_lines = [ln for ln in _lines(caplog) if ln["path"] == "tool"]
    assert len(tool_lines) == 1
    assert tool_lines[0]["conversation_id"] == conv
    assert tool_lines[0]["fused_candidates"]["leg_chunk_digests"]["vector"]


@pytest.mark.parametrize("n_both, query", [
    (6, QUERY),
    (11, "temporary site access gravel barrier"),
])
def test_results_and_model_output_are_byte_identical_with_and_without_trace(
        monkeypatch, tmp_path, n_both, query):
    """The tool's results, their order and the model-visible output are the
    same bytes whether the trace is written or not."""
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "0")
    without = _run_handler(query, "ws-p-1700000000000")
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    with_trace = _run_handler(query, "ws-p-1700000000000")
    assert without["result"]["results"], "the search must find rows"
    assert _bytes(with_trace) == _bytes(without)


# ── Fails open: a trace failure never touches the search ────────────────────


def _boom(*_a, **_k):
    raise RuntimeError("synthetic trace failure")


@pytest.mark.parametrize("n_both, query", [
    (6, QUERY),
    (11, "temporary site access gravel barrier"),
])
def test_a_leg_record_failure_leaves_the_search_result_untouched(
        monkeypatch, tmp_path, caplog, n_both, query):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    good = _run_handler(query, "ws-p-1700000000000")
    monkeypatch.setattr(vs, "fused_candidates_record", _boom)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        broken = _run_handler(query, "ws-p-1700000000000")
    assert good["result"]["results"]
    assert _bytes(broken) == _bytes(good)
    # The line is still written; it states no fusion instead of failing.
    line = [ln for ln in _lines(caplog) if ln["path"] == "tool"][0]
    assert line["fused_candidates"] is None and line["lexical"] == {}


@pytest.mark.parametrize("n_both, query", [
    (6, QUERY),
    (11, "temporary site access gravel barrier"),
])
def test_an_emit_failure_leaves_the_search_result_untouched(
        monkeypatch, tmp_path, n_both, query):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    good = _run_handler(query, "ws-p-1700000000000")
    monkeypatch.setattr(inject, "_public_fused", _boom)
    broken = _run_handler(query, "ws-p-1700000000000")
    assert good["result"]["results"]
    assert _bytes(broken) == _bytes(good)
