"""The fusion's leg summary is computed only while the retrieval trace is on.

``_rrf_combine`` summarises the fused order by leg (``fusion_trace.leg_summary``)
for the pre_injection trace line, and nothing else reads that summary. With
``RAG_RETRIEVAL_TRACE`` off, no summary is computed or recorded, and the
ranking that leaves the retrieval is the same as with the trace on.

Every input is constructed here: invented texts, ids and scores. The real
``VectorStore.search`` and ``_rrf_combine`` run; only the two legs and the
side lookups are stubbed (the S-R7 fixture's helpers).
"""
from __future__ import annotations

import pytest

from app.core.rag import fusion_trace as ft
from app.core.rag import retriever
from app.core.rag import vector_store as vs
from app.core.rag.vector_store import Chunk
from tests.test_fused_rank_is_chunk_score import (
    PID, QUERY, _chunk, _install, _vector_leg,
)


@pytest.fixture
def turn():
    token = vs.start_turn_memo()
    try:
        yield
    finally:
        vs.end_turn_memo(token)


def _legs(n_vec: int, n_both: int):
    semantic = _vector_leg(n_vec)
    lone = _chunk("lex-only", 0.09, kind="lexical", n=n_vec + 7)
    agree = [Chunk(**{**c.__dict__, "score": round(0.08 - i * 0.001, 6)})
             for i, c in enumerate(semantic[:n_both])]
    return semantic, [lone] + agree


def _count_summaries(monkeypatch):
    calls = []
    real = ft.leg_summary

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(ft, "leg_summary", counting)
    return calls


def _run(monkeypatch, tmp_path, trace_value, n_vec, n_both):
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", trace_value)
    semantic, lexical = _legs(n_vec, n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    calls = _count_summaries(monkeypatch)
    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    return pooled, calls


@pytest.mark.parametrize("off", ["0", "off"])
@pytest.mark.parametrize("n_vec,n_both", [(30, 8), (45, 15)])
def test_trace_off_computes_and_records_no_leg_summary(
        monkeypatch, tmp_path, turn, off, n_vec, n_both):
    _, calls = _run(monkeypatch, tmp_path, off, n_vec, n_both)

    assert calls == []
    assert vs.fusion_records(PID) == []
    assert vs.fused_candidates_record(PID) is None


@pytest.mark.parametrize("on", ["1", "true"])
@pytest.mark.parametrize("n_vec,n_both", [(30, 8), (45, 15)])
def test_trace_on_still_records_the_summary(
        monkeypatch, tmp_path, turn, on, n_vec, n_both):
    _, calls = _run(monkeypatch, tmp_path, on, n_vec, n_both)

    assert calls, "the trace is on: the fusion must be summarised"
    rec = vs.fused_candidates_record(PID)
    assert rec is not None
    assert rec["n_candidates"] == n_vec + 1
    assert rec["n_both_legs"] == n_both


@pytest.mark.parametrize("n_vec,n_both", [(30, 8), (45, 15)])
def test_the_ranking_is_the_same_with_the_trace_on_or_off(
        monkeypatch, tmp_path, n_vec, n_both):
    orders = {}
    for value in ("1", "0"):
        token = vs.start_turn_memo()
        try:
            pooled, _ = _run(monkeypatch, tmp_path, value, n_vec, n_both)
        finally:
            vs.end_turn_memo(token)
        orders[value] = [(c.chunk_id, getattr(c, "fused_rank", None))
                         for c in pooled]

    assert orders["1"] == orders["0"]
    assert orders["0"], "the retrieval returned nothing to compare"
