"""S-R7b: the pre_injection trace carries a summary of the fused candidates.

Reciprocal-rank fusion with k=60 ranks a chunk found by BOTH legs above a
rank-1 chunk found by one leg (2/(60+r) > 1/61 for every r <= 50 per leg),
so the best candidate only one leg found can sit below every agreeing
candidate. The trace now shows that: how many candidates each leg found
alone, and where the best lexical-only and best vector-only candidate ended
up (fused rank, scored position, pool position, handed position, or the step
that dropped it). Counts and ranks only; no ranking changes.

Every input is constructed here: invented texts, ids and scores. The real
``VectorStore.search`` and ``_rrf_combine`` run; only the two legs and the
side lookups are stubbed. The helpers are the S-R7 fixture's.
"""
from __future__ import annotations

import json
import logging

import pytest

from app.core.rag import fusion_trace as ft
from app.core.rag import retriever
from app.core.rag import vector_store as vs
from app.core.rag.inject import _emit_retrieval_trace
from app.core.rag.vector_store import Chunk
from tests.test_fused_rank_is_chunk_score import (
    PID, QUERY, _chunk, _ids, _install, _vector_leg,
)


@pytest.fixture
def turn():
    token = vs.start_turn_memo()
    try:
        yield
    finally:
        vs.end_turn_memo(token)


def _as_lexical(c: Chunk, score: float) -> Chunk:
    return Chunk(**{**c.__dict__, "score": score})


def _legs(n_both: int, *, lexical_only_first: bool = True):
    """50 vector hits; the lexical leg returns one lexical-only chunk ``x``
    at rank 1 followed by the first ``n_both`` vector hits."""
    semantic = _vector_leg(50)
    x = _chunk("x", 0.09, kind="lexical", n=101)
    agree = [_as_lexical(c, round(0.08 - i * 0.001, 6))
             for i, c in enumerate(semantic[:n_both])]
    lexical = [x] + agree if lexical_only_first else agree + [x]
    return semantic, lexical


# ── 1. counts, and the lexical-only rank-1 hit buried under agreement ──────

@pytest.mark.parametrize("n_both", [10, 15])
def test_lexical_only_rank_one_is_counted_and_followed(
        monkeypatch, tmp_path, turn, n_both):
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    rec = vs.fused_candidates_record(PID)

    assert rec["n_candidates"] == 51
    assert rec["n_both_legs"] == n_both
    assert rec["n_lexical_only"] == 1
    assert rec["n_vector_only"] == 50 - n_both
    lex = rec["best_lexical_only"]
    # Every agreeing candidate outranks the lexical leg's rank-1 hit.
    assert lex["leg_rank"] == 1
    assert lex["fused_rank"] == n_both + 1
    assert lex["in_fused_list"] is True
    vec = rec["best_vector_only"]
    assert vec["leg_rank"] == n_both + 1
    assert vec["fused_rank"] == n_both + 2
    # Its fate is stated, and agrees with the pool the retriever returned.
    assert "x" not in _ids(pooled)
    assert lex["pool_position"] is None
    assert lex["scored_position"] > len(pooled)
    assert lex["dropped_by"] == ft.BELOW_TARGET


# ── 2. legs that agree: no single-leg candidate, the pool is unchanged ─────

@pytest.mark.parametrize("count", [20, 7])
def test_agreeing_legs_trace_no_single_leg_and_same_pool(
        monkeypatch, tmp_path, count):
    semantic = _vector_leg(count)
    lexical = [_as_lexical(c, round(0.09 - i * 0.001, 6))
               for i, c in enumerate(semantic)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    outside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    token = vs.start_turn_memo()
    try:
        inside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
        rec = vs.fused_candidates_record(PID)
    finally:
        vs.end_turn_memo(token)

    assert _ids(inside) == _ids(outside)
    assert [c.score for c in inside] == [c.score for c in outside]
    assert rec["n_both_legs"] == count == rec["n_candidates"]
    assert rec["n_lexical_only"] == rec["n_vector_only"] == 0
    assert rec["best_lexical_only"] is None
    assert rec["best_vector_only"] is None


# ── 3. tracing changes no ranking when a single-leg hit exists ─────────────

@pytest.mark.parametrize("n_both", [0, 12])
def test_trace_on_and_off_give_the_same_pool(monkeypatch, tmp_path, n_both):
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)

    outside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    token = vs.start_turn_memo()
    try:
        inside, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
        rec = vs.fused_candidates_record(PID)
    finally:
        vs.end_turn_memo(token)
    assert _ids(inside) == _ids(outside)
    assert rec["best_lexical_only"]["fused_rank"] == n_both + 1


# ── 4. a keep-loop drop is named ───────────────────────────────────────────

@pytest.mark.parametrize("n_both", [3, 8])
def test_keep_loop_drop_reason_is_recorded(monkeypatch, tmp_path, turn, n_both):
    semantic, lexical = _legs(n_both)
    # The default noise pattern matches an Office lock-file name.
    lexical[0] = Chunk(**{**lexical[0].__dict__, "doc_id": "~$lock"})
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=30)
    lex = vs.fused_candidates_record(PID)["best_lexical_only"]
    assert "x" not in _ids(pooled)
    assert lex["scored_position"] is not None
    assert lex["dropped_by"] == "noise_filename"


# ── 5. the fusion summary and the follow, on their own ─────────────────────

class _C:
    def __init__(self, cid):
        self.chunk_id = cid


@pytest.mark.parametrize("fused_k", [2, 3])
def test_best_single_leg_past_the_fused_cut_is_overfetch_cut(fused_k):
    order = [_C("b1"), _C("b2"), _C("b3"), _C("lx"), _C("vx")]
    sem = {"b1": 1, "b2": 2, "b3": 3, "vx": 4}
    bm = {"b1": 2, "b2": 3, "b3": 4, "lx": 1}
    summary = ft.leg_summary(order, sem, bm, fused_k)
    assert (summary["n_both_legs"], summary["n_lexical_only"],
            summary["n_vector_only"]) == (3, 1, 1)
    out = ft.Follow(summary).finish(order[:fused_k], order[:fused_k], order[:1])
    for leg, rank in (("lexical_only", 4), ("vector_only", 5)):
        entry = out[f"best_{leg}"]
        assert entry["fused_rank"] == rank
        assert entry["in_fused_list"] is False
        assert entry["dropped_by"] == ft.OVERFETCH_CUT


@pytest.mark.parametrize("handed_has_it", [True, False])
def test_public_view_has_no_chunk_id_and_states_handed(handed_has_it):
    order = [_C("b1"), _C("lx")]
    summary = ft.leg_summary(order, {"b1": 1}, {"b1": 1, "lx": 1}, 10)
    rec = ft.Follow(summary).finish(order, order, order)
    handed = order if handed_has_it else order[:1]
    view = ft.public_view(rec, handed)
    lex = view["best_lexical_only"]
    assert "chunk_id" not in lex
    assert lex["pool_position"] == 2
    if handed_has_it:
        assert lex["handed_position"] == 2 and lex["dropped_by"] is None
    else:
        assert lex["handed_position"] is None
        assert lex["dropped_by"] == ft.NOT_HANDED
    assert view["best_vector_only"] is None


@pytest.mark.parametrize("reason_order", [("rerank",), ("below",)])
def test_follow_tells_rerank_cut_from_below_target(reason_order):
    order = [_C("b1"), _C("lx")]
    summary = ft.leg_summary(order, {"b1": 1}, {"b1": 1, "lx": 1}, 10)
    loop = order if reason_order == ("rerank",) else order[:1]
    out = ft.Follow(summary).finish(order, loop, order[:1])
    expected = ft.RERANK_CUT if reason_order == ("rerank",) else ft.BELOW_TARGET
    assert out["best_lexical_only"]["dropped_by"] == expected


# ── 6. the pre_injection line carries it ───────────────────────────────────

@pytest.mark.parametrize("n_both", [10, 14])
def test_pre_injection_line_carries_fused_candidates(
        monkeypatch, tmp_path, turn, caplog, n_both):
    semantic, lexical = _legs(n_both)
    _install(monkeypatch, tmp_path, semantic, lexical)
    monkeypatch.setenv("RAG_RETRIEVAL_TRACE", "1")
    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)

    with caplog.at_level(logging.INFO, logger="app.rag.retrieval_trace"):
        _emit_retrieval_trace(
            query_label="user_message", project_id=PID,
            conversation_id=None, agent_name="t", retrieved=pooled,
            handed=pooled[:5], threshold_fired=False, identifier_miss=False,
            top_score=0.8,
        )
    line = next(r.getMessage() for r in caplog.records
                if "rag_retrieval_trace" in r.getMessage())
    payload = json.loads(line.split(" ", 1)[1])
    fc = payload["fused_candidates"]
    assert fc["n_both_legs"] == n_both
    assert fc["n_lexical_only"] == 1
    assert fc["best_lexical_only"]["fused_rank"] == n_both + 1
    assert fc["best_lexical_only"]["handed_position"] is None
    assert "chunk_id" not in json.dumps(fc)


@pytest.mark.parametrize("handed", [[], [_C("b1")]])
def test_no_fusion_traces_none(handed):
    """Vector-only degrade or the lexical-only path: no fusion ran."""
    follow = ft.Follow(None)
    follow.drop(_C("b1"), "noise_filename")
    assert follow.finish(handed, handed, handed) is None
    assert ft.public_view(None, handed) is None
