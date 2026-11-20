"""S-R7c: each leg's best single-leg hits are represented in the pool.

With k=60 a single-leg rank-1 hit scores 1/61 = 0.0164 and a chunk at rank
50 in both legs scores 2/110 = 0.0182, so agreement at any depth outranks
the best hit of one leg. ``guarantee_leg_share`` places the best
``RAG_FUSION_LEG_SHARE`` single-leg candidates of each leg at their own leg
rank; everything else keeps its fused order, and agreeing legs give the
same order as plain fusion.

Every input is constructed here (invented ids, texts, scores). The real
``VectorStore.search`` / ``_rrf_combine`` / ``retrieve_with_filter`` run;
only the legs and side lookups are stubbed (the S-R7 fixture). The two
pool tests fail on main: the single-leg rank-1 hit is fused at rank
n_both + 1 and falls below the k=10 pool.
"""
from __future__ import annotations

import pytest

from app.core.rag import retriever
from app.core.rag import vector_store as vs
from app.core.rag.vector_store import Chunk, guarantee_leg_share
from tests.test_fused_rank_is_chunk_score import (
    PID, QUERY, _chunk, _ids, _install, _vector_leg,
)


def _as_lexical(c: Chunk, score: float) -> Chunk:
    return Chunk(**{**c.__dict__, "score": score})


# ── 1. a lexical-only rank-1 hit is pooled despite 10+ agreeing hits ───────

@pytest.mark.parametrize("n_both", [10, 20])
def test_lexical_only_rank_one_is_pooled(monkeypatch, tmp_path, n_both):
    semantic = _vector_leg(50)
    x = _chunk("x", 0.09, kind="lexical", n=101)
    lexical = [x] + [_as_lexical(c, round(0.08 - i * 0.001, 6))
                     for i, c in enumerate(semantic[:n_both])]
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    assert "x" in _ids(pooled), _ids(pooled)


# ── 2. the same for a vector-only rank-1 hit ───────────────────────────────

@pytest.mark.parametrize("n_both", [10, 20])
def test_vector_only_rank_one_is_pooled(monkeypatch, tmp_path, n_both):
    semantic = _vector_leg(50)
    # The lexical leg returns vector hits 2..n_both+1 and a tail of its own;
    # it never returns the vector leg's rank-1 hit v0.
    lexical = [_as_lexical(c, round(0.08 - i * 0.001, 6))
               for i, c in enumerate(semantic[1:n_both + 1])]
    lexical += [_chunk(f"l{i}", round(0.03 - i * 0.001, 6), kind="lexical",
                       n=300 + i) for i in range(5)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    assert "v0" in _ids(pooled), _ids(pooled)


# ── 3. legs that agree give the same pool as plain fusion ──────────────────

@pytest.mark.parametrize("count", [20, 7])
def test_agreeing_legs_pool_unchanged(monkeypatch, tmp_path, count):
    semantic = _vector_leg(count)
    lexical = [_as_lexical(c, round(0.09 - i * 0.001, 6))
               for i, c in enumerate(semantic)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "0")
    plain, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", "2")
    shared, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    assert _ids(shared) == _ids(plain)
    assert [c.score for c in shared] == [c.score for c in plain]


# ── 4. the reorder on its own ──────────────────────────────────────────────

class _C:
    def __init__(self, cid):
        self.chunk_id = cid


def _order(ids):
    return [_C(i) for i in ids]


@pytest.mark.parametrize("share, expected", [
    # b* in both legs; lx lexical rank 1, lz lexical rank 2; vx vector rank 1.
    (0, ["b1", "b2", "b3", "vx", "lx", "lz"]),
    (1, ["vx", "lx", "b1", "b2", "b3", "lz"]),
    (2, ["vx", "lx", "lz", "b1", "b2", "b3"]),
])
def test_share_places_single_leg_hits_by_leg_rank(share, expected):
    order = _order(["b1", "b2", "b3", "vx", "lx", "lz"])
    sem = {"vx": 1, "b1": 2, "b2": 3, "b3": 4}
    bm = {"lx": 1, "lz": 2, "b1": 3, "b2": 4, "b3": 5}
    out = guarantee_leg_share(order, sem, bm, share)
    assert [c.chunk_id for c in out] == expected


@pytest.mark.parametrize("share", [1, 3])
def test_a_single_leg_hit_fused_higher_is_never_moved_down(share):
    # lx is lexical rank 4 but already fused at position 1.
    order = _order(["lx", "b1", "b2"])
    out = guarantee_leg_share(order, {"b1": 1, "b2": 2},
                              {"b1": 1, "b2": 2, "lx": 4}, share)
    assert [c.chunk_id for c in out] == ["lx", "b1", "b2"]


@pytest.mark.parametrize("raw, expected", [
    ("", vs.FUSION_LEG_SHARE_DEFAULT), ("3", 3), ("-1", 0), ("x", 2),
])
def test_fusion_leg_share_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", raw)
    assert vs.fusion_leg_share() == expected
