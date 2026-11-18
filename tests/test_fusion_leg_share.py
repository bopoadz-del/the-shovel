"""Each leg's top chunks open the fused list, by leg rank (the window).

With k=60 a single-leg rank-1 hit scores 1/61 = 0.0164 and a chunk at rank
50 in both legs scores 2/110 = 0.0182, so agreement at any depth outranks
the best hit of one leg. ``guarantee_leg_share`` puts the union of each
leg's top ``RAG_FUSION_LEG_SHARE`` chunks (found by the other leg or not)
in the first W places (P1), never drops a chunk from that window because
the other leg found it too (P2), orders the window by fused score, then
leg rank, then id (P3), and keeps everything else in fused order (P4).

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


# ── 4. the window on its own: P1-P4 on invented lists ──────────────────────

class _C:
    def __init__(self, cid):
        self.chunk_id = cid


def _order(ids):
    return [_C(i) for i in ids]


def _ranks(ids):
    return {c: r for r, c in enumerate(ids, 1)}


def _fused(sem, bm):
    """Plain fusion of two invented legs (k=60, stable, lexical first)."""
    sr, br = _ranks(sem), _ranks(bm)
    ids = list(dict.fromkeys(bm + sem))
    score = {c: (1 / (60 + sr[c]) if c in sr else 0) + (1 / (60 + br[c]) if c in br else 0)
             for c in ids}
    return sorted(ids, key=lambda c: -score[c]), sr, br, score


def _run(sem, bm, share):
    order, sr, br, _ = _fused(sem, bm)
    return [c.chunk_id for c in guarantee_leg_share(_order(order), sr, br, share)]


def _window(sem, bm, share):
    return set(sem[:share]) | set(bm[:share])


# Two shapes. A: the vector leg's first chunk ``top`` is found by the lexical
# leg only deep down, under chunks both legs rank moderately. B: each leg
# ranks first the chunk the other ranks second, plus single-leg tails.
_SHAPE_A = (["top", "s2", "s3", "m1", "m2", "m3"],
            ["x1", "m1", "m2", "m3"] + [f"f{i}" for i in range(40)] + ["top"])
_SHAPE_B = (["a", "b", "p", "q", "r"], ["b", "a", "u", "q", "w"])
SHAPES = [_SHAPE_A, _SHAPE_B]


@pytest.mark.parametrize("sem, bm", SHAPES)
@pytest.mark.parametrize("share", [1, 2])
def test_p1_each_legs_top_chunks_fill_the_first_w_places(sem, bm, share):
    out = _run(sem, bm, share)
    window = _window(sem, bm, share)
    assert set(out[:len(window)]) == window, out


@pytest.mark.parametrize("sem, bm", SHAPES)
@pytest.mark.parametrize("share", [1, 2])
def test_p2_the_other_leg_finding_a_chunk_never_takes_it_out(sem, bm, share):
    for mine, other, mine_is_sem in ((sem, bm, True), (bm, sem, False)):
        for x in mine[:share]:
            if x not in other:
                continue
            dropped = [c for c in other if c != x]
            alone = _run(sem, dropped, share) if mine_is_sem else _run(dropped, bm, share)
            alone_window = _window(sem, dropped, share) if mine_is_sem else _window(dropped, bm, share)
            full = _run(sem, bm, share)
            assert x in full[:len(_window(sem, bm, share))], (x, full)
            assert x in alone[:len(alone_window)], (x, alone)


@pytest.mark.parametrize("sem, bm, share, head", [
    # A: top is in both legs (1/61 + 1/105) so it leads; x1 is lexical rank 1.
    (*_SHAPE_A, 1, ["top", "x1"]),
    # B: a and b tie on fused score and on leg rank 1: id decides.
    (*_SHAPE_B, 1, ["a", "b"]),
])
def test_p3_window_order_is_fused_score_then_leg_rank_then_id(sem, bm, share, head):
    assert _run(sem, bm, share)[:len(head)] == head


@pytest.mark.parametrize("sem, bm", SHAPES)
@pytest.mark.parametrize("share", [1, 2])
def test_p4_outside_the_window_keeps_fused_order_and_is_a_permutation(sem, bm, share):
    order, _, _, _ = _fused(sem, bm)
    out = _run(sem, bm, share)
    window = _window(sem, bm, share)
    assert sorted(out) == sorted(order)
    assert [c for c in out if c not in window] == [c for c in order if c not in window]


@pytest.mark.parametrize("sem, bm", SHAPES)
def test_share_zero_changes_nothing(sem, bm):
    order, _, _, _ = _fused(sem, bm)
    assert _run(sem, bm, 0) == order


@pytest.mark.parametrize("rank", [0, -3])
def test_share_zero_changes_nothing_whatever_the_rank_numbering(rank):
    """Share 0 is the kill switch: it returns the fused order as given even
    when a leg map numbers from 0 or below (a caller's convention, not
    this function's), where ``rank <= share`` would otherwise admit it."""
    order = _order(["p", "q", "r"])
    sem, bm = {"r": rank, "p": 2}, {"q": 1}
    out = guarantee_leg_share(order, sem, bm, 0)
    assert [c.chunk_id for c in out] == ["p", "q", "r"]


@pytest.mark.parametrize("sem, bm", SHAPES)
def test_a_window_chunk_never_needs_the_other_leg_to_lose_it(sem, bm):
    """Shape A's ``top``: vector #1, lexical #45. Plain fusion puts the
    moderate both-leg chunks above it; the window puts it first."""
    plain, _, _, _ = _fused(sem, bm)
    out = _run(sem, bm, 2)
    first = sem[0]
    assert out.index(first) < len(_window(sem, bm, 2))
    assert out.index(first) <= plain.index(first)


@pytest.mark.parametrize("raw, expected", [
    ("", vs.FUSION_LEG_SHARE_DEFAULT), ("3", 3), ("-1", 0), ("x", 2),
])
def test_fusion_leg_share_setting(monkeypatch, raw, expected):
    monkeypatch.setenv("RAG_FUSION_LEG_SHARE", raw)
    assert vs.fusion_leg_share() == expected


# ── 5. through the real search path ────────────────────────────────────────

@pytest.mark.parametrize("n_both", [10, 20])
def test_vector_rank_one_found_deep_by_lexical_is_pooled_first(monkeypatch, tmp_path, n_both):
    """Real ``VectorStore.search`` / ``_rrf_combine`` / ``retrieve_with_filter``."""
    semantic = _vector_leg(50)
    top = semantic[0]
    # The lexical leg: two lexical-only hits, then vector hits 5..n_both+4
    # (so v1..v4 are vector-only), then v0 only at the very end.
    lexical = [_chunk(f"l{i}", round(0.09 - i * 0.001, 6), kind="lexical", n=200 + i)
               for i in range(2)]
    lexical += [_as_lexical(c, round(0.08 - i * 0.001, 6))
                for i, c in enumerate(semantic[5:n_both + 5])]
    lexical += [_as_lexical(top, 0.001)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    ids = _ids(pooled)
    # Window = {v0, v1} | {l0, l1}; v0 is in both legs, so it leads.
    assert ids[0] == "v0", ids
    assert set(ids[:4]) == {"v0", "v1", "l0", "l1"}, ids
