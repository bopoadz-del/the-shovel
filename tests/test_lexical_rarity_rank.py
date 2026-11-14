"""The lexical leg ranks by the rarity of the matched terms, mildly length-normed.

Score = sum of idf over the query lexemes a chunk holds, divided by
1 + ln(distinct lexemes in the chunk); idf = ln(1 + 1/f) from the store's own
statistics, an unlisted lexeme taking the least listed frequency. The bounded
top-k stays exact: these pin the bound on brute-forced corpora, then rank
real chunks on PostgreSQL. Every corpus is built here; none reads project data.
"""
from __future__ import annotations

import itertools
import math
import os
import random
import time
import uuid

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

import pytest

from app.core.rag import vector_store as vs

_PG = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")
pg_only = pytest.mark.skipif(not _PG, reason="the PostgreSQL lexical score")


# ── the weights (no database) ───────────────────────────────────────────────


@pytest.mark.parametrize("freq, rare, listed", [
    ({"work": 0.6, "site": 0.3}, "culvert", "work"),
    ({"concret": 0.2, "cover": 0.05, "pump": 0.01}, "headwal", "concret"),
])
def test_an_unlisted_lexeme_takes_the_least_listed_frequency(freq, rare, listed):
    w = vs.lexeme_weights([rare, listed], freq)
    floor = min(freq.values())
    assert w[rare] == pytest.approx(math.log(1 + 1 / floor))
    assert w[listed] == pytest.approx(math.log(1 + 1 / freq[listed]))
    assert w[rare] > w[listed]


@pytest.mark.parametrize("freq", [{}, None])
def test_without_statistics_every_lexeme_weighs_the_same(freq):
    w = vs.lexeme_weights(["a", "b", "c"], freq)
    assert set(w.values()) == {1.0}
    assert vs.lexical_score_scale(w) == pytest.approx(vs._LEXICAL_SCORE_SCALE / 3)


@pytest.mark.parametrize("tsq, want", [
    ("'2.5' <-> 'to' | '3m'", ["2.5", "3m", "to"]),
    ("'o''brien' | 'pile' | 'pile'", ["o'brien", "pile"]),
])
def test_every_lexeme_of_a_tsquery_is_scored(tsq, want):
    assert vs.tsquery_lexemes(tsq) == want


# ── the bound, brute-forced (no database) ───────────────────────────────────


def _score(held, chunk_size, weights):
    """The score the SQL computes, for a chunk holding ``held`` query lexemes."""
    s = sum(weights[x] for x in held) * vs.lexical_score_scale(weights)
    return s / (1 + math.log(max(chunk_size, 1)))


def _admits(tsq, present):
    groups = [g.strip().strip("()") for g in tsq.split(" | ")]
    return any(
        all(term.strip().strip("'") in present for term in g.split(" & "))
        for g in groups
    )


@pytest.mark.parametrize("n, seed", [(4, 1), (7, 2), (11, 3)])
def test_no_chunk_left_out_scores_above_the_bound(n, seed):
    """Every chunk a candidate query leaves out scores at most the bound."""
    rng = random.Random(seed)
    lexemes = [f"lx{i}" for i in range(n)]
    weights = vs.lexeme_weights(lexemes, {x: rng.uniform(0.001, 0.9) for x in lexemes})
    for rare in range(n):
        for t in range(1, n + 1):
            tsq = vs.bm25_candidate_tsquery(lexemes, t, weights, rare)
            if tsq is None:
                continue
            bound = vs.bm25_excluded_bound(lexemes, weights, t, rare)
            for size in range(1, n + 1):
                for held in itertools.combinations(lexemes, size):
                    if _admits(tsq, set(held)):
                        continue
                    # The smallest chunk that holds them scores the most.
                    assert _score(held, size, weights) <= bound + 1e-12, (rare, t, held)


@pytest.mark.parametrize("n, seed", [(5, 4), (9, 5)])
def test_the_rare_lexemes_admit_alone(n, seed):
    rng = random.Random(seed)
    lexemes = [f"lx{i}" for i in range(n)]
    weights = vs.lexeme_weights(lexemes, {x: rng.uniform(0.001, 0.9) for x in lexemes})
    order = vs.rarest_first(lexemes, weights)
    for rare in range(1, n):
        tsq = vs.bm25_candidate_tsquery(lexemes, n, weights, rare)
        for x in order[:rare]:
            assert _admits(tsq, {x}), (rare, x)
        assert not _admits(tsq, {order[rare]})


@pytest.mark.parametrize("kth, holds", [(0.05, True), (0.05 * (1 - 1e-6), False)])
def test_the_bound_needs_the_kth_score_to_reach_it(kth, holds):
    assert vs.bm25_rank_bound_holds(kth, 0.05 / (1 + 1e-9)) is holds


# ── ranking real chunks (PostgreSQL) ────────────────────────────────────────


def _store_with(monkeypatch, chunks, freq):
    from app.core.rag.embeddings import get_embedder, reset_embedder_cache

    reset_embedder_cache()
    embedder = get_embedder()
    store = vs.get_store(dim=embedder.dim)
    project = f"lexr-{uuid.uuid4().hex[:10]}"
    for doc_id, body in chunks:
        store.upsert_chunks(project, doc_id, [body], embedder.encode([body]))
    # The statistics the weights come from, fixed for the test (and restored
    # after it): the score is under test here, the floor rule is pinned above.
    monkeypatch.setattr(store, "_lexeme_freq_cache", (time.monotonic(), dict(freq)), raising=False)
    return store, project


_FILLER = (
    "the contractor shall carry out the works in accordance with the "
    "contract and the instructions of the engineer and shall provide all "
    "labour plant and materials required for the completion of the works "
)


@pg_only
@pytest.mark.parametrize("query, short, long_, freq", [
    (
        "culvert headwall works contractor engineer site",
        "Culvert headwall: precast units.",
        "Works by the contractor on site as the engineer directs. " + _FILLER * 6,
        # A real statistics list runs down to rare lexemes ('kerb' here).
        {"work": 0.6, "contractor": 0.5, "engin": 0.4, "site": 0.4, "kerb": 0.002},
    ),
    (
        "pile cutoff level drawings general requirements",
        "Pile cutoff level 2 m below.",
        "General requirements and drawings for the level of the works. " + _FILLER * 6,
        {"draw": 0.5, "general": 0.4, "requir": 0.5, "level": 0.3, "gulli": 0.001},
    ),
])
def test_a_short_chunk_with_the_rare_terms_outranks_a_long_common_one(
    monkeypatch, query, short, long_, freq,
):
    store, project = _store_with(
        monkeypatch, [("d-short", short), ("d-long", long_)], freq,
    )
    got = [c.doc_id for c in store.bm25_search(project, query, 5)]
    assert got[:2] == ["d-short", "d-long"], got


@pg_only
@pytest.mark.parametrize("query, a, b", [
    ("pump station commissioning", "Pump station.", "Pump station commissioning."),
    ("manhole cover grade", "Manhole.", "Manhole cover grade D400 frames."),
])
def test_holding_more_of_the_query_still_ranks_higher(monkeypatch, query, a, b):
    """Rarity does not undo coverage: the chunk with more terms wins."""
    store, project = _store_with(monkeypatch, [("d-a", a), ("d-b", b)], {})
    got = [c.doc_id for c in store.bm25_search(project, query, 5)]
    assert got[:2] == ["d-b", "d-a"], got


@pg_only
@pytest.mark.parametrize("query", [
    "the contractor shall carry out the works",
    "materials required for completion",
])
def test_an_all_common_question_still_returns_its_chunks(monkeypatch, query):
    """No drop: every chunk the tsquery matches can still be returned."""
    bodies = [("d1", _FILLER), ("d2", "Works schedule."), ("d3", "Unrelated pump.")]
    store, project = _store_with(monkeypatch, bodies, {"work": 0.9, "contractor": 0.9})
    got = {c.doc_id for c in store.bm25_search(project, query, 10)}
    assert "d1" in got and "d3" not in got
    scores = [c.score for c in store.bm25_search(project, query, 10)]
    assert all(0 < s <= vs._LEXICAL_SCORE_SCALE for s in scores), scores
