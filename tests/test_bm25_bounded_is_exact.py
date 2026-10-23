"""The bounded BM25 leg returns exactly what ranking every match returns.

``VectorStore._bm25_bounded`` ranks only the chunks a GIN lookup admits and
accepts the result only when the k-th rank proves no other chunk could beat
it. These tests pin the three facts that proof rests on, then compare the
store's results with the full ``ORDER BY ts_rank`` over every match on a
seeded corpus, row for row.
"""
from __future__ import annotations

import itertools
import os
import random

import pytest
from sqlalchemy import event, text

from app.core.rag import vector_store as vs

_PG = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")
pg_only = pytest.mark.skipif(not _PG, reason="ts_rank / GIN are PostgreSQL features")


# ── the pieces of the proof (no database) ────────────────────────────────


def test_only_a_plain_or_of_lexemes_is_bounded():
    assert vs.tsquery_or_lexemes("'commenc' | 'date' | 'date'") == ["commenc", "date"]
    assert vs.tsquery_or_lexemes("'concret'") == ["concret"]
    assert vs.tsquery_or_lexemes("") == []
    assert vs.tsquery_or_lexemes(None) == []
    # A phrase (a compound token), AND, NOT, prefix or weight: full scan.
    for other in ("'foo' <-> 'bar' | 'baz'", "'a' & 'b'", "!'a' | 'b'",
                  "'a':* | 'b'", "'a':A | 'b'"):
        assert vs.tsquery_or_lexemes(other) is None, other


def _dnf_admits(tsq: str, present: set) -> bool:
    groups = [g.strip().strip("()") for g in tsq.split(" | ")]
    return any(
        all(term.strip().strip("'") in present for term in g.split(" & "))
        for g in groups
    )


@pytest.mark.parametrize("n", [2, 3, 5, 8, 12, 17])
def test_candidate_query_admits_every_chunk_with_t_lexemes(n):
    """Pigeonhole: any chunk holding t of the n lexemes satisfies the DNF."""
    lexemes = [f"lx{i}" for i in range(n)]
    rng = random.Random(n)
    freq = {lx: rng.random() for lx in lexemes}
    for t in range(1, n + 1):
        tsq = vs.bm25_candidate_tsquery(lexemes, t, freq)
        assert tsq is not None
        assert tsq.count("|") + 1 <= vs._BM25_MAX_CONJUNCTIONS
        subsets = (
            itertools.combinations(lexemes, t) if n <= 8
            else (rng.sample(lexemes, t) for _ in range(300))
        )
        for subset in subsets:
            assert _dnf_admits(tsq, set(subset)), (n, t, subset)


def test_bound_needs_the_kth_rank_to_clear_t_minus_one_terms():
    cap = vs._TS_RANK_TERM_CAP
    n = 7
    # A chunk with 3 of 7 lexemes ranks below 3 * cap / 7.
    assert vs.bm25_rank_bound_holds(3 * cap / n, n, 4)
    assert not vs.bm25_rank_bound_holds(3 * cap / n - 1e-9, n, 4)
    assert not vs.bm25_rank_bound_holds(1.0, 0, 2)


@pg_only
def test_one_lexeme_never_contributes_the_cap():
    """The per-lexeme cap holds however often the lexeme occurs."""
    from app.core.db import get_engine

    with get_engine().connect() as conn:
        for reps in (1, 2, 5, 50, 300, 2000):
            rank = conn.execute(text(
                "SELECT ts_rank(to_tsvector('english', repeat('culvert ', :r)), "
                "'culvert'::tsquery)"
            ), {"r": reps}).scalar()
            assert 0 < rank < vs._TS_RANK_TERM_CAP, (reps, rank)


# ── the store against the full ranking (PostgreSQL) ───────────────────────

_STOP = "the of and to in for shall be is with on by as or".split()
_DOMAIN = (
    "contractor works engineer contract date period clause specification "
    "drawing concrete steel cover drainage pipe trench excavation backfill "
    "pump station rate quantity bill item amount total particulars data "
    "conditions completion defects notification retention payment "
    "certificate variation commencement access site cement aggregate "
    "reinforcement formwork curing membrane waterproofing slab beam column "
    "foundation pile manhole chamber gully kerb asphalt subgrade compaction"
).split()
_WORDS = _STOP + [f"zc{i:03d}" for i in range(120)] + _DOMAIN + [
    f"zq{i:04x}" for i in range(2000)
]


def _seed(engine, table: str, project: str, docs: int, per: int, seed: int) -> None:
    rng = random.Random(seed)
    with engine.begin() as conn:
        rows = []
        for d in range(docs):
            for i in range(per):
                words = [
                    _WORDS[min(len(_WORDS) - 1, int(len(_WORDS) ** rng.random()) - 1)]
                    for _ in range(60 + i % 7)
                ]
                rows.append({
                    "cid": f"{project}:d{d}:{i}", "p": project, "d": f"{project}-d{d}",
                    "i": i, "t": " ".join(words),
                })
        conn.execute(text(f"""
            INSERT INTO {table} (chunk_id, project_id, doc_id, chunk_index, text,
                embedding, created_at, embedding_model, embedding_dim,
                embedding_normalized)
            VALUES (:cid, :p, :d, :i, :t,
                (SELECT array_agg(0.01::real) FROM generate_series(1, 256))::vector,
                '2026-01-01T00:00:00Z', 'fake', 256, true)
        """), rows)


@pg_only
def test_bounded_bm25_matches_ranking_every_match():
    from app.core.db import get_engine
    from app.core.rag.embeddings import get_embedder

    engine = get_engine()
    store = vs.get_store(dim=get_embedder().dim)
    table = store._table_name
    _seed(engine, table, "bm-a", docs=30, per=60, seed=1)
    _seed(engine, table, "bm-b", docs=10, per=40, seed=2)
    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT").execute(
            text(f"ANALYZE {table}")
        )

    rng = random.Random(7)
    queries = [
        "What is the defects notification period stated in the contract data",
        "concrete cover for the manhole chamber",
        "the of and",  # stopwords only
        "pile pile pile foundation",  # duplicate terms
        "zq0001 zq0002",  # rare terms, few matches
    ] + [
        " ".join(rng.sample(_DOMAIN + _STOP + _WORDS[:400], rng.randint(2, 16)))
        for _ in range(40)
    ]

    bounded_used = []
    full_scans = []
    searches = 0

    def _seen(conn, cursor, statement, parameters, context, executemany):
        if "CAST(%(cand)s AS tsquery)" in statement:
            bounded_used.append(statement)
        elif "c.text_search @@ q AND c.project_id = %(project_id)s" in statement:
            full_scans.append(statement)

    full_sql = text(
        f"SELECT c.chunk_id, ts_rank(c.text_search, q) AS rank "
        f"FROM {table} c, websearch_to_tsquery('english', :q) AS q "
        "WHERE c.text_search @@ q AND c.project_id = :p "
        "ORDER BY rank DESC, c.chunk_id LIMIT :k"
    )
    event.listen(engine, "before_cursor_execute", _seen)
    try:
        for project in ("bm-a", "bm-b"):
            for q in queries:
                for k in (5, 50):
                    searches += 1
                    got = store.bm25_search(project, q, k)
                    with engine.connect() as conn:
                        want = conn.execute(full_sql, {
                            "q": vs._sanitize_websearch_query(q), "p": project, "k": k,
                        }).all()
                    assert [(c.chunk_id, c.score) for c in got] == [
                        (r.chunk_id, float(r.rank)) for r in want
                    ], (project, q, k)
    finally:
        event.remove(engine, "before_cursor_execute", _seen)
    # The comparison above is only worth something if the bounded path ran
    # and answered many searches without the full scan. (A small project or
    # k=50 over few matches legitimately falls back to it.)
    assert len(bounded_used) > 20
    assert searches - len(full_scans) >= 40, (len(full_scans), searches)
