"""The bounded BM25 leg returns exactly what ranking every match returns.

``VectorStore._bm25_bounded`` ranks only the chunks a GIN lookup admits and
accepts the result only when the k-th rank proves no other chunk could beat
it. These tests pin the facts that proof rests on, then compare the store's
results with the score computed in Python for every match on a seeded corpus,
row for row.
"""
from __future__ import annotations

import itertools
import math
import os
import random

import pytest
from sqlalchemy import event, text

from app.core.rag import vector_store as vs

_PG = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")
pg_only = pytest.mark.skipif(not _PG, reason="text_search / GIN are PostgreSQL features")


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
    weights = {lx: rng.random() for lx in lexemes}
    for t in range(1, n + 1):
        tsq = vs.bm25_candidate_tsquery(lexemes, t, weights)
        assert tsq is not None
        assert tsq.count("|") + 1 <= vs._BM25_MAX_CONJUNCTIONS
        subsets = (
            itertools.combinations(lexemes, t) if n <= 8
            else (rng.sample(lexemes, t) for _ in range(300))
        )
        for subset in subsets:
            assert _dnf_admits(tsq, set(subset)), (n, t, subset)


@pytest.mark.parametrize("weights", [
    {"a": 3.0, "b": 2.0, "c": 1.0, "d": 0.5},
    {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
])
def test_bound_is_the_t_minus_one_heaviest_after_the_rare(weights):
    lex = sorted(weights)
    scale = vs.lexical_score_scale(weights)
    order = vs.rarest_first(lex, weights)
    # Admitting the rarest alone, a chunk left out holds at most t-1 of the rest.
    assert vs.bm25_excluded_bound(lex, weights, 3, 1) == pytest.approx(
        (weights[order[1]] + weights[order[2]]) * scale
    )
    assert vs.bm25_excluded_bound(lex, weights, 1, 0) == 0.0
    assert not vs.bm25_rank_bound_holds(0.0, 1e-6)


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


def _tie_groups(items, scores):
    """Chunk ids grouped by score (to 9 significant digits), the last group
    dropped: the LIMIT may cut a run of equal scores at a different chunk."""
    ids = [getattr(x, "chunk_id", x) for x in items]
    groups = {}
    for cid, sc in zip(ids, scores):
        groups.setdefault(float(f"{sc:.9g}"), set()).add(cid)
    if scores:
        groups.pop(float(f"{scores[-1]:.9g}"), None)
    return groups


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

    # The reference ranks EVERY match in Python from each chunk's lexemes,
    # with the score's definition written out independently of the store's SQL.
    all_sql = text(
        f"SELECT c.chunk_id, tsvector_to_array(c.text_search) AS lexemes, "
        f"websearch_to_tsquery('english', :q)::text AS tsq "
        f"FROM {table} c, websearch_to_tsquery('english', :q) AS q "
        "WHERE c.text_search @@ q AND c.project_id = :p"
    )
    store._lexeme_freq_cache = None  # weights from the statistics just analysed

    def _want(project, q, k):
        with engine.connect() as conn:
            rows = conn.execute(all_sql, {
                "q": vs._sanitize_websearch_query(q), "p": project,
            }).all()
            freq = store._lexeme_frequencies(conn)
        if not rows:
            return []
        lex = vs.tsquery_lexemes(rows[0].tsq)
        w = vs.lexeme_weights(lex, freq)
        total = sum(w.values())
        scored = []
        for r in rows:
            held = set(r.lexemes)
            s_ = sum(w[x] for x in lex if x in held) * 0.1 / total
            scored.append((s_ / (1 + math.log(max(len(held), 1))), r.chunk_id))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[:k]

    event.listen(engine, "before_cursor_execute", _seen)
    try:
        for project in ("bm-a", "bm-b"):
            for q in queries:
                for k in (5, 50):
                    searches += 1
                    got = store.bm25_search(project, q, k)
                    want = _want(project, q, k)
                    assert [c.score for c in got] == pytest.approx(
                        [x[0] for x in want], rel=1e-9, abs=1e-15,
                    ), (project, q, k)
                    # Same chunks; within a run of equal scores the database's
                    # and Python's last-bit rounding may order ties apart.
                    assert _tie_groups(got, [c.score for c in got]) == _tie_groups(
                        [x[1] for x in want], [x[0] for x in want],
                    ), (project, q, k)
    finally:
        event.remove(engine, "before_cursor_execute", _seen)
    # The comparison above is only worth something if the bounded path ran
    # and answered many searches without the full scan. (A small project or
    # k=50 over few matches legitimately falls back to it.)
    assert len(bounded_used) > 20
    assert searches - len(full_scans) >= 40, (len(full_scans), searches)
