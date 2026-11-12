"""S-A5: the bill-rows pool ranks bill documents by priced-row density,
prefers documents the ingest typed as a bill, and can rank an item inside
those documents alone (``bm25_search(..., doc_ids=...)``).

Every test builds its own corpus and runs each decision on two inputs: a
project with a typed bill and one without; a document filter that matches
and one that matches nothing; a dense bill against a sparse one, swapped.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from types import SimpleNamespace

import pytest

from app.core.rag import vector_store as vs
from app.core.rag.evidence_actions import (
    _bill_rows_retrieve,
    bill_rows_action,
    rank_bill_documents,
)
from app.core.rag.evidence_budget import cap_chunks
from app.core.rag.retriever import priced_row_count


def _c(doc_id, text, score=1.0, chunk_id=None):
    return SimpleNamespace(
        doc_id=doc_id, text=text, score=score, chunk_id=chunk_id or uuid.uuid4().hex,
    )


def _priced(doc_id, desc, qty, rate, score=1.0):
    """A tab-separated priced row: qty x rate = amount."""
    return _c(doc_id, f"{desc}\tm2\t{qty}\t{rate:.2f}\t{qty * rate:.2f}", score)


def _unpriced(doc_id, desc, score=1.0):
    return _c(doc_id, f"{desc} forecast 2026 delivery by phase, see note 4.", score)


# ── density ranking ──────────────────────────────────────────────────────

def _dense_and_sparse(dense, sparse):
    """``dense``: 3 of 3 chunks priced; ``sparse``: 1 of 4 priced."""
    return [
        _unpriced(sparse, "Membrane"), _unpriced(sparse, "Membrane"),
        _unpriced(sparse, "Membrane"), _priced(sparse, "Membrane", 10, 4.0),
        _priced(dense, "Membrane", 120, 6.5), _priced(dense, "Kerb", 30, 12.0),
        _priced(dense, "Sub-base", 44, 3.0),
    ]


@pytest.mark.parametrize("dense,sparse", [("doc-a", "doc-b"), ("doc-b", "doc-a")])
def test_denser_priced_document_ranks_first_whatever_its_name(dense, sparse):
    ranked = rank_bill_documents(_dense_and_sparse(dense, sparse), row_count=priced_row_count)
    assert ranked == [dense, sparse]


def test_unpriced_untyped_document_is_not_a_bill():
    pool = [_unpriced("notes", "Membrane"), _priced("bill", "Membrane", 5, 2.0)]
    assert rank_bill_documents(pool, row_count=priced_row_count) == ["bill"]


# ── typed-bill preference ────────────────────────────────────────────────

def test_typed_bill_ranks_before_a_denser_untyped_priced_document():
    # "typed" prints 1 priced row in 2 chunks (density 0.5); "sheet" 3 in 3.
    pool = [
        _priced("sheet", "Membrane", 10, 4.0), _priced("sheet", "Kerb", 3, 2.0),
        _priced("sheet", "Sub-base", 7, 3.0),
        _unpriced("typed", "Membrane"), _priced("typed", "Membrane", 120, 6.5),
    ]
    with_type = rank_bill_documents(pool, row_count=priced_row_count, typed_bills={"typed"})
    without = rank_bill_documents(pool, row_count=priced_row_count, typed_bills=())
    assert with_type == ["typed", "sheet"]
    assert without == ["sheet", "typed"]


def test_typed_bill_without_priced_rows_is_kept_but_ranks_after_priced_documents():
    pool = [_unpriced("typed", "Preamble"), _priced("sheet", "Membrane", 10, 4.0)]
    assert rank_bill_documents(pool, row_count=priced_row_count, typed_bills={"typed"}) == [
        "sheet", "typed",
    ]
    assert rank_bill_documents(pool, row_count=priced_row_count, typed_bills=()) == ["sheet"]


# ── the action: a typed bill crowded out of the global pool ──────────────

def _fake_bm25(corpus, calls):
    """Fake ``store.bm25_search``: word overlap, by score, optional doc filter."""
    def _search(project_id, query, k=50, doc_ids=None):
        calls.append(None if doc_ids is None else tuple(doc_ids))
        words = {w.lower() for w in query.split() if len(w) > 3}
        hits = [
            c for c in corpus
            if words & {w.lower().strip(".,") for w in c.text.split()}
            and (doc_ids is None or c.doc_id in set(doc_ids))
        ]
        return sorted(hits, key=lambda c: c.score, reverse=True)[:k]
    return _search


def _crowded_corpus():
    """A forecast spreadsheet with one priced line out-ranks a typed bill."""
    forecast = [_unpriced("forecast", "Geotextile membrane", 9.0 - i * 0.1) for i in range(6)]
    forecast.append(_priced("forecast", "Geotextile membrane", 4, 1.5, 8.0))
    bill = [_priced("typed-bill", "Geotextile membrane", 3120, 6.75, 0.5)]
    return forecast + bill


def test_typed_bill_crowded_out_of_the_pool_is_ranked_in_and_handed_first():
    corpus = _crowded_corpus()
    calls: list = []
    handed = _bill_rows_retrieve(
        _fake_bm25(corpus, calls), "p", 7, priced_row_count, "geotextile membrane",
        typed_bills=lambda _pid: frozenset({"typed-bill"}),
    )
    assert "21060.00" in handed
    assert handed.index("21060.00") < handed.index("6.00")
    assert calls == [None, ("typed-bill",)]


def test_project_without_a_typed_bill_keeps_the_plain_pool():
    corpus = _crowded_corpus()
    calls: list = []
    handed = _bill_rows_retrieve(
        _fake_bm25(corpus, calls), "p", 7, priced_row_count, "geotextile membrane",
        typed_bills=lambda _pid: frozenset(),
    )
    assert "21060.00" not in handed  # k=7: the bill is outside the plain pool
    assert "6.00" in handed
    assert calls == [None]


def test_a_doc_filter_that_matches_nothing_leaves_the_pool_unchanged():
    corpus = _crowded_corpus()
    plain_calls: list = []
    plain = _bill_rows_retrieve(
        _fake_bm25(corpus, plain_calls), "p", 7, priced_row_count, "geotextile membrane",
    )
    calls: list = []
    filtered = _bill_rows_retrieve(
        _fake_bm25(corpus, calls), "p", 7, priced_row_count, "geotextile membrane",
        typed_bills=lambda _pid: frozenset({"no-such-doc"}),
    )
    assert filtered == plain
    assert calls == [None, ("no-such-doc",)]


def test_typed_lookup_runs_once_per_action_and_a_failure_falls_back():
    corpus = _crowded_corpus()
    seen: list = []

    def _lookup(pid):
        seen.append(pid)
        return ["typed-bill"]

    action = bill_rows_action(
        _fake_bm25(corpus, []), "p", row_count=priced_row_count, k=7, typed_bills=_lookup,
    )
    assert "21060.00" in action("geotextile membrane")
    assert "21060.00" in action("membrane geotextile")
    assert seen == ["p"]

    def _broken(_pid):
        raise RuntimeError("documents table unavailable")

    broken = bill_rows_action(
        _fake_bm25(corpus, []), "p", row_count=priced_row_count, k=7, typed_bills=_broken,
    )
    assert "21060.00" not in broken("geotextile membrane")
    assert "6.00" in broken("geotextile membrane")


# ── cap_chunks keeps a caller's ranking when told to ─────────────────────

def test_presorted_keeps_the_given_order_and_default_sorts_by_score():
    low, high = _c("a", "alpha", score=0.1), _c("b", "beta", score=0.9)
    kept, _ = cap_chunks([low, high], total_tokens=1000, per_doc_tokens=1000, presorted=True)
    assert [c.doc_id for c in kept] == ["a", "b"]
    kept, _ = cap_chunks([low, high], total_tokens=1000, per_doc_tokens=1000)
    assert [c.doc_id for c in kept] == ["b", "a"]


# ── bm25_search doc_ids on the SQLite backend (real FTS5) ────────────────

@pytest.fixture
def sqlite_store(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    from app.core.rag.embeddings import get_embedder, reset_embedder_cache

    reset_embedder_cache()
    embedder = get_embedder()
    tmpdir = tempfile.mkdtemp(prefix="bm25_scope_")
    db_path = os.path.join(tmpdir, f"v_{uuid.uuid4().hex}.db")
    vs.reset_store_cache()
    store = vs.VectorStore(db_path=db_path, dim=embedder.dim)
    for doc_id, text in [
        ("bill", "Geotextile separator membrane m2 3120 6.75 21060.00"),
        ("spec", "Geotextile membrane shall be laid with 300 mm laps"),
        ("draw", "Geotextile membrane extent, see grid C to F"),
    ]:
        store.upsert_chunks("proj", doc_id, [text], embedder.encode([text]))
    yield store
    store.close()
    reset_embedder_cache()


def test_sqlite_doc_ids_filter_restricts_and_none_does_not(sqlite_store):
    everything = {c.doc_id for c in sqlite_store.bm25_search("proj", "geotextile membrane", k=10)}
    scoped = {
        c.doc_id
        for c in sqlite_store.bm25_search("proj", "geotextile membrane", k=10, doc_ids=["bill"])
    }
    assert everything == {"bill", "spec", "draw"}
    assert scoped == {"bill"}


@pytest.mark.parametrize("doc_ids", [[], ["no-such-doc"]])
def test_sqlite_doc_ids_filter_that_matches_nothing_returns_empty(sqlite_store, doc_ids):
    assert sqlite_store.bm25_search("proj", "geotextile membrane", k=10, doc_ids=doc_ids) == []


# ── bm25 doc scope on the PostgreSQL statement path (no database) ────────

def test_doc_scope_binds_ids_and_empty_scope_matches_nothing():
    assert vs._doc_scope(None) == ("", {})
    assert vs._doc_scope([]) == (None, {})
    sql, params = vs._doc_scope(["d2", "d1", "d2"])
    assert sql == " AND c.doc_id IN (:bm25_doc_0, :bm25_doc_1)"
    assert params == {"bm25_doc_0": "d1", "bm25_doc_1": "d2"}


class _PgSession:
    def __init__(self):
        self.full_params: list = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        if "::text" in sql and "FROM" not in sql:
            return SimpleNamespace(scalar=lambda: "'geotextil'")
        self.full_params.append((sql, dict(params)))
        return SimpleNamespace(all=lambda: [SimpleNamespace(chunk_id=params.get("bm25_doc_0"))])


def test_postgres_scope_reaches_the_statement_and_the_turn_memo_key():
    store = object.__new__(vs.VectorStore)
    store._table_name = "chunks_test"
    session = _PgSession()
    token = vs.start_turn_memo()
    try:
        a = store._bm25_postgres_rows(
            session, "p", "geotextile", 5, " AND c.doc_id IN (:bm25_doc_0)",
            {"bm25_doc_0": "bill-a"},
        )
        b = store._bm25_postgres_rows(
            session, "p", "geotextile", 5, " AND c.doc_id IN (:bm25_doc_0)",
            {"bm25_doc_0": "bill-b"},
        )
    finally:
        vs.end_turn_memo(token)
    assert [r.chunk_id for r in a] == ["bill-a"]
    assert [r.chunk_id for r in b] == ["bill-b"]
    assert len(session.full_params) == 2
    assert all("c.doc_id IN (:bm25_doc_0)" in sql for sql, _ in session.full_params)


# ── the runtime's typed-bill lookup reads documents.doc_type ─────────────

@pytest.mark.parametrize("docs,expected", [
    (
        [
            {"id": "d1", "original_name": "x.pdf", "doc_type": "boq"},
            {"id": "d2", "original_name": "y.xlsx", "doc_type": "document"},
        ],
        ["d1"],
    ),
    (
        [{"id": "d3", "original_name": "z.pdf", "doc_type": "specification"}],
        [],
    ),
])
def test_runtime_lookup_returns_only_typed_bills(monkeypatch, docs, expected):
    from app.agents import runtime
    from app.core import projects

    monkeypatch.setattr(projects, "list_document_names", lambda _pid: docs)
    assert runtime._typed_bill_doc_ids("p") == expected
