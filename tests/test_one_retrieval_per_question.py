"""One question retrieves once: identical searches inside a turn are answered
from the turn's memo, and the hidden-documents filter is an index lookup.

Live 2026-10-07: one question issued BM25 eight times (raw question and its
stripped variant reduce to the same tsquery; the model's search tool repeated
the turn's retrieval), and every BM25 call scanned the whole documents table
for the hidden-document filter. Synthetic corpus.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from app.core.rag import vector_store as vs

_ON_POSTGRES = os.getenv("PYTEST_USE_POSTGRES", "").strip().lower() in ("1", "true", "yes")
PID = "synthetic-memo-project"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    from app.core.rag import embeddings as emb
    emb.reset_embedder_cache()
    vs.reset_store_cache()
    e = emb.Embedder(model_name="fake")
    st = vs.get_store(dim=e.dim)
    texts = [f"Clause {i}: the contractor shall give notice of suspension {i} days before work stops."
             for i in range(30)]
    st.upsert_chunks(PID, "doc-memo", texts, e.encode(texts))
    yield st, e
    emb.reset_embedder_cache()
    vs.reset_store_cache()


def _count_statements(st, fn):
    from sqlalchemy import event
    seen = []
    engine = st._session_factory()().get_bind()

    def rec(conn, cursor, statement, params, context, many):
        seen.append(statement)
    event.listen(engine, "before_cursor_execute", rec)
    try:
        fn()
    finally:
        event.remove(engine, "before_cursor_execute", rec)
    return seen


def test_the_memo_answers_a_repeated_search_once_and_only_inside_a_turn(store):
    st, e = store
    vec = e.encode(["notice of suspension"])[0]

    def two_searches():
        a = st.search(PID, vec, k=5, query_text="What notice of suspension?")
        b = st.search(PID, vec, k=5, query_text="What notice of suspension?")
        return a, b

    outside = _count_statements(st, two_searches)
    token = vs.start_turn_memo()
    try:
        inside = _count_statements(st, two_searches)
    finally:
        vs.end_turn_memo(token)
    assert len(inside) < len(outside), (len(inside), len(outside))
    assert vs._TURN_MEMO.get() is None


def test_the_memo_hands_out_fresh_chunks(store):
    st, e = store
    vec = e.encode(["notice of suspension"])[0]
    token = vs.start_turn_memo()
    try:
        first = st.search(PID, vec, k=3, query_text="notice")
        for c in first:
            c.score = -99.0  # a reranker adjusting scores in place
        second = st.search(PID, vec, k=3, query_text="notice")
    finally:
        vs.end_turn_memo(token)
    assert second and all((c.score or 0) != -99.0 for c in second)


def test_memoised_runs_compute_once_per_key_and_never_outside_a_turn():
    calls = []

    def compute():
        calls.append(1)
        return ["row"]

    assert vs._memoised(("k",), compute) == ["row"]
    assert vs._memoised(("k",), compute) == ["row"]
    assert len(calls) == 2  # no turn: no memo
    token = vs.start_turn_memo()
    try:
        vs._memoised(("k",), compute)
        vs._memoised(("k",), compute)
        vs._memoised(("other",), compute)
    finally:
        vs.end_turn_memo(token)
    assert len(calls) == 4


def test_the_index_and_the_filter_use_one_condition():
    from pathlib import Path
    mig = Path(__file__).resolve().parent.parent / "alembic" / "versions" / "0025_documents_hidden_index.py"
    src = mig.read_text(encoding="utf-8")
    assert "hidden_document_predicate(postgres=True)" in src
    from app.core.ingest_status import NO_CHUNK_STATUSES
    pred = vs.hidden_document_predicate("d.")
    assert pred.startswith("(d.retrieval_visible IS FALSE OR d.ingest_status IN (")
    assert all(f"'{s}'" in pred for s in NO_CHUNK_STATUSES)
    assert "= 0" in vs.hidden_document_predicate(postgres=False)


@pytest.mark.skipif(not _ON_POSTGRES, reason="index use is a PostgreSQL planner property (test-postgres job)")
def test_the_hidden_document_filter_reads_the_index_not_the_table():
    import json
    from sqlalchemy import text
    from app.core.db import get_engine

    engine = get_engine()
    with engine.begin() as conn:
        idx = conn.execute(text(
            "SELECT indexdef FROM pg_indexes WHERE indexname = 'idx_documents_hidden_from_retrieval'"
        )).scalar()
        assert idx, "alembic 0025 did not create the index"
        conn.execute(text("SET LOCAL enable_seqscan = off"))
        plan = conn.execute(text(
            "EXPLAIN (FORMAT JSON) SELECT 1 FROM documents d WHERE "
            + vs.hidden_document_predicate("d.")
        )).scalar()
    plan = json.loads(plan) if isinstance(plan, str) else plan

    def names(n):
        yield n.get("Index Name")
        for c in n.get("Plans", []):
            yield from names(c)
    assert "idx_documents_hidden_from_retrieval" in set(names(plan[0]["Plan"]))
