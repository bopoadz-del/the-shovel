"""Whole-document reads return the same rows, without reading the embedding.

``chunks_for_docs`` used to load every chunk of the asked documents as full
ORM rows -- embedding included, ~1.5 KB each, stored out of line -- and cut
the per-document window in Python. Recall reads the same volumes several
times a turn. It now selects only the returned columns and cuts the window
in SQL. These tests hold it to the old slicing, row for row, for every
window shape the recall code uses, with a retired document in the mix.
"""
from __future__ import annotations

import pytest
from sqlalchemy import event

PID = "cr-proj"
DOCS = {"cr-a": 37, "cr-b": 5, "cr-c": 12, "cr-hidden": 9}


@pytest.fixture()
def store():
    from app.core.db import SessionLocal
    from app.core.models import Document, Project
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    emb = get_embedder()
    st = get_store(dim=emb.dim)
    for did, n in DOCS.items():
        texts = [f"{did} row {i} text" for i in range(n)]
        st.upsert_chunks(PID, did, texts, emb.encode(texts))
    with SessionLocal() as session:
        if session.get(Project, PID) is None:
            session.add(Project(
                id=PID, name=PID, client=None, status="active",
                aconex_connected=False, user_id="system",
                created_at="2026-01-01T00:00:00Z",
            ))
            session.flush()
        for did in DOCS:
            doc = session.get(Document, did)
            if doc is None:
                doc = Document(
                    id=did, project_id=PID, original_name=f"{did}.pdf",
                    doc_type="document", doc_role="other", size=0,
                    uploaded_at="2026-01-01T00:00:00Z",
                )
                session.add(doc)
            doc.retrieval_visible = did != "cr-hidden"
        session.commit()
    st._visibility_ready = None
    return st


def _old_slicing(ids, per, off, from_end, all_rows):
    """The pre-change Python slicing over all rows ordered (doc_id, index)."""
    unique = list(dict.fromkeys(d for d in ids if d))
    rows = [
        (did, i) for did in sorted(unique) if did in DOCS and did != "cr-hidden"
        for i in range(DOCS[did])
    ]
    if all_rows:
        return rows
    if from_end:
        out = []
        for did in unique:
            group = [r for r in rows if r[0] == did]
            end = len(group) - off
            if end <= 0:
                continue
            out.extend(group[max(0, end - per):end])
        return out
    out, skipped, taken = [], {}, {}
    for r in rows:
        if skipped.get(r[0], 0) < off:
            skipped[r[0]] = skipped.get(r[0], 0) + 1
            continue
        if taken.get(r[0], 0) >= per:
            continue
        taken[r[0]] = taken.get(r[0], 0) + 1
        out.append(r)
    return out


@pytest.mark.parametrize("from_end", [False, True])
@pytest.mark.parametrize("per", [1, 3, 12, 400, 1_000_000])
@pytest.mark.parametrize("off", [0, 2, 5, 40])
def test_windows_match_the_old_slicing(store, per, off, from_end):
    ids = ["cr-c", "cr-a", "cr-hidden", "cr-missing", "cr-a", "cr-b"]
    got = store.chunks_for_docs(PID, ids, k_per_doc=per, offset=off, from_end=from_end)
    assert [(c.doc_id, c.chunk_index) for c in got] == _old_slicing(
        ids, per, off, from_end, all_rows=False,
    )
    for c in got:
        assert c.text == f"{c.doc_id} row {c.chunk_index} text"
        assert c.score == 0.0


def test_all_rows_matches_the_old_read(store):
    ids = ["cr-b", "cr-hidden", "cr-a"]
    got = store.chunks_for_docs(PID, ids, all_rows=True)
    assert [(c.doc_id, c.chunk_index) for c in got] == _old_slicing(
        ids, 0, 0, False, all_rows=True,
    )


def test_following_rows_skip_only_the_retired_document(store):
    got = store.chunks_following(PID, [("cr-a", 3), ("cr-hidden", 1), ("cr-b", 4)], n=2)
    assert [(c.doc_id, c.chunk_index) for c in got] == [("cr-a", 4), ("cr-a", 5)]


def test_chunk_reads_never_select_the_embedding(store):
    from sqlalchemy.engine import Engine

    # Every engine: a module that reloads app.core.db leaves the store on an
    # engine object this test could not otherwise name.
    engine = Engine
    seen = []

    def _rec(conn, cursor, statement, parameters, context, executemany):
        if store._table_name in statement:
            seen.append(statement)

    event.listen(engine, "before_cursor_execute", _rec)
    try:
        store.chunks_for_docs(PID, ["cr-a"], all_rows=True)
        store.chunks_for_docs(PID, ["cr-a"], k_per_doc=4, from_end=True)
        store.chunks_following(PID, [("cr-a", 1)])
    finally:
        event.remove(engine, "before_cursor_execute", _rec)
    assert len(seen) == 3
    for statement in seen:
        assert "embedding" not in statement, statement
