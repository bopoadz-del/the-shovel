"""R1: retrieval_visible hides a superseded extract on BOTH hybrid legs.

D1: top-5 must contain the corrected id and not the stale one.
Reversible by flipping the flag. Hard delete is never used.
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app.core.rag.embeddings import get_embedder, reset_embedder_cache
from app.core.rag.vector_store import VectorStore, reset_store_cache


PROJECT_ID = "p_vis"
STALE_ID = "5a1e0001"
LIVE_ID = "5a1e0002"
QUERY = "D1LETTER signatory token"
STALE_TEXT = "D1LETTER signatory token stale extract yours sincerely"
LIVE_TEXT = "D1LETTER signatory token corrected extract yours sincerely"


def _reload(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_embedder_cache()
    reset_store_cache()
    return projects, db_mod, users_mod


def _seed_pair(projects, users):
    users.ensure_user_exists("u1")
    projects.create_project(name="Vis", client="C", user_id="u1")
    proj = projects.list_projects("u1")[0]
    pid = proj["id"]
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        for did, name in ((STALE_ID, "stale.docx"), (LIVE_ID, "live.docx")):
            session.add(
                Document(
                    id=did,
                    project_id=pid,
                    original_name=name,
                    stored_as=name,
                    file_path=None,
                    doc_type="document",
                    doc_role="other",
                    size=139671,
                    uploaded_at="2026-01-01T00:00:00+00:00",
                    content_sha256="ab" * 32,
                    retrieval_visible=True,
                )
            )
        session.commit()
    return pid


def _index_pair(pid, embedder):
    from app.core.db import get_database_url

    url = get_database_url()
    path = url.replace("sqlite:///", "") if url.startswith("sqlite") else url
    store = VectorStore(db_path=path, dim=embedder.dim)
    store._visibility_ready = None
    for doc_id, text in ((STALE_ID, STALE_TEXT), (LIVE_ID, LIVE_TEXT)):
        vecs = embedder.encode([text])
        store.upsert_chunks(pid, doc_id, [text], vecs)
    store._visibility_ready = None
    return store


@pytest.fixture
def vis_store(monkeypatch, tmp_path):
    projects, _db, users = _reload(monkeypatch, tmp_path)
    projects.init_db()
    pid = _seed_pair(projects, users)
    embedder = get_embedder()
    store = _index_pair(pid, embedder)
    projects.supersede_document(STALE_ID, LIVE_ID)
    store._visibility_ready = None
    yield store, embedder, pid, projects
    store.close()


def _ids(chunks):
    return [c.doc_id for c in chunks]


def test_seed_hides_stale_without_deleting(vis_store):
    store, _emb, pid, projects = vis_store
    stale = projects.get_document(STALE_ID)
    live = projects.get_document(LIVE_ID)
    assert stale is not None
    assert live is not None
    assert stale["superseded_by"] == LIVE_ID
    assert stale["retrieval_visible"] is False
    assert live["retrieval_visible"] is True
    assert projects.get_document(STALE_ID)["id"] == STALE_ID


def test_vector_leg_drops_hidden_doc(vis_store):
    store, embedder, pid, _p = vis_store
    q = embedder.encode([QUERY])[0]
    hits = store._semantic_search(pid, q, k=5)
    ids = _ids(hits)
    assert LIVE_ID in ids
    assert STALE_ID not in ids


def test_bm25_leg_drops_hidden_doc(vis_store):
    store, _emb, pid, _p = vis_store
    hits = store.bm25_search(pid, QUERY, k=5)
    ids = _ids(hits)
    assert LIVE_ID in ids
    assert STALE_ID not in ids


def test_identifier_search_drops_hidden_doc(vis_store):
    store, _emb, pid, _p = vis_store
    hits = store.identifier_search(pid, ["D1LETTER"], k=5)
    ids = _ids(hits)
    assert LIVE_ID in ids
    assert STALE_ID not in ids


# ── the read paths the three tests above did not cover ────────────────────
#
# Live d8d9573. The Contract Data PDF ``d0c00002`` was retired in favour of
# ``d0c00003`` that morning. Asked "Who is the Engineer under this contract?",
# results #3-#5 were chunks of the RETIRED document. The vector and BM25 legs
# and identifier_search all hide it; the rescue paths fetch through two other
# doors, and neither had a lock: chunks_for_docs() and
# chunks_containing_all(). The file lookup that feeds the first one,
# documents_matching_title_phrase(), listed retired files as well.

def test_chunks_for_docs_drops_hidden_doc(vis_store):
    """Even when the caller passes the retired id explicitly. A caller's list
    of ids comes from a filename lookup; it is not a licence."""
    store, _emb, pid, _p = vis_store
    hits = store.chunks_for_docs(pid, [STALE_ID, LIVE_ID])
    assert _ids(hits) == [LIVE_ID]


def test_chunks_containing_all_drops_hidden_doc(vis_store):
    store, _emb, pid, _p = vis_store
    hits = store.chunks_containing_all(pid, ["d1letter", "signatory"])
    assert _ids(hits) == [LIVE_ID]
    scoped = store.chunks_containing_all(
        pid, ["d1letter", "signatory"], doc_ids=[STALE_ID, LIVE_ID],
    )
    assert _ids(scoped) == [LIVE_ID]


def test_chunks_following_drops_hidden_doc(vis_store):
    """The "read the next chunk" door: found by the invariant test below, not
    by a live failure."""
    store, embedder, pid, _p = vis_store
    for doc_id in (STALE_ID, LIVE_ID):
        texts = [f"{doc_id} page one", f"{doc_id} page two continues"]
        store.upsert_chunks(pid, doc_id, texts, embedder.encode(texts))
    store._visibility_ready = None
    hits = store.chunks_following(pid, [(STALE_ID, 0), (LIVE_ID, 0)])
    assert _ids(hits) == [LIVE_ID]


def test_the_title_lookup_does_not_list_a_retired_file(vis_store):
    _store, _emb, pid, projects = vis_store
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        for did in (STALE_ID, LIVE_ID):
            session.get(Document, did).original_name = f"{did} Contract Data.pdf"
        session.commit()
    found = projects.documents_matching_title_phrase(pid, "contract data")
    assert [d["id"] for d in found] == [LIVE_ID]


def test_no_way_of_reading_chunks_returns_a_retired_document(vis_store):
    """The invariant, not the list. Every public VectorStore method that hands
    back chunks is called here; a new one added without the filter fails this
    test the day it is written, instead of being found on live months later
    the way these two were."""
    import inspect

    store, embedder, pid, _p = vis_store
    q = embedder.encode([QUERY])[0]
    calls = {
        "search": lambda: store.search(pid, q, k=5, query_text=QUERY),
        "bm25_search": lambda: store.bm25_search(pid, QUERY, k=5),
        "identifier_search": lambda: store.identifier_search(pid, ["D1LETTER"], k=5),
        "chunks_for_docs": lambda: store.chunks_for_docs(pid, [STALE_ID, LIVE_ID]),
        "chunks_containing_all": lambda: store.chunks_containing_all(pid, ["d1letter"]),
    }
    # Exempt, with the reason on the record rather than a silent gap:
    # photo_chunks rows are not documents (doc_id is the chunk's own id), so
    # there is no documents row to retire and nothing for the flag to gate.
    exempt = {"bm25_search_photos"}
    returns_chunks = {
        name for name, fn in inspect.getmembers(VectorStore, inspect.isfunction)
        if not name.startswith("_")
        and "List[Chunk]" in str(inspect.signature(fn).return_annotation)
    } - exempt
    assert returns_chunks <= set(calls) | {"chunks_following"}, (
        f"VectorStore methods that return chunks but are not covered by the "
        f"retired-document invariant: {sorted(returns_chunks - set(calls))}"
    )
    for name, call in calls.items():
        ids = _ids(call())
        assert STALE_ID not in ids, f"{name}() returned a retired document"
        assert LIVE_ID in ids, f"{name}() lost the live document"


def test_d1_hybrid_top5_contains_live_not_stale(vis_store, monkeypatch):
    store, embedder, pid, _p = vis_store
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    q = embedder.encode([QUERY])[0]
    hits = store.search(pid, q, k=5, query_text=QUERY)
    ids = _ids(hits)
    assert LIVE_ID in ids
    assert STALE_ID not in ids


def test_flag_flip_restores_stale_in_both_legs(vis_store):
    store, embedder, pid, projects = vis_store
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, STALE_ID)
        row.retrieval_visible = True
        session.commit()
    store._visibility_ready = None

    q = embedder.encode([QUERY])[0]
    vec_ids = _ids(store._semantic_search(pid, q, k=5))
    bm25_ids = _ids(store.bm25_search(pid, QUERY, k=5))
    assert STALE_ID in vec_ids
    assert STALE_ID in bm25_ids
