"""Input rules of two readers the session-scope change (#400) touched.

* ``projects.documents_matching_filename_terms``: with no usable term (every
  term shorter than three characters, or carrying a LIKE wildcard) it
  returns nothing, even when no overlap is required -- never the whole
  project.
* ``projects.backfill_chunk_counts_from_table``: a run for one project scans
  and repairs that project's rows only, a chat-session file's included
  (maintenance sees every row), and leaves every other project alone.

Every project, file name and chunk text is invented here.
"""
from __future__ import annotations

import uuid

import pytest

from app.core import projects as store


def _project(tag: str) -> str:
    return store.create_project(f"ReaderRule-{tag}-{uuid.uuid4().hex[:6]}")["id"]


@pytest.mark.parametrize("terms,kw", [
    (["zq", "%x"], {"require_all": True}),
    ([], {"min_terms": 0}),
])
def test_no_usable_term_matches_nothing(terms, kw):
    pid = _project("terms")
    store.add_document(pid, "zq_invented_halvard_register.md")
    store.add_document(pid, "invented_halvard_schedule.md")
    assert store.documents_matching_filename_terms(pid, terms, **kw) == []


@pytest.mark.parametrize("apply", [False, True])
def test_a_project_backfill_touches_that_project_only(apply):
    from app.core.db import SessionLocal
    from app.core.models import Document
    from app.core.rag.embeddings import get_embedder
    from app.core.rag.vector_store import get_store

    emb = get_embedder()
    st = get_store(dim=emb.dim)
    mine, other = _project("mine"), _project("other")
    own = store.add_document(mine, "invented_corbel_notes.md")["id"]
    sess = store.add_document(mine, "invented_corbel_session.md", metadata={
        "session_id": f"ws-bf-{uuid.uuid4().hex[:6]}", "uploader_id": "u-bf",
        "origin": "uploaded"})["id"]
    theirs = store.add_document(other, "invented_corbel_other.md")["id"]
    for pid, did, texts in ((mine, own, ["corbel bearing one"]),
                            (mine, sess, ["corbel bearing two", "corbel bearing three"]),
                            (other, theirs, ["corbel bearing four"])):
        st.upsert_chunks(pid, did, texts, emb.encode(texts))
    with SessionLocal() as s:
        for did in (own, sess, theirs):
            s.get(Document, did).chunk_count = 0
        s.commit()

    got = store.backfill_chunk_counts_from_table(project_id=mine, apply=apply)

    assert (got["scanned"], got["stale"], got["updated"]) == (2, 2, 2 if apply else 0)
    with SessionLocal() as s:
        counts = {did: s.get(Document, did).chunk_count for did in (own, sess, theirs)}
    assert counts == ({own: 1, sess: 2, theirs: 0} if apply
                      else {own: 0, sess: 0, theirs: 0})
