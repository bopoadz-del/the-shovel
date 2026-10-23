"""List continuation: the items an in-pool introduction opens reach the top-k.

General mechanism (``recall_list_continuations`` in the retriever). The
chunker often splits a clause from its list: chunk N ends "... shall be as
follows", chunk N+1 opens with the items. Retrieval ranks the introduction
(it carries the question's words) and never pools the items (they carry none
of them), so the answer stops at "as follows". When a top-k chunk ends on an
open introduction that shares the question's terms, the next chunk of the same
document is fetched and, if it opens with a run of list items, pooled ahead of
the introduction.

Replaces a path keyed on one clause number and on the expected list items
themselves. Every text below is invented.
"""
from __future__ import annotations

import importlib

import pytest

from app.core.rag.vector_store import Chunk

ASK = (
    "In what order of priority do the approved submittals rank? Name the "
    "first three."
)
OTHER_ASK = "What is the retention percentage held on interim payments?"

INTRO_TEXT = (
    "4.2 Priority of Submittals\n"
    "Where approved submittals conflict, their order of priority shall be "
    "as follows:"
)
LIST_TEXT = (
    "Site Instructions\n"
    "Approved Shop Drawings\n"
    "Method Statements\n"
    "Material Approval Requests\n"
    "Inspection Requests"
)
TOC_TEXT = (
    "Volume 3 Quality Plan - Contents\n"
    "4.2 Priority of Submittals\n"
    "4.3 Review periods"
)
OTHER_TEXT = (
    "Submittals shall be reviewed by the engineer within fourteen days of "
    "receipt and returned with a review status."
)
UNRELATED_INTRO = "The design review procedure is as follows:"

PID = "synthetic-list-project"
VOL_DOC = "quality-plan"
TOC_DOC = "quality-plan-toc"
OTHER_DOC = "review-procedure"


def _chunk(cid, doc_id, score, text, chunk_index=0):
    return Chunk(
        chunk_id=cid, project_id=PID, doc_id=doc_id, chunk_index=chunk_index,
        text=text, score=score,
    )


def _install(monkeypatch, *, list_in_semantic=False, intro=INTRO_TEXT):
    from app.core.rag import retriever as ret

    intro_c = _chunk("intro", VOL_DOC, 0.91, intro, chunk_index=6)
    toc = _chunk("toc", TOC_DOC, 0.88, TOC_TEXT)
    other = _chunk("oth", OTHER_DOC, 0.84, OTHER_TEXT)
    listed = _chunk("list", VOL_DOC, 0.22, LIST_TEXT, chunk_index=7)
    semantic = [intro_c, toc, other] + ([listed] if list_in_semantic else [])

    def fake_following(self, project_id, anchors, n=1):
        return [listed for doc_id, idx in anchors or []
                if doc_id == VOL_DOC and int(idx) == 6]

    from app.core.rag import vector_store as vs
    patches = {
        "search": lambda self, project_id, qvec, k, query_text=None: [
            Chunk(**{**c.__dict__}) for c in semantic][:k],
        "identifier_search": lambda self, project_id, identifiers, k=20: [],
        "chunks_for_docs": lambda self, project_id, doc_ids, **kw: [],
        "chunks_containing_all": lambda self, project_id, needles, k=20, doc_ids=None: [],
        "chunks_following": fake_following,
        "count": lambda self, pid=None: 4,
        "_verify_embedding_identity": lambda self: None,
    }
    for name, fn in patches.items():
        monkeypatch.setattr(vs.VectorStore, name, fn)
    names = {VOL_DOC: "Quality Plan Volume 3.pdf", TOC_DOC: "Quality Plan contents.pdf",
             OTHER_DOC: "Submittal review procedure.pdf"}
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda did: names.get(did, ""))
    monkeypatch.setattr("app.core.projects.documents_matching_title_phrase",
                        lambda pid, phrase, limit=8: [])
    monkeypatch.setattr("app.core.projects.documents_matching_filename_terms",
                        lambda *a, **k: [])
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    monkeypatch.delenv("RAG_LAYERED", raising=False)
    return ret


def test_the_list_after_an_open_introduction_is_pooled_first(monkeypatch):
    ret = _install(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(ASK, PID, k=5)
    assert chunks and chunks[0].chunk_id == "list", [c.chunk_id for c in chunks]
    assert "Approved Shop Drawings" in chunks[0].text


def test_a_list_already_pooled_is_lifted_over_its_introduction(monkeypatch):
    ret = _install(monkeypatch, list_in_semantic=True)
    chunks, _ = ret.retrieve_with_filter(ASK, PID, k=5)
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "list", ids


def test_a_question_about_something_else_does_not_take_the_list(monkeypatch):
    ret = _install(monkeypatch)
    chunks, _ = ret.retrieve_with_filter(OTHER_ASK, PID, k=5)
    assert "list" not in [c.chunk_id for c in chunks]


def test_an_introduction_that_shares_no_terms_is_not_followed(monkeypatch):
    ret = _install(monkeypatch, intro=UNRELATED_INTRO)
    chunks, _ = ret.retrieve_with_filter(ASK, PID, k=5)
    assert "list" not in [c.chunk_id for c in chunks]


@pytest.mark.parametrize("text, expected", [
    (INTRO_TEXT, True),
    (UNRELATED_INTRO, True),
    ("The documents listed below apply.\n", False),
    ("Approvals shall follow the documents listed below", True),
    (INTRO_TEXT + "\n" + LIST_TEXT, False),
    (OTHER_TEXT, False),
])
def test_open_introduction_is_read_from_the_chunk_end(text, expected):
    from app.core.rag.retriever import chunk_is_open_list_intro

    assert chunk_is_open_list_intro(text) is expected


def test_a_list_opening_is_a_run_of_short_items():
    from app.core.rag.retriever import chunk_opens_with_list

    assert chunk_opens_with_list(LIST_TEXT)
    assert chunk_opens_with_list("(a) Site Instructions; (b) Method Statements; (c) Drawings")
    assert chunk_opens_with_list("1. Site Instructions\n2. Method Statements")
    assert not chunk_opens_with_list(OTHER_TEXT)
    assert not chunk_opens_with_list("Site Instructions")


@pytest.fixture
def project_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod

    importlib.reload(db_mod)
    from app.core import projects as projects_mod

    pm = importlib.reload(projects_mod)
    pm._initialized = False
    pm.init_db()
    return pm


def test_chunks_following_returns_the_next_same_doc_index(project_store, monkeypatch):
    from app.core.rag import embeddings as _emb, vector_store as _vs

    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    _emb.reset_embedder_cache()
    _vs.reset_store_cache()
    from app.core.rag.embeddings import Embedder
    from app.core.rag.vector_store import get_store

    e = Embedder(model_name="fake")
    store = get_store(dim=e.dim)
    p = project_store.create_project("List neighbour SQL")
    pid = p["id"]
    vol = project_store.add_document(pid, "Quality Plan Volume 3.pdf", size=40)
    texts = [TOC_TEXT, INTRO_TEXT, LIST_TEXT, OTHER_TEXT]
    store.upsert_chunks(pid, vol["id"], texts, e.encode(texts))
    hits = store.chunks_following(pid, [(vol["id"], 1)], n=1)
    assert len(hits) == 1
    assert hits[0].chunk_index == 2
    assert "Method Statements" in hits[0].text
    assert store.chunks_following(pid, [], n=1) == []
    assert store.chunks_following(pid, [(vol["id"], 99)], n=1) == []
