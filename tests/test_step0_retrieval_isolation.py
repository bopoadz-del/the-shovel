"""STEP 0 — retrieval isolation regression + golden tests.

The invariants these lock (owner directive, 2026-07-17):

  0a  A project-scoped query can NEVER surface another project's own chunks,
      even when that other project's chunk outscores everything the active
      project has. Structural: the retriever only ever *asks* the vector store
      for the active project + the designated general-knowledge projects. A
      future "search every project, then rank" refactor must break these tests.

  0b  An empty/thin project does not receive another project's own documents.
      Shared general-knowledge stays readable. When the operator selects the
      Master Corpus itself, those chunks are OWN.

  regression  The live leak was a client corpus (drive_archive / example_infra_pack)
      listed in RAG_GENERAL_KNOWLEDGE_PROJECTS, silently merged into every other
      project. The Master-Corpus source is now structurally barred from the GK
      merge even when a stale env still lists it.

Harness mirrors tests/test_gk_ranking_knobs.py: the vector store's ``search`` is
faked so scores are exact, and queries avoid identifier/lexical overlap so the
identifier bonus and GK lexical bonus stay at zero.
"""
from __future__ import annotations

import pytest


ACTIVE = "p_active"
GK = "curated_kb"                 # a genuinely-general knowledge project
MASTER = "master_src"            # the Master-Corpus / client fallback corpus
OTHER = "p_other_client"        # a third project — neither GK nor fallback


def _chunk(chunk_id, project_id, score, text="filler content"):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=chunk_id, project_id=project_id,
                 doc_id=f"doc-{chunk_id}", chunk_index=0,
                 text=text, score=score)


def _install(monkeypatch, per_project, *, gk=GK, master=None, record=None):
    """Wire a fake vector store returning fixed candidates per project.

    ``per_project`` maps project_id -> list of Chunks. ``record`` (optional
    list) captures every project_id the retriever calls ``search`` with, so a
    test can assert the retriever never even *asks* for an unrelated project.
    """
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        if record is not None:
            record.append(project_id)
        return list(per_project.get(project_id, []))

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    # search is fully faked, so the store is just a shell — neutralise the
    # on-disk embedding-identity guard, which would otherwise trip on rows a
    # prior test wrote to the shared SQLite DB (the store cache + DB rows only
    # reset between tests under Postgres CI, not local SQLite).
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None,
    )
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", gk)
    if master is None:
        monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    else:
        monkeypatch.setenv("MASTER_CORPUS_SOURCE_PROJECT_ID", master)
    for var in ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
                "RAG_GK_LEXICAL_FOLD"):
        monkeypatch.delenv(var, raising=False)
    return ret


# ── 0a — structural isolation ─────────────────────────────────────────────

def test_unrelated_project_never_queried_even_when_it_would_outscore(monkeypatch):
    """Golden 0a: OTHER holds a 0.99 chunk that would dominate any ranking.
    A query scoped to ACTIVE must never even ASK the store for OTHER, so that
    chunk is structurally unreachable regardless of score."""
    asked: list = []
    ret = _install(
        monkeypatch,
        {
            ACTIVE: [_chunk("a1", ACTIVE, 0.55)],
            OTHER: [_chunk("leak", OTHER, 0.99)],   # never should be fetched
            GK: [],
        },
        master=MASTER,
        record=asked,
    )
    chunks, _ = ret.retrieve_with_filter("some neutral query", ACTIVE, k=5)
    assert OTHER not in asked, f"retriever asked for an unrelated project: {asked}"
    assert all(c.project_id != OTHER for c in chunks)
    assert [c.chunk_id for c in chunks] == ["a1"]
    assert chunks[0].layer == "own"


# ── 0b — labeled Master-Corpus fallback ───────────────────────────────────

def test_empty_project_does_not_receive_another_projects_own_chunks(monkeypatch):
    """0b: an empty project must not be served another project's own rows."""
    asked: list = []
    ret = _install(
        monkeypatch,
        {ACTIVE: [], MASTER: [_chunk("m1", MASTER, 0.90)], GK: []},
        master=MASTER,
        record=asked,
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert MASTER not in asked
    assert all(c.project_id != MASTER for c in chunks)
    assert chunks == []


def test_thin_project_does_not_pull_another_projects_own_chunks(monkeypatch):
    """A weak own match is not an invitation to read another project."""
    asked: list = []
    ret = _install(
        monkeypatch,
        {ACTIVE: [_chunk("a_weak", ACTIVE, 0.20)],
         MASTER: [_chunk("m1", MASTER, 0.90)], GK: []},
        master=MASTER,
        record=asked,
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert MASTER not in asked
    assert all(c.project_id != MASTER for c in chunks)
    assert [c.chunk_id for c in chunks] == ["a_weak"]
    assert chunks[0].layer == "own"


def test_strong_project_does_not_pull_master_corpus(monkeypatch):
    """0b core / the ha_long -> the client project fix: a populated project with a confident
    own match does NOT surface the Master Corpus, even though the Master chunk
    (0.99) outscores the own chunk (0.80). No silent cross-corpus."""
    ret = _install(
        monkeypatch,
        {ACTIVE: [_chunk("a1", ACTIVE, 0.80)],
         MASTER: [_chunk("m_big", MASTER, 0.99)], GK: []},
        master=MASTER,
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert [c.chunk_id for c in chunks] == ["a1"]
    assert all(c.layer != "master_corpus" for c in chunks)


def test_master_corpus_is_own_layer_when_it_is_the_active_project(monkeypatch):
    """When the operator selects the Master Corpus itself (project_id resolves
    to the source), its chunks are OWN, and it must not also self-fall-back."""
    ret = _install(
        monkeypatch,
        {MASTER: [_chunk("m1", MASTER, 0.30)], GK: []},
        master=MASTER,
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", MASTER, k=5)
    assert [c.chunk_id for c in chunks] == ["m1"]
    assert chunks[0].layer == "own"


# ── regression — client corpus in GK env is structurally excluded ─────────

def test_master_source_excluded_from_gk_merge_even_when_listed(monkeypatch):
    """The live leak: MASTER listed in RAG_GENERAL_KNOWLEDGE_PROJECTS would
    silently merge into every populated project. It must be stripped from the
    GK merge set (it can only surface as the disclosed empty/thin fallback)."""
    ret = _install(
        monkeypatch,
        {ACTIVE: [_chunk("a1", ACTIVE, 0.80)],
         MASTER: [_chunk("m_big", MASTER, 0.99)], GK: [_chunk("g1", GK, 0.10)]},
        gk=f"{GK},{MASTER}",           # stale/misconfigured env lists MASTER as GK
        master=MASTER,
    )
    # Sanity: the exclusion happens in the id resolver itself.
    assert MASTER not in ret._general_knowledge_project_ids()
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert all(c.project_id != MASTER for c in chunks), "client corpus leaked via GK"


def test_no_master_env_preserves_empty_project_returns_nothing(monkeypatch):
    """CI / self-host default (no MASTER_CORPUS_SOURCE_PROJECT_ID): an empty
    project returns [] — the fallback is opt-in by config, so the pre-existing
    'unindexed project' contract is unchanged."""
    ret = _install(
        monkeypatch,
        {ACTIVE: [], MASTER: [_chunk("m1", MASTER, 0.90)], GK: []},
        master=None,
    )
    chunks, _ = ret.retrieve_with_filter("neutral query", ACTIVE, k=5)
    assert chunks == []


# ── disclosure wiring: inject audit + runtime banner + sources panel ──────

def test_rag_inject_does_not_mark_foreign_own_rows_as_fallback(monkeypatch):
    ret = _install(
        monkeypatch,
        {ACTIVE: [], MASTER: [_chunk("m1", MASTER, 0.90)], GK: []},
        master=MASTER,
    )
    assert ret  # installed
    monkeypatch.setenv("RAG_CONFIDENCE_THRESHOLD", "0.4")
    from app.core.rag import inject as inj
    sys_msg, audit = inj.rag_inject(
        user_message="neutral query", project_id=ACTIVE,
        conversation_id=None, user_id=None, agent_name="project-assistant",
    )
    assert audit.get("fallback_used") is not True
    assert all(c.get("layer") != "master_corpus" for c in audit.get("chunks", []))
    assert all(c.get("project_id") != MASTER for c in audit.get("chunks", []))
    assert not sys_msg or "master_corpus" not in (sys_msg.get("content") or "").lower()


def test_postprocess_prepends_fallback_banner():
    from app.agents.runtime import _postprocess_answer, _MASTER_CORPUS_FALLBACK_NOTE
    out = _postprocess_answer("Here is the answer.", None, [], fallback_used=True)
    assert out.startswith(_MASTER_CORPUS_FALLBACK_NOTE)
    # Idempotent: a second pass over already-banner'd text doesn't double it.
    twice = _postprocess_answer(out, None, [], fallback_used=True)
    assert twice.count(_MASTER_CORPUS_FALLBACK_NOTE.strip()) == 1
    # Off by default: no banner when not a fallback answer.
    plain = _postprocess_answer("Here is the answer.", None, [], fallback_used=False)
    assert _MASTER_CORPUS_FALLBACK_NOTE.strip() not in plain


def test_sources_panel_tags_master_corpus_layer():
    from app.agents.runtime import _build_sources_from_audit
    audit = {
        "project_id": ACTIVE,
        "fallback_used": True,
        "chunks": [
            {"doc_id": "doc-m1", "chunk_index": 0, "chunk_id": "m1",
             "project_id": MASTER, "score": 0.9, "layer": "master_corpus"},
        ],
    }
    sources = _build_sources_from_audit(audit, final_text="")
    assert sources and sources[0]["layer"] == "master_corpus"
    assert sources[0]["layer_label"] == "Master Corpus (fallback)"
