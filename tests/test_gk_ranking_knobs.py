"""Tests for the GK-contamination ranking knobs (RAG_AUDIT_V2 follow-up).

Three independent, env-driven, DEFAULT-OFF knobs in
``retrieve_with_filter``:

  * RAG_GK_SCORE_MARGIN — GK survives only when it beats the project's
    best candidate by the margin (empty projects keep the GK fallback)
  * RAG_OWN_DOC_BOOST   — additive boost on active-project chunks before
    the merged re-rank (ownership, not recency)
  * RAG_GK_TOPK_CAP     — at most N GK chunks in the final top-K, excess
    slots backfilled by the next-best project chunks

All three are gated on intent: they apply for intent=None / DOC_LOOKUP
intents and are bypassed for CALC/KB intents (calculation features depend
on GK winning). With no env vars set, results must be identical to the
pre-knob merge — that invariant is the first test.

The vector store is mocked (test_rag_injection.py pattern) so scores are
exact; queries avoid identifier tokens and lexical overlap with chunk
text so the identifier bonus and _gk_lexical_bonus stay at zero.
"""
from __future__ import annotations

import pytest


GK = "training_material"
ACTIVE = "p_active"
KNOB_VARS = ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP")


def _chunk(chunk_id, project_id, score, text="filler content"):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=chunk_id, project_id=project_id,
                 doc_id=f"doc-{chunk_id}", chunk_index=0,
                 text=text, score=score)


def _install(monkeypatch, active_chunks, gk_chunks):
    """Wire a fake vector store returning fixed candidates per project,
    with the fake embedder and a benign doc-name resolver."""
    from app.core.rag import retriever as ret

    def fake_search(self, project_id, qvec, k, query_text=None):
        if project_id == ACTIVE:
            return list(active_chunks)
        if project_id == GK:
            return list(gk_chunks)
        return []

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(ret, "_doc_name_for_id", lambda _id: "real.pdf",
                        raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", GK)
    for var in KNOB_VARS:
        monkeypatch.delenv(var, raising=False)
    return ret


def test_defaults_off_results_identical_to_plain_merge(monkeypatch):
    """No knob env vars set: pure score-descending merge, GK included,
    exactly as before this feature (the non-negotiable invariant)."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80)],
        gk_chunks=[_chunk("gk1", GK, 0.92), _chunk("gk2", GK, 0.55)],
    )
    chunks, dropped = ret.retrieve_with_filter("query", ACTIVE, k=3)
    assert dropped == 0
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk1", 0.92), ("ap1", 0.80), ("gk2", 0.55),
    ]


def test_margin_filters_gk_below_bar_keeps_gk_above(monkeypatch):
    """H1: with margin 0.2 and best project score 0.80, a GK chunk at
    0.92 (< 1.00) is filtered out while one at 1.05 (>= 1.00) survives."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80)],
        gk_chunks=[_chunk("gk_low", GK, 0.92), _chunk("gk_high", GK, 1.05)],
    )
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "0.2")
    # The margin guards questions framed inside the project.
    chunks, _ = ret.retrieve_with_filter("query about this contract", ACTIVE, k=3)
    ids = [c.chunk_id for c in chunks]
    # gk_high clears the margin and is kept; a project-framed question puts
    # the project's best chunk first (project layer first).
    assert ids == ["ap1", "gk_high"]
    assert "gk_low" not in ids


def test_margin_leaves_gk_untouched_for_empty_project(monkeypatch):
    """H1: an unindexed project must not merge GK fallback candidates."""
    ret = _install(
        monkeypatch,
        active_chunks=[],
        gk_chunks=[_chunk("gk1", GK, 0.60), _chunk("gk2", GK, 0.10)],
    )
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "5.0")
    chunks, _ = ret.retrieve_with_filter("query", ACTIVE, k=3)
    assert [c.chunk_id for c in chunks] == []


def test_own_doc_boost_reorders_project_above_gk(monkeypatch):
    """H2: an additive ownership boost lifts the project's own chunk over
    a higher-raw-score GK chunk, and the reported score reflects it."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80)],
        gk_chunks=[_chunk("gk1", GK, 0.92)],
    )
    monkeypatch.setenv("RAG_OWN_DOC_BOOST", "0.5")
    chunks, _ = ret.retrieve_with_filter("query", ACTIVE, k=2)
    assert [c.chunk_id for c in chunks] == ["ap1", "gk1"]
    assert chunks[0].score == pytest.approx(1.30)
    assert chunks[1].score == pytest.approx(0.92)


def test_gk_topk_cap_backfills_with_project_chunks(monkeypatch):
    """H3: with cap=2 and k=5 the top-5 keeps only the 2 best GK chunks;
    the slots the excess GK chunks would have taken go to the next-best
    project chunks."""
    ret = _install(
        monkeypatch,
        active_chunks=[
            _chunk("ap1", ACTIVE, 0.70), _chunk("ap2", ACTIVE, 0.65),
            _chunk("ap3", ACTIVE, 0.60), _chunk("ap4", ACTIVE, 0.55),
        ],
        gk_chunks=[
            _chunk("gk1", GK, 0.95), _chunk("gk2", GK, 0.90),
            _chunk("gk3", GK, 0.85), _chunk("gk4", GK, 0.80),
        ],
    )
    monkeypatch.setenv("RAG_GK_TOPK_CAP", "2")
    chunks, _ = ret.retrieve_with_filter("query", ACTIVE, k=5)
    # Uncapped top-5 would be gk1,gk2,gk3,gk4,ap1.
    assert [c.chunk_id for c in chunks] == ["gk1", "gk2", "ap1", "ap2", "ap3"]


def test_gk_topk_cap_drops_excess_when_no_project_chunks_remain(monkeypatch):
    """H3: on an unindexed project the GK pool is not merged at all."""
    ret = _install(
        monkeypatch,
        active_chunks=[],
        gk_chunks=[_chunk("gk1", GK, 0.95), _chunk("gk2", GK, 0.90),
                   _chunk("gk3", GK, 0.85)],
    )
    monkeypatch.setenv("RAG_GK_TOPK_CAP", "1")
    chunks, _ = ret.retrieve_with_filter("query", ACTIVE, k=5)
    assert [c.chunk_id for c in chunks] == []


@pytest.mark.parametrize("intent", ["calculation", "standards", "knowledge"])
def test_calc_kb_intent_bypasses_all_knobs(monkeypatch, intent):
    """CALC/KB intents preserve the current GK-wins behavior even with
    every knob set aggressively — calculation features depend on it."""
    from app.core.rag.retriever import CALC_KB_INTENTS
    assert intent in CALC_KB_INTENTS
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80)],
        gk_chunks=[_chunk("gk1", GK, 0.92), _chunk("gk2", GK, 0.55)],
    )
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "5.0")
    monkeypatch.setenv("RAG_OWN_DOC_BOOST", "9.0")
    monkeypatch.setenv("RAG_GK_TOPK_CAP", "0")
    chunks, _ = ret.retrieve_with_filter("query", ACTIVE, k=3, intent=intent)
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk1", 0.92), ("ap1", 0.80), ("gk2", 0.55),
    ]


def test_doc_lookup_intent_applies_knobs(monkeypatch):
    """DOC_LOOKUP intents behave like intent=None: knobs active."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80)],
        gk_chunks=[_chunk("gk1", GK, 0.92)],
    )
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "5.0")
    chunks, _ = ret.retrieve_with_filter(
        "query about this contract", ACTIVE, k=3, intent="document_lookup",
    )
    assert [c.chunk_id for c in chunks] == ["ap1"]


def test_rag_search_request_accepts_optional_intent():
    """The /v1/rag/search model takes intent for eval runs, defaulting to
    None so existing callers are unaffected."""
    from app.routers.rag import RagSearchRequest
    req = RagSearchRequest(query="q", project_id="p")
    assert req.intent is None
    req = RagSearchRequest(query="q", project_id="p", intent="calculation")
    assert req.intent == "calculation"
