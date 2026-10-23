"""Tests for RAG_GK_LEXICAL_FOLD (TASK 2 — the EOT residual).

RAG_AUDIT_V3 showed the GK lexical bonus (+0.25/term, cap +1.2) lifts
curated notes ~+0.5 above any project chunk, so a user's own document
loses the H1 margin comparison to a bonused GK note even under cfg7.
The fold makes the H1 margin gate compare on the GK chunk's RAW fused
score (lexical bonus subtracted); survivors keep the bonus for ordering.

Contract under test:

  * flag unset (default): bit-identical to pre-fold behavior, even with
    the cfg7 knobs on — the bonused score faces the margin as before
  * flag on: a GK chunk whose raw score fails the margin but whose
    bonused score would pass is gated out; one whose raw score clears
    the margin survives WITH its bonus intact for ordering
  * CALC/KB intents bypass the fold entirely (same exemption as the
    H knobs)
  * empty project: GK-only fallback unaffected (margin gate skipped)
  * garbage flag values mean OFF (same contract as the H knobs)

Mocked-store pattern from test_gk_ranking_knobs.py: scores are exact.
Unlike that file, the query here deliberately overlaps GK chunk text so
_gk_lexical_bonus fires; active-project chunk text stays overlap-free.
"""
from __future__ import annotations

import pytest


GK = "training_material"
ACTIVE = "p_active"
KNOB_VARS = (
    "RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
    "RAG_GK_LEXICAL_FOLD",
)

# Query terms (>=4 chars, non-stopword): extension, time, notice, period ->
# significant terms are {extension, notice, period} ("time" is 4 chars...
# actually len("time") == 4 so it also counts). GK text below contains
# "extension", "notice", "period" -> 3 overlapping terms = +0.75 bonus.
QUERY = "extension notice period"
GK_OVERLAP_TEXT = (
    "FIDIC 2017: the Contractor gives notice of an extension claim within "
    "a 28 day period of awareness."
)
GK_PLAIN_TEXT = "curated reference filler with no query overlap"
ACTIVE_TEXT = "own contract filler content"


def _chunk(chunk_id, project_id, score, text):
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


def _set_cfg7(monkeypatch):
    monkeypatch.setenv("RAG_GK_SCORE_MARGIN", "0.15")
    monkeypatch.setenv("RAG_OWN_DOC_BOOST", "0.1")
    monkeypatch.setenv("RAG_GK_TOPK_CAP", "2")


def test_lexical_bonus_fires_in_this_fixture(monkeypatch):
    """Sanity: the query/text pairing produces the +0.75 bonus the other
    tests rely on (3 overlapping significant terms x 0.25)."""
    ret = _install(monkeypatch, [], [])
    terms = ret._significant_terms(QUERY)
    assert ret._gk_lexical_bonus(terms, GK_OVERLAP_TEXT) == pytest.approx(0.75)
    assert ret._gk_lexical_bonus(terms, ACTIVE_TEXT) == 0.0


def test_flag_unset_identical_to_pre_fold_even_with_cfg7(monkeypatch):
    """Flag unset: the bonused GK score faces the margin, exactly as
    before this feature. GK raw 0.70 + bonus 0.75 = 1.45 clears the bar
    (0.80 + 0.1 boost + 0.15 margin = 1.05) and wins rank 1 — the EOT
    failure mode this task exists to fix, preserved verbatim when OFF."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[_chunk("gk1", GK, 0.70, GK_OVERLAP_TEXT)],
    )
    _set_cfg7(monkeypatch)
    chunks, dropped = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert dropped == 0
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk1", pytest.approx(1.45)), ("ap1", pytest.approx(0.90)),
    ]


def test_flag_on_gates_gk_that_passes_only_via_bonus(monkeypatch):
    """Flag on + cfg7: gk_low raw 0.70 fails the 1.05 bar even though its
    bonused 1.45 would pass — gated out. gk_high raw 1.10 clears the bar
    and keeps its bonus for ordering (final score 1.85, rank 1)."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[
            _chunk("gk_low", GK, 0.70, GK_OVERLAP_TEXT),
            _chunk("gk_high", GK, 1.10, GK_OVERLAP_TEXT),
        ],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk_high", pytest.approx(1.85)), ("ap1", pytest.approx(0.90)),
    ]


def test_flag_on_own_doc_reaches_rank_1_in_eot_shape(monkeypatch):
    """The acceptance shape: project's own contract chunk semantically
    strong (0.80), GK FIDIC note weaker raw (0.70) but keyword-dense —
    with the fold the user's own doc is rank 1."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("own_contract", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[_chunk("gk_fidic", GK, 0.70, GK_OVERLAP_TEXT)],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert [c.chunk_id for c in chunks] == ["own_contract"]


def test_flag_on_gk_without_bonus_unaffected(monkeypatch):
    """A GK chunk with no lexical overlap has nothing to fold: it faces
    the margin on its plain score exactly as under cfg7 alone."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[
            _chunk("gk_plain_high", GK, 1.10, GK_PLAIN_TEXT),
            _chunk("gk_plain_low", GK, 0.95, GK_PLAIN_TEXT),
        ],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    # Framed inside the project, so the margin applies.
    chunks, _ = ret.retrieve_with_filter(QUERY + " under this contract", ACTIVE, k=3)
    # bar = 0.80 + 0.1 + 0.15 = 1.05: 1.10 survives, 0.95 gated; a
    # project-framed question puts the project's best chunk first.
    assert [c.chunk_id for c in chunks] == ["ap1", "gk_plain_high"]


@pytest.mark.parametrize("intent", ["calculation", "standards", "knowledge"])
def test_calc_kb_intent_bypasses_fold_and_knobs(monkeypatch, intent):
    """CALC/KB intents: full bypass — bonused GK-wins behavior even with
    the flag and every knob set (same exemption as the H knobs)."""
    from app.core.rag.retriever import CALC_KB_INTENTS
    assert intent in CALC_KB_INTENTS
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[_chunk("gk1", GK, 0.70, GK_OVERLAP_TEXT)],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3, intent=intent)
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk1", pytest.approx(1.45)), ("ap1", pytest.approx(0.80)),
    ]


def test_flag_on_empty_project_skips_gk_merge(monkeypatch):
    """Empty project: GK merge is skipped, so lexical fold has no effect."""
    ret = _install(
        monkeypatch,
        active_chunks=[],
        gk_chunks=[
            _chunk("gk1", GK, 0.60, GK_OVERLAP_TEXT),
            _chunk("gk2", GK, 0.10, GK_PLAIN_TEXT),
        ],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert [(c.chunk_id, c.score) for c in chunks] == []


@pytest.mark.parametrize("garbage", ["2", "yep", "enabled", "tru"])
def test_garbage_flag_value_means_off(monkeypatch, garbage):
    """Garbage values disable the fold (logged), same contract as the H
    knobs: results identical to flag-unset."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[_chunk("gk1", GK, 0.70, GK_OVERLAP_TEXT)],
    )
    _set_cfg7(monkeypatch)
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", garbage)
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert [c.chunk_id for c in chunks] == ["gk1", "ap1"]


def test_flag_on_without_margin_is_a_noop(monkeypatch):
    """The fold modifies the H1 comparison only; with no margin set there
    is no gate to fold into, so the flag alone changes nothing."""
    ret = _install(
        monkeypatch,
        active_chunks=[_chunk("ap1", ACTIVE, 0.80, ACTIVE_TEXT)],
        gk_chunks=[_chunk("gk1", GK, 0.70, GK_OVERLAP_TEXT)],
    )
    monkeypatch.setenv("RAG_GK_LEXICAL_FOLD", "1")
    chunks, _ = ret.retrieve_with_filter(QUERY, ACTIVE, k=3)
    assert [(c.chunk_id, c.score) for c in chunks] == [
        ("gk1", pytest.approx(1.45)), ("ap1", pytest.approx(0.80)),
    ]
