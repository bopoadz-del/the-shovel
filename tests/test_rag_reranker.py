"""Cross-encoder reranker (RAG_RERANKER) — planted-truth + degrade shapes.

Why this exists: the 2026-08-01 live measurement showed doc-recall 41% @5 but
53% @50 on drive_archive — correct documents retrieved but ranked 6th-50th,
where the top-5 cut discards them. The reranker collects a deeper candidate
pool and lets a cross-encoder pick the best k.

Test shapes (STANDING TEST DOCTRINE — new shapes, not battery re-runs):
  * PLANTED TRUTH: the correct chunk is buried at the bottom of the cosine
    order; a relevance-aware scorer must surface it into the top-k.
  * CONSERVATION: flag OFF is byte-identical to the pre-reranker pipeline,
    and the gates (noise, revision suppression) hold with the flag ON — a
    suppressed chunk can never be resurrected by a good rerank score.
  * OUTAGE: model load failure and scoring failure both degrade to the
    original cosine order, never to an error or empty result.

No real cross-encoder is loaded anywhere here — a fake with a .predict()
keyed on text overlap stands in, so CI needs no model download.
"""
import pytest

from app.core.rag import reranker

# ── fakes ────────────────────────────────────────────────────────────────────

class _OverlapScorer:
    """Fake CrossEncoder: score = word overlap between query and passage."""

    def predict(self, pairs, show_progress_bar=False):
        out = []
        for q, t in pairs:
            qw = set(q.lower().split())
            tw = set(t.lower().split())
            out.append(float(len(qw & tw)))
        return out


class _Boom:
    def predict(self, pairs, show_progress_bar=False):
        raise RuntimeError("scoring exploded")


class _NeverCalled:
    def predict(self, pairs, show_progress_bar=False):  # pragma: no cover
        raise AssertionError("reranker must not be invoked with the flag off")


class _FakeChunk:
    def __init__(self, cid, text):
        self.chunk_id = cid
        self.text = text

    def __repr__(self):
        return f"<{self.chunk_id}>"


def _use_model(monkeypatch, model):
    monkeypatch.setattr(reranker, "_get_model", lambda: model)


# ── unit: rerank() ───────────────────────────────────────────────────────────

def test_planted_truth_surfaces_from_the_bottom(monkeypatch):
    """The needle sits LAST in cosine order; the scorer must lift it to #1."""
    _use_model(monkeypatch, _OverlapScorer())
    noise = [_FakeChunk(f"n{i}", "unrelated boilerplate clause text") for i in range(29)]
    needle = _FakeChunk("needle", "manhole spacing shall not exceed forty metres")
    got = reranker.rerank("what is the maximum manhole spacing", noise + [needle], k=5)
    assert got[0].chunk_id == "needle"
    assert len(got) == 5


def test_scoring_failure_degrades_to_original_order(monkeypatch):
    _use_model(monkeypatch, _Boom())
    chunks = [_FakeChunk(f"c{i}", f"text {i}") for i in range(10)]
    got = reranker.rerank("q", chunks, k=3)
    assert [c.chunk_id for c in got] == ["c0", "c1", "c2"]


def test_model_unavailable_degrades_to_original_order(monkeypatch):
    _use_model(monkeypatch, None)
    chunks = [_FakeChunk(f"c{i}", f"text {i}") for i in range(4)]
    got = reranker.rerank("q", chunks, k=2)
    assert [c.chunk_id for c in got] == ["c0", "c1"]


def test_single_candidate_short_circuits_without_model(monkeypatch):
    # len<=1 must not even consult the model.
    _use_model(monkeypatch, _NeverCalled())
    only = [_FakeChunk("solo", "x")]
    assert reranker.rerank("q", only, k=5) == only


# ── unit: knobs ──────────────────────────────────────────────────────────────

def test_enabled_default_off(monkeypatch):
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert reranker.enabled() is False


@pytest.mark.parametrize("val,want", [
    ("1", True), ("true", True), ("on", True), ("0", False), ("", False),
    ("false", False),
])
def test_enabled_parsing(monkeypatch, val, want):
    monkeypatch.setenv("RAG_RERANKER", val)
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    assert reranker.enabled() is want


@pytest.mark.parametrize("val,want", [
    ("1", True), ("true", True), ("yes", True), ("on", True),
    ("false", False), ("0", False),
])
def test_rerank_enabled_is_the_canonical_flag(monkeypatch, val, want):
    monkeypatch.setenv("RERANK_ENABLED", val)
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    assert reranker.enabled() is want


def test_rerank_enabled_wins_over_alias(monkeypatch):
    """An operator pinning RERANK_ENABLED=false must not be flipped by alias."""
    monkeypatch.setenv("RERANK_ENABLED", "false")
    monkeypatch.setenv("RAG_RERANKER", "1")
    assert reranker.enabled() is False
    monkeypatch.setenv("RERANK_ENABLED", "true")
    monkeypatch.setenv("RAG_RERANKER", "0")
    assert reranker.enabled() is True


def test_default_candidate_depth_is_hybrid_top_50(monkeypatch):
    monkeypatch.delenv("RAG_RERANK_CANDIDATES", raising=False)
    assert reranker.DEFAULT_CANDIDATES == 50
    assert reranker.candidate_depth(5) == 50


def test_candidate_depth_never_below_k(monkeypatch):
    monkeypatch.setenv("RAG_RERANK_CANDIDATES", "10")
    assert reranker.candidate_depth(25) == 25
    assert reranker.candidate_depth(5) == 10
    monkeypatch.setenv("RAG_RERANK_CANDIDATES", "garbage")
    assert reranker.candidate_depth(5) == reranker.DEFAULT_CANDIDATES


# ── retriever integration (fixture shape of test_revision_currency) ─────────

ACTIVE = "p_active"


def _chunk(cid, score, text="filler"):
    from app.core.rag.vector_store import Chunk
    return Chunk(chunk_id=cid, project_id=ACTIVE, doc_id=f"doc-{cid}",
                 chunk_index=0, text=text, score=score)


def _install(monkeypatch, chunks, names=None):
    from app.core.rag import retriever as ret
    names = names or {}

    def fake_search(self, project_id, qvec, k, query_text=None):
        return [c for c in chunks if c.project_id == project_id]

    monkeypatch.setattr("app.core.rag.vector_store.VectorStore.search", fake_search)
    monkeypatch.setattr(
        "app.core.rag.vector_store.VectorStore._verify_embedding_identity",
        lambda self: None)
    monkeypatch.setattr(ret, "_doc_name_for_id",
                        lambda _id: names.get(_id, ""), raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "curated_kb")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    for v in ("RAG_GK_SCORE_MARGIN", "RAG_OWN_DOC_BOOST", "RAG_GK_TOPK_CAP",
              "RAG_GK_LEXICAL_FOLD", "RAG_AUTHORITY_WEIGHT", "RAG_LAYER_WEIGHT",
              "RAG_LAYERED", "RAG_RERANK_CANDIDATES"):
        monkeypatch.delenv(v, raising=False)
    return ret


def test_flag_off_pipeline_untouched(monkeypatch):
    """CONSERVATION: with both flags unset the result is the plain cosine
    top-k and the model is never consulted."""
    monkeypatch.delenv("RAG_RERANKER", raising=False)
    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    _use_model(monkeypatch, _NeverCalled())
    pool = [_chunk(f"c{i}", 0.9 - i * 0.01) for i in range(10)]
    ret = _install(monkeypatch, pool)
    got, _ = ret.retrieve_with_filter("any query", ACTIVE, k=3)
    assert [c.chunk_id for c in got] == ["c0", "c1", "c2"]


def test_flag_on_surfaces_buried_truth_through_retriever(monkeypatch):
    """PLANTED TRUTH end-to-end: the relevant chunk is 20th by cosine; with
    the flag on it must reach the top-3."""
    monkeypatch.setenv("RERANK_ENABLED", "true")
    _use_model(monkeypatch, _OverlapScorer())
    pool = [_chunk(f"c{i}", 0.9 - i * 0.01, text="unrelated clause") for i in range(19)]
    pool.append(_chunk("needle", 0.60,
                       text="pipe bedding class B granular material specification"))
    ret = _install(monkeypatch, pool)
    got, _ = ret.retrieve_with_filter(
        "what bedding class and granular material does the pipe specification require",
        ACTIVE, k=3)
    assert "needle" in [c.chunk_id for c in got]


def test_flag_on_cannot_resurrect_revision_suppressed_chunk(monkeypatch):
    """CONSERVATION: gates run before the reranker. A lower revision of the
    same drawing stays suppressed even when its text is the best match."""
    monkeypatch.setenv("RAG_RERANKER", "1")
    _use_model(monkeypatch, _OverlapScorer())
    stale = _chunk("revA", 0.90, text="manhole spacing forty metres maximum")
    current = _chunk("revC", 0.89, text="unrelated general note")
    ret = _install(monkeypatch, [stale, current],
                   {"doc-revA": "A-101_RevA.pdf", "doc-revC": "A-101_RevC.pdf"})
    got, _ = ret.retrieve_with_filter("manhole spacing maximum", ACTIVE, k=5)
    ids = [c.chunk_id for c in got]
    assert "revA" not in ids, "reranker must not resurrect a suppressed revision"
    assert "revC" in ids


def test_flag_on_model_down_returns_cosine_topk(monkeypatch):
    """OUTAGE end-to-end: flag on but the model can't load — behave exactly
    like flag off rather than erroring or returning empty."""
    monkeypatch.setenv("RAG_RERANKER", "1")
    _use_model(monkeypatch, None)
    pool = [_chunk(f"c{i}", 0.9 - i * 0.01) for i in range(10)]
    ret = _install(monkeypatch, pool)
    got, _ = ret.retrieve_with_filter("any query", ACTIVE, k=3)
    assert [c.chunk_id for c in got] == ["c0", "c1", "c2"]
