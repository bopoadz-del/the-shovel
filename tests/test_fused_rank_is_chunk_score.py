"""S-R7: the fused rank becomes ``chunk.score`` on the cosine scale.

Reciprocal-rank fusion orders the hybrid candidates, but before this change
``chunk.score`` stayed the raw score of the leg that found the chunk. A chunk
only the lexical leg found kept its lexical rank (``ts_rank`` < 0.1) as its
score, so every later step -- which ranks, thresholds and adds bonuses on
``.score`` -- sorted it under every vector hit (cosine ~0.7-0.8) and it never
reached the handed set, whatever its fused position.

Every input below is constructed here: invented texts, invented ids, invented
scores. The query carries no identifier, so the identifier bonus stays off,
and every chunk shares the query's term pair, so the term-rescue gate closes
as it does live.

T1 (and its twin) and T4 fail on main; T2, T3 and the top-score guard pass on
main and must keep passing.
"""
from __future__ import annotations

import pytest

from app.core.rag import retriever
from app.core.rag import vector_store as vs
from app.core.rag.handed_set import select_handed
from app.core.rag.vector_store import Chunk, _rrf_combine

PID = "synthetic-fusion-project"
QUERY = "how is temporary site access kept open during the works"

_WORDS = (
    "gravel", "barrier", "signage", "lighting", "drainage", "fencing",
    "marshal", "crossing", "ramp", "gate", "kerb", "cone", "beacon",
    "matting", "plate", "hoarding", "turnstile", "bollard", "mirror", "siren",
)


def _text(n: int, kind: str) -> str:
    """A distinct invented passage sharing the query's term pair."""
    a = _WORDS[n % len(_WORDS)]
    b = _WORDS[(n * 7 + 3) % len(_WORDS)]
    return (
        f"{kind} passage {n}: temporary site access uses {a} and {b} "
        f"arrangements reviewed by the site team, variant {n * 13 % 97}."
    )


def _chunk(cid: str, score: float, *, doc: str | None = None, kind="vector",
           n: int = 0) -> Chunk:
    return Chunk(
        chunk_id=cid, project_id=PID, doc_id=doc or f"doc-{cid}",
        chunk_index=0, text=_text(n, kind), score=score,
    )


def _vector_leg(count: int, top: float = 0.80, bottom: float = 0.70,
                docs: int | None = None) -> list[Chunk]:
    step = (top - bottom) / max(count - 1, 1)
    out = []
    for i in range(count):
        doc = f"vdoc-{i % docs}" if docs else None
        out.append(_chunk(f"v{i}", round(top - i * step, 6), doc=doc, n=i))
    return out


def _copy(c: Chunk) -> Chunk:
    return Chunk(**{**c.__dict__})


def _install(monkeypatch, tmp_path, semantic, lexical):
    """The REAL ``VectorStore.search`` (and so the real ``_rrf_combine``)
    runs; only its two legs and the side lookups are stubbed."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    monkeypatch.setenv("RAG_GENERAL_KNOWLEDGE_PROJECTS", "")
    monkeypatch.delenv("MASTER_CORPUS_SOURCE_PROJECT_ID", raising=False)
    corpus = list({c.chunk_id: c for c in list(semantic) + list(lexical)}.values())

    def _semantic_search(self, project_id, q, k):
        return [_copy(c) for c in semantic if c.project_id == project_id][:k]

    def bm25_search(self, project_id, query, k=50, doc_ids=None):
        return [_copy(c) for c in lexical if c.project_id == project_id][:k]

    def identifier_search(self, project_id, identifiers, k=20):
        return []

    def chunks_containing_all(self, project_id, needles, k=20, doc_ids=None):
        return []

    def chunks_for_docs(self, project_id, doc_ids, **_kw):
        return [_copy(c) for c in corpus
                if c.project_id == project_id and c.doc_id in doc_ids]

    def chunks_following(self, project_id, anchors, n=1):
        return []

    def count(self, project_id=None):
        return len(corpus)

    def has_chunks(self, project_id):
        return bool(corpus)

    for name, fn in (
        ("_semantic_search", _semantic_search), ("bm25_search", bm25_search),
        ("identifier_search", identifier_search),
        ("chunks_containing_all", chunks_containing_all),
        ("chunks_for_docs", chunks_for_docs),
        ("chunks_following", chunks_following), ("count", count),
        ("has_chunks", has_chunks),
    ):
        monkeypatch.setattr(vs.VectorStore, name, fn)
    monkeypatch.setattr(retriever, "_doc_name_for_id", lambda did: f"{did}.pdf")


def _store_search(semantic_dim_query: str = QUERY, k: int = 60):
    store = vs.get_store()
    qvec = retriever.get_embedder().encode([semantic_dim_query])[0]
    return store.search(PID, qvec, k=k, query_text=semantic_dim_query)


def _ids(chunks):
    return [c.chunk_id for c in chunks]


# ── T1: a lexical rank-1 hit reaches the pool and the handed set ─────────────

@pytest.mark.parametrize("x_vector_rank", [None, 45])
def test_lexical_rank_one_hit_is_pooled_and_handed(monkeypatch, tmp_path, x_vector_rank):
    """``None``: X is found by the lexical leg only. ``45``: X is also the
    vector leg's 45th hit. Either way the lexical leg ranks it first, its
    fused position is at the top, and it must be pooled and handed."""
    semantic = _vector_leg(50)
    if x_vector_rank is not None:
        x = semantic[x_vector_rank - 1]
        x_id = x.chunk_id
    else:
        x_id = "x"
        x = _chunk(x_id, 0.06, kind="lexical", n=101)
    lexical = [Chunk(**{**x.__dict__, "score": 0.06})] + [
        _chunk(f"l{i}", round(0.05 - i * 0.001, 6), kind="lexical", n=200 + i)
        for i in range(9)
    ]
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    assert x_id in _ids(pooled), _ids(pooled)
    handed, _ = select_handed(pooled, k=5)
    assert x_id in _ids(handed), _ids(handed)


# ── T2: a vector rank-1 hit is still handed; the top score is the top cosine ──

@pytest.mark.parametrize("lexical_kind", ["lexical_only_rows", "empty"])
def test_vector_rank_one_hit_is_still_handed(monkeypatch, tmp_path, lexical_kind):
    semantic = _vector_leg(50)
    lexical = (
        [_chunk(f"l{i}", round(0.06 - i * 0.001, 6), kind="lexical", n=300 + i)
         for i in range(10)]
        if lexical_kind == "lexical_only_rows" else []
    )
    _install(monkeypatch, tmp_path, semantic, lexical)

    pooled, _ = retriever.retrieve_with_filter(QUERY, PID, k=10)
    handed, _ = select_handed(pooled, k=5)
    assert "v0" in _ids(handed), _ids(handed)
    raw = _store_search()
    assert max(c.score for c in raw) == pytest.approx(0.80)


# ── T3: when both legs agree, the remap is the identity ─────────────────────

@pytest.mark.parametrize("count, top, bottom", [(20, 0.80, 0.70), (7, 0.66, 0.41)])
def test_agreeing_legs_keep_vector_order_and_cosines(monkeypatch, tmp_path,
                                                     count, top, bottom):
    semantic = _vector_leg(count, top=top, bottom=bottom)
    lexical = [Chunk(**{**c.__dict__, "score": 0.09 - i * 0.001})
               for i, c in enumerate(semantic)]
    _install(monkeypatch, tmp_path, semantic, lexical)

    out = _store_search()
    assert _ids(out) == _ids(semantic)
    assert [c.score for c in out] == [c.score for c in semantic]


# ── guard: the pool's top score equals its top cosine ───────────────────────

@pytest.mark.parametrize("top, bottom", [(0.80, 0.70), (0.52, 0.31)])
def test_pool_top_score_equals_top_cosine(monkeypatch, tmp_path, top, bottom):
    semantic = _vector_leg(30, top=top, bottom=bottom)
    lexical = [_chunk("lx", 0.07, kind="lexical", n=400)] + [
        Chunk(**{**c.__dict__, "score": 0.05}) for c in semantic[20:]
    ]
    _install(monkeypatch, tmp_path, semantic, lexical)

    out = _store_search()
    assert max(c.score for c in out) == pytest.approx(top)


# ── T4: the search tool returns a document only the lexical leg found ───────

@pytest.mark.parametrize("vector_docs", [3, 6])
def test_search_tool_returns_lexical_only_document(monkeypatch, tmp_path, vector_docs):
    from app.core import doc_index

    semantic = _vector_leg(vector_docs * 3, docs=vector_docs)
    lexical = [
        _chunk("a0", 0.06, doc="lexdoc", kind="lexical", n=500),
        _chunk("a1", 0.05, doc="lexdoc", kind="lexical", n=501),
    ]
    _install(monkeypatch, tmp_path, semantic, lexical)

    rows, trace = doc_index._search_project_documents_traced(PID, QUERY, top_k=2)
    docs = [r["document_id"] for r in rows]
    assert "lexdoc" in docs, docs
    assert len(docs) == 2, docs


# ── the fusion on its own ───────────────────────────────────────────────────

@pytest.mark.parametrize("vec_scores, lex_scores", [
    ([0.8, 0.7], [0.05, 0.04, 0.03]),
    ([0.61], [0.09, 0.02]),
])
def test_rrf_keeps_leg_scores_and_fused_rank(vec_scores, lex_scores):
    semantic = [_chunk(f"v{i}", s, n=i) for i, s in enumerate(vec_scores)]
    bm25 = [_chunk(f"l{i}", s, kind="lexical", n=50 + i)
            for i, s in enumerate(lex_scores)]
    out = _rrf_combine(semantic, bm25, top_k=10)
    assert [c.fused_rank for c in out] == list(range(1, len(out) + 1))
    by_id = {c.chunk_id: c for c in out}
    for i, s in enumerate(vec_scores):
        assert by_id[f"v{i}"].vector_score == s
        assert by_id[f"v{i}"].lexical_score is None
    for i, s in enumerate(lex_scores):
        assert by_id[f"l{i}"].lexical_score == s
        assert by_id[f"l{i}"].vector_score is None
    # Past the vector leg's length the last cosine repeats.
    expected = [vec_scores[min(i, len(vec_scores) - 1)] for i in range(len(out))]
    assert [c.score for c in out] == expected


@pytest.mark.parametrize("lex_scores", [[0.06, 0.04], [0.09, 0.08, 0.01]])
def test_empty_vector_leg_leaves_lexical_scores(lex_scores):
    """Out of scope for S-R7: with no vector leg there is no cosine scale,
    so the lexical scores are left as the leg gave them."""
    bm25 = [_chunk(f"l{i}", s, kind="lexical", n=60 + i)
            for i, s in enumerate(lex_scores)]
    out = _rrf_combine([], bm25, top_k=10)
    assert [c.score for c in out] == lex_scores
    assert [c.fused_rank for c in out] == list(range(1, len(lex_scores) + 1))


@pytest.mark.parametrize("field_name, value", [
    ("vector_score", 0.71), ("lexical_score", 0.0612), ("fused_rank", 3),
])
def test_leg_fields_never_reach_the_wire(field_name, value):
    c = _chunk("w", 0.5)
    setattr(c, field_name, value)
    assert field_name not in c.to_dict()


@pytest.mark.parametrize("vector_score, lexical_score, fused_rank", [
    (0.71, None, 3),
    (None, 0.0612, 1),
])
def test_trace_lines_carry_leg_scores_and_fused_rank(vector_score, lexical_score,
                                                     fused_rank):
    from app.core.rag import inject

    c = _chunk("t", 0.75)
    c.source_name = "invented.pdf"
    c.vector_score, c.lexical_score, c.fused_rank = (
        vector_score, lexical_score, fused_rank,
    )
    want = {
        "vector_score": vector_score, "lexical_score": lexical_score,
        "fused_rank": fused_rank,
    }
    for entry in (inject._trace_chunk(2, c), inject.trace_chunk_ids(2, c)):
        assert {k: entry[k] for k in want} == want
        assert entry["score"] == 0.75
        assert "text" not in entry
