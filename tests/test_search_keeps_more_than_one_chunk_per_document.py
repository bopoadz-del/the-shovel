"""A specification split into nine parts must not contribute one chunk each.

Live SET5 P1a, measured 6 times on dd4323e: **1/6**. Five runs answered "I
don't have that in the retrieved excerpts" about a clause the sixth run quoted
in full -- "compaction of the backfill to minimum 98% of maximum dry density of
the modified proctor test". The content is in the corpus and was reachable
about one time in six.

Two limits in ``_search_project_documents_sync`` compounded to cause it:

1. results were collapsed to the single best chunk per document (``best`` was
   keyed by ``doc_id``), with a comment assuming "each doc has 2-10 chunks".
   The specification here is nine PDFs of hundreds of chunks each -- the
   failing run cited chunk 1242, chunk 1245, chunk 689, one per part -- so a
   clause anywhere else in the same file was discarded whenever a neighbouring
   chunk scored marginally higher;
2. each surviving chunk was then cut to its first 50 words.

Five documents times fifty words is the whole pipe into the model for a 3,354
document corpus.

The retriever already fetches ``max(top_k * 4, 20)`` candidates and the
collapse threw fifteen of them away. This keeps more of what was already paid
for: the over-fetch is deliberately unchanged, so the reranker pool contract
in ``test_doc_index_rerank_pool.py`` still holds, and no extra retrieval work
is done.

Document diversity is still preserved -- the result rows are still one per
document, ranked by that document's best chunk.
"""
from __future__ import annotations

import pytest

from app.core import doc_index


class _Chunk:
    def __init__(self, doc_id, text, score):
        self.doc_id = doc_id
        self.text = text
        self.score = score
        self.layer = "own"


# Part 4 holds the answer, but not in its top-scoring chunk.
CLAUSE = ("compaction of the backfill to minimum 98 percent of maximum dry "
          "density of the modified proctor test at near optimum moisture")
CHUNKS = [
    _Chunk("spec2", "cable duct trench soft ground sand cover after tamping " * 6, 0.91),
    _Chunk("spec2", CLAUSE, 0.72),
    _Chunk("spec2", "gradation fines content between 4 and 12 percent", 0.60),
    _Chunk("spec2", "marker tape at 100 mm below finished ground level", 0.55),
    _Chunk("spec3", "embankment compaction testing to the MOT standard " * 6, 0.88),
    _Chunk("spec3", "method of measurement for earthworks", 0.50),
]


@pytest.fixture
def fake_corpus(monkeypatch):
    import app.core.rag.retriever as retr
    monkeypatch.setattr(retr, "retrieve_with_filter",
                        lambda query, project_id, k=5, **_kw: (list(CHUNKS), 0))
    monkeypatch.setattr(retr, "_doc_name_for_id", lambda d: f"Specification ({d}).pdf")
    monkeypatch.setattr(doc_index, "_load_index", lambda _pid: {"documents": []})


def _snippets(results):
    return " ".join(r["snippet"] for r in results)


# ── the live defect ────────────────────────────────────────────────────────

async def test_the_clause_survives_a_higher_scoring_neighbour(fake_corpus):
    results = await doc_index.search_project_documents("p1", "backfill compaction", top_k=5)
    assert "modified proctor" in _snippets(results).lower(), (
        "the clause was discarded by the one-chunk-per-document collapse")


async def test_a_clause_past_the_fiftieth_word_is_not_cut_off(fake_corpus, monkeypatch):
    long_lead = "preliminary wording " * 40  # 80 words before the figure
    monkeypatch.setattr(
        __import__("app.core.rag.retriever", fromlist=["x"]), "retrieve_with_filter",
        lambda query, project_id, k=5, **_kw: ([_Chunk("spec2", long_lead + CLAUSE, 0.9)], 0))
    results = await doc_index.search_project_documents("p1", "backfill compaction", top_k=5)
    assert "98 percent" in _snippets(results)


# ── the diversity the collapse was there to protect ────────────────────────

async def test_results_are_still_one_row_per_document(fake_corpus):
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    ids = [r["document_id"] for r in results]
    assert len(ids) == len(set(ids)), "a document must not take two result rows"
    assert set(ids) == {"spec2", "spec3"}


async def test_documents_are_still_ranked_by_their_best_chunk(fake_corpus):
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    assert results[0]["document_id"] == "spec2"
    assert results[0]["score"] == pytest.approx(0.91)


async def test_top_k_still_limits_the_documents_returned(fake_corpus):
    results = await doc_index.search_project_documents("p1", "backfill", top_k=1)
    assert len(results) == 1


# ── bounded, and switchable ────────────────────────────────────────────────

async def test_the_chunks_per_document_are_capped(fake_corpus, monkeypatch):
    monkeypatch.setenv("SEARCH_CHUNKS_PER_DOC", "2")
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    spec2 = next(r for r in results if r["document_id"] == "spec2")
    # Two kept of the four available: the duct text and the clause.
    assert "4 and 12 percent" not in spec2["snippet"]
    assert "modified proctor" in spec2["snippet"].lower()


async def test_one_per_document_is_restorable(fake_corpus, monkeypatch):
    monkeypatch.setenv("SEARCH_CHUNKS_PER_DOC", "1")
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    spec2 = next(r for r in results if r["document_id"] == "spec2")
    assert "modified proctor" not in spec2["snippet"].lower(), (
        "the kill-switch must reproduce the old behaviour exactly")


async def test_a_nonsense_setting_falls_back_to_the_default(fake_corpus, monkeypatch):
    monkeypatch.setenv("SEARCH_CHUNKS_PER_DOC", "not a number")
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    assert "modified proctor" in _snippets(results).lower()


async def test_the_row_shape_is_unchanged(fake_corpus):
    results = await doc_index.search_project_documents("p1", "backfill", top_k=5)
    for r in results:
        assert set(r) >= {"document_id", "filename", "snippet", "score", "origin"}
        assert isinstance(r["score"], float)
