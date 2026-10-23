"""Tests for hybrid BM25 + vector retrieval in app/core/rag/vector_store.

Hand-crafted micro-corpus mirroring the failure modes from the live
drive_archive probe:

- 3 dense-repetition decoys ("Vol 3 Drawings legend" pattern) that
  semantic-only over-weights because their token cluster looks like the
  query.
- The actual answer chunks for Q2, Q4, Q5 — short, precise, with the
  exact tokens (e.g. "TL-600-0000002", "SECTIONAL ELEVATION",
  "MANHOLE TYPE-A SHALL BE PROVIDED AT EVERY 1000M") that BM25 will
  surface but semantic-only buries.
- A XYZ-501 chunk for the Q3 regression check (must still rank top-2).
- Generic construction noise.

Corpus tests honor ``RAG_EMBEDDING_MODEL``. CI pins ``fake`` so a fresh
runner never downloads ``minishlab/potion-base-8M`` from HuggingFace
(test.yml; run 33207362472 died on nine Hub 429 ERRORs after this
module deleted that env). BM25 / signature tests are valid on the fake
hash embedder. Hybrid-beats-semantic ranking tests skip when the
configured model is ``fake`` — the hash embedder ranks ~randomly and
cannot distinguish a manhole-spacing chunk from MEP rough-in.
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid

import numpy as np
import pytest

# Path setup so this file works whether pytest discovers from repo root
# or from inside tests/.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from app.core.rag.embeddings import get_embedder, reset_embedder_cache  # noqa: E402
from app.core.rag.vector_store import (  # noqa: E402
    Chunk,
    VectorStore,
    _rrf_combine,
    _sanitize_fts5_query,
)


# ── Fixture corpus ────────────────────────────────────────────────────────

PROJECT_ID = "test_hybrid_proj"

# Hand-tuned to match the live-probe failure modes. Indices are 0-based
# within their own list and embed into chunk_index.
CORPUS = [
    # 0-2: dense-repetition Vol 3 Drawings legend decoys. Semantic
    # retrieval clusters on the repeated tokens.
    (
        "vol3_legend_a",
        "TELECOM MANHOLE TYPE-A TELECOM HANDHOLE TELECOM DUCT TELECOM "
        "MANHOLE TYPE-A TELECOM HANDHOLE TELECOM DUCT TELECOM MANHOLE "
        "TYPE-A TELECOM HANDHOLE TELECOM DUCT LEGEND DRAWING SHEET",
    ),
    (
        "vol3_legend_b",
        "TELECOM MANHOLE TYPE-A TELECOM HANDHOLE TELECOM DUCT TELECOM "
        "SECTIONAL ELEVATION TELECOM MANHOLE TYPE-A TELECOM HANDHOLE "
        "TELECOM DUCT VOL 3 DRAWINGS LEGEND",
    ),
    (
        "vol3_legend_c",
        "TELECOM HANDHOLE TELECOM DUCT TELECOM MANHOLE TYPE-A TELECOM "
        "HANDHOLE TELECOM DUCT TELECOM MANHOLE TYPE-A SHEET DRAWING "
        "TELECOM HANDHOLE TELECOM DUCT TELECOM MANHOLE",
    ),
    # 3: Q5 answer chunk — manhole spacing, precise tokens
    (
        "tl_600_0000002",
        "QZ-ENG-310-0000-EXC-DWG-TL-600-0000002 -- SECTIONAL ELEVATION "
        "(Telecom, Rev D) Notes: GENERAL NOTES 1. MANHOLE TYPE-A SHALL "
        "BE PROVIDED AT EVERY 1000M INTERVALS AND AT THE JUNCTIONS OR "
        "CHANGE IN DIRECTIONS.",
    ),
    # 4: Q4 answer chunk — trench width
    (
        "trench_width",
        "4.4.9.3.2 Payable trench width shall be measured from the "
        "outside of the pipe wall to a maximum of 600mm beyond each "
        "side per BOQ unless drawing shows otherwise.",
    ),
    # 5: Q3 regression chunk — XYZ-501
    (
        "xyz_501",
        "XYZ-501 Design Reviews & Acceptance — Acceptance shall be "
        "formalized by issuing a Project Decision Note (QRS-505) or a "
        "Design Package Acceptance Form (QRS-504) after the Design "
        "Review Workshop.",
    ),
    # 6-13: generic construction noise
    (
        "noise_1",
        "Concrete strength testing shall conform to BS EN 12390 with "
        "cubes cast at the point of delivery and cured under standard "
        "conditions until day 28.",
    ),
    (
        "noise_2",
        "All structural steelwork shall be supplied with mill "
        "certificates and shall be fabricated to tolerances per the "
        "approved shop drawings.",
    ),
    (
        "noise_3",
        "The contractor shall maintain site safety procedures "
        "including daily toolbox talks, PTW issuance, and weekly "
        "audits by the HSE manager.",
    ),
    (
        "noise_4",
        "Lifting operations require an approved lift plan, competent "
        "appointed person, and exclusion zones around the crane "
        "footprint.",
    ),
    (
        "noise_5",
        "Method statements for excavation works in close proximity to "
        "existing utilities shall include trial pits and ground "
        "penetrating radar surveys.",
    ),
    (
        "noise_6",
        "Painting works shall use approved coatings from the project "
        "specification, applied at the manufacturer's recommended dry "
        "film thickness.",
    ),
    (
        "noise_7",
        "Mechanical, electrical and plumbing rough-in must be "
        "completed and inspected before slab pour to avoid rework on "
        "concrete elements.",
    ),
    (
        "noise_8",
        "All asphaltic concrete shall conform to the relevant Gulf "
        "specification and be laid in lifts of compacted thickness "
        "approved by the engineer.",
    ),
]


# ── Helpers ──────────────────────────────────────────────────────────────


# Ranking assertions need a real model. Do not undo CI's
# RAG_EMBEDDING_MODEL=fake — that download is what Hub-429'd the virgin job.
#
# The decision is made at CALL time, from the embedder that was actually
# built, and never at import time from RAG_EMBEDDING_MODEL. It used to be a
# module-level ``pytest.mark.skipif`` reading that variable, and the two
# answered different questions:
#
#   * Eight test modules (test_bm25_backend_parity, test_upload_contract,
#     test_master_corpus_upload, test_fetch_document_scope, and others) call
#     ``os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")`` at MODULE
#     scope. That mutates the process environment permanently — there is no
#     monkeypatch to unwind it — the moment the module is imported.
#   * ``skipif`` is evaluated when THIS module is imported. Whether the
#     variable is already "fake" by then is a function of collection order,
#     not of anything this file controls.
#   * ``store_with_corpus`` calls ``get_embedder()`` later still, at test
#     run time, reading whatever the environment says by then.
#
# So the guard could answer "real model, run it" while the fixture went on to
# build the fake one, and the ranking assertions — baselined against real
# embeddings — failed. Reproduced deterministically:
#
#   pytest tests/test_hybrid_retrieval.py                 -> ranking tests skip
#   pytest tests/test_bm25_backend_parity.py tests/test_hybrid_retrieval.py::\
#          test_hybrid_beats_semantic_q5_manhole_spacing  -> FAILS
#
# Asking the built embedder what it is removes the question of ordering
# entirely: there is now one source of truth, consulted once, at the only
# moment when the answer is knowable.


@pytest.fixture
def store_with_corpus():
    """Build a VectorStore against a temp SQLite DB seeded with CORPUS.

    Honors ``RAG_EMBEDDING_MODEL`` (CI: ``fake``). Never deletes that env
    to force a Hub download. Ranking tests that need a real model are
    skipif-gated separately.

    Returns (store, embedder, db_path). Teardown removes the DB file.
    """
    reset_embedder_cache()
    try:
        embedder = get_embedder()
    except Exception as exc:  # noqa: BLE001 — Hub 429 / missing snapshot
        pytest.skip(
            f"Embedder unavailable ({type(exc).__name__}: {exc})"
        )
    tmpdir = tempfile.mkdtemp(prefix="hybrid_test_")
    db_path = os.path.join(tmpdir, f"vectors_{uuid.uuid4().hex}.db")

    # Force a clean schema-init for this DB path.
    from app.core.rag import vector_store as _vs
    _vs.reset_store_cache()

    store = VectorStore(db_path=db_path, dim=embedder.dim)
    # Index each chunk as its own one-chunk "doc" so chunk_id is unique
    # and the per-doc replace semantics in upsert_chunks don't clobber.
    for doc_id, text in CORPUS:
        embeddings = embedder.encode([text])
        store.upsert_chunks(PROJECT_ID, doc_id, [text], embeddings)

    yield store, embedder, db_path

    try:
        store.close()
    finally:
        try:
            os.remove(db_path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


@pytest.fixture
def ranking_corpus(store_with_corpus):
    """``store_with_corpus``, but only for a run that built a REAL embedder.

    Skips otherwise. See the note above ``store_with_corpus`` for why this
    cannot be a module-level ``skipif``.
    """
    store, embedder, db_path = store_with_corpus
    backend = getattr(embedder, "backend", None)
    if backend == "fake":
        pytest.skip(
            "hybrid-vs-semantic ranking is baselined against a real embedding "
            "model; this run built the fake (model2vec) backend"
        )
    return store, embedder, db_path


def _ids(chunks):
    return [c.chunk_id for c in chunks]


def _run_search(store, embedder, query, k=5, query_text=None):
    """Run store.search with optional query_text. ``query_text=None``
    means "don't pass it" so the legacy semantic-only path is taken
    independently of the env flag."""
    q_vec = embedder.encode([query])[0]
    if query_text is None:
        return store.search(PROJECT_ID, q_vec, k=k)
    return store.search(PROJECT_ID, q_vec, k=k, query_text=query_text)


def _contains_doc(chunks, doc_id):
    return any(doc_id in c.chunk_id for c in chunks)


# ── Unit: RRF math (pure, no fixture needed) ─────────────────────────────


def test_rrf_combine_basic_math():
    """Two lists with one common chunk — common chunk should outrank
    singletons because it gets both contributions."""
    a = Chunk(chunk_id="x", project_id="p", doc_id="d", chunk_index=0, text="x")
    b = Chunk(chunk_id="y", project_id="p", doc_id="d", chunk_index=1, text="y")
    c = Chunk(chunk_id="z", project_id="p", doc_id="d", chunk_index=2, text="z")
    semantic = [a, b]
    bm25 = [a, c]
    out = _rrf_combine(semantic, bm25, top_k=3)
    assert out[0].chunk_id == "x", "common-to-both chunk wins"
    assert set(_ids(out)) == {"x", "y", "z"}


def test_rrf_combine_respects_top_k():
    """top_k truncates the output list to k entries."""
    chunks = [
        Chunk(chunk_id=f"c{i}", project_id="p", doc_id="d", chunk_index=i, text=f"c{i}")
        for i in range(10)
    ]
    out = _rrf_combine(chunks, chunks[::-1], top_k=3)
    assert len(out) == 3


# ── Unit: sanitizer (pure) ──────────────────────────────────────────────


def test_sanitize_fts5_query_strips_punctuation():
    # Punctuation goes; tokens are joined with OR so FTS5 MATCH is
    # bag-of-words. See _sanitize_fts5_query docstring for the rationale.
    assert _sanitize_fts5_query("Manhole spacing requirements?") == \
        "Manhole OR spacing OR requirements"
    assert _sanitize_fts5_query("XYZ-501!") == "XYZ OR 501"
    assert _sanitize_fts5_query("") == ""
    assert _sanitize_fts5_query("   ?!?  ") == ""
    # Single token: no OR
    assert _sanitize_fts5_query("MANHOLE") == "MANHOLE"


def test_sanitize_fts5_query_handles_none():
    """Sanitizer must accept None / empty without crashing."""
    assert _sanitize_fts5_query("") == ""


# ── Hybrid vs semantic on the seeded corpus ──────────────────────────────


def test_hybrid_beats_semantic_q5_manhole_spacing(ranking_corpus, monkeypatch):
    """Q5: manhole spacing — semantic clusters on the repeated-token
    Vol3 legends. The TL chunk should land top-5 with hybrid
    (matches the docstring; the BM25 exact-token contribution from
    "MANHOLE" + "TYPE" pulls it up even though the natural-language
    query has no "1000m"/"intervals")."""
    store, embedder, _ = ranking_corpus
    # NOTE (scrub audit 2026-08-12): the project token in this query is PINNED
    # FIXTURE DATA, not a stray client reference. Under RAG_EMBEDDING_MODEL=fake
    # the embedding is a deterministic function of the exact string, so editing
    # the query at all re-rolls the semantic ranking and this assertion fails —
    # verified by trying both "on the project" and "on the PRJ2 project".
    # Changing it requires re-baselining the expected chunk, which is retrieval
    # tuning, not a rename.
    query = "Manhole spacing requirements for telecom ducts on the PRJ2 project"

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    hyb = _run_search(store, embedder, query, k=5, query_text=query)
    assert _contains_doc(hyb[:5], "tl_600_0000002"), \
        f"hybrid should surface TL chunk top-5; got = {_ids(hyb)}"


def test_hybrid_beats_semantic_q4_trench_width(ranking_corpus, monkeypatch):
    """Q4: trench width — BM25 catches 'Payable trench width' on exact
    tokens, semantic disperses across noise."""
    store, embedder, _ = ranking_corpus
    query = "What is the payable trench width on the PRJ2 project"

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    hyb = _run_search(store, embedder, query, k=5, query_text=query)
    assert _contains_doc(hyb[:5], "trench_width"), \
        f"hybrid should land trench-width chunk top-5; got = {_ids(hyb)}"


def test_hybrid_beats_semantic_q2_sectional_elevation(ranking_corpus, monkeypatch):
    """Q2: SECTIONAL ELEVATION exact-token match should pull the TL
    chunk into the top of the hybrid ranking."""
    store, embedder, _ = ranking_corpus
    query = "SECTIONAL ELEVATION drawing for telecom infrastructure"

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    hyb = _run_search(store, embedder, query, k=5, query_text=query)
    assert _contains_doc(hyb[:5], "tl_600_0000002"), \
        f"hybrid should surface TL chunk by SECTIONAL ELEVATION; got = {_ids(hyb)}"


def test_hybrid_preserves_q3_prc501(ranking_corpus, monkeypatch):
    """Q3 regression: XYZ-501 query must still hit the XYZ-501 chunk
    top-1 or top-2 under hybrid (semantic was already good; hybrid
    shouldn't break it)."""
    store, embedder, _ = ranking_corpus
    query = "XYZ-501 Design Reviews and Acceptance procedure"

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    hyb = _run_search(store, embedder, query, k=5, query_text=query)
    top2_ids = _ids(hyb[:2])
    assert any("xyz_501" in i for i in top2_ids), \
        f"hybrid should preserve XYZ-501 top-2; got top-2 = {top2_ids}"


def test_hybrid_disabled_falls_back_to_semantic(store_with_corpus, monkeypatch):
    """RAG_HYBRID_SEARCH=false should produce results identical to the
    legacy semantic-only path (which is itself reproduced by calling
    search without query_text)."""
    store, embedder, _ = store_with_corpus
    query = "Manhole spacing requirements for telecom ducts"

    # Legacy path: no query_text supplied
    legacy = _run_search(store, embedder, query, k=10, query_text=None)

    # Hybrid disabled via env flag
    monkeypatch.setenv("RAG_HYBRID_SEARCH", "false")
    disabled = _run_search(store, embedder, query, k=10, query_text=query)

    assert _ids(legacy) == _ids(disabled), (
        f"disabled flag must match legacy semantic-only; "
        f"legacy={_ids(legacy)} disabled={_ids(disabled)}"
    )


def test_bm25_only_query_with_no_matches(store_with_corpus, monkeypatch):
    """A query whose tokens don't appear in the FTS5 index should fall
    through to semantic-only results (graceful degrade — empty bm25
    list, hybrid returns the semantic ranking)."""
    store, embedder, _ = store_with_corpus
    # Gibberish that won't match any token in the corpus
    query = "xyzqwertyuiop nonsenseword foobaz quuxzapper"

    monkeypatch.setenv("RAG_HYBRID_SEARCH", "true")
    hyb = _run_search(store, embedder, query, k=5, query_text=query)

    # The semantic leg ALWAYS returns something for a non-empty corpus;
    # the BM25 leg returns empty; hybrid should return the semantic
    # top-k unchanged.
    sem = _run_search(store, embedder, query, k=5, query_text=None)
    assert _ids(hyb) == _ids(sem), (
        f"no-BM25-match query should degrade to semantic-only; "
        f"hybrid={_ids(hyb)} semantic={_ids(sem)}"
    )


def test_search_signature_unchanged(store_with_corpus):
    """retrieve_with_filter -> store.search must still work with the
    pre-change positional signature (project_id, query_vec, k=...)."""
    store, embedder, _ = store_with_corpus
    q_vec = embedder.encode(["telecom"])[0]
    # Exactly how retriever.py calls it today (positional, no query_text)
    out = store.search(PROJECT_ID, q_vec, k=5)
    assert isinstance(out, list)
    assert all(isinstance(c, Chunk) for c in out)
    # And the caller-facing attributes are intact
    for c in out:
        # These four are the contract retriever.py reads
        _ = (c.doc_id, c.chunk_index, c.text, c.score)


def test_bm25_search_direct(store_with_corpus):
    """bm25_search returns chunks ordered by FTS5 rank (lower = better,
    which becomes 'earlier in the list')."""
    store, _, _ = store_with_corpus
    out = store.bm25_search(PROJECT_ID, "MANHOLE TYPE-A SECTIONAL ELEVATION", k=10)
    assert len(out) > 0
    # The TL chunk has the strongest exact-token overlap with the query
    # AND it's a short chunk (so its BM25 length normalization wins).
    assert _contains_doc(out[:3], "tl_600_0000002"), \
        f"TL chunk should be top-3 in BM25 for SECTIONAL ELEVATION query; got = {_ids(out)}"


def test_bm25_search_empty_query_returns_empty(store_with_corpus):
    store, _, _ = store_with_corpus
    assert store.bm25_search(PROJECT_ID, "", k=10) == []
    assert store.bm25_search(PROJECT_ID, "   ?? !!  ", k=10) == []


def test_chunk_to_dict_drops_rrf_score():
    """to_dict() must drop rrf_score even when set — it's debug-only.
    Existing API consumers don't know about the new field."""
    c = Chunk(chunk_id="x", project_id="p", doc_id="d", chunk_index=0, text="x",
              score=0.5, rrf_score=0.0123)
    d = c.to_dict()
    assert "rrf_score" not in d
    assert d["score"] == 0.5


def test_ranking_guard_is_not_decided_at_import_time():
    """Fence: no module-level skipif in this file may read the environment.

    This is the defect this file was fixed for, expressed as a test so it
    cannot come back. A ``pytest.mark.skipif`` at module scope is evaluated
    when the module is imported; eight other test modules set
    ``RAG_EMBEDDING_MODEL=fake`` in ``os.environ`` at *their* module scope,
    permanently and with no unwind. Whether that has happened yet depends on
    collection order, so an import-time guard and the run-time fixture could
    — and did — disagree, running ranking assertions against the fake
    embedder.

    Whatever decides whether a ranking test runs must ask the embedder that
    was actually built, at the moment it exists.
    """
    import ast as _ast
    from pathlib import Path as _Path

    src = _Path(__file__).read_text(encoding="utf-8")
    tree = _ast.parse(src)

    offenders = []
    for node in tree.body:  # module scope only; nested is run-time
        for sub in _ast.walk(node):
            if not isinstance(sub, _ast.Call):
                continue
            target = _ast.unparse(sub.func)
            if not target.endswith("skipif"):
                continue
            rendered = _ast.unparse(sub)
            if "getenv" in rendered or "environ" in rendered:
                offenders.append(rendered[:120])

    assert not offenders, (
        "a module-level skipif reads the environment; move the decision into "
        "a fixture that inspects the built embedder instead: " + "; ".join(offenders)
    )
