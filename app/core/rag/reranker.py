"""Cross-encoder reranker — flag-gated second-stage ranking (RAG_RERANKER).

STATUS: DORMANT BY VERDICT — ships flag-OFF and must stay off with the
default model. The offline proof (2026-08-02, scripts/rerank_offline_proof.py
against the live master-corpus top-50) measured ms-marco-MiniLM reranking as
NEGATIVE: doc-recall@5 fell 47% -> 43%. MS-MARCO web-passage training
transfers badly to OCR-flavoured construction chunks. Full decision record +
enable-later checklist: docs/rag-reranker.md.

WHY IT EXISTS ANYWAY. The 2026-08-01 live measurement put doc-recall at 41%
@5 but 53% @50 — part of the gap is documents ranked 6th-50th, which is a
second-stage-ranking shape. The hook is model-agnostic: a DOMAIN-suitable
cross-encoder can be trialled later purely by env change (repo precedent for
shipping tested-but-dormant capability: layered RAG). Prove any candidate
offline against live top-50 FIRST; enable only on a measured lift.

HOW. ``rerank(query, chunks, k)`` scores each (query, chunk.text) pair with
a small cross-encoder and returns the top-k chunks by that score. The
retriever calls it AFTER every existing gate (noise filter, revision
suppression, GK cap) has already shaped the candidate pool — the reranker
only reorders survivors, it can never resurrect a suppressed or filtered
chunk.

OPS. Everything is env-gated and fails open:

  RERANK_ENABLED=false           canonical flag (default OFF — flag off is
                                 byte-identical to today's pipeline).
                                 Truthy values: 1 / true / yes / on.
  RAG_RERANKER=1                 alias kept for existing tests / .env files;
                                 ignored when RERANK_ENABLED is set.
  RAG_RERANKER_MODEL             cross-encoder checkpoint (default
                                 cross-encoder/ms-marco-MiniLM-L-6-v2,
                                 ~90MB — sized for the 512Mi web box)
  RAG_RERANK_CANDIDATES          candidate pool depth (default 50 — hybrid
                                 top-50, then cross-encoder picks top-k)

The model is lazy-loaded on first use (a process-wide singleton, cached
under HF_HOME like the embedder) so a flag-off boot pays zero memory. Any
failure — model missing, sentence-transformers absent, scoring error —
logs a warning and returns the original ordering: the reranker may only
ever improve on cosine order, never take the pipeline down with it.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATES = 50
_TRUTHY = {"1", "true", "yes", "on"}

# Chunk text is bounded per pair so one pathological chunk cannot blow the
# cross-encoder's input budget (the model truncates at 512 tokens anyway;
# this just avoids tokenizing megabyte strings to find that out).
_PAIR_TEXT_LIMIT = 2000

_lock = threading.Lock()
_model: object | None = None
_model_name: str | None = None
_load_failed = False


def enabled() -> bool:
    """True when RERANK_ENABLED (canonical) or RAG_RERANKER is truthy.

    Default OFF. ``RERANK_ENABLED`` wins when both are set so an operator
    can pin the canonical flag without leftover alias values flipping it.
    """
    canonical = os.getenv("RERANK_ENABLED")
    if canonical is not None and canonical.strip() != "":
        return canonical.strip().lower() in _TRUTHY
    return os.getenv("RAG_RERANKER", "").strip().lower() in _TRUTHY


def candidate_depth(k: int) -> int:
    """Candidate pool size the retriever should collect before reranking.

    Never below ``k`` — reranking fewer candidates than the caller asked to
    receive would silently shrink results.
    """
    try:
        n = int(os.getenv("RAG_RERANK_CANDIDATES", "") or DEFAULT_CANDIDATES)
    except ValueError:
        n = DEFAULT_CANDIDATES
    return max(k, n)


def _get_model() -> object | None:
    """Lazy process-wide cross-encoder singleton; None when unavailable.

    A failed load is remembered so a broken environment logs once and then
    degrades silently instead of re-attempting (and re-failing) per query.
    """
    global _model, _model_name, _load_failed
    name = os.getenv("RAG_RERANKER_MODEL", "").strip() or DEFAULT_MODEL
    with _lock:
        if _model is not None and _model_name == name:
            return _model
        if _load_failed and _model_name == name:
            return None
        try:
            from sentence_transformers import CrossEncoder
            logger.info("reranker: loading cross-encoder %s", name)
            _model = CrossEncoder(name, max_length=512)
            _model_name = name
            _load_failed = False
            return _model
        except Exception as exc:  # noqa: BLE001 — degrade, never break retrieval
            logger.warning("reranker: cross-encoder load failed (%s); "
                           "reranking disabled this process: %s", name, exc)
            _model = None
            _model_name = name
            _load_failed = True
            return None


def rerank(query: str, chunks: Sequence, k: int) -> list:
    """Return the top-``k`` of ``chunks`` by cross-encoder relevance to ``query``.

    Chunks are whatever the retriever passes (objects with a ``.text``
    attribute); they are returned unmodified, only reordered and truncated.
    On ANY failure the original order's first ``k`` is returned — the
    reranker is an upgrade path, never a failure path.
    """
    chunks = list(chunks)
    if len(chunks) <= 1 or k <= 0:
        return chunks[:k]
    model = _get_model()
    if model is None:
        return chunks[:k]
    try:
        pairs = [(query, (getattr(c, "text", "") or "")[:_PAIR_TEXT_LIMIT])
                 for c in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        order = sorted(range(len(chunks)), key=lambda i: -float(scores[i]))
        return [chunks[i] for i in order[:k]]
    except Exception as exc:  # noqa: BLE001 — degrade, never break retrieval
        logger.warning("reranker: scoring failed, keeping original order: %s", exc)
        return chunks[:k]
