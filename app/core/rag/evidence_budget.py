"""Token budget for the agentic figure-retrieval evidence loop.

``rag_inject`` caps the first pass (``MAX_RAG_TOKENS``), but the evidence
loop that augments an uncovered project-figure / priced-BOQ ask merges
further retrieved chunks into the prompt with NO bound. A priced bill whose
first pass missed injected the whole spreadsheet (hundreds of chunks) and
exceeded the model's context window (HTTP 400). This module bounds that
merge generally: a configurable total cap and a per-document cap, keeping the
highest-scoring chunks whole (never truncated mid-chunk) and recording a
``context_truncated`` provenance note when rows are dropped.

Nothing here names a project, an item or a column.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

#: Provenance marker appended to the merged evidence when the budget drops
#: rows. General wording: the answer layer must say the table was sampled,
#: never that a value or item is absent.
CONTEXT_TRUNCATED_NOTE = (
    "\n[context_truncated: the retrieved rows exceeded the injection budget "
    "and were sampled. The highest-scoring rows plus the sheet/bill headers "
    "were kept and the rest dropped. State that the table was sampled; do "
    "not state that a value, item or section is absent because of this.]"
)


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw)
    except ValueError:
        # Unset, empty, whitespace-only and non-numeric all mean "default".
        return default
    if value <= 0:
        return default
    return value


def max_total_tokens() -> int:
    """Total injected-token budget for the evidence-loop merge (configurable)."""
    return _env_int("AGENTIC_EVIDENCE_MAX_TOKENS", 120_000)


def max_tokens_per_doc() -> int:
    """Per-document injected-token budget for the evidence-loop merge."""
    return _env_int("AGENTIC_EVIDENCE_MAX_TOKENS_PER_DOC", 60_000)


def estimate_tokens(text: str) -> int:
    """Cheap 4-chars-per-token proxy (matches ``inject._estimate_tokens``)."""
    return max(1, len(text or "") // 4)


def _chunk_score(chunk: Any) -> float:
    raw = getattr(chunk, "score", None)
    if raw is None or isinstance(raw, bool):
        return 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    # A non-numeric score (a stray string) is unranked, never a crash.
    return 0.0


def _chunk_tokens(chunk: Any) -> int:
    return estimate_tokens(getattr(chunk, "text", "") or "")


def _chunk_doc_id(chunk: Any) -> str:
    return getattr(chunk, "doc_id", "") or ""


def cap_chunks(
    chunks: Sequence[Any],
    *,
    total_tokens: int | None = None,
    per_doc_tokens: int | None = None,
) -> tuple[list[Any], int]:
    """Keep the highest-scoring chunks, whole, under per-doc + total caps.

    Returns ``(kept, dropped_count)``. ``kept`` is in descending score order.
    A chunk is included or excluded whole; every chunk of a single document is
    bounded by ``per_doc_tokens`` while the union is bounded by
    ``total_tokens``. Caps default to the module's configured env values.
    """
    total = max_total_tokens() if total_tokens is None else int(total_tokens)
    per_doc = (
        max_tokens_per_doc() if per_doc_tokens is None else int(per_doc_tokens)
    )
    ordered = sorted(chunks, key=_chunk_score, reverse=True)
    per_doc_used: dict = {}
    kept: list[Any] = []
    used = 0
    for chunk in ordered:
        cost = _chunk_tokens(chunk)
        doc_id = _chunk_doc_id(chunk)
        if per_doc_used.get(doc_id, 0) + cost > per_doc:
            continue
        if used + cost > total:
            continue
        per_doc_used[doc_id] = per_doc_used.get(doc_id, 0) + cost
        used += cost
        kept.append(chunk)
    return kept, len(chunks) - len(kept)


def cap_evidence_text(
    text: str,
    *,
    max_tokens: int | None = None,
) -> tuple[str, bool]:
    """Bound a merged evidence string to ``max_tokens``, keeping the front.

    The evidence loop concatenates in relevance order (first pass, then round
    1, 2, 3), so keeping the front keeps the most relevant content. Returns
    ``(text, truncated)``; when truncated the caller appends
    :data:`CONTEXT_TRUNCATED_NOTE`. This is a whole-turn safety net on top of
    the chunk-level budget, not a replacement for it.
    """
    cap = max_total_tokens() if max_tokens is None else int(max_tokens)
    if estimate_tokens(text or "") <= cap:
        return text, False
    # Keep the front under the cap, leaving room for the truncation note.
    budget = max(1, cap)
    kept = (text or "")[: budget * 4]
    return kept, True
