"""The injection token cap must fit a full top-k of TODAY'S chunks.

Live-fire campaign phase 3, F20. After the dual-query fix put the tender
needle at rank 3 of 5 (score 0.580), live chat still answered "the context
does not contain" -- truthfully, because ``apply_token_cap`` dropped it.
Simulated against the exact live candidate set:

    1. score=0.581  est_tok=793  cum=793   KEPT
    2. score=0.581  est_tok=737  cum=1530  DROPPED (cap 1500)
    3. score=0.580  est_tok=862  DROPPED   <-- the needle
    4. score=0.579  est_tok=808  DROPPED
    5. score=0.571  est_tok=940  DROPPED

The 1500 default was calibrated for the 512-char chunker; the doc-reindex
path emits ~3,000-char chunks, so the cap admitted exactly ONE chunk and
made retrieval quality irrelevant. The default now fits a full k=5 of
current-size chunks; the cap still bounds a runaway injection and the env
override still wins.
"""
from __future__ import annotations

from app.core.rag.inject import apply_token_cap
from app.core.rag.vector_store import Chunk


def _chunk(i, chars, score):
    return Chunk(chunk_id=f"c{i}", project_id="p", doc_id="d1",
                 chunk_index=i, text="x" * chars, score=score)


def _live_shaped_set():
    """Five chunks with the exact char sizes measured live 2026-08-15."""
    sizes = [3174, 2951, 3448, 3232, 3763]
    return [_chunk(i, sz, 0.581 - i * 0.002) for i, sz in enumerate(sizes)]


def test_a_full_topk_of_current_size_chunks_survives_the_default_cap(monkeypatch):
    """RED before the fix: the default kept exactly one of these five."""
    monkeypatch.delenv("MAX_RAG_TOKENS", raising=False)
    kept, _total = apply_token_cap(_live_shaped_set())
    assert len(kept) == 5, f"cap dropped {5 - len(kept)} of 5 current-size chunks"


def test_the_rank3_needle_specifically_is_not_dropped(monkeypatch):
    """The live failure: rank 3 held the answer and was excluded whole."""
    monkeypatch.delenv("MAX_RAG_TOKENS", raising=False)
    chunks = _live_shaped_set()
    kept, _ = apply_token_cap(chunks)
    assert chunks[2].chunk_id in {c.chunk_id for c in kept}


def test_the_cap_still_bounds_a_runaway_injection(monkeypatch):
    monkeypatch.delenv("MAX_RAG_TOKENS", raising=False)
    many = [_chunk(i, 3200, 0.9 - i * 0.01) for i in range(40)]
    kept, total = apply_token_cap(many)
    assert len(kept) < 40
    assert total <= 6000


def test_the_env_override_still_wins(monkeypatch):
    monkeypatch.setenv("MAX_RAG_TOKENS", "800")
    kept, total = apply_token_cap(_live_shaped_set())
    assert len(kept) == 1
    assert total <= 800
