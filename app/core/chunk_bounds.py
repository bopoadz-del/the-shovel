"""A floor and a ceiling on every chunk the indexer writes (RQ-2).

The embedder sees at most a 512-token window (the cross-encoder reranker and
the BERT-family tokenizer the static embedder was distilled from both stop
there). At about four characters per English token that window is ~2,000
characters, so a longer chunk is embedded on its head only and its tail is
invisible to vector search. A very short chunk (a lone heading, one cell)
carries too little text to embed into a useful neighbourhood.

* **ceiling** -- a chunk longer than ``CHUNK_MAX_CHARS`` (default 2000) is
  split at the coarsest boundary that fits: blank line, then line (a
  spreadsheet or bill row), then sentence end, then whitespace. A hard cut
  is the last resort for a single unbroken run.
* **floor** -- a prose chunk shorter than ``CHUNK_MIN_CHARS`` (default 200)
  is joined to its next neighbour (the previous one for the last chunk) when
  the joined text still fits under the ceiling. A short chunk that cannot be
  joined without breaking the ceiling is kept as is: text is never dropped.
  Unit chunks -- one bill or spreadsheet row, one Contract Data particulars
  block, one drawing schedule -- are a deliberate unit of retrieval and get
  the ceiling only (``merge=False``): joining two rows would blur which row
  states the figure.

Both values are read from the environment at call time; ``0`` turns the
rule off. Nothing here reads a document name or id: the rules look only at
the chunk text's own length and boundaries.
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Any

DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_CHARS = 2000

# Coarsest boundary first. Each pattern matches the separator between pieces.
_BOUNDARIES = (
    re.compile(r"\n\s*\n"),          # paragraph
    re.compile(r"\n"),               # line / table row
    re.compile(r"(?<=[.!?;])\s+"),   # sentence end
    re.compile(r"\s+"),              # word
)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, "").strip()))
    except ValueError:
        return default


def chunk_max_chars() -> int:
    """``CHUNK_MAX_CHARS`` (default 2000; 0 = no ceiling)."""
    return _env_int("CHUNK_MAX_CHARS", DEFAULT_MAX_CHARS)


def chunk_min_chars() -> int:
    """``CHUNK_MIN_CHARS`` (default 200; 0 = no floor)."""
    return _env_int("CHUNK_MIN_CHARS", DEFAULT_MIN_CHARS)


def _pack(pieces: list[str], sep: str, ceiling: int) -> list[str]:
    """Greedy: join consecutive pieces with ``sep`` while they fit."""
    out: list[str] = []
    cur = ""
    for piece in pieces:
        if not piece:
            continue
        joined = f"{cur}{sep}{piece}" if cur else piece
        if cur and len(joined) > ceiling:
            out.append(cur)
            cur = piece
        else:
            cur = joined
    if cur:
        out.append(cur)
    return out


def split_oversize(text: str, ceiling: int, _level: int = 0) -> list[str]:
    """``text`` cut into pieces of at most ``ceiling`` chars at natural boundaries."""
    if ceiling <= 0 or len(text) <= ceiling:
        return [text]
    if _level >= len(_BOUNDARIES):
        return [text[i:i + ceiling] for i in range(0, len(text), ceiling)]
    pattern = _BOUNDARIES[_level]
    sep = "\n\n" if _level == 0 else ("\n" if _level == 1 else " ")
    pieces = [p.strip() for p in pattern.split(text) if p and p.strip()]
    if len(pieces) <= 1:
        return split_oversize(text, ceiling, _level + 1)
    out: list[str] = []
    for packed in _pack(pieces, sep, ceiling):
        out.extend(split_oversize(packed, ceiling, _level + 1))
    return out


def merge_undersize(chunks: list[str], floor: int, ceiling: int) -> list[str]:
    """Join each chunk shorter than ``floor`` to a neighbour when it still fits."""
    if floor <= 0:
        return list(chunks)
    out: list[str] = []
    carry = ""
    for chunk in chunks:
        if carry:
            joined = f"{carry}\n{chunk}"
            if ceiling <= 0 or len(joined) <= ceiling:
                chunk = joined
            else:
                out.append(carry)
            carry = ""
        if len(chunk.strip()) < floor:
            carry = chunk
        else:
            out.append(chunk)
    if carry:
        if out and (ceiling <= 0 or len(out[-1]) + 1 + len(carry) <= ceiling):
            out[-1] = f"{out[-1]}\n{carry}"
        else:
            out.append(carry)
    return out


def bound_chunks(
    chunks: Iterable[str],
    stats: dict[str, Any] | None = None,
    *,
    merge: bool = True,
) -> list[str]:
    """Every chunk under ``CHUNK_MAX_CHARS``; with ``merge``, over the floor too.

    The ceiling is applied first so a merge never rebuilds an oversize chunk.
    ``merge=False`` is for unit chunks (rows, particulars, schedules).
    ``stats`` (optional) receives ``split_from`` / ``merged`` counts.
    """
    ceiling = chunk_max_chars()
    floor = chunk_min_chars() if merge else 0
    if ceiling and floor >= ceiling:
        floor = 0  # a floor at or above the ceiling cannot be honoured
    src = [c for c in chunks if c is not None]
    split: list[str] = []
    split_from = 0
    for chunk in src:
        parts = split_oversize(chunk, ceiling)
        if len(parts) > 1:
            split_from += 1
        split.extend(parts)
    out = merge_undersize(split, floor, ceiling)
    if stats is not None:
        stats["split_from"] = stats.get("split_from", 0) + split_from
        stats["merged"] = stats.get("merged", 0) + (len(split) - len(out))
    return out


def bounds_summary(lengths: Iterable[int]) -> dict[str, Any]:
    """Counts of chunk lengths against the configured bounds, plus p50/p95/max."""
    vals = sorted(int(n) for n in lengths)
    ceiling = chunk_max_chars()
    floor = chunk_min_chars()

    def _pct(q: float) -> int | None:
        if not vals:
            return None
        return vals[min(len(vals) - 1, int(q * (len(vals) - 1) + 0.5))]

    return {
        "chunks": len(vals),
        "min_chars": floor,
        "max_chars": ceiling,
        "over_ceiling": sum(1 for v in vals if ceiling and v > ceiling),
        "under_floor": sum(1 for v in vals if floor and v < floor),
        "p50": _pct(0.5),
        "p95": _pct(0.95),
        "max": vals[-1] if vals else None,
    }
