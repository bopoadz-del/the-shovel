#!/usr/bin/env python3
"""Regenerate the training dataset as RAG-grounded examples.

For each row in the source JSONL we run the instruction through the same
``retrieve_with_filter`` + 0.4 threshold pipeline used by the live
runtime, and emit a new row that carries the retrieved context alongside
the original instruction / response. The model then trains on
``(instruction, context) -> response`` instead of ``instruction ->
response``, so it learns to reason over retrieved evidence rather than
recall facts from its weights.

Rows where retrieval scores below the threshold (or returns nothing)
are kept but tagged ``source = no_rag_hit:<original_source>`` and given
an empty context string — preserving dataset size so the training
distribution doesn't collapse to only well-retrieved facts.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")

# Path tweak so we can run from the repo root via .venv\Scripts\python.exe
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.rag.retriever import retrieve_with_filter  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT = "globalkb"
DEFAULT_THRESHOLD = 0.4
DEFAULT_TOP_K = 3


def _format_context(chunks) -> str:
    """Concatenate retrieved chunks the same way the live inject layer
    does — doc_id + chunk index + score tag, then the chunk text. Keeps
    the format stable so the trained model sees the same surface
    structure at training time and at inference."""
    parts = []
    for c in chunks:
        score = c.score or 0.0
        parts.append(
            f"[doc_id={c.doc_id} chunk={c.chunk_index} score={score:.3f}]\n{c.text.strip()}"
        )
    return "\n\n".join(parts)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/learning/training_scenarios_merged.jsonl"))
    parser.add_argument("--out", type=Path,
                        default=Path("data/learning/training_scenarios_rag_grounded.jsonl"))
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = [json.loads(l) for l in args.source.open(encoding="utf-8") if l.strip()]
    logger.info("loaded %d rows from %s", len(rows), args.source)

    grounded: List[dict] = []
    chunk_doc_hits = collections.Counter()
    hits = 0
    misses = 0
    skipped_thresh = 0

    for i, r in enumerate(rows):
        instr = (r.get("instruction") or "").strip()
        resp = (r.get("response") or "").strip()
        src = r.get("source", "?")
        if not instr or not resp:
            continue

        chunks, _noise = retrieve_with_filter(instr, PROJECT, k=args.k)
        if chunks:
            top_score = max(c.score or 0 for c in chunks)
        else:
            top_score = 0.0

        if not chunks or top_score < args.threshold:
            grounded.append({
                "instruction": instr,
                "context": "",
                "response": resp,
                "source": f"no_rag_hit:{src}",
                "source_detail": r.get("source_detail", ""),
            })
            if not chunks:
                misses += 1
            else:
                skipped_thresh += 1
            continue

        # Cap retrieved chunks to the requested K (the live inject would
        # also run a token cap; for training rows we trust K is bounded
        # enough that token caps would never fire on three chunks).
        kept = chunks[: args.k]
        for c in kept:
            chunk_doc_hits[c.doc_id] += 1

        grounded.append({
            "instruction": instr,
            "context": _format_context(kept),
            "response": resp,
            "source": src,
            "source_detail": r.get("source_detail", ""),
        })
        hits += 1

        if (i + 1) % 200 == 0:
            logger.info("  ... %d / %d (%d hits, %d misses)",
                        i + 1, len(rows), hits, misses + skipped_thresh)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in grounded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print()
    print("=" * 60)
    print("REGENERATION SUMMARY")
    print("=" * 60)
    print(f"Total rows in:               {len(rows)}")
    print(f"Total rows out:              {len(grounded)}")
    print(f"  RAG hits (>= threshold):   {hits}")
    print(f"  Sub-threshold (kept):      {skipped_thresh}")
    print(f"  No retrieval (kept):       {misses}")
    print(f"  Hit rate:                  {hits/max(1,len(grounded))*100:.1f}%")
    print()
    print("Top docs by chunk-hit frequency:")
    for doc, n in chunk_doc_hits.most_common(5):
        print(f"  {doc:<40} {n}")
    print()
    print(f"Output: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
