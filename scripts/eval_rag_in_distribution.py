#!/usr/bin/env python3
"""RAG-pipeline in-distribution eval — same 10 questions as
``eval_adapter_in_distribution.py``, but instead of querying a LoRA
adapter directly we:

1. Retrieve top-K chunks for each query from the ``globalkb`` project
   (data/knowledge/*.md indexed via index_chunks).
2. Format the chunks as a system message (same shape as the live
   ``app/core/rag/inject.py`` does for chat_stream).
3. Call the LLM (DeepSeek for local; could swap to Ollama later) with
   the system message + the user query.
4. Score the response with the same substring-match rules as the
   adapter eval, so the two numbers are directly comparable.

The point: does RAG + a capable base LLM beat the best LoRA result on
the same 10 in-distribution questions?
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_adapter_in_distribution import _QUERIES, _score  # noqa: E402

os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")

from app.core.rag.retriever import retrieve_with_filter  # noqa: E402
from app.core.rag.inject import format_chunks_as_system_message, apply_token_cap  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT = "globalkb"


@dataclass
class RagSample:
    text: str
    chunks_used: int
    top_score: float
    docs_hit: List[str]


def _ollama_chat(system_msg: str, user_msg: str, model: str) -> str:
    """Local Ollama chat call (non-streaming)."""
    base_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("message", {}).get("content", "").strip()


def _retrieve_and_format(query: str, k: int = 5) -> tuple:
    """Top-K chunks + system-message-formatted context as a string."""
    chunks, noise = retrieve_with_filter(query, PROJECT, k=k)
    if not chunks:
        return "", 0, 0.0, []
    kept, _ = apply_token_cap(chunks)
    msg = format_chunks_as_system_message(kept, total_candidates=len(chunks))
    sys_content = msg["content"] if isinstance(msg, dict) else str(msg)
    top = max(c.score or 0 for c in kept)
    docs = sorted({c.doc_id for c in kept})
    return sys_content, len(kept), top, docs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--k", type=int, default=5,
                        help="Top-K chunks to retrieve per query.")
    parser.add_argument("--model", default="qwen3-coder:480b-cloud",
                        help="Local Ollama model tag.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rows = []
    for i, q in enumerate(_QUERIES, 1):
        logger.info("[%d/%d] %s", i, len(_QUERIES), q.query[:80])
        sys_msg, n_chunks, top, docs = _retrieve_and_format(q.query, args.k)
        if not sys_msg:
            sample = RagSample(text="(no chunks retrieved)", chunks_used=0, top_score=0.0, docs_hit=[])
        else:
            text = _ollama_chat(sys_msg, q.query, model=args.model)
            sample = RagSample(text=text, chunks_used=n_chunks, top_score=top, docs_hit=docs)
        score = _score(sample.text, q)
        rows.append((q, sample, score))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pass_count = sum(1 for _, _, s in rows if s["passed"])

    with args.out.open("w", encoding="utf-8") as f:
        f.write("# RAG in-distribution eval\n\n")
        f.write(f"- project: `{PROJECT}` (data/knowledge/*.md indexed)\n")
        f.write(f"- LLM: `{args.model}` (no LoRA adapter)\n")
        f.write(f"- top-K per query: {args.k}\n")
        f.write(f"- generated: {_dt.datetime.utcnow().isoformat()}Z\n\n")
        f.write(f"## Summary\n\n- **RAG pass: {pass_count} / {len(rows)}**\n")
        f.write(f"- (best LoRA result was 3/10 — Test 1, Qwen-4B LR 1e-5)\n\n")
        f.write("## Per-question\n\n| # | Source | Verdict | Top score | Doc hit |\n|---|---|---|---|---|\n")
        for i, (q, s, sc) in enumerate(rows, 1):
            verdict = "PASS" if sc["passed"] else "FAIL"
            docs_str = ",".join(s.docs_hit) if s.docs_hit else "—"
            f.write(f"| {i} | `{q.source}` | {verdict} | {s.top_score:.3f} | {docs_str} |\n")
        f.write("\n---\n\n")
        for i, (q, s, sc) in enumerate(rows, 1):
            f.write(f"## Q{i} — `{q.source}`\n\n")
            f.write(f"**Query:** {q.query}\n\n")
            f.write(f"**Notes (ground truth):** {q.notes}\n\n")
            f.write(f"**Expected:** {q.expected}    **Forbidden:** {q.forbidden}\n\n")
            f.write(f"**Retrieved:** {s.chunks_used} chunks, top_score={s.top_score:.3f}, docs={s.docs_hit}\n\n")
            f.write(f"### RAG + {args.model} — **{'PASS' if sc['passed'] else 'FAIL'}**\n\n")
            f.write("```\n" + s.text + "\n```\n\n")
            f.write(f"- found: {sc['found_expected']}\n")
            f.write(f"- missing: {sc['missing_expected']}\n")
            f.write(f"- forbidden_hits: {sc['forbidden_hits']}\n\n")
            f.write("---\n\n")

    logger.info("RAG pass: %d / %d -> %s", pass_count, len(rows), args.out)
    print(str(args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
