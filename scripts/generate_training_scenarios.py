#!/usr/bin/env python3
"""Generate (question, expert-answer) training pairs from a project's
indexed documents (PR 25).

Reads chunks from the doc_index for a given ``project_id``, sends each
chunk to the chat block (LLM) with a prompt asking for synthetic Q&A
in the construction expert voice, and writes the pairs as JSONL ready
for ``scripts/finetune_router.py``.

Pipeline:

    1. Hydration ingests Drive / local folders / dropbox into the
       platform (see app/core/learning/hydration.py — Drive ingest is
       now recursive in this PR).
    2. doc_index extracts text + chunks per document.
    3. THIS SCRIPT walks the chunks, calls the LLM per chunk,
       parses + filters the response, writes JSONL.
    4. finetune_router.py consumes the JSONL.

Why generate synthetic Q&A instead of using documents directly?

  Documents are declarative: "Concrete cover for slabs in moderate
  exposure shall be 30mm minimum per ACI 318." The fine-tune target
  is conversational: "what's the cover for a slab in moderate
  exposure?" → "30mm minimum per ACI 318." Same fact, different
  shape. Direct document chunks would teach the model to reproduce
  the SOP, not to answer questions about it.

CLI:
    python scripts/generate_training_scenarios.py \\
        --project-id <project_id> \\
        --out data/learning/training_scenarios.jsonl \\
        [--questions-per-chunk 3] \\
        [--min-chunk-chars 200] \\
        [--max-chunks 1000] \\
        [--provider deepseek|local_lora|offline_template]

Honest non-verification: when the chat block returns the offline
template (no LLM reachable), this script silently degrades to zero
output rather than producing garbage Q&A. The provider check in the
filtering step catches that.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Dict, Iterator, List, Optional

# Resolve the `app` package when invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


# Prompt template — separate constant so tests can introspect it and so
# operators can swap it via --prompt-file without code edits.
_DEFAULT_PROMPT = """You are a senior construction project manager reviewing internal SOPs and project documents.

Below is an excerpt from {source}. Generate {n} question-answer pairs that a
project engineer or graduate engineer might genuinely ask about this material.

Constraints:
- Each question must be specific and answerable from the excerpt.
- Each answer must be 2-4 sentences, factually grounded in the excerpt, in
  the voice of an experienced PM (no preamble like "Based on the text...").
- Skip generic questions ("what does this say about safety?") in favor of
  concrete ones ("what's the minimum concrete cover for slabs in marine
  exposure?").
- Output STRICT JSONL — one JSON object per line, no surrounding prose,
  no markdown fences. Schema: {{"instruction": "...", "response": "..."}}

Excerpt:
\"\"\"
{chunk}
\"\"\"

JSONL:"""


def iter_chunks_for_project(
    project_id: str,
    min_chars: int = 200,
    max_chunks: Optional[int] = None,
) -> Iterator[Dict[str, str]]:
    """Yield chunk dicts ``{text, source}`` from the doc_index for a project.

    ``source`` is the document's filename (e.g.
    ``"600-Procurement & Contracts/standard_form_subcontract.pdf"``) when
    the doc_index recorded one, otherwise the document_id.
    """
    from app.core import doc_index
    from app.core.rag.vector_store import get_store

    index = doc_index._load_index(project_id)
    if not index:
        return
    docs = [d for d in index.get("documents", []) if d.get("document_id")]
    # The index ledger names the documents; chunk text lives only in the
    # vector store (0018), read per document in index order.
    store = get_store()
    texts_by_doc = store.doc_chunk_texts(project_id, [d["document_id"] for d in docs])
    count = 0
    for doc in docs:
        filename = doc.get("filename") or doc.get("document_id") or "<unknown>"
        for chunk in texts_by_doc.get(doc["document_id"]) or []:
            text = str(chunk or "").strip()
            if len(text) < min_chars:
                continue
            yield {"text": text, "source": filename}
            count += 1
            if max_chunks and count >= max_chunks:
                return


async def _generate_for_chunk(
    chunk: Dict[str, str],
    questions_per_chunk: int,
    provider_hint: str,
) -> List[Dict[str, str]]:
    """Send one chunk to the chat block, parse JSONL out of the response.

    Returns an empty list when:
    - the chat block returned the offline_template (no LLM available)
    - the response wasn't parseable as JSONL
    - the response had no valid {instruction, response} rows
    """
    from app.blocks import BLOCK_REGISTRY

    cls = BLOCK_REGISTRY.get("chat")
    if cls is None:
        logger.warning("chat block not in registry — cannot generate scenarios")
        return []

    block = cls()
    prompt = _DEFAULT_PROMPT.format(
        source=chunk["source"], n=questions_per_chunk, chunk=chunk["text"]
    )
    envelope = await block.execute({"text": prompt}, {
        "max_tokens": 1500,
        "temperature": 0.7,
    })
    inner = envelope.get("result") if isinstance(envelope, dict) else {}
    if not isinstance(inner, dict):
        return []

    # Refuse the offline template path — the deterministic fallback can't
    # actually generate Q&A. Better to skip than write garbage.
    provider = inner.get("provider", "")
    if provider in ("offline_template", "", None):
        return []
    if provider_hint and provider_hint != "any" and provider != provider_hint:
        # Operator forced a specific provider; skip this chunk if we got
        # something else. Lets them re-run cleanly after configuring the
        # provider they wanted.
        return []

    raw = (inner.get("response") or inner.get("text") or "").strip()
    if not raw:
        return []

    pairs: List[Dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip code-fence noise the LLM might wrap output in despite the prompt
        if line.startswith("```"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        instruction = (obj.get("instruction") or "").strip()
        response = (obj.get("response") or "").strip()
        # Light quality filters: skip refusals / one-liners
        if len(instruction) < 10 or len(response) < 30:
            continue
        pairs.append({
            "instruction": instruction,
            "response": response,
            "source": chunk["source"],
        })
    return pairs


async def _run(
    project_id: str,
    out_path: str,
    questions_per_chunk: int,
    min_chunk_chars: int,
    max_chunks: Optional[int],
    provider_hint: str,
) -> int:
    chunks = list(iter_chunks_for_project(
        project_id, min_chars=min_chunk_chars, max_chunks=max_chunks
    ))
    if not chunks:
        logger.error(
            "no chunks found for project %s — has the project been hydrated?",
            project_id,
        )
        return 1

    logger.info("found %d chunks ≥ %d chars for project %s",
                len(chunks), min_chunk_chars, project_id)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Accumulate in memory so the validation pipeline can dedupe / drop
    # before any rows hit disk. At 500-row scale this is well under a MB.
    rows: List[Dict[str, str]] = []
    skipped = 0
    for i, chunk in enumerate(chunks, start=1):
        pairs = await _generate_for_chunk(chunk, questions_per_chunk, provider_hint)
        if not pairs:
            skipped += 1
            continue
        rows.extend(pairs)
        if i % 20 == 0:
            logger.info("processed %d/%d chunks -> %d pairs (%d skipped)",
                        i, len(chunks), len(rows), skipped)

    if not rows:
        logger.error(
            "ZERO pairs generated. Check that the chat block has a working "
            "LLM provider (KIMI_API_KEY or GROQ_API_KEY set, or a local model configured)."
        )
        return 1

    # Validate before writing. The validator is deterministic given the
    # input rows, so re-running this step is safe.
    kept_rows, validation_report = _validate_scenarios(rows)
    print("== validation ==", file=sys.stderr)
    for k, v in validation_report.items():
        print(f"  {k} = {v}", file=sys.stderr)

    # Also surface top contributors so the operator can sanity-check.
    by_doc: Dict[str, int] = {}
    for r in kept_rows:
        key = r.get("source") or "?"
        by_doc[key] = by_doc.get(key, 0) + 1
    top = sorted(by_doc.items(), key=lambda kv: kv[1], reverse=True)[:5]
    print(f"  top sources: {top}", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept_rows)} rows to {out_path}", file=sys.stderr)

    logger.info(
        "wrote %d validated pairs to %s (%d generated, %d chunks, %d skipped)",
        len(kept_rows), out_path, len(rows), len(chunks), skipped,
    )
    if not kept_rows:
        logger.error(
            "ZERO pairs survived validation. Check the validation report above."
        )
        return 1

    # ── Sample preview (PR #25 review fix #3) ─────────────────────────
    # Synthetic Q&A quality is the load-bearing risk on this pipeline.
    # The docs recommend reading 20 random rows by eye before paying for
    # a fine-tune; that check is easy to skip. Print 5 random rows here
    # so the operator sees a quality signal without an extra command —
    # if the first 5 look wrong or generic, the whole run is suspect.
    _print_sample(out_path, n=5)
    return 0


def _print_sample(out_path: str, n: int = 5) -> None:
    """Print n random rows from the output to stderr for an at-a-glance
    quality check. Stderr so it doesn't pollute pipe-to-file usage."""
    import random as _random
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read back for sample preview: %s", exc)
        return
    if not rows:
        return
    sample = _random.sample(rows, k=min(n, len(rows)))
    print("\n── %d random samples (read these before fine-tuning) ──" % len(sample), file=sys.stderr)
    for i, r in enumerate(sample, start=1):
        q = (r.get("instruction") or "").strip()
        a = (r.get("response") or "").strip()
        src = r.get("source") or "?"
        print(f"\n[{i}] source: {src}", file=sys.stderr)
        print(f"    Q: {q[:200]}{'…' if len(q) > 200 else ''}", file=sys.stderr)
        print(f"    A: {a[:300]}{'…' if len(a) > 300 else ''}", file=sys.stderr)
    print("", file=sys.stderr)


def _validate_scenarios(rows: List[Dict[str, str]]) -> tuple:
    """Apply the validation pipeline. Returns ``(kept_rows, report)``.

    Drops:
    * empty instruction or response
    * response under 29 chars (too short to be a real answer)
    * duplicate responses (cosine >= 0.85 against any kept row)

    Uses the platform embedder (``app.core.rag.embeddings``) when
    available; falls back to string-equality dedupe when it isn't, so
    the script can still validate offline / in CI without optional ML
    deps. Both code paths are correct; the fallback is just coarser.
    """
    out: List[Dict[str, str]] = []
    report = {
        "input": len(rows),
        "dropped_empty": 0,
        "dropped_short": 0,
        "dropped_duplicates": 0,
    }
    # Stage 1: drop empties / too-short.
    stage1: List[Dict[str, str]] = []
    for r in rows:
        instr = (r.get("instruction") or "").strip()
        resp = (r.get("response") or "").strip()
        if not instr or not resp:
            report["dropped_empty"] += 1
            continue
        if len(resp) < 29:
            report["dropped_short"] += 1
            continue
        stage1.append(r)

    # Stage 2: dedupe by response cosine. Use the platform embedder
    # if available; fall back to a string-equality dedupe if not.
    try:
        from app.core.rag.embeddings import Embedder, get_embedder
        if not Embedder.available():
            raise RuntimeError("embedder not available")
        embedder = get_embedder()
        responses = [r["response"] for r in stage1]
        vecs = embedder.encode(responses)
    except Exception:
        seen = set()
        for r in stage1:
            key = r["response"]
            if key in seen:
                report["dropped_duplicates"] += 1
                continue
            seen.add(key)
            out.append(r)
        report["kept"] = len(out)
        return out, report

    import numpy as np
    kept_vecs = []
    for r, v in zip(stage1, vecs):
        keep = True
        for kv in kept_vecs:
            # Cosine assumes unit-normalized embeddings (zvec is).
            cos = float(np.dot(v, kv))
            if cos >= 0.85:
                keep = False
                report["dropped_duplicates"] += 1
                break
        if keep:
            kept_vecs.append(v)
            out.append(r)
    report["kept"] = len(out)
    return out, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True, help="Project to source chunks from")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.getenv("DATA_DIR", "./data"), "learning", "training_scenarios.jsonl"
        ),
    )
    parser.add_argument("--questions-per-chunk", type=int, default=3)
    parser.add_argument("--min-chunk-chars", type=int, default=200)
    parser.add_argument(
        "--max-chunks", type=int, default=None,
        help="Stop after this many chunks (default: process all)",
    )
    parser.add_argument(
        "--provider", default="any",
        choices=("any", "kimi", "groq", "local_ollama", "local_llama_cpp", "local_lora", "ollama"),
        help="Require this provider on chat responses (default: any non-offline)",
    )
    args = parser.parse_args()

    # When the operator forces 'ollama' (the cloud-style provider routed via
    # _llm_config), propagate that selection to the chat block via the env
    # var it reads. Without this, the chat block would pick kimi/groq
    # based on whichever API key is configured and the provider-hint match
    # at _generate_for_chunk would drop every chunk.
    if args.provider == "ollama":
        os.environ["LLM_PROVIDER"] = "ollama"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    return asyncio.run(_run(
        project_id=args.project_id,
        out_path=args.out,
        questions_per_chunk=args.questions_per_chunk,
        min_chunk_chars=args.min_chunk_chars,
        max_chunks=args.max_chunks,
        provider_hint=args.provider,
    ))


if __name__ == "__main__":
    sys.exit(main())
