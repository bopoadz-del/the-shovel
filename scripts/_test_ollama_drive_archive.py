"""Smoke test: retrieve from drive_archive + answer via local Ollama.

Bypasses the Fork UI. Uses the production retrieve_with_filter against
project_id=drive_archive then posts the RAG context to qwen2.5:7b-instruct
on the local Ollama server.

Run: .venv/Scripts/python.exe scripts/_test_ollama_drive_archive.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
os.environ.setdefault("RAG_EMBEDDING_MODEL", "minishlab/potion-base-8M")
os.environ.setdefault("DATA_DIR", os.path.join(_REPO, "data"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import requests

from app.core.rag.retriever import retrieve_with_filter

PROJECT = "drive_archive"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen2.5:7b-instruct"

QUERIES = [
    "What is the EXC drawing-number format used on the the client project project?",
    "What does the SECTIONAL ELEVATION telecom drawing show?",
    "What is the procedure for design review acceptance under PRC-951?",
    "What is the payable trench width specification for the water supply pipe?",
    "Manhole spacing requirements for telecom ducts on the the client project project?",
]


def _format_rag_context(chunks) -> str:
    parts = []
    for c in chunks:
        snippet = (c.text or "").strip().replace("\n", " ")
        parts.append(
            f"[doc_id={c.doc_id} score={(c.score or 0):.3f}]\n{snippet[:600]}"
        )
    return "\n\n".join(parts)


def _ask_ollama(query: str, context: str) -> tuple[str, float]:
    system = (
        "You are a construction-engineering assistant. Answer strictly from the "
        "provided context. If the context does not contain the answer, say "
        "'Not found in retrieved documents.' Be concise."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 400},
    }
    t = time.monotonic()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    elapsed = time.monotonic() - t
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", ""), elapsed


def main() -> None:
    for i, q in enumerate(QUERIES, 1):
        print(f"\n{'=' * 90}")
        print(f"Q{i}: {q}")
        print("=" * 90)

        t = time.monotonic()
        chunks, noise = retrieve_with_filter(q, PROJECT, k=3)
        retr = time.monotonic() - t
        if not chunks:
            print(f"[retrieval] no chunks (noise_filtered={noise}); skipping LLM")
            continue
        print(f"[retrieval] {len(chunks)} chunks in {retr:.2f}s; top score={chunks[0].score:.3f}")
        for c in chunks:
            print(f"  - doc_id={c.doc_id} chunk={c.chunk_index} score={(c.score or 0):.3f}")
            snippet = (c.text or "").strip().replace("\n", " ")[:140]
            print(f"    snippet: {snippet}")

        ctx = _format_rag_context(chunks)
        try:
            answer, llm_t = _ask_ollama(q, ctx)
        except Exception as e:
            print(f"[ollama] FAILED: {e!r}")
            continue
        print(f"\n[ollama qwen2.5:7b-instruct in {llm_t:.1f}s]")
        print(answer.strip())


if __name__ == "__main__":
    main()
