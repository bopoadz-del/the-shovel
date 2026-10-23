#!/usr/bin/env python3
"""P0d — one-document ingestion proof with the approved embedder.

Downloads a test document's bytes (here a synthetic .txt), extracts text,
chunks it, embeds with `BAAI/bge-small-en-v1.5`, writes into the `v2`
namespace via the normal doc_index pipeline, and verifies:

  1. chunk_count > 0
  2. the chunks are retrievable via the same embedder
  3. a zero-chunk document fails loud (ZERO_CHUNK banner)

This is the controlled proof that the chosen embedder works end-to-end
through the normal pipeline before Phase-1 mass re-ingestion begins.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

# Force the approved embedder and the new namespace BEFORE any module imports.
os.environ.setdefault("RAG_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
os.environ.setdefault("RAG_VECTOR_NAMESPACE", "v2")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _write_txt(path: Path, content: bytes) -> None:
    path.write_bytes(content)


def main() -> int:
    from app.core import file_crypto, projects as projects_mod
    from app.core import doc_index
    from app.core.rag import embeddings as _emb, retriever as _rag

    _emb.reset_embedder_cache()

    print("=" * 70)
    print("P0d — one-document ingestion proof")
    print(f"embedder: {os.environ['RAG_EMBEDDING_MODEL']}")
    print(f"namespace: {os.environ['RAG_VECTOR_NAMESPACE']}")
    print("=" * 70)

    # 1. Load embedder and verify identity.
    t0 = time.monotonic()
    embedder = _emb.get_embedder()
    identity = embedder.identity
    print(f"embedder identity: {identity}")
    print(f"query instruction: {embedder.query_instruction!r}")
    print(f"embedder load time: {time.monotonic() - t0:.1f}s")

    # 2. Create project and a real .txt document.
    with tempfile.TemporaryDirectory() as tmpdir:
        proj = projects_mod.create_project("P0d BGE Proof")
        project_id = proj["id"]
        print(f"project: {project_id}")

        doc_text = (
            "Project-specific note — Pier P-217 structural concrete.\n"
            "Concrete grade C50/60, cement content 385 kg/m3 CEM I 42.5N, "
            "water/cement ratio 0.38, target slump 180 mm. "
            "Trial mix approved by the Engineer on revision C.\n"
        )
        doc_path = Path(tmpdir) / "P-217-pier-mix-design.txt"
        _write_txt(doc_path, doc_text.encode("utf-8"))

        doc = projects_mod.add_document(
            project_id,
            "P-217-pier-mix-design.txt",
            file_path=str(doc_path),
            size=doc_path.stat().st_size,
        )
        document_id = doc["id"]
        print(f"document: {document_id}")

        # 3. Run the normal indexing pipeline.
        print(f"RAG available before index: {_rag.available()}")
        t0 = time.monotonic()
        result = doc_index.index_document(project_id, document_id)
        elapsed = time.monotonic() - t0
        print(f"index_document result: {result}")
        print(f"index time: {elapsed:.1f}s")

        # 4. Fail-loud checks.
        if result.get("status") == "error":
            print("[FAIL] index_document returned an error", file=sys.stderr)
            return 1
        total_chunks = result.get("total_chunks", 0)
        rag_indexed = result.get("rag_indexed", 0)
        if total_chunks == 0:
            print("[FAIL] total_chunks == 0", file=sys.stderr)
            return 1
        if rag_indexed == 0:
            print("[FAIL] rag_indexed == 0", file=sys.stderr)
            return 1
        print(f"[OK] indexed {total_chunks} chunk(s), RAG wrote {rag_indexed}")

        # 5. Retrieval check.
        query = "What is the cement content for the Pier P-217 concrete mix?"
        results = _rag.retrieve(query, project_id, k=3)
        if not results:
            print("[FAIL] retrieve returned no results", file=sys.stderr)
            return 1
        top = results[0]
        print(f"[OK] retrieve top-1 doc={top.doc_id} score={top.score:.4f}")
        print(f"     text head: {top.text[:120]}...")
        if "385" not in top.text:
            print("[FAIL] top result does not contain expected fact '385'", file=sys.stderr)
            return 1
        print("[OK] expected fact found in top result")

        # 6. Fail-loud: zero-chunk document must error.
        empty_path = Path(tmpdir) / "empty.txt"
        _write_txt(empty_path, b"")
        empty_doc = projects_mod.add_document(
            project_id,
            "empty.txt",
            file_path=str(empty_path),
            size=0,
        )
        empty_result = doc_index.index_document(project_id, empty_doc["id"])
        print(f"empty doc result: {empty_result}")
        if empty_result.get("error") != "ZERO_CHUNK":
            print("[FAIL] empty supported file did not fail loud", file=sys.stderr)
            return 1
        print("[OK] zero-chunk file failed loud with ZERO_CHUNK")

    print("=" * 70)
    print("P0d PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
