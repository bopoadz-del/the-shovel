"""RAG layer — persistent retrieval over indexed project documents.

Three modules, one responsibility each:

* ``embeddings`` — wraps sentence-transformers/all-MiniLM-L6-v2 (384-dim,
  small, fast, decent quality). Public entrypoint :func:`get_embedder`
  returns a process-cached instance.
* ``vector_store`` — SQLite-backed chunk store. Uses ``sqlite-vec``'s
  ``vec0`` virtual table for ANN search when the extension is loadable;
  otherwise falls back to numpy cosine similarity over all rows (slower
  but works without the C extension).
* ``retriever`` — high-level ``retrieve(query, project_id, k)`` composing
  the two; the unit the chat block and ``/v1/rag/search`` route both call.

All three are import-safe even without the optional deps installed
(``sentence-transformers``, ``sqlite-vec`` per ``requirements-rag.txt``).
:meth:`Embedder.available` and the route's 503 are the user-facing signals
that RAG isn't ready; nothing else breaks.
"""

__all__ = []  # public surface is the submodules, not this package
