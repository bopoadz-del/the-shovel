"""The three retrieval actions wired to the real primitives (item 2).

Each action has the ``RetrievalAction`` shape ``(plan_item_text) -> evidence
text`` the fetch loop consumes. The adapters here turn the real primitives —
semantic ``retrieve_with_filter`` (returns chunks), lexical ``bm25_search``
(returns chunks), structural ``rows_matching`` (returns sheet rows) — into
that shape via top-level query helpers, so the helpers are directly testable.
Nothing here knows a project, an item or a column.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, Sequence

from app.core.rag.evidence_retrieval import RetrievalAction
from app.core.rag.structural_retrieval import render_rows, rows_matching


def chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else ""


def chunks_to_text(chunks) -> str:
    """Concatenate chunk texts into one evidence string."""
    return "\n".join(chunk_text(c) for c in chunks if chunk_text(c).strip())


def _semantic_query(retrieve_fn, project_id, k, item):
    chunks, _ = retrieve_fn(item, project_id, k=k)
    return chunks_to_text(chunks)


def _lexical_query(bm25_fn, project_id, k, item):
    return chunks_to_text(bm25_fn(project_id, item, k=k))


def _structural_query(sheets, headers, item):
    parts = []
    for name, rows in sheets.items():
        matched = rows_matching(rows, item)
        if not matched:
            continue
        parts.append(f"--- {name} ---\n" + render_rows(matched, headers.get(name, ())))
    return "\n".join(parts)


def semantic_action(
    retrieve_fn: Callable[..., tuple],
    project_id: str,
    k: int = 5,
) -> RetrievalAction:
    """Semantic: the existing top-k. ``retrieve_fn`` returns ``(chunks, n)``."""
    return partial(_semantic_query, retrieve_fn, project_id, k)


def lexical_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    k: int = 50,
) -> RetrievalAction:
    """Lexical: BM25 over the same index. ``bm25_fn`` returns chunks."""
    return partial(_lexical_query, bm25_fn, project_id, k)


def structural_action(
    sheets: Mapping[str, Sequence[Sequence[str]]],
    headers: Mapping[str, Sequence[str]] | None = None,
) -> RetrievalAction:
    """Structural: all rows of a sheet whose cells match the item.

    ``sheets`` maps a sheet name to its rows (the WBS ``XLSXDocument.sheets``
    shape); ``headers`` optionally maps a sheet name to its header row.
    """
    return partial(_structural_query, sheets, headers or {})
