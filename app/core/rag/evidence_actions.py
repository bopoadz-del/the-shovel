"""The three retrieval actions wired to the real primitives (item 2).

Each action has the ``RetrievalAction`` shape ``(plan_item_text) -> evidence
text`` the fetch loop consumes. The adapters here turn the real primitives —
semantic ``retrieve_with_filter`` (returns chunks), lexical ``bm25_search``
(returns chunks), structural ``rows_matching`` (returns sheet rows) — into
that shape. They are unit-testable with fake primitives; nothing here knows a
project, an item or a column.
"""
from __future__ import annotations

from functools import partial
from typing import Callable, Mapping, Sequence

from app.core.rag.evidence_budget import (
    CONTEXT_TRUNCATED_NOTE,
    cap_chunks,
    max_tokens_per_doc,
    max_total_tokens,
)
from app.core.rag.evidence_retrieval import RetrievalAction
from app.core.rag.structural_retrieval import render_rows, rows_matching

#: A chunk has ``text``; a plain string passes through; anything without a
#: usable text yields the empty string (so it never fabricates evidence).
def chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    text = getattr(chunk, "text", None)
    return text if isinstance(text, str) else ""


def chunks_to_text(chunks) -> str:
    """Concatenate chunk texts into one evidence string."""
    return "\n".join(chunk_text(c) for c in chunks if chunk_text(c).strip())


def _budgeted_text(chunks, *, total_tokens=None, per_doc_tokens=None) -> str:
    """Highest-scoring chunks, budgeted, joined into one evidence string.

    Applies the same per-document + total token budget the loop's merged
    output respects, so no single fetch action can inject a whole document.
    ``chunks`` may hold ``Chunk`` objects or lightweight fakes carrying
    ``text``/``score``/``doc_id``; a plain string list still passes through.
    """
    raw = list(chunks or [])
    if not raw:
        return ""
    # Plain strings carry no score/doc_id: budget on the concatenation only.
    if all(isinstance(c, str) for c in raw):
        text = chunks_to_text(raw)
        from app.core.rag.evidence_budget import cap_evidence_text
        text, truncated = cap_evidence_text(
            text,
            max_tokens=total_tokens if total_tokens is not None else max_total_tokens(),
        )
        return text + (CONTEXT_TRUNCATED_NOTE if truncated else "")
    kept, dropped = cap_chunks(
        raw,
        total_tokens=total_tokens if total_tokens is not None else max_total_tokens(),
        per_doc_tokens=per_doc_tokens if per_doc_tokens is not None else max_tokens_per_doc(),
    )
    text = chunks_to_text(kept)
    return text + (CONTEXT_TRUNCATED_NOTE if dropped else "")


def _semantic_retrieve(retrieve_fn, project_id: str, k: int, item: str) -> str:
    chunks, _ = retrieve_fn(item, project_id, k=k)
    return _budgeted_text(chunks)


def semantic_action(
    retrieve_fn: Callable[..., tuple],
    project_id: str,
    k: int = 5,
) -> RetrievalAction:
    """Semantic: the existing top-k. ``retrieve_fn`` returns ``(chunks, n)``."""
    return partial(_semantic_retrieve, retrieve_fn, project_id, k)


def _lexical_retrieve(bm25_fn, project_id: str, k: int, item: str) -> str:
    return _budgeted_text(bm25_fn(project_id, item, k=k))


def lexical_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    k: int = 50,
) -> RetrievalAction:
    """Lexical: BM25 over the same index. ``bm25_fn`` returns chunks."""
    return partial(_lexical_retrieve, bm25_fn, project_id, k)


def _rows_for_item(
    sheets: Mapping[str, Sequence[Sequence[str]]],
    headers: Mapping[str, Sequence[str]] | None,
    item: str,
) -> str:
    """All rows of every sheet whose cells match ``item``, as TSV sections."""
    parts: list[str] = []
    for name, rows in sheets.items():
        matched = rows_matching(rows, item)
        if not matched:
            continue
        head = (headers or {}).get(name, ())
        parts.append(f"--- {name} ---\n" + render_rows(matched, head))
    return "\n".join(parts)


def structural_action(
    sheets: Mapping[str, Sequence[Sequence[str]]],
    headers: Mapping[str, Sequence[str]] | None = None,
) -> RetrievalAction:
    """Structural: all rows of a sheet whose cells match the item.

    ``sheets`` maps a sheet name to its rows (the WBS ``XLSXDocument.sheets``
    shape); ``headers`` optionally maps a sheet name to its header row.
    """
    return partial(_rows_for_item, sheets, headers)
