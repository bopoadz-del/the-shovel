"""The three retrieval actions wired to the real primitives (item 2).

Each action has the ``RetrievalAction`` shape ``(plan_item_text) -> evidence
text`` the fetch loop consumes. The adapters here turn the real primitives —
semantic ``retrieve_with_filter`` (returns chunks), lexical ``bm25_search``
(returns chunks), structural ``rows_matching`` (returns sheet rows) — into
that shape. They are unit-testable with fake primitives; nothing here knows a
project, an item or a column.
"""
from __future__ import annotations

import logging
import os
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

logger = logging.getLogger(__name__)

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
    # Plain strings (and the empty case) carry no score/doc_id: budget the
    # concatenation as one text, so no single fetch can inject a whole
    # document and the empty result is still the empty string.
    if not raw or all(isinstance(c, str) for c in raw):
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


def _handed_docs(chunks) -> list[str]:
    """Distinct document ids of ``chunks``, in order (for the trace line)."""
    out: list[str] = []
    for chunk in chunks or ():
        doc_id = getattr(chunk, "doc_id", "") or ""
        if doc_id and doc_id not in out:
            out.append(doc_id)
    return out


def _trace(action: str, item: str, chunks) -> None:
    """One CloudWatch line per action call: which documents it handed in.

    ``rag_evidence_action_trace`` makes a figure turn diagnosable from the
    logs: whether the loop ran, which action fetched what, and from which
    documents -- without the evidence text itself.
    """
    raw = list(chunks or ())
    logger.info(
        "rag_evidence_action_trace action=%s item=%r chunks=%d docs=%s",
        action, (item or "")[:120], len(raw), _handed_docs(raw)[:12],
    )


def _semantic_retrieve(retrieve_fn, project_id: str, k: int, item: str) -> str:
    chunks, _ = retrieve_fn(item, project_id, k=k)
    _trace("semantic", item, chunks)
    return _budgeted_text(chunks)


def semantic_action(
    retrieve_fn: Callable[..., tuple],
    project_id: str,
    k: int = 5,
) -> RetrievalAction:
    """Semantic: the existing top-k. ``retrieve_fn`` returns ``(chunks, n)``."""
    return partial(_semantic_retrieve, retrieve_fn, project_id, k)


def _lexical_retrieve(bm25_fn, project_id: str, k: int, item: str) -> str:
    chunks = bm25_fn(project_id, item, k=k)
    _trace("lexical", item, chunks)
    return _budgeted_text(chunks)


def lexical_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    k: int = 50,
) -> RetrievalAction:
    """Lexical: BM25 over the same index. ``bm25_fn`` returns chunks."""
    return partial(_lexical_retrieve, bm25_fn, project_id, k)


def bill_pool_k() -> int:
    """Depth of the lexical candidate pool the bill-rows action scopes."""
    raw = (os.getenv("AGENTIC_BILL_POOL_K") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return 200
    return value if value > 0 else 200


def bill_document_ids(chunks, *, row_count: Callable[[str], int]) -> set[str]:
    """Documents whose chunks in ``chunks`` print priced rows.

    The selection signal is the chunk content itself: ``row_count`` counts
    rows whose quantity x rate = amount. A document is bill-type when its
    candidate chunks print at least one such row. Nothing here reads a
    filename, a title or a project.
    """
    rows_by_doc: dict[str, int] = {}
    for chunk in chunks or ():
        doc_id = getattr(chunk, "doc_id", "") or ""
        if not doc_id:
            continue
        rows_by_doc[doc_id] = rows_by_doc.get(doc_id, 0) + row_count(chunk_text(chunk))
    return {doc_id for doc_id, rows in rows_by_doc.items() if rows > 0}


def _bill_rows_retrieve(
    bm25_fn, project_id: str, k: int, row_count: Callable[[str], int], item: str,
) -> str:
    pool = list(bm25_fn(project_id, item, k=k) or ())
    bills = bill_document_ids(pool, row_count=row_count)
    scoped = [c for c in pool if (getattr(c, "doc_id", "") or "") in bills]
    _trace("bill_rows", item, scoped)
    return _budgeted_text(scoped)


def bill_rows_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    *,
    row_count: Callable[[str], int],
    k: int | None = None,
) -> RetrievalAction:
    """Document-scoped lexical: the item's rows in the project's bills.

    The global lexical/semantic rankings let a long priced bill lose every
    item to drawings and specifications that merely mention it. This action
    takes a deeper lexical pool for the item, keeps only the documents whose
    candidate chunks print priced rows (``row_count``: qty x rate = amount),
    and hands those chunks in under the same per-document + total budget.
    """
    depth = bill_pool_k() if k is None else int(k)
    return partial(_bill_rows_retrieve, bm25_fn, project_id, depth, row_count)


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
