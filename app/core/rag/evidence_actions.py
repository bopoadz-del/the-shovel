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
from functools import lru_cache, partial
from typing import Callable, Iterable, Mapping, Sequence

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


def _budgeted_text(
    chunks, *, total_tokens=None, per_doc_tokens=None, presorted=False,
) -> str:
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
        presorted=presorted,
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
    rows_by_doc, _ = _priced_rows_by_doc(chunks, row_count)
    return {doc_id for doc_id, rows in rows_by_doc.items() if rows > 0}


def _priced_rows_by_doc(chunks, row_count: Callable[[str], int]):
    """``({doc: priced rows}, {doc: candidate chunks})`` over ``chunks``."""
    rows_by_doc: dict[str, int] = {}
    chunks_by_doc: dict[str, int] = {}
    for chunk in chunks or ():
        doc_id = getattr(chunk, "doc_id", "") or ""
        if not doc_id:
            continue
        rows_by_doc[doc_id] = rows_by_doc.get(doc_id, 0) + row_count(chunk_text(chunk))
        chunks_by_doc[doc_id] = chunks_by_doc.get(doc_id, 0) + 1
    return rows_by_doc, chunks_by_doc


def rank_bill_documents(
    chunks,
    *,
    row_count: Callable[[str], int],
    typed_bills: Iterable[str] = (),
) -> list[str]:
    """Candidate bill documents in ``chunks``, best first.

    Kept: every document whose candidate chunks print a priced row, plus every
    document in ``typed_bills`` (the ingest's own bill classification) that
    has a candidate chunk. Order:

    1. documents that print priced rows before documents that print none;
    2. among those, a typed bill before an untyped document;
    3. then priced rows per candidate chunk (density), highest first;
    4. ties keep first-seen order (the lexical rank of the pool).

    So a document that prints one priced row among many unpriced chunks
    ranks below a bill whose candidate chunks are mostly priced rows.
    """
    typed = {str(d) for d in (typed_bills or ()) if d}
    rows_by_doc, chunks_by_doc = _priced_rows_by_doc(chunks, row_count)
    kept = [d for d in chunks_by_doc if rows_by_doc[d] > 0 or d in typed]
    # ``sorted`` is stable: equal keys keep the pool's first-seen order.
    return sorted(
        kept,
        key=lambda d: bill_rank_key(rows_by_doc[d], chunks_by_doc[d], d in typed),
    )


def bill_rank_key(priced_rows: int, candidate_chunks: int, typed: bool) -> tuple:
    """Sort key of one candidate bill document (smaller sorts first).

    Prints priced rows, then typed as a bill, then priced rows per candidate
    chunk (highest first).
    """
    density = priced_rows / candidate_chunks if candidate_chunks > 0 else 0.0
    return (priced_rows <= 0, not typed, -density)


def _chunk_identity(chunk):
    chunk_id = getattr(chunk, "chunk_id", None)
    return ("id", chunk_id) if chunk_id else ("obj", id(chunk))


def _merge_pools(*pools) -> list:
    """Concatenate chunk pools; the first occurrence of a chunk wins."""
    seen: set = set()
    out: list = []
    for pool in pools:
        for chunk in pool or ():
            ident = _chunk_identity(chunk)
            if ident in seen:
                continue
            seen.add(ident)
            out.append(chunk)
    return out


def typed_bill_ids(
    lookup: Callable[[str], Iterable[str]], project_id: str,
) -> frozenset[str]:
    """The project's typed bill ids from ``lookup``; empty when it fails.

    A failed lookup means "no typed bill known" (logged), so the action falls
    back to the plain lexical pool instead of breaking the loop.
    """
    try:
        return frozenset(str(d) for d in (lookup(project_id) or ()) if d)
    except Exception:  # noqa: BLE001 -- a lookup failure must not break the loop
        logger.warning(
            "bill_rows typed-bill lookup failed project=%s; "
            "ranking the lexical pool only", project_id, exc_info=True,
        )
        return frozenset()


def _bill_rows_retrieve(
    bm25_fn, project_id: str, k: int, row_count: Callable[[str], int], item: str,
    *, typed_bills: Callable[[str], Iterable[str]] | None = None,
) -> str:
    pool = list(bm25_fn(project_id, item, k=k) or ())
    typed = frozenset(typed_bills(project_id)) if typed_bills is not None else frozenset()
    if typed:
        # Rank the item inside the typed bills alone, so the rest of the
        # project cannot crowd the bill's own rows out of the top-k.
        scoped = list(bm25_fn(project_id, item, k=k, doc_ids=sorted(typed)) or ())
        pool = _merge_pools(scoped, pool)
    ranked = rank_bill_documents(pool, row_count=row_count, typed_bills=typed)
    order = {doc_id: i for i, doc_id in enumerate(ranked)}
    handed = [c for c in pool if (getattr(c, "doc_id", "") or "") in order]
    # Stable: inside one document the pool's lexical order is kept.
    handed.sort(key=lambda c: order[getattr(c, "doc_id", "") or ""])
    _trace("bill_rows", item, handed)
    return _budgeted_text(handed, presorted=True)


def bill_rows_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    *,
    row_count: Callable[[str], int],
    k: int | None = None,
    typed_bills: Callable[[str], Iterable[str]] | None = None,
) -> RetrievalAction:
    """Document-scoped lexical: the item's rows in the project's bills.

    The global lexical/semantic rankings let a long priced bill lose every
    item to drawings and specifications that merely mention it. This action
    takes a deeper lexical pool for the item, keeps the documents whose
    candidate chunks print priced rows (``row_count``: qty x rate = amount),
    ranks them by priced rows per candidate chunk, and hands their chunks in
    that order under the same per-document + total budget.

    ``typed_bills(project_id)``, when supplied, returns the ids of the
    documents the ingest classified as a bill (``documents.doc_type``). When
    it returns any, the item is also ranked inside those documents alone
    (``bm25_fn(..., doc_ids=...)``) and a typed bill that prints priced rows
    ranks ahead of an untyped one. When it returns none, or is not supplied,
    the pool is the plain lexical pool ranked by density only.
    """
    depth = bill_pool_k() if k is None else int(k)
    # The action runs once per uncovered plan item per round and the
    # documents table does not change inside one loop: look up once.
    cache = (
        lru_cache(maxsize=None)(partial(typed_bill_ids, typed_bills))
        if typed_bills is not None else None
    )
    return partial(
        _bill_rows_retrieve, bm25_fn, project_id, depth, row_count, typed_bills=cache,
    )


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
