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
from types import SimpleNamespace
from typing import Callable, Iterable, Mapping, Sequence

from app.core.rag.bill_sections import (
    in_named_span,
    item_rows,
    named_sections,
    section_label,
    section_map,
)
from app.core.rag.bill_sections import words as section_words
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


def _section_tagged(chunk, label: str) -> SimpleNamespace:
    """A copy of ``chunk`` whose text leads with its bill section ``label``.

    Carries the attributes the ranking, dedup and budget read, so a tagged
    chunk ranks, dedups and budgets like the chunk it copies.
    """
    return SimpleNamespace(
        doc_id=getattr(chunk, "doc_id", "") or "",
        chunk_id=getattr(chunk, "chunk_id", None),
        chunk_index=getattr(chunk, "chunk_index", None),
        score=getattr(chunk, "score", None),
        text=f"{label}\n{chunk_text(chunk)}",
    )


def bill_section_maps(
    fetch: Callable[[str, list[str]], Sequence], project_id: str, doc_ids,
) -> dict[str, tuple[list, list]]:
    """``{doc_id: (section map, chunks)}`` for every bill the fetch returns.

    ``fetch(project_id, doc_ids)`` returns every chunk of those documents
    (each with ``doc_id``/``chunk_index``/``text``). A bill that prints no
    section heading maps to ``[]``: its rows are still read, labelled
    "section unknown". A failed fetch means "no bill read" (logged): the
    action then ranks the lexical pool as it did before.
    """
    ids = sorted(str(d) for d in (doc_ids or ()) if d)
    if not ids:
        return {}
    try:
        chunks = list(fetch(project_id, ids) or ())
    except Exception:  # noqa: BLE001 -- a fetch failure must not break the loop
        logger.warning(
            "bill_rows section-map fetch failed project=%s; ranking without "
            "sections", project_id, exc_info=True,
        )
        return {}
    by_doc: dict[str, list] = {}
    for chunk in chunks:
        if getattr(chunk, "chunk_index", None) is None:
            continue
        by_doc.setdefault(getattr(chunk, "doc_id", "") or "", []).append(chunk)
    out: dict[str, tuple[list, list]] = {}
    for doc_id, doc_chunks in by_doc.items():
        sections = section_map(
            (int(c.chunk_index), chunk_text(c)) for c in doc_chunks
        )
        if doc_id:
            out[doc_id] = (sections, doc_chunks)
    return out


def _named_by_doc(maps, item: str, ask: str) -> dict[str, set[str]]:
    """``{doc_id: section numbers}`` the plan item (else the ask) names."""
    named: dict[str, set[str]] = {}
    for doc_id, (sections, _chunks) in maps.items():
        numbers = named_sections(item, sections) or named_sections(ask, sections)
        if numbers:
            named[doc_id] = numbers
    return named


def _scope_to_named_sections(pool, maps, named, item: str, row_count) -> list:
    """The pool with each bill whose section the ask names cut to that span.

    Only a bill in ``named`` is restricted: its chunks outside the named
    spans are excluded. Every other document (a bill whose sections the ask
    does not name, a bill with no headings, a spec) is kept as it was. A
    named section's own priced chunks that print an item word join the pool
    after its lexical hits, so the section's rows are not lost to the
    document-wide top-k.
    """
    def inside(chunk) -> bool:
        doc_id = getattr(chunk, "doc_id", "") or ""
        if doc_id not in named:
            return True
        return in_named_span(
            getattr(chunk, "chunk_index", None), maps[doc_id][0],
            named.get(doc_id, ()),
        )

    scoped = [c for c in pool if inside(c)]
    item_words = section_words(item)
    extra = []
    for doc_id, numbers in named.items():
        sections, doc_chunks = maps[doc_id]
        for chunk in doc_chunks:
            if not in_named_span(chunk.chunk_index, sections, numbers):
                continue
            text = chunk_text(chunk)
            if item_words & section_words(text) and row_count(text) > 0:
                extra.append(chunk)
    return _merge_pools(scoped, extra)


def _tag_sections(chunks, maps) -> list:
    """Every chunk of a mapped bill, its text led by its section label."""
    out = []
    for chunk in chunks:
        doc_id = getattr(chunk, "doc_id", "") or ""
        label = ""
        if doc_id in maps:
            label = section_label(getattr(chunk, "chunk_index", None), maps[doc_id][0])
        out.append(_section_tagged(chunk, label) if label else chunk)
    return out


#: Leads the complete row list of a bill item. General words only: it tells
#: the model what the list is and how a total over it is stated.
ITEM_ROWS_NOTE = (
    "[bill item rows: every priced row of the project's bill(s) whose "
    "description matches the item \"{item}\", one per line, each led by "
    "the bill section it sits in ({count} rows, not sampled). An item can "
    "recur in several sections with no billed sum for it. When the bill "
    "prints no total for the asked item, the answer is the sum of the listed "
    "rows that are that item: list the rows per section, then state the sum "
    "as computed from those rows. A document-level total guard applies only "
    "to a claimed overall contract or BOQ total, never to a sum computed from "
    "rows listed here.]"
)


#: Leads a bill's evidence when the ingest recorded no verified overall total
#: for it (``doc_index.boq_total_status``). It states what the parser
#: observed, never a guess at how the document was produced.
BILL_TOTAL_NOTE = (
    "[bill total status: {document} has no verified overall total -- the "
    "parser did not cross-check its rows against a total the bill states "
    "(observed: {reason}). Do not state an overall contract or BOQ total for "
    "this bill. A sum computed from listed rows is still stated as computed "
    "from those rows.]"
)


def _total_status_text(total_status, doc_ids) -> str:
    """One ``BILL_TOTAL_NOTE`` per handed bill whose total is unverified.

    ``total_status(doc_id)`` returns the document's BOQ verified-total
    attribute (optionally with a ``document`` display name) or None. None
    (never staged as a BOQ) and ``verified: True`` add nothing; a failed read
    adds nothing and is logged, so the loop never breaks on it.
    """
    if total_status is None:
        return ""
    lines: list[str] = []
    for doc_id in doc_ids:
        try:
            attr = total_status(doc_id)
        except Exception:  # noqa: BLE001 -- a status read must not break the loop
            logger.warning("bill total-status read failed doc=%s", doc_id, exc_info=True)
            continue
        if not isinstance(attr, Mapping) or attr.get("verified") is not False:
            continue
        lines.append(BILL_TOTAL_NOTE.format(
            document=str(attr.get("document") or doc_id),
            reason=str(attr.get("reason") or "unrecorded").replace("_", " "),
        ))
    return "\n".join(lines)


def _item_rows_text(maps, named, item: str, row_segments) -> tuple[str, dict]:
    """The item's complete row list over every bill read, and its counts.

    A bill whose section the ask names gives only that section's rows; every
    other bill gives all its matching rows. Nothing is ranked or cut here:
    the rows are the asked item's own evidence, so the chunk budget does not
    apply to them.
    """
    lines: list[str] = []
    counts: dict[str, int] = {}
    for doc_id in sorted(maps):
        sections, doc_chunks = maps[doc_id]
        rows = item_rows(
            ((int(c.chunk_index), chunk_text(c)) for c in doc_chunks),
            sections, item, row_segments=row_segments,
            numbers=named.get(doc_id, ()),
        )
        if rows:
            counts[doc_id] = len(rows)
            lines.extend(rows)
    if not lines:
        return "", counts
    note = ITEM_ROWS_NOTE.format(item=(item or "").strip()[:120], count=len(lines))
    return note + "\n" + "\n".join(lines), counts


def _bill_rows_retrieve(
    bm25_fn, project_id: str, k: int, row_count: Callable[[str], int], item: str,
    *, typed_bills: Callable[[str], Iterable[str]] | None = None,
    section_maps: Callable[[frozenset], Mapping] | None = None,
    ask: str = "",
    row_segments: Callable[[str], Sequence[str]] | None = None,
    total_status: Callable[[str], Mapping | None] | None = None,
) -> str:
    pool = list(bm25_fn(project_id, item, k=k) or ())
    typed = frozenset(typed_bills(project_id)) if typed_bills is not None else frozenset()
    if typed:
        # Rank the item inside the typed bills alone, so the rest of the
        # project cannot crowd the bill's own rows out of the top-k.
        scoped = list(bm25_fn(project_id, item, k=k, doc_ids=sorted(typed)) or ())
        pool = _merge_pools(scoped, pool)
    if not any(row_count(chunk_text(c)) > 0 for c in pool):
        # No candidate chunk prints a priced row: the item is not a bill
        # item (a clause, a drawing, a contract datum). Semantic and lexical
        # carry it; handing the bill's unpriced pages only spends budget.
        _trace("bill_rows", item, ())
        return ""
    maps = section_maps(typed) if (section_maps is not None and typed) else {}
    named = _named_by_doc(maps, item, ask) if maps else {}
    if named:
        pool = _scope_to_named_sections(pool, maps, named, item, row_count)
    ranked = rank_bill_documents(pool, row_count=row_count, typed_bills=typed)
    order = {doc_id: i for i, doc_id in enumerate(ranked)}
    handed = [c for c in pool if (getattr(c, "doc_id", "") or "") in order]
    # Stable: inside one document the pool's lexical order is kept.
    handed.sort(key=lambda c: order[getattr(c, "doc_id", "") or ""])
    if maps:
        handed = _tag_sections(handed, maps)
    _trace("bill_rows", item, handed)
    rows_text, row_counts = (
        _item_rows_text(maps, named, item, row_segments)
        if (maps and row_segments is not None) else ("", {})
    )
    if named or row_counts:
        logger.info(
            "rag_bill_section_trace item=%r sections=%s rows=%s",
            (item or "")[:120], {d: sorted(n) for d, n in named.items()},
            row_counts,
        )
    chunks_text = _budgeted_text(handed, presorted=True)
    status_text = _total_status_text(total_status, _handed_docs(handed))
    return "\n".join(t for t in (status_text, rows_text, chunks_text) if t)


def bill_rows_action(
    bm25_fn: Callable[..., Sequence],
    project_id: str,
    *,
    row_count: Callable[[str], int],
    k: int | None = None,
    typed_bills: Callable[[str], Iterable[str]] | None = None,
    bill_chunks: Callable[[str, list[str]], Sequence] | None = None,
    ask: str = "",
    row_segments: Callable[[str], Sequence[str]] | None = None,
    total_status: Callable[[str], Mapping | None] | None = None,
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

    An item whose pool prints no priced row at all is not a bill item (a
    clause, a drawing, a contract datum): the action hands nothing and
    semantic/lexical carry it.

    ``bill_chunks(project_id, doc_ids)``, when supplied, returns every chunk
    of the typed bills; each bill's section map is read from them
    (``bill_sections.section_map``). When the plan item, else the ``ask``,
    names a section, that bill's pool is restricted to the section's span
    before ranking; with no section named nothing is filtered. Every handed
    chunk of a mapped bill leads with its section label.

    ``row_segments(text)``, when supplied with ``bill_chunks``, splits a
    chunk into its priced rows; the action then leads its evidence with the
    item's COMPLETE row list over every bill read (``ITEM_ROWS_NOTE``): every
    matching row, each labelled with its section ("section unknown" for a
    bill with no headings), not ranked and not cut by the chunk budget, which
    applies to the chunks that follow it.

    ``total_status(doc_id)``, when supplied, reads a handed bill's BOQ
    verified-total attribute (``doc_index.boq_total_status``). Every handed
    bill recorded as unverified leads the evidence with ``BILL_TOTAL_NOTE``:
    no overall total is claimed for it. Unverified is never read as a cause
    (it does not mean scanned); a bill with no attribute adds nothing.

    Each item is fetched once per ask: a later round reuses the result.
    """
    depth = bill_pool_k() if k is None else int(k)
    # The action runs once per uncovered plan item per round and the
    # documents table does not change inside one loop: look up once.
    cache = (
        lru_cache(maxsize=None)(partial(typed_bill_ids, typed_bills))
        if typed_bills is not None else None
    )
    maps = (
        lru_cache(maxsize=None)(partial(bill_section_maps, bill_chunks, project_id))
        if bill_chunks is not None else None
    )
    return lru_cache(maxsize=None)(partial(
        _bill_rows_retrieve, bm25_fn, project_id, depth, row_count,
        typed_bills=cache, section_maps=maps, ask=ask or "",
        row_segments=row_segments,
        total_status=(
            lru_cache(maxsize=None)(total_status) if total_status is not None else None
        ),
    ))


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
