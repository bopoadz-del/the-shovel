"""Plan and (opt-in) write ``doc_rows`` for documents indexed before the table.

Additive only: rows go into ``doc_rows`` through
:func:`app.core.doc_rows.store_document_rows` with ``keep_existing=True``, so a
document that already has rows is left as it is. No document, chunk,
embedding or file is changed, and nothing is read a second way: the sources
hand over what the index already read (a spreadsheet's rows, a bill's saved
word boxes), never a new extraction.

What reads a document is injected (:class:`Sources`), so the live runner and
the tests build their inputs the same way. The plan is read-only and hashed;
``execute`` writes only when the caller passes the live plan's hash (the plan
is rebuilt just before the write) and then plans again, and the run is
complete only when that second plan has nothing left to write.

Per document the plan states why it writes nothing, when it does not:

* ``kept``: the document already has rows;
* ``derived_of``: the sheet was written out from another document of the
  project that is itself planned (its ``derived_from``), so its rows would be
  the same reading counted twice;
* ``chunk_mismatch``: a sheet whose rows read now are not as many as its
  stored row chunks (none read while chunks are stored included), so a
  row's chunk ordinal would cite the wrong chunk;
* ``pages_missing``: a bill whose saved word boxes do not cover every page
  (the backfill never reads a page again);
* ``no_rows``: the reader returned no row.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.core import doc_rows as dr

#: The kinds of table document the backfill reads.
SHEET = "sheet"
BILL = "bill"
KINDS = (SHEET, BILL)

KEPT = "kept"
DERIVED_OF = "derived_of"
CHUNK_MISMATCH = "chunk_mismatch"
PAGES_MISSING = "pages_missing"
NO_ROWS = "no_rows"
UNKNOWN_KIND = "unknown_kind"


@dataclass(frozen=True)
class BackfillDoc:
    """One table document of the project, as the caller found it."""

    doc_id: str
    kind: str
    derived_from: str = ""


class Sources(Protocol):
    """What the backfill reads; the live runner wires the stored readers."""

    def documents(self, project_id: str) -> Iterable[BackfillDoc]: ...
    def has_rows(self, project_id: str, doc_id: str) -> bool: ...
    def sheet_rows(self, doc: BackfillDoc) -> Sequence[Any]: ...
    def row_chunk_count(self, doc: BackfillDoc) -> int: ...
    def page_coverage(self, doc: BackfillDoc) -> tuple[int, int]: ...
    def bill_rows(self, doc: BackfillDoc) -> Sequence[Any]: ...


def _rows_digest(rows: Sequence[dr.DocRow]) -> str:
    return hashlib.sha256(json.dumps(
        [[r.table_no, r.row_no, list(r.path), sorted(r.fields.items()), r.page,
          r.sheet, r.sheet_row, r.chunk_ordinal] for r in rows],
    ).encode("utf-8")).hexdigest()


def _read(doc: BackfillDoc, sources: Sources, entry: dict[str, Any]) -> list[dr.DocRow]:
    """The document's rows, or ``[]`` with ``entry["skip_reason"]`` set."""
    if doc.kind == SHEET:
        read = list(sources.sheet_rows(doc))
        chunks = sources.row_chunk_count(doc)
        entry["row_chunks"] = chunks
        if len(read) != chunks:
            entry["skip_reason"] = CHUNK_MISMATCH
            return []
        return dr.rows_from_spreadsheet(doc.doc_id, read)
    if doc.kind == BILL:
        cached, in_file = sources.page_coverage(doc)
        entry["pages_cached"], entry["pages_in_file"] = cached, in_file
        if in_file <= 0 or cached < in_file:
            entry["skip_reason"] = PAGES_MISSING
            return []
        return dr.rows_from_bill(doc.doc_id, list(sources.bill_rows(doc)))
    entry["skip_reason"] = UNKNOWN_KIND
    return []


def _build(project_id: str, sources: Sources) -> tuple[dict[str, Any], dict[str, list[dr.DocRow]]]:
    docs = sorted(sources.documents(project_id), key=lambda d: d.doc_id)
    planned = {d.doc_id for d in docs if d.kind in KINDS}
    entries: list[dict[str, Any]] = []
    to_write: dict[str, list[dr.DocRow]] = {}
    for doc in docs:
        entry: dict[str, Any] = {"doc_id": doc.doc_id, "kind": doc.kind,
                                 "rows_to_write": 0, "skip_reason": ""}
        entries.append(entry)
        if doc.derived_from and doc.derived_from != doc.doc_id and doc.derived_from in planned:
            entry["skip_reason"] = DERIVED_OF
            entry["derived_of"] = doc.derived_from
            continue
        if sources.has_rows(project_id, doc.doc_id):
            entry["skip_reason"] = KEPT
            continue
        rows = _read(doc, sources, entry)
        if entry["skip_reason"]:
            continue
        if not rows:
            entry["skip_reason"] = NO_ROWS
            continue
        entry["rows_to_write"] = len(rows)
        entry["rows_digest"] = _rows_digest(rows)
        entry["states"] = {str(t): c.state for t, c in sorted(dr.reconcile_tables(rows).items())}
        to_write[doc.doc_id] = rows
    digest = hashlib.sha256(json.dumps(
        {"project_id": project_id,
         "docs": [[e["doc_id"], e["kind"], e["rows_to_write"], e.get("rows_digest", "")]
                  for e in entries]},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    plan = {
        "project_id": project_id,
        "documents": entries,
        "documents_to_write": len(to_write),
        "rows_to_write": sum(e["rows_to_write"] for e in entries),
        "plan_hash": digest,
    }
    return plan, to_write


def plan_rows_backfill(project_id: str, sources: Sources) -> dict[str, Any]:
    """Read-only: what the backfill would write for ``project_id``, hashed."""
    if not project_id:
        raise ValueError("project_id is required")
    return _build(project_id, sources)[0]


def execute_rows_backfill(
    project_id: str, plan_hash: str, sources: Sources,
    store: Callable[..., int] = dr.store_document_rows,
) -> dict[str, Any]:
    """Write the planned rows when ``plan_hash`` is the live plan's hash.

    ``complete`` is true only when the plan made after the write has nothing
    left to write.
    """
    if not project_id:
        raise ValueError("project_id is required")
    plan, to_write = _build(project_id, sources)
    result: dict[str, Any] = {"plan": plan, "executed": False, "written": {},
                              "kept": [], "refused": None, "after": None, "complete": False}
    if plan_hash != plan["plan_hash"]:
        result["refused"] = "plan_hash does not match the live plan; nothing written"
        return result
    for doc_id, rows in to_write.items():
        n = store(project_id, doc_id, rows, keep_existing=True)
        if n:
            result["written"][doc_id] = n
        else:
            result["kept"].append(doc_id)
    after = _build(project_id, sources)[0]
    result["executed"] = True
    result["after"] = after
    result["complete"] = after["rows_to_write"] == 0
    return result
