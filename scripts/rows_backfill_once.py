"""PLAN (or, with the plan's hash, EXECUTE) the additive ``doc_rows`` backfill
for ONE project, inside the ingest task.

The ingest worker hands over to :func:`main_from_env` when
``ROWS_BACKFILL_PROJECT_ID`` is set (the hand-over line and the workflow are
the platform's; this module only offers the entry point). Selectors travel
as environment names, like ``bill_plan_once``:

* ``ROWS_BACKFILL_PROJECT_ID`` (required);
* ``ROWS_BACKFILL_EXECUTE_HASH`` (optional, a 64-hex plan hash): without it
  the run only plans; with it the run writes through
  :func:`app.core.doc_rows_backfill.execute_rows_backfill`, which rebuilds
  the plan and refuses unless its hash is this one.

What each document is comes from its own stored state, never its name:

* a sheet: its stored name has a spreadsheet extension the row reader reads
  (``doc_index._SPREADSHEET_EXTS``);
* a bill: a PDF whose word boxes the bill plan saved under its work directory
  (``bill_plan_once.default_work_dir``). Its rows are read back from those
  boxes only; a page with no saved boxes makes the bill ``pages_missing``.
  Nothing is OCRed or read from the page again.

A sheet's stored row chunks are counted by content: a read row counts when
the chunk text the index makes for it (``doc_index._spreadsheet_row_chunks``)
is among the document's stored chunk texts. A row whose chunk was split,
refused or re-indexed differently does not count, so the backfill skips that
sheet (``chunk_mismatch``) rather than cite a wrong chunk.

The last line is ``ROWSBACKFILLSUMMARY <json>``: counts, codes, ids and
hashes only (``bill_plan_once.public_summary``'s value rule). No row text,
description or file name is printed.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core import doc_rows_backfill as rb  # noqa: E402
from scripts import bill_plan_once as bpo  # noqa: E402

SUMMARY_PREFIX = "ROWSBACKFILLSUMMARY "
MODE_PLAN = "plan"
MODE_EXECUTE = "execute"
ERR_MISSING_PROJECT = "missing_project"
ERR_BAD_HASH = "bad_execute_hash"
ERR_RAISED = "backfill_raised"
_HASH_RE = re.compile(r"[0-9a-f]{64}")
#: Per-document plan fields the summary carries (ids, kinds, codes, counts).
DOC_FIELDS = ("doc_id", "kind", "rows_to_write", "skip_reason", "derived_of",
              "row_chunks", "pages_cached", "pages_in_file", "states")


def document_kind(original_name: str, saved_pages: int, sheet_exts: Iterable[str]) -> str:
    """``sheet``, ``bill`` or ``""`` (not a table document for the backfill).

    A sheet by its stored extension; a bill is a PDF with saved word boxes.
    """
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext in set(sheet_exts):
        return rb.SHEET
    if ext == ".pdf" and saved_pages > 0:
        return rb.BILL
    return ""


def stored_row_chunk_count(row_chunks: Sequence[str], stored: Iterable[str]) -> int:
    """How many of the rows' own chunk texts are stored for the document."""
    have = set(stored)
    return sum(1 for chunk in row_chunks if chunk in have)


class LiveSources:
    """:class:`app.core.doc_rows_backfill.Sources` over the live stores.

    Every reader is injected with a production default, so a test builds the
    same object from its own inputs.
    """

    def __init__(
        self,
        *,
        data_dir: str | None,
        list_documents: Callable[[str], list[dict[str, Any]]] | None = None,
        load_rows: Callable[[str, str], list[Any]] | None = None,
        materialize: Callable[[dict[str, Any]], str | None] | None = None,
        chunk_texts: Callable[[str, str], list[str]] | None = None,
        cached_pages: Callable[[str], int] = bpo.cached_pages,
        read_saved_rows: Callable[[str, int], Sequence[Any]] | None = None,
    ) -> None:
        self.data_dir = data_dir
        self._list = list_documents or _list_documents
        self._load_rows = load_rows or _load_rows
        self._materialize = materialize or _materialize
        self._chunk_texts = chunk_texts or _chunk_texts
        self._cached_pages = cached_pages
        self._read_saved_rows = read_saved_rows or _read_saved_rows
        self._docs: dict[str, dict[str, Any]] = {}
        self._project = ""

    def _work_dir(self, doc_id: str) -> str:
        return bpo.default_work_dir(doc_id, self.data_dir)

    def documents(self, project_id: str) -> Iterable[rb.BackfillDoc]:
        from app.core.doc_index import _SPREADSHEET_EXTS
        from app.core.projects import coerce_document_metadata

        self._project = project_id
        out: list[rb.BackfillDoc] = []
        for doc in self._list(project_id):
            doc_id = str(doc.get("id") or "")
            if not doc_id:
                continue
            kind = document_kind(str(doc.get("original_name") or ""),
                                 self._cached_pages(self._work_dir(doc_id)), _SPREADSHEET_EXTS)
            if not kind:
                continue
            self._docs[doc_id] = doc
            meta = coerce_document_metadata(doc.get("metadata"))
            out.append(rb.BackfillDoc(doc_id, kind, str(meta.get("derived_from") or "")))
        return out

    def has_rows(self, project_id: str, doc_id: str) -> bool:
        return bool(self._load_rows(project_id, doc_id))

    def _sheet_args(self, doc: rb.BackfillDoc) -> tuple[str, str, str] | None:
        stored = self._docs.get(doc.doc_id) or {}
        name = str(stored.get("original_name") or "")
        path = self._materialize(stored)
        if not path:
            return None
        return path, name, os.path.splitext(name)[1].lower()

    def sheet_rows(self, doc: rb.BackfillDoc) -> Sequence[Any]:
        from app.core.doc_index import spreadsheet_rows

        args = self._sheet_args(doc)
        return spreadsheet_rows(*args) if args else []

    def row_chunk_count(self, doc: rb.BackfillDoc) -> int:
        from app.core.doc_index import _spreadsheet_row_chunks

        args = self._sheet_args(doc)
        if not args:
            return 0
        return stored_row_chunk_count(_spreadsheet_row_chunks(*args),
                                      self._chunk_texts(self._project, doc.doc_id))

    def page_coverage(self, doc: rb.BackfillDoc) -> tuple[int, int]:
        stored = self._docs.get(doc.doc_id) or {}
        in_file = stored.get("pages_in_file")
        total = in_file if isinstance(in_file, int) and not isinstance(in_file, bool) else 0
        return self._cached_pages(self._work_dir(doc.doc_id)), total

    def bill_rows(self, doc: rb.BackfillDoc) -> Sequence[Any]:
        return self._read_saved_rows(self._work_dir(doc.doc_id), self.page_coverage(doc)[1])


def _list_documents(project_id: str) -> list[dict[str, Any]]:
    from app.core import projects

    return projects.list_documents(project_id, all_sessions=True)


def _load_rows(project_id: str, doc_id: str) -> list[Any]:
    from app.core import doc_rows

    return doc_rows.load_rows(project_id, doc_id)


def _materialize(doc: dict[str, Any]) -> str | None:
    from app.core import projects

    return projects.materialize_document_path(doc) if doc else None


def _chunk_texts(project_id: str, doc_id: str) -> list[str]:
    from sqlalchemy import text

    from app.core.db import SessionLocal
    from app.core.ingest_reconcile import _guard_chunk_tables

    out: list[str] = []
    with SessionLocal() as session:
        for table in _guard_chunk_tables(session):
            out += [str(t or "") for (t,) in session.execute(
                text(f"SELECT text FROM {table} WHERE project_id = :pid AND doc_id = :did"),
                {"pid": project_id, "did": doc_id},
            )]
    return out


def _read_saved_rows(work_dir: str, pages_in_file: int) -> list[Any]:
    """The bill's rows from its saved word boxes, every page in order."""
    from app.core import bill_derived
    from app.core.bill_table_geometry import extract_bill_rows

    words = []
    for index in range(pages_in_file):
        read = bill_derived.cached_page_read(bill_derived.cached_page_path(work_dir, index), index)
        words.extend(read.words)
    return extract_bill_rows(words)


def plan_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Counts, codes, ids and hashes of a plan; per-document fields only."""
    docs = [{k: d[k] for k in DOC_FIELDS if k in d} for d in plan.get("documents") or []]
    return {
        "project_id": plan.get("project_id"),
        "plan_hash": plan.get("plan_hash"),
        "documents_to_write": plan.get("documents_to_write"),
        "rows_to_write": plan.get("rows_to_write"),
        "documents": docs,
    }


def run(project_id: str, execute_hash: str | None, sources: rb.Sources,
        store: Callable[..., int] | None = None) -> dict[str, Any]:
    """The report of one plan or execute run (never prints)."""
    if execute_hash is None:
        return {"ok": True, "mode": MODE_PLAN,
                **plan_summary(rb.plan_rows_backfill(project_id, sources))}
    kwargs = {"store": store} if store is not None else {}
    result = rb.execute_rows_backfill(project_id, execute_hash, sources, **kwargs)
    return {
        "ok": bool(result["executed"] and result["complete"]),
        "mode": MODE_EXECUTE,
        "refused": result["refused"] is not None,
        "written": result["written"],
        "kept": result["kept"],
        "complete": result["complete"],
        "plan": plan_summary(result["plan"]),
        "after": plan_summary(result["after"]) if result["after"] else None,
    }


def main_from_env(
    env: Mapping[str, str] | None = None,
    sources: rb.Sources | None = None,
    log: Callable[[str], None] = print,
    store: Callable[..., int] | None = None,
) -> int:
    env = os.environ if env is None else env
    project_id = (env.get("ROWS_BACKFILL_PROJECT_ID") or "").strip()
    if not project_id:
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_MISSING_PROJECT}))
        return 2
    raw = (env.get("ROWS_BACKFILL_EXECUTE_HASH") or "").strip()
    if raw and not _HASH_RE.fullmatch(raw):
        log(SUMMARY_PREFIX + json.dumps({"ok": False, "error": ERR_BAD_HASH}))
        return 2
    sources = sources or LiveSources(data_dir=env.get("DATA_DIR"))
    try:
        report = run(project_id, raw or None, sources, store)
    except Exception as exc:  # noqa: BLE001 - the summary carries the class only
        report = {"ok": False, "error": ERR_RAISED, "error_class": type(exc).__name__}
    log(SUMMARY_PREFIX + json.dumps(bpo._public_value(report), sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main_from_env())
