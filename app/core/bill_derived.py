"""Derived spreadsheet for an image-only bill: its own document row (P-B3).

The owner's ruling (13 Nov 05:55Z): when a document classified as a bill BY
ITS OWN CONTENT is image-only, the corpus build produces a derived ``.xlsx``
(one sheet per bill section, canonical columns, ``Check`` and ``Source Page``),
stores it as its own documents row linked to the source (``derived_from`` in
metadata), and indexes it through the existing spreadsheet row-chunk path.
The source PDF row stays for provenance; its ``retrieval_visible`` is not
touched here.

Two steps, both refused inside a web request (the derivation renders and OCRs
every page; it runs in the ingest task or a dispatched ops task only):

* :func:`plan_bill_sheet` classifies and extracts rows and reports, per
  section path, the row count, the rows whose quantity x rate does not
  reconcile and the page range, plus a ``plan_hash``. It adds no row and
  indexes nothing.
* :func:`derive_bill_sheet` takes that hash, rebuilds the plan, refuses when
  it differs, and only then adds and indexes the derived row.

Pages are rendered and OCR'd one at a time; only each page's word boxes are
kept (in ``work_dir``, one JSON file per page), so a run over a page range
resumes where an earlier one stopped and a later run reads the boxes back
instead of OCRing again.

Pieces this module only joins, never re-implements:

* ``bill_classify.classify_image_only_bill`` - is it an image-only bill;
* ``bill_table_geometry.extract_bill_rows`` / ``write_bill_workbook`` - word
  boxes to rows to sheets;
* ``projects.materialize_document_path`` - readable bytes for a row (local,
  preview cache or Drive);
* ``projects.add_document`` (``reingest_of`` supersedes a previous derived
  row instead of deleting it) and ``doc_index.index_document``.

Nothing here names a bill, a section, a project, a page or a file.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable, Sequence
from typing import Any

from app.core import bill_classify as bc
from app.core import bill_table_geometry as g

logger = logging.getLogger(__name__)

DERIVATION = "p-b3_bill_sheet"

SKIP_NOT_PDF = "not_pdf"
SKIP_IS_DERIVED = "is_derived_document"
SKIP_NO_BYTES = "no_readable_bytes"
SKIP_NO_ROWS = "no_rows_extracted"
SKIP_PLAN_MISMATCH = "plan_hash_mismatch"
SKIP_PARTIAL_PLAN = "plan_not_whole_document"

Classify = Callable[[str], bc.BillVerdict]
PageWords = Callable[[str, Sequence[int]], list[g.Word]]


class WebRequestRefused(RuntimeError):
    """The derivation was called on behalf of an HTTP request."""


def refuse_in_web_request() -> None:
    from app.infra.monitoring import in_web_request

    if in_web_request():
        raise WebRequestRefused(
            "bill sheet derivation renders every page; it runs in the ingest "
            "or ops task, never in a web request",
        )


def _page_count(file_path: str) -> int:
    import fitz  # PyMuPDF

    from app.core import file_crypto

    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            return int(doc.page_count or 0)
        finally:
            doc.close()


def _page_words(
    file_path: str, index: int, page_words: PageWords, work_dir: str | None,
) -> list[g.Word]:
    """One page's word boxes: read back from ``work_dir`` or OCR'd and saved."""
    cached = os.path.join(work_dir, f"page_{index:05d}.json") if work_dir else None
    if cached and os.path.exists(cached):
        with open(cached, encoding="utf-8") as fh:
            return [g.Word(**w) for w in json.load(fh)]
    words = page_words(file_path, [index])
    if cached:
        tmp = cached + ".part"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump([dataclasses.asdict(w) for w in words], fh)
        os.replace(tmp, cached)
    return words


def _workbook_bytes(rows: Sequence[g.BillRow]) -> tuple[bytes, dict[str, int]]:
    fd, tmp = tempfile.mkstemp(suffix=".xlsx")
    os.close(fd)
    try:
        counts = g.write_bill_workbook(rows, tmp)
        with open(tmp, "rb") as fh:
            return fh.read(), counts
    finally:
        os.unlink(tmp)


def _previous_derived(project_id: str, source_id: str) -> str | None:
    """The visible derived row for ``source_id``, if one exists."""
    from app.core import projects

    for doc in projects.list_documents(project_id):
        if doc.get("retrieval_visible") is False:
            continue
        meta = projects.coerce_document_metadata(doc.get("metadata"))
        if meta.get("derived_from") == source_id:
            return doc["id"]
    return None


def _sections(rows: Sequence[g.BillRow]) -> list[dict[str, Any]]:
    """Per section path, in first-seen order: rows, unreconciled, pages."""
    out: dict[tuple[str, ...], dict[str, Any]] = {}
    for r in rows:
        key = tuple(r.path) or (r.section,)
        s = out.setdefault(key, {"path": list(key), "rows": 0, "unreconciled": 0,
                                 "pages": [r.source_page, r.source_page]})
        s["rows"] += 1
        s["unreconciled"] += 1 if r.check else 0
        s["pages"] = [min(s["pages"][0], r.source_page), max(s["pages"][1], r.source_page)]
    return list(out.values())


def _plan_hash(document_id: str, pages: tuple[int, int], rows: Sequence[g.BillRow]) -> str:
    body = json.dumps(
        {"document_id": document_id, "pages": list(pages),
         "rows": [dataclasses.asdict(r) for r in rows]},
        sort_keys=True, default=list,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _prepare(
    document_id: str,
    pages: tuple[int, int] | None,
    classify: Classify | None,
    page_words: PageWords | None,
    block,
    work_dir: str | None,
) -> dict[str, Any]:
    """Shared by plan and execute: a skip dict, or the plan with its rows."""
    from app.core import projects

    refuse_in_web_request()
    doc = projects.get_document(document_id) or {}
    meta = projects.coerce_document_metadata(doc.get("metadata"))
    if meta.get("derived_from"):
        return {"derived": False, "reason": SKIP_IS_DERIVED}
    name = doc.get("original_name") or ""
    if os.path.splitext(name)[1].lower() != ".pdf":
        return {"derived": False, "reason": SKIP_NOT_PDF}
    file_path = projects.materialize_document_path(doc)
    if not file_path:
        return {"derived": False, "reason": SKIP_NO_BYTES}

    verdict = (classify or bc.classify_image_only_bill)(file_path)
    if not verdict.is_bill:
        return {"derived": False, "reason": verdict.reason}

    total = _page_count(file_path)
    first, stop = pages if pages is not None else (0, total)
    first, stop = max(0, first), min(total, stop)
    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
    words: list[g.Word] = []
    for index in range(first, stop):
        words += _page_words(file_path, index, page_words or bc._ocr_sampled_pages, work_dir)
    rows = g.extract_bill_rows(words, block)
    if not rows:
        return {"derived": False, "reason": SKIP_NO_ROWS, "bill_verdict": verdict.reason}
    return {
        "doc": doc,
        "verdict": verdict,
        "rows": rows,
        "plan": {
            "derived": False,
            "reason": verdict.reason,
            "source_id": document_id,
            "pages": [first, stop],
            "pages_in_file": total,
            "rows": len(rows),
            "unreconciled": sum(1 for r in rows if r.check),
            "sections": _sections(rows),
            "plan_hash": _plan_hash(document_id, (first, stop), rows),
        },
    }


def plan_bill_sheet(
    project_id: str,
    document_id: str,
    *,
    pages: tuple[int, int] | None = None,
    classify: Classify | None = None,
    page_words: PageWords | None = None,
    block=None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """What the derivation would write, without writing it. ``pages`` is a
    0-based half-open range (default: every page)."""
    prepared = _prepare(document_id, pages, classify, page_words, block, work_dir)
    return prepared.get("plan", prepared)


def derive_bill_sheet(
    project_id: str,
    document_id: str,
    *,
    plan_hash: str,
    classify: Classify | None = None,
    page_words: PageWords | None = None,
    block=None,
    work_dir: str | None = None,
) -> dict[str, Any]:
    """Add and index the derived spreadsheet row for a whole-document plan
    whose hash is ``plan_hash``. Returns what happened, as counts and ids."""
    from app.core import byte_store, doc_index, projects

    prepared = _prepare(document_id, None, classify, page_words, block, work_dir)
    plan = prepared.get("plan")
    if plan is None:
        return prepared
    if plan["pages"] != [0, plan["pages_in_file"]]:
        return {"derived": False, "reason": SKIP_PARTIAL_PLAN}
    if plan["plan_hash"] != plan_hash:
        return {"derived": False, "reason": SKIP_PLAN_MISMATCH, "plan_hash": plan["plan_hash"]}
    rows, verdict, name = prepared["rows"], prepared["verdict"], prepared["doc"]["original_name"]

    data, sheets = _workbook_bytes(rows)
    stem = os.path.splitext(os.path.basename(name))[0]
    derived_name = f"{stem}.bill.xlsx"
    path = byte_store.allocate_path(f"{uuid.uuid4().hex[:12]}_{derived_name}")
    byte_store.write_bytes(path, data)
    previous = _previous_derived(project_id, document_id)
    row = projects.add_document(
        project_id,
        derived_name,
        stored_as=os.path.basename(path),
        file_path=path,
        size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
        metadata={
            "derived_from": document_id,
            "derivation": DERIVATION,
            "bill_verdict": verdict.reason,
            "plan_hash": plan_hash,
            "rows": len(rows),
            "rows_flagged": plan["unreconciled"],
            "sheets": sheets,
        },
        reingest_of=previous,
    )
    indexed = doc_index.index_document(
        project_id, row["id"], profile=doc_index.PROFILE_CORPUS_BUILD,
    )
    projects.update_document_metadata(document_id, {"derived_document": row["id"]})
    return {
        "derived": True,
        "reason": verdict.reason,
        "document_id": row["id"],
        "superseded": previous,
        "rows": len(rows),
        "rows_flagged": plan["unreconciled"],
        "sheets": len(sheets),
        "chunks": int(indexed.get("total_chunks") or 0),
        "index_status": indexed.get("status"),
    }
