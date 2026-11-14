"""Derived spreadsheet for an image-only bill: its own document row (P-B3).

The owner's ruling (13 Nov 05:55Z): when a document classified as a bill BY
ITS OWN CONTENT is image-only, the corpus build produces a derived ``.xlsx``
(one sheet per bill section, canonical columns, ``Check`` and ``Source Page``),
stores it as its own documents row linked to the source (``derived_from`` in
metadata), and indexes it through the existing spreadsheet row-chunk path.
The source PDF row stays for provenance; its ``retrieval_visible`` is not
touched here.

Pieces this module only joins, never re-implements:

* ``bill_classify.classify_image_only_bill`` - is it an image-only bill;
* ``bill_table_geometry.extract_bill_rows`` / ``write_bill_workbook`` - word
  boxes to rows to sheets;
* ``projects.add_document`` (``reingest_of`` supersedes a previous derived
  row instead of deleting it) and ``doc_index.index_document``.

Nothing here names a bill, a section, a project, a page or a file.
"""

from __future__ import annotations

import hashlib
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
SKIP_NO_FILE = "no_file_path"
SKIP_NO_ROWS = "no_rows_extracted"

Classify = Callable[[str], bc.BillVerdict]
PageWords = Callable[[str, Sequence[int]], list[g.Word]]


def _page_count(file_path: str) -> int:
    import fitz  # PyMuPDF

    from app.core import file_crypto

    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            return int(doc.page_count or 0)
        finally:
            doc.close()


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


def derive_bill_sheet(
    project_id: str,
    document_id: str,
    *,
    classify: Classify | None = None,
    page_words: PageWords | None = None,
    block=None,
) -> dict[str, Any]:
    """Classify one source document and, when it is an image-only bill, add
    and index its derived spreadsheet row. Returns what happened, as counts
    and ids (``derived`` True/False plus ``reason``)."""
    from app.core import byte_store, doc_index, projects

    doc = projects.get_document(document_id) or {}
    meta = projects.coerce_document_metadata(doc.get("metadata"))
    if meta.get("derived_from"):
        return {"derived": False, "reason": SKIP_IS_DERIVED}
    name = doc.get("original_name") or ""
    if os.path.splitext(name)[1].lower() != ".pdf":
        return {"derived": False, "reason": SKIP_NOT_PDF}
    file_path = doc.get("file_path")
    if not file_path:
        return {"derived": False, "reason": SKIP_NO_FILE}

    verdict = (classify or bc.classify_image_only_bill)(file_path)
    if not verdict.is_bill:
        return {"derived": False, "reason": verdict.reason}

    words = (page_words or bc._ocr_sampled_pages)(
        file_path, range(_page_count(file_path)),
    )
    rows = g.extract_bill_rows(words, block)
    if not rows:
        return {"derived": False, "reason": SKIP_NO_ROWS, "bill_verdict": verdict.reason}

    data, sheets = _workbook_bytes(rows)
    stem = os.path.splitext(os.path.basename(name))[0]
    derived_name = f"{stem}.bill.xlsx"
    path = byte_store.allocate_path(f"{uuid.uuid4().hex[:12]}_{derived_name}")
    byte_store.write_bytes(path, data)
    previous = _previous_derived(project_id, document_id)
    flagged = sum(1 for r in rows if r.check)
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
            "rows": len(rows),
            "rows_flagged": flagged,
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
        "rows_flagged": flagged,
        "sheets": len(sheets),
        "chunks": int(indexed.get("total_chunks") or 0),
        "index_status": indexed.get("status"),
    }
