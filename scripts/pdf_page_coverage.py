#!/usr/bin/env python3
"""Read-only PDF page-coverage diagnosis for a list of document ids.

For each id (a full id or an unambiguous prefix) this opens the stored file
the way ingest does (``file_crypto.open_plaintext`` + PyMuPDF) and prints:

* the page count and how many pages carry a text layer at or above the
  extractor's OCR threshold, and how many are image-only below it;
* the runs of image-only pages and the page where the text layer stops
  (the first page of a trailing image-only run);
* where the OCR page budget the extractor would apply runs out;
* chars by page range;
* the ingest gates the file meets: whole-PDF skip, OCR size skip, batched
  extraction, and whether the pdfplumber table probe would skip it (size,
  page count, largest page side).

Nothing here OCRs a page, re-extracts a document or writes to the database:
the SQL is parameterised SELECTs and each page is read with
``page.get_text()`` only. Documents are chosen by ``COVERAGE_DOC_IDS`` (comma
or space separated ids or prefixes, at most ``_MAX_DOCS``); no document
identifier or path lives in this repository.

The report is written to a workflow log on a public repository, so it prints
document ids, counts, page numbers and product-written codes ONLY: never a
file name, a path, a Drive path, a project id or any free text read from the
row, the metadata, the file or an exception message (an exception is named
by its class). ``withhold`` is the last step before printing: it replaces any
remaining occurrence of the row's own name, stem, path or project id.

Per document it also prints the index's own page coverage (stored chunks by
their ``page`` locator: which pages carry a chunk and how much text), read
from the chunk table, so it answers even when the source file is gone.

With ``COVERAGE_PAGELESS_SUMMARY`` set, each id is an anchor instead: the
run prints, for the anchor document's own project, how many chunks carry no
``page`` locator, by document id (counts and ids only, at most ``_MAX_DOCS``
documents listed). The project is read from the anchor's row, never taken
from an input, and never printed.

Per document it also prints the stored failure evidence (the ledger reason,
the parsed ``attempts_exhausted`` mark and the failure keys of the metadata)
and, once per run, the extraction budgets read from the product helpers, so
"why does this file hold 0 chunks" is answered from the row and the gate,
not from a guess.
"""
from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from typing import Any

# Invoked through the entrypoint's WORKER_COMMAND; make `app` importable for
# `python -m scripts...`, `python scripts/...` and direct runs alike.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

# Read-only. The id prefix is a bind parameter. The name and path columns are
# read to open the file and to withhold them from the output, never printed.
_COLS = (
    "SELECT id, project_id, original_name, size, file_path, ingest_status, "
    "chunk_count, pages_in_file, pages_ocrd, ingest_status_reason, metadata "
    "FROM documents "
)
# ``LIKE`` on the string id (prefix; the id is validated by parse_ids).
_DOC_SQL = _COLS + "WHERE id LIKE :p ORDER BY id"
# Metadata keys that carry failure evidence (matched on the key name).
_FAIL_KEY_RE = re.compile(r"error|fail|skip|reason|attempt|truncat|timeout|oom|limit", re.IGNORECASE)
# A product-written code (a reason segment, a metadata value): lower snake
# case. Anything else read from the row is free text, printed as its length.
_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
# Documents per run: the report is a public log, keep it short.
_MAX_DOCS = 20
_WITHHELD = "<withheld>"
_MIN_WITHHOLD_CHARS = 4

# Bounded output: at most this many chars-by-range rows per document.
_MAX_RANGE_ROWS = 25
_MIN_RANGE_PAGES = 10
_ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")


def parse_ids(raw: str) -> list[str]:
    """Split the env value into ids, dropping blanks and duplicates (order
    kept). Raises ValueError on a token that is not a plain id, so a stray
    ``%`` or quote never reaches the LIKE pattern."""
    out: list[str] = []
    for tok in re.split(r"[\s,]+", raw or ""):
        if not tok:
            continue
        if not _ID_RE.match(tok):
            raise ValueError(f"not a document id: {tok!r}")
        if tok not in out:
            out.append(tok)
    return out


def _metadata(raw: Any) -> dict[str, Any]:
    from app.core.projects import coerce_document_metadata

    return coerce_document_metadata(raw)


def terminal_meaning(reason: str | None) -> str | None:
    """What the ``attempts_exhausted`` mark in ``reason`` means under the
    product's own retry rule; None when the row carries no such mark."""
    from app.core import ingest_status as st

    m = re.search(re.escape(st.ATTEMPTS_MARK) + r"([^:|]+):(\d+):(\w+)", reason or "")
    if not m:
        return None
    version, n, tail = m.group(1), int(m.group(2)), m.group(3)
    if version == st.EXTRACTOR_VERSION:
        after = ("this IS the current extractor: the row stays closed until its "
                 "source bytes change or EXTRACTOR_VERSION is bumped")
    else:
        after = f"the current extractor is {st.EXTRACTOR_VERSION}, so the mark no longer closes the row"
    return (f"closed after {n} attempt(s) by extractor {version} ({tail}); "
            f"INGEST_MAX_ATTEMPTS now {st.max_attempts()}; {after}")


def safe_reason(reason: str | None) -> str:
    """The ``|``-separated reason with each product code kept and any other
    segment (free text that may quote a name or a path) printed as its length.
    The attempts mark is read by ``terminal_meaning``, not echoed."""
    from app.core import ingest_status as st

    if not reason:
        return "-"
    out = []
    for seg in str(reason).split("|"):
        if seg.startswith(st.ATTEMPTS_MARK):
            out.append("attempts_mark")
        elif _CODE_RE.match(seg):
            out.append(seg)
        else:
            out.append(f"<text len={len(seg)}>")
    return "|".join(out)


def safe_value(value: Any) -> str:
    """A metadata value as a count or a shape: numbers and booleans print,
    a product code prints, other text prints as its length, a mapping prints
    its numeric members under their keys, a list its length."""
    if isinstance(value, (bool, int, float)):
        return str(value)
    if value is None:
        return "-"
    if isinstance(value, str):
        return value if _CODE_RE.match(value) else f"<text len={len(value)}>"
    if isinstance(value, dict):
        parts = [f"{k}={v}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
                 if _KEY_RE.match(str(k)) and isinstance(v, (bool, int, float))]
        return "{" + ", ".join(parts[:10]) + f"}} keys={len(value)}"
    if isinstance(value, (list, tuple)):
        return f"<list len={len(value)}>"
    return f"<{type(value).__name__}>"


def failure_evidence(reason: str | None, meta: dict[str, Any]) -> list[str]:
    """The stored evidence of why a row holds what it holds, as codes and
    counts only (see the module docstring)."""
    lines = [f"   reason: {safe_reason(reason)}"]
    meaning = terminal_meaning(reason)
    if meaning:
        lines.append(f"   terminal: {meaning}")
    for key in sorted(k for k in meta if _KEY_RE.match(str(k)) and _FAIL_KEY_RE.search(str(k))):
        lines.append(f"   meta.{key}: {safe_value(meta[key])}")
    return lines


def withhold(lines: list[str], doc_row: tuple) -> list[str]:
    """Last step before printing: replace any occurrence of this row's file
    name, its stem, its stored path (and that path's file name and stem), its
    Drive path and its project id. Values shorter than ``_MIN_WITHHOLD_CHARS``
    are skipped so a one-letter stem cannot garble the counts."""
    pid, name, fpath = doc_row[1], doc_row[2], doc_row[4]
    meta = _metadata(doc_row[10]) if len(doc_row) > 10 else {}
    found: set[str] = set()
    for v in (name, fpath, pid, meta.get("drive_path")):
        if not v:
            continue
        v = str(v)
        base = v.replace("\\", "/").rsplit("/", 1)[-1]
        found.update({v, base, os.path.splitext(base)[0]})
    found = {f for f in found if len(f) >= _MIN_WITHHOLD_CHARS}
    if not found:
        return lines
    pat = re.compile("|".join(re.escape(f) for f in sorted(found, key=len, reverse=True)),
                     re.IGNORECASE)
    return [pat.sub(_WITHHELD, ln) for ln in lines]


def budgets() -> dict[str, Any]:
    """The extraction budgets that apply to every file, read from the product
    helpers (call-time env) so the report cannot drift from ingest."""
    from app.core import doc_index as di
    from app.core import extract_isolated as iso
    from app.core import ingest_status as st

    file_mem, file_timeout = di._file_budget()
    return {
        "pdf_max_size_mb": di.pdf_max_size_mb(),
        "file_mem_mb": file_mem,
        "file_timeout_s": file_timeout,
        "extract_child_mem_mb": iso._MEM_MB,
        "extract_child_timeout_s": iso._TIMEOUT_S,
        "max_attempts": st.max_attempts(),
        "extractor_version": st.EXTRACTOR_VERSION,
    }


def _runs(pages: Iterable[int]) -> list[tuple[int, int]]:
    """Collapse sorted 1-based page numbers into inclusive (start, end) runs."""
    runs: list[tuple[int, int]] = []
    for p in pages:
        if runs and p == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], p)
        else:
            runs.append((p, p))
    return runs


def range_size(n_pages: int) -> int:
    """Pages per chars-by-range row: whole pages, at most _MAX_RANGE_ROWS rows."""
    if n_pages <= 0:
        return _MIN_RANGE_PAGES
    per = -(-n_pages // _MAX_RANGE_ROWS)
    return max(_MIN_RANGE_PAGES, per)


def coverage(page_chars: list[int], threshold: int, ocr_budget: int) -> dict[str, Any]:
    """Pure page-coverage summary from per-page text-layer char counts.

    ``page_chars[i]`` is ``len(page.get_text().strip())`` for page i+1. A page
    below ``threshold`` is image-only to the extractor (it would OCR it while
    the budget lasts). ``ocr_budget`` is the number of image-only pages the
    extractor would OCR; the first image-only page past it is where the index
    silently stops gaining text.
    """
    n = len(page_chars)
    text_pages = [i + 1 for i, c in enumerate(page_chars) if c >= threshold]
    image_pages = [i + 1 for i, c in enumerate(page_chars) if c < threshold]
    last_text = text_pages[-1] if text_pages else 0
    # First page of the trailing image-only run; None when the last page has text.
    text_stops_at = last_text + 1 if last_text < n else None
    budget = max(0, int(ocr_budget))
    budget_exhausted_at = image_pages[budget] if len(image_pages) > budget else None
    step = range_size(n)
    ranges = []
    for start in range(0, n, step):
        chunk = page_chars[start:start + step]
        ranges.append({
            "pages": (start + 1, start + len(chunk)),
            "chars": sum(chunk),
            "text_pages": sum(1 for c in chunk if c >= threshold),
        })
    return {
        "pages": n,
        "text_pages": len(text_pages),
        "image_only_pages": len(image_pages),
        "image_only_runs": _runs(image_pages),
        "last_text_page": last_text,
        "text_stops_at": text_stops_at,
        "ocr_budget": budget,
        "ocr_budget_exhausted_at": budget_exhausted_at,
        "image_only_past_budget": max(0, len(image_pages) - budget),
        "chars_total": sum(page_chars),
        "chars_by_range": ranges,
    }


def ingest_gates(file_path: str, filename: str) -> dict[str, Any]:
    """The gates ingest applies to this PDF, read from the product helpers
    (call-time env), so the diagnosis cannot drift from the extractor."""
    from app.core import doc_index as di

    size_mb = di._document_size_mb(file_path)
    max_mb = di.pdf_max_size_mb()
    ocr_max = di.pdf_ocr_max_size_mb(filename)
    ocr_skipped = ocr_max > 0 and size_mb > ocr_max
    return {
        "size_mb": round(size_mb, 1),
        "skipped_too_large": max_mb > 0 and size_mb > max_mb,
        "pdf_max_size_mb": max_mb,
        "ocr_max_size_mb": ocr_max,
        "ocr_skipped_over_size": ocr_skipped,
        "ocr_page_cap": 0 if ocr_skipped else di.pdf_ocr_page_cap(filename),
        "batched": di._pdf_should_batch(file_path),
        "table_probe_runs": di._pdf_tables_enabled(file_path),
        "tables_max_pages": int(os.getenv("PDF_TABLES_MAX_PAGES", "200")),
        "tables_max_page_pt": float(os.getenv("PDF_TABLES_MAX_PAGE_PT", "1250")),
        "ocr_threshold": di._PDF_OCR_THRESHOLD,
    }


def read_page_chars(file_path: str) -> tuple[list[int], float, str | None]:
    """Per-page text-layer char counts, the largest page side in points, and
    the error that ended the read early (None when every page was read).
    Never OCRs. A mid-document exception keeps the pages already counted and
    names the page it stopped at, which is the defect this diagnosis hunts."""
    import fitz

    from app.core import file_crypto

    chars: list[int] = []
    max_side = 0.0
    error: str | None = None
    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            for i in range(doc.page_count):
                try:
                    page = doc.load_page(i)
                    max_side = max(max_side, page.rect.width, page.rect.height)
                    chars.append(len((page.get_text() or "").strip()))
                except Exception as exc:  # report it (by class), do not hide it
                    error = f"page {i + 1}: {type(exc).__name__}"
                    break
        finally:
            doc.close()
    return chars, max_side, error


# Stored chunks of one document by page (read-only). The table name comes from
# the product's single chokepoint for the namespaced chunk table, which
# validates the identifier; project and document id are bind parameters, so
# the read uses the (project_id, doc_id) index.
def _stored_pages_sql(table: str) -> str:
    return (
        f"SELECT page, COUNT(*), COALESCE(SUM(LENGTH(text)), 0) FROM {table} "
        "WHERE project_id = :pr AND doc_id = :d GROUP BY page"
    )


def stored_chunk_table() -> str:
    from app.core.models import rag_chunk_table_name

    return rag_chunk_table_name(os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip())


def stored_page_coverage(rows: Iterable[tuple], pages_in_file: int | None) -> dict[str, Any]:
    """Pure summary of the index's own page locators for one document.

    ``rows`` is ``(page, chunks, chars)`` per stored page (page None = chunks
    with no locator). The span is the file's page count when the row records
    it, else the highest page that carries a chunk; ``pages_without_chunk``
    are the pages of that span no chunk points at. This answers "is page N
    indexed, and with how much text" from the index alone, so it still
    answers when the source bytes are gone."""
    per_page: dict[int, tuple[int, int]] = {}
    no_page = (0, 0)
    for page, n, chars in rows:
        n, chars = int(n or 0), int(chars or 0)
        if page is None or int(page) < 1:
            no_page = (no_page[0] + n, no_page[1] + chars)
            continue
        a, b = per_page.get(int(page), (0, 0))
        per_page[int(page)] = (a + n, b + chars)
    top = max(per_page) if per_page else 0
    span = max(int(pages_in_file or 0), top)
    missing = [p for p in range(1, span + 1) if p not in per_page]
    step = range_size(span)
    by_range = []
    for start in range(1, span + 1, step):
        end = min(span, start + step - 1)
        hits = [per_page[p] for p in range(start, end + 1) if p in per_page]
        by_range.append({
            "pages": (start, end),
            "pages_with_chunk": len(hits),
            "chunks": sum(h[0] for h in hits),
            "chars": sum(h[1] for h in hits),
        })
    return {
        "chunks": sum(v[0] for v in per_page.values()) + no_page[0],
        "chunks_without_page": no_page[0],
        "chars_without_page": no_page[1],
        "pages_with_chunk": len(per_page),
        "span": span,
        "span_from": "pages_in_file" if pages_in_file and int(pages_in_file) >= top else "highest_chunk_page",
        "pages_without_chunk": len(missing),
        "pages_without_chunk_runs": _runs(missing),
        "by_range": by_range,
    }


def render_stored(cov: dict[str, Any]) -> list[str]:
    runs = cov["pages_without_chunk_runs"]
    shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs[:40])
    more = f" (+{len(runs) - 40} more runs)" if len(runs) > 40 else ""
    lines = [
        (f"   stored: chunks={cov['chunks']} pages_with_chunk={cov['pages_with_chunk']} "
         f"span={cov['span']} ({cov['span_from']}) pages_without_chunk={cov['pages_without_chunk']} "
         f"chunks_without_page={cov['chunks_without_page']} "
         f"chars_without_page={cov['chars_without_page']}"),
        f"   stored pages_without_chunk_runs: {shown or '-'}{more}",
        "   stored pages | pages_with_chunk | chunks | chars",
    ]
    for r in cov["by_range"]:
        a, b = r["pages"]
        lines.append(f"   {a}-{b} | {r['pages_with_chunk']} | {r['chunks']} | {r['chars']}")
    return lines


def stored_lines(doc_row: tuple) -> list[str]:
    """The index's own page coverage for one document; a failed read is
    printed, never raised, so the file-side diagnosis still prints."""
    did, pid, pif = doc_row[0], doc_row[1], doc_row[7]
    from app.core.db import SessionLocal

    try:
        sql = _stored_pages_sql(stored_chunk_table())
        with SessionLocal() as s:
            rows = s.execute(text(sql), {"pr": pid, "d": did}).all()
    except Exception as exc:  # report it (by class), do not hide it
        return [f"   stored: READ FAILED: {type(exc).__name__}"]
    return render_stored(stored_page_coverage([tuple(r) for r in rows], pif))


# Chunks with no page locator, by document, across ONE project (read-only).
# The project is the anchor document's own (read from its row, never from an
# input), and it is a bind parameter, so the read uses the project index.
def _pageless_sql(table: str) -> str:
    return (
        "SELECT doc_id, "
        "SUM(CASE WHEN page IS NULL OR page < 1 THEN 1 ELSE 0 END), COUNT(*) "
        f"FROM {table} WHERE project_id = :pr GROUP BY doc_id"
    )


def pageless_summary(rows: Iterable[tuple], cap: int = _MAX_DOCS) -> dict[str, Any]:
    """Pure count of chunks without a page locator, by document id.

    ``rows`` is ``(doc_id, chunks_without_page, chunks)`` per document of one
    project. A chunk with no page cannot be cited by page, so this is the
    size of that defect in the working set: how many documents carry one,
    how many chunks, and the ``cap`` documents with the most (ties by id)."""
    docs = [(str(d), int(n or 0), int(t or 0)) for d, n, t in rows]
    hit = sorted((r for r in docs if r[1] > 0), key=lambda r: (-r[1], r[0]))
    return {
        "documents": len(docs),
        "documents_with_pageless": len(hit),
        "chunks": sum(r[2] for r in docs),
        "pageless_chunks": sum(r[1] for r in hit),
        "top": hit[:cap],
    }


def render_pageless(anchor: str, summ: dict[str, Any]) -> list[str]:
    lines = [
        (f"== pageless in the project of id {anchor}: documents={summ['documents']} "
         f"documents_with_pageless={summ['documents_with_pageless']} "
         f"chunks={summ['chunks']} pageless_chunks={summ['pageless_chunks']}"),
        "   doc_id | pageless_chunks | chunks",
    ]
    for did, n, t in summ["top"]:
        lines.append(f"   {did} | {n} | {t}")
    more = summ["documents_with_pageless"] - len(summ["top"])
    if more > 0:
        lines.append(f"   (+{more} more documents with pageless chunks)")
    return lines


def pageless_lines(anchor: str, doc_row: tuple) -> list[str]:
    """The pageless count for the anchor document's project; a failed read
    is printed by class, never raised. The project id is never printed."""
    from app.core.db import SessionLocal

    try:
        sql = _pageless_sql(stored_chunk_table())
        with SessionLocal() as s:
            rows = s.execute(text(sql), {"pr": doc_row[1]}).all()
    except Exception as exc:  # report it (by class), do not hide it
        return [f"== pageless in the project of id {anchor}: READ FAILED: {type(exc).__name__}"]
    return render_pageless(anchor, pageless_summary([tuple(r) for r in rows]))


def render(doc_row: tuple, cov: dict[str, Any] | None, gates: dict[str, Any] | None,
           max_side: float, error: str | None) -> list[str]:
    did, _pid, _name, size, _fpath, status, nchunks, pif, pocr, reason, meta = doc_row
    lines = [
        f"== doc {did}",
        (f"   db: size={size} ingest_status={status} chunks={nchunks} "
         f"pages_in_file={pif} pages_ocrd={pocr}"),
    ]
    lines.extend(failure_evidence(reason, _metadata(meta)))
    if gates and gates["skipped_too_large"]:
        lines.append(
            f"   SIZE GATE: {gates['size_mb']} MB > pdf_max_size_mb="
            f"{gates['pdf_max_size_mb']}: ingest skips the whole PDF, "
            "its text pages included"
        )
    if gates:
        lines.append(
            "   gates: size_mb={size_mb} skipped_too_large={skipped_too_large} "
            "ocr_skipped_over_size={ocr_skipped_over_size} (ocr_max_mb={ocr_max_size_mb}) "
            "ocr_page_cap={ocr_page_cap} batched={batched} "
            "table_probe_runs={table_probe_runs} (max_pages={tables_max_pages}, "
            "max_page_pt={tables_max_page_pt}) ocr_threshold={ocr_threshold}".format(**gates)
        )
    if cov is not None:
        lines.append(
            f"   pages={cov['pages']} text_pages={cov['text_pages']} "
            f"image_only={cov['image_only_pages']} last_text_page={cov['last_text_page']} "
            f"text_stops_at={cov['text_stops_at']} max_page_side_pt={max_side:.0f}"
        )
        lines.append(
            f"   ocr_budget={cov['ocr_budget']} ocr_budget_exhausted_at_page="
            f"{cov['ocr_budget_exhausted_at']} image_only_past_budget="
            f"{cov['image_only_past_budget']} chars_total={cov['chars_total']}"
        )
        runs = cov["image_only_runs"]
        shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in runs[:40])
        more = f" (+{len(runs) - 40} more runs)" if len(runs) > 40 else ""
        lines.append(f"   image_only_runs: {shown or '-'}{more}")
        lines.append("   pages | chars | text_pages")
        for r in cov["chars_by_range"]:
            a, b = r["pages"]
            lines.append(f"   {a}-{b} | {r['chars']} | {r['text_pages']}")
    if error:
        lines.append(f"   READ STOPPED EARLY: {error}")
    return lines


def diagnose(doc_row: tuple) -> list[str]:
    _did, _pid, name, _size, fpath, *_ = doc_row
    if not fpath:
        return render(doc_row, None, None, 0.0, "no stored file path on the row")
    if not os.path.exists(fpath):
        return render(doc_row, None, None, 0.0, "file not on this filesystem")
    if os.path.splitext(name or fpath)[1].lower() != ".pdf":
        return render(doc_row, None, None, 0.0, "not a PDF: page read skipped")
    try:
        gates = ingest_gates(fpath, name or "")
        chars, max_side, error = read_page_chars(fpath)
    except Exception as exc:
        return render(doc_row, None, None, 0.0, f"open failed: {type(exc).__name__}")
    cov = coverage(chars, gates["ocr_threshold"], gates["ocr_page_cap"])
    return render(doc_row, cov, gates, max_side, error)


def report(doc_row: tuple) -> list[str]:
    """Every printed line for one document, already withheld."""
    return withhold(diagnose(doc_row) + stored_lines(doc_row), doc_row)


# Every line the script prints carries this mark; the workflow publishes the
# marked lines only, never the rest of the container stream (entrypoint and
# library output it does not control).
MARK = "PPC| "


def emit(line: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    stream.write(f"{MARK}{line}" + chr(10))
    stream.flush()


def _one(prefix: str) -> bool:
    """Report one id; True when it printed exactly one document."""
    from app.core.db import SessionLocal

    with SessionLocal() as s:
        rows = s.execute(text(_DOC_SQL), {"p": prefix + "%"}).all()
    if len(rows) != 1:
        emit(f"== id {prefix}: {len(rows)} matching documents, need exactly 1 -- skipped")
        return False
    row = tuple(rows[0])
    lines = (withhold(pageless_lines(prefix, row), row) if pageless_mode()
             else report(row))
    for line in lines:
        emit(line)
    return True


def pageless_mode() -> bool:
    """``COVERAGE_PAGELESS_SUMMARY``: each id is an anchor; print the
    pageless-chunk count of its project instead of the per-page report."""
    return os.environ.get("COVERAGE_PAGELESS_SUMMARY", "").strip().lower() in ("1", "true", "yes")


def main() -> int:
    if os.environ.get("COVERAGE_DRIVE_PATH", "").strip():
        # Withdrawn: a path selector puts a client folder name into a public
        # run log. The value is not echoed.
        emit("COVERAGE_DRIVE_PATH is withdrawn: select by document id", err=True)
        return 2
    try:
        ids = parse_ids(os.environ.get("COVERAGE_DOC_IDS", ""))
    except ValueError:
        emit("COVERAGE_DOC_IDS holds a token that is not a document id", err=True)
        return 2
    if not ids:
        emit("COVERAGE_DOC_IDS is required", err=True)
        return 2
    if len(ids) > _MAX_DOCS:
        emit(f"{len(ids)} ids given, at most {_MAX_DOCS} per run", err=True)
        return 2
    try:
        emit("budgets: " + " ".join(f"{k}={v}" for k, v in budgets().items()))
    except Exception as exc:  # by class only: a message can quote a path
        emit(f"budgets: unreadable ({type(exc).__name__})")
    failed = 0
    for prefix in ids:
        try:
            ok = _one(prefix)
        except Exception as exc:  # by class only: a message can quote a path or a name
            emit(f"== id {prefix}: failed ({type(exc).__name__})")
            ok = False
        failed += 0 if ok else 1
    emit(f"done: ids={len(ids)} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
