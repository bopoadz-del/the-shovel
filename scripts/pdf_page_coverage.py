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
the SQL is one parameterised SELECT and each page is read with
``page.get_text()`` only. Ids arrive in ``COVERAGE_DOC_IDS`` (comma or space
separated); no document identifier lives in this repository.
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

# Read-only. The prefix is a bind parameter; ``LIKE`` on the string id.
_DOC_SQL = (
    "SELECT id, project_id, original_name, size, file_path, ingest_status, "
    "chunk_count, pages_in_file, pages_ocrd FROM documents "
    "WHERE id LIKE :p ORDER BY id"
)

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
                except Exception as exc:  # report it, do not hide it
                    error = f"page {i + 1}: {type(exc).__name__}: {str(exc)[:200]}"
                    break
        finally:
            doc.close()
    return chars, max_side, error


def render(doc_row: tuple, cov: dict[str, Any] | None, gates: dict[str, Any] | None,
           max_side: float, error: str | None) -> list[str]:
    did, pid, name, size, _fpath, status, nchunks, pif, pocr = doc_row
    lines = [
        f"== doc {did} | project {pid} | {name}",
        (f"   db: size={size} ingest_status={status} chunks={nchunks} "
         f"pages_in_file={pif} pages_ocrd={pocr}"),
    ]
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
    if not fpath or not os.path.exists(fpath):
        return render(doc_row, None, None, 0.0, f"file not on this filesystem: {fpath!r}")
    try:
        gates = ingest_gates(fpath, name or "")
        chars, max_side, error = read_page_chars(fpath)
    except Exception as exc:
        return render(doc_row, None, None, 0.0, f"open failed: {type(exc).__name__}: {str(exc)[:200]}")
    cov = coverage(chars, gates["ocr_threshold"], gates["ocr_page_cap"])
    return render(doc_row, cov, gates, max_side, error)


def main() -> int:
    try:
        ids = parse_ids(os.environ.get("COVERAGE_DOC_IDS", ""))
    except ValueError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if not ids:
        sys.stderr.write("COVERAGE_DOC_IDS is required\n")
        return 2
    from app.core.db import SessionLocal

    failed = 0
    for prefix in ids:
        with SessionLocal() as s:
            rows = s.execute(text(_DOC_SQL), {"p": prefix + "%"}).all()
        if len(rows) != 1:
            print(f"== id {prefix}: {len(rows)} matching documents, need exactly 1 -- skipped")
            failed += 1
            continue
        for line in diagnose(tuple(rows[0])):
            print(line, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
