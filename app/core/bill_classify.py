"""Is this image-only PDF a bill? Decided by its own content (P-B3).

The owner's ruling (13 Nov 05:55Z) converts an image-only bill into a derived
spreadsheet. That conversion runs only on documents that ARE bills, and the
decision must come from the page content, never from the file name.

Content evidence, read from OCR word boxes of a deterministic page sample:

* a column-header line that resolves through the BOQ processor's aliasing
  (a description column plus at least one figure column) -- the same test
  ``bill_table_geometry.detect_header`` applies when it extracts rows;
* enough measured rows under that header whose quantity or amount is a number;
* where a row prints quantity, rate and amount, the arithmetic holds
  (qty x rate == amount within rounding) on most of them. A priced bill
  reconciles; a table that merely has similar headings does not.

A PDF with a text layer is never a candidate: the derived sheet exists only
for a bill the text path cannot index.

Nothing here names a bill, a section, an item, a page or a file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from app.core import bill_table_geometry as g

# A table with fewer measured rows than this across the sample is not
# evidence of a bill (a cover page or a summary with one figure line).
_MIN_MEASURED_ROWS = 3
# Share of priced rows (qty, rate and amount all numeric) that must reconcile.
_MIN_RECONCILED_SHARE = 0.5

NOT_BILL_TEXT_LAYER = "has_text_layer"
NOT_BILL_NO_HEADER = "no_column_header"
NOT_BILL_FEW_ROWS = "too_few_measured_rows"
NOT_BILL_UNRECONCILED = "priced_rows_do_not_reconcile"
BILL_PRICED = "priced_rows_reconcile"
BILL_UNPRICED = "measured_rows_without_prices"


@dataclass(frozen=True)
class BillVerdict:
    is_bill: bool
    reason: str
    pages_sampled: int = 0
    header_pages: int = 0
    measured_rows: int = 0
    priced_rows: int = 0
    reconciled_rows: int = 0


def _is_measured(row: g.BillRow) -> bool:
    return any(
        g.parse_number(row.fields.get(f, "")) is not None
        for f in ("quantity", "total")
    )


def _is_priced(row: g.BillRow) -> bool:
    return all(
        g.parse_number(row.fields.get(f, "")) is not None
        for f in ("quantity", "rate", "total")
    )


def bill_verdict(words: Iterable[g.Word], block=None) -> BillVerdict:
    """Classify OCR word boxes from one or more sampled pages.

    Each page is read on its own (columns are not carried between sampled
    pages, which need not be adjacent).
    """
    block = block or g._resolver()
    by_page: dict[int, list[g.Word]] = {}
    for w in words:
        by_page.setdefault(w.page, []).append(w)

    header_pages = 0
    rows: list[g.BillRow] = []
    for page in sorted(by_page):
        page_words = by_page[page]
        if any(g.detect_header(line, block) for line in g.group_lines(page_words)):
            header_pages += 1
            rows += g.extract_bill_rows(page_words, block)

    measured = [r for r in rows if _is_measured(r)]
    priced = [r for r in measured if _is_priced(r)]
    reconciled = [r for r in priced if not g.check_row(r.fields)]
    counts = {
        "pages_sampled": len(by_page),
        "header_pages": header_pages,
        "measured_rows": len(measured),
        "priced_rows": len(priced),
        "reconciled_rows": len(reconciled),
    }
    if header_pages == 0:
        return BillVerdict(False, NOT_BILL_NO_HEADER, **counts)
    if len(measured) < _MIN_MEASURED_ROWS:
        return BillVerdict(False, NOT_BILL_FEW_ROWS, **counts)
    if not priced:
        return BillVerdict(True, BILL_UNPRICED, **counts)
    if len(reconciled) < _MIN_RECONCILED_SHARE * len(priced):
        return BillVerdict(False, NOT_BILL_UNRECONCILED, **counts)
    return BillVerdict(True, BILL_PRICED, **counts)


PageWords = Callable[[str, Sequence[int]], list[g.Word]]


def _ocr_sampled_pages(file_path: str, indices: Sequence[int]) -> list[g.Word]:
    """Render each sampled page and OCR it to word boxes, one page at a time."""
    import fitz  # PyMuPDF
    from PIL import Image

    from app.core import file_crypto

    words: list[g.Word] = []
    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            for index in indices:
                pix = doc[index].get_pixmap(dpi=200)
                image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                words += g.ocr_page_words(image, page=index + 1)
                del pix, image
        finally:
            doc.close()
    return words


def classify_image_only_bill(
    file_path: str,
    *,
    sample_pages: int = 5,
    page_words: PageWords | None = None,
    block=None,
) -> BillVerdict:
    """Verdict for one PDF: a bill only when it has no text layer AND its
    sampled pages carry bill content (see :func:`bill_verdict`).

    ``page_words`` maps (path, 0-based page indices) to word boxes; the default
    renders and OCRs the pages. The sample is the same deterministic spread
    ``pdf_has_text_layer`` reads (first, last, evenly spaced between).
    """
    import fitz  # PyMuPDF

    from app.core import doc_index, file_crypto

    if doc_index.pdf_has_text_layer(file_path, sample_pages=sample_pages):
        return BillVerdict(False, NOT_BILL_TEXT_LAYER)
    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            total = int(doc.page_count or 0)
        finally:
            doc.close()
    indices = doc_index._sample_page_indices(total, sample_pages)
    words = (page_words or _ocr_sampled_pages)(file_path, indices)
    return bill_verdict(words, block)
