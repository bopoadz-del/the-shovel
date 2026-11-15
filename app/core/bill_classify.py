"""Is this PDF a bill the derived sheet should carry? Decided by its own content (P-B3).

The owner's ruling (13 Nov 05:55Z) converts an image-only bill into a derived
spreadsheet. That conversion runs only on documents that ARE bills, and the
decision must come from the page content, never from the file name.

Content evidence, read from the word boxes of a deterministic page sample:

* a column-header line that resolves through the BOQ processor's aliasing
  (a description column plus at least one figure column) -- the same test
  ``bill_table_geometry.detect_header`` applies when it extracts rows;
* enough measured rows under that header whose quantity or amount is a number;
* where a row prints quantity, rate and amount, the arithmetic holds
  (qty x rate == amount within rounding) on most of them. A priced bill
  reconciles; a table that merely has similar headings does not.

Word boxes come per PAGE from the best source that page has (see
:func:`read_pages`): a page whose text layer passes the index's own OCR test
(``doc_index._PDF_OCR_THRESHOLD``) gives its words with their boxes from that
layer and is never OCR'd; any other page is rendered and OCR'd. Both land in
the same pixel space, so the geometry downstream cannot tell them apart. A
file is therefore judged page by page, never by whether some page has text.

Candidacy is decided by what the index already holds, not by the file kind:
a bill whose rows the text path indexed as row chunks (its BOQ parse recorded
priced rows, :func:`rows_indexed_as_rows`) is never a candidate; one whose
row parse did not succeed is, when its content passes the verdict above.

Nothing here names a bill, a section, an item, a page or a file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from app.core import bill_table_geometry as g

# A table with fewer measured rows than this across the sample is not
# evidence of a bill (a cover page or a summary with one figure line).
_MIN_MEASURED_ROWS = 3
# Share of priced rows (qty, rate and amount all numeric) that must reconcile.
_MIN_RECONCILED_SHARE = 0.5

NOT_CANDIDATE_ROWS_INDEXED = "rows_indexed_as_rows"
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
    #: Diagnostic counts (no labels, no text). They explain a verdict; the
    #: verdict never reads them. ``pages_read`` counts every page judged
    #: (``pages_sampled`` only those that gave words), by word source, and
    #: the pages that gave none. ``lines_tested`` lines met the header test;
    #: ``header_lines`` passed it and every other line is counted under the
    #: first reason it failed (``bill_table_geometry.HEADER_REJECTIONS``).
    pages_read: int = 0
    pages_text_layer: int = 0
    pages_ocr: int = 0
    pages_without_words: int = 0
    lines_tested: int = 0
    header_lines: int = 0
    rejected_fewer_than_3_cells: int = 0
    rejected_no_description_column: int = 0
    rejected_no_figure_column: int = 0


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
    lines_tested = 0
    header_lines = 0
    rejected = dict.fromkeys(g.HEADER_REJECTIONS, 0)
    rows: list[g.BillRow] = []
    for page in sorted(by_page):
        page_words = by_page[page]
        lines = g.group_lines(page_words)
        lines_tested += len(lines)
        page_headers = 0
        for line in lines:
            columns, why = g.header_test(line, block)
            if columns:
                page_headers += 1
            else:
                rejected[why] += 1
        header_lines += page_headers
        if page_headers:
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
        "lines_tested": lines_tested,
        "header_lines": header_lines,
        **{f"rejected_{why}": n for why, n in rejected.items()},
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


#: Page word sources. ``text_layer``: the page's own digital text, with its
#: boxes. ``ocr``: the page rendered at :data:`RENDER_DPI` and OCR'd.
SOURCE_TEXT_LAYER = "text_layer"
SOURCE_OCR = "ocr"
#: Render resolution for OCR; text-layer boxes (PDF points) are scaled into
#: the same pixel space so one geometry serves both sources.
RENDER_DPI = 200


@dataclass(frozen=True)
class PageRead:
    """One page's word boxes and the source that produced them."""

    index: int          # 0-based page index
    source: str         # SOURCE_TEXT_LAYER or SOURCE_OCR
    words: tuple[g.Word, ...]


PageSource = Callable[[str, Sequence[int]], list[PageRead]]


def page_has_text_layer(page) -> bool:
    """The index's own per-page test: a text layer shorter than
    ``doc_index._PDF_OCR_THRESHOLD`` is image-only and gets OCR'd there too."""
    from app.core import doc_index

    return len((page.get_text() or "").strip()) >= doc_index._PDF_OCR_THRESHOLD


def text_layer_words(page, number: int) -> list[g.Word]:
    """Word boxes from a page's text layer, in the rendered image's pixel space
    (display orientation, :data:`RENDER_DPI`)."""
    import fitz  # PyMuPDF

    scale = RENDER_DPI / 72.0
    to_display = page.rotation_matrix
    out: list[g.Word] = []
    for x0, y0, x1, y1, raw, *_rest in page.get_text("words"):
        text = (raw or "").strip()
        if not text:
            continue
        r = fitz.Rect(x0, y0, x1, y1) * to_display
        out.append(g.Word(
            text=text, page=number,
            left=r.x0 * scale, top=r.y0 * scale,
            width=r.width * scale, height=r.height * scale,
        ))
    return out


def _ocr_page(page, number: int) -> list[g.Word]:
    from PIL import Image

    pix = page.get_pixmap(dpi=RENDER_DPI)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return g.ocr_page_words(image, page=number)
    finally:
        del pix, image


def read_pages(file_path: str, indices: Sequence[int]) -> list[PageRead]:
    """Each asked page's word boxes from its best source, one page at a time."""
    import fitz  # PyMuPDF

    from app.core import file_crypto

    out: list[PageRead] = []
    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            for index in indices:
                page = doc[index]
                if page_has_text_layer(page):
                    words, source = text_layer_words(page, index + 1), SOURCE_TEXT_LAYER
                else:
                    words, source = _ocr_page(page, index + 1), SOURCE_OCR
                out.append(PageRead(index, source, tuple(words)))
        finally:
            doc.close()
    return out


def rows_indexed_as_rows(document: dict | None) -> bool:
    """Whether the index already holds this document's bill rows as row
    chunks: its BOQ parse succeeded with priced rows (the attribute
    ``doc_index`` records next to the line-item chunks it emits). A parse
    that failed, raised, found no priced rows or never ran leaves none."""
    from app.core import doc_index

    priced = (doc_index.boq_total_status(document) or {}).get("priced_rows")
    return isinstance(priced, (int, float)) and not isinstance(priced, bool) and priced > 0


def classify_bill_pdf(
    file_path: str,
    *,
    sample_pages: int = 5,
    page_source: PageSource | None = None,
    block=None,
) -> BillVerdict:
    """Verdict for one PDF from its sampled pages' content (see
    :func:`bill_verdict`), each page read from its own best source.

    ``page_source`` maps (path, 0-based page indices) to :class:`PageRead`;
    the default is :func:`read_pages`. The sample is the deterministic spread
    ``doc_index._sample_page_indices`` gives (first, last, evenly between).
    """
    import fitz  # PyMuPDF

    from app.core import doc_index, file_crypto

    with file_crypto.open_plaintext(file_path) as readable:
        doc = fitz.open(readable)
        try:
            total = int(doc.page_count or 0)
        finally:
            doc.close()
    indices = doc_index._sample_page_indices(total, sample_pages)
    return verdict_of_reads((page_source or read_pages)(file_path, indices), block)


def verdict_of_reads(reads: Sequence[PageRead], block=None) -> BillVerdict:
    """The verdict over exactly the pages in ``reads``, with their source counts.

    :func:`classify_bill_pdf` passes its small page sample (the cheap gate);
    the bill PLAN and the derivation pass every page they read, so a bundle
    whose covers, preambles and summaries fall on the sample is judged by its
    measured pages too. Same tests, same bars, either way."""
    verdict = bill_verdict([w for r in reads for w in r.words], block)
    return replace(verdict, **read_source_counts(reads))


def read_source_counts(reads: Sequence[PageRead]) -> dict[str, int]:
    """Pages judged, by word source, and how many gave no words.

    ``bill_verdict`` sees only words, so a page with none (a blank page, or
    one OCR could not read) is invisible to its ``pages_sampled``; this
    counts it from the reads themselves."""
    return {
        "pages_read": len(reads),
        "pages_text_layer": sum(1 for r in reads if r.source == SOURCE_TEXT_LAYER),
        "pages_ocr": sum(1 for r in reads if r.source == SOURCE_OCR),
        "pages_without_words": sum(1 for r in reads if not r.words),
    }
