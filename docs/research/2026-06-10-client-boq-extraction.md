# the client BOQ extraction — research note

Source: Kimi research dispatch 2026-06-10 (`/tmp/kimi-fleet/20260610-011403-206/`).

## Hypothesis (ranked)

1. **Reading-order collapse in dense multi-column tables** — highest
   likelihood. PyMuPDF `get_text()` emits text in content-stream order,
   not visual reading order. BOQ columns (Item / Description / Unit /
   Qty / Rate / Amount) interleave, producing the "garbled chunks"
   symptom. Fix: **pdfplumber** with `extract_table()` + visual tolerances.

2. **Scanned image pages with no text layer** — high. Tender annexes,
   stamped approvals, faxed rate sheets. Our 30-char fallback threshold
   is brittle: a page with just header/footer text passes the gate and
   OCR is skipped. Fix: **Tesseract 5 via pytesseract, `lang='ara+eng'`**,
   render to 300 DPI.

3. **Lattice tables with ruled borders + merged cells** — moderate.
   PyMuPDF has no table semantics. Fix: **camelot-py Lattice mode**
   (OpenCV line detection, outputs pandas DataFrames).

## Two-pass strategy (mixed scanned + digital)

- **Pass 1 page triage**: `page.get_text("text")`; if >200 coherent
  chars AND contains BOQ keywords (Qty, Unit, Rate, SR), mark digital;
  else render to 300 DPI image and mark scanned.
- **Pass 2 extraction**: digital → pdfplumber (fallback camelot for
  heavily ruled sheets); scanned → Tesseract `ara+eng`. Normalize
  both into a unified markdown-table or CSV structure before chunking.

## Targets for a 9.7 MB BOQ

- Chunk count: 80-150 (chunk by work section, e.g. "Division 03 - Concrete")
- Average chunk length: 1000-2000 chars (~250-500 tokens)
- "Good" gate: >90% of chunks contain recognizable qty/item/unit triples,
  garbled-char rate <5%
- If <50 chunks OR avg <300 chars → over-fragmented, table structure
  destroyed.

## Acceptance criteria for a follow-up PR

- `app/core/doc_index.py` PDF branch gets a pdfplumber fallback that
  activates when PyMuPDF returns a chunk count <50 OR avg chunk length
  <300 chars
- Add `RAG_PDF_TABLE_FALLBACK=auto|on|off` env (default `auto`)
- Tesseract `ara+eng` only fires on pages with <30 text-layer chars
  (current behaviour kept for non-BOQ docs)
- Validate via the new `/v1/admin/debug/doc-extract` endpoint on
  the client doc `c6dae280`: target 80+ chunks, avg >1000 chars

## Live diagnostic (2026-06-10, against production)

`GET /v1/admin/debug/doc-extract?project_id=3f6f28b2&document_id=c6dae280`
returned:

- `pdf_page_count`: 16 (the file is 9.7 MB but only 16 pages — embedded
  images / fonts inflate the bytes; the actual text is ~24K chars)
- `pdf_first_pages_chars`: [176, 1459, 1529, 1676, 1923, 1982, 1680,
  1597, 1419, 1566] — page 1 is a thin header (176 chars), pages 2-10
  carry the BOQ rows
- `indexed_chunk_count`: 8
- `indexed_chunks_avg_chars`: 2798 (way over Kimi's target of 1000-2000)

Chunks contain real BOQ line items + prices, e.g. `D 999.1 Nr 897
1,275.00 1,143,675.00`. The first chunk also contains Arabic mojibake
(`'yLjUgLiiflJIg jIojIlUIBtJgiall`) — that is the Arabic header line
being mangled by PyMuPDF's get_text() because the PDF stores Arabic
glyphs without proper CMAP. Tesseract with `lang='ara+eng'` (rendered
at 300 DPI) is the right fix for that subset of pages.

The chat-side symptom ("I was not able to process it") is consistent
with this state: chunks exist, but each chunk is so big (700 tokens)
that finding a specific rate buried inside one is hard for the model.
**Two distinct improvements needed:**

1. **Smaller chunks** — re-chunk by BOQ section header / line group,
   not by 500-word window. Target 100-300 chars per chunk so a
   specific rate is in a focused chunk.
2. **Arabic OCR layer** — Tesseract `ara+eng` for pages with
   detectable Arabic-glyph mojibake (heuristic: if get_text() output
   contains characters outside the basic-Latin + standard-symbol
   ranges at a rate >5%, treat the page as Arabic and OCR).

Both improvements are scoped to PDF documents with mixed Arabic-Latin
content; they should NOT touch the working extraction for English-only
PDFs (RFP/BOD on the Anthropic project).
