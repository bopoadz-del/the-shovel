# Drawing Reader — Design Spec

**Goal:** Replace `app/blocks/drawing_qto.py` body with a deterministic, font-size-aware spatial parser for construction CAD drawings (DWG→PDF) so the Drive corpus indexer produces useful RAG chunks instead of CAD-tag soup.

**Author:** Operator (architecture), Claude (drafting)
**Status:** Draft for operator review
**Blocks:** Drive corpus big-batch indexing (#116). 2,092 of 2,997 docs in the Drive inventory are CAD-plotted drawings.

---

## Problem

The current pipeline routes drawing PDFs through `PDFBlockV2`'s bare `fitz.page.get_text()`. That returns every text label in the PDF — title-block, notes, callouts, dimensions, *and* the thousands of tiny CAD tag-IDs scattered across the drawing — concatenated without spatial reasoning. The resulting chunks look like:

```
S7  S4  S12  S8
PACKAGE C  PACKAGE 2  L.O.W
DE10-RC-01  DE10-PU-03
PROPOSED DETACHABLE BOLLARDS @2.00m INTERVAL
MATCH LINE : FOR REFERENCE REFER TO SHEET NO : 02
```

This is unusable for retrieval: useful strings (title block, notes, match lines) drown in label soup and any query keyword-matches against the noise.

Confirmed by the pilot on 9 drawings (project `drive_archive_drawings_test`, audit log `data/logs/drive_indexer_audit_drawings_pilot.jsonl`).

## Out of Scope (v1)

- **Vision-LLM understanding** of sheet contents (qwen2-vl, Claude Vision). Deferred to v2 if v1 quality is insufficient.
- **Schedules** embedded inside drawings (door schedules, equipment schedules). Deferred.
- **Drawing-type classification** (plan / section / elevation / detail) via vision. Deferred to v2.
- **Geometry-based QTO** (areas, volumes, line lengths). The existing geometry code paths in `drawing_qto.py` that operate on `fitz.page.get_drawings()` may stay if they exist and are working; v1 focuses on the **text** extraction failure.

## Design

### Library choice

**Use `pdfplumber`**. It exposes per-character `x0, y0, x1, y1, size, fontname` and per-page width/height — exactly what spatial parsing needs. pymupdf (`fitz`) could do this too via `get_text("dict")`, but `pdfplumber`'s API is tighter for the bbox-cluster + font-size workflow.

`pdfplumber>=0.11.0` is already listed (commented) in `requirements.txt`. Uncomment + install.

### Architecture (5 steps)

**Step 1 — Page region split.**
- Read page width and height from `page.width`, `page.height`.
- **Title block zone:** bottom 15% of page height, full width. (ISO 5457 standard DWG title block location.)
- **Drawing zone:** everything above the title block zone.
- For **landscape** pages (`page.rotation == 90 or 270`, or `width > height`), the same split applies in the rotated frame — pdfplumber returns coordinates in the rotated frame already.

**Step 2 — Title-block structured extraction.**

Within the title-block zone, extract these fields via proximity-grouped text clusters (text characters within 20px vertically of each other = same field):

| Field | Detection rule |
|---|---|
| `drawing_number` | Regex match: `[A-Z]{2,}-[A-Z]{2,}-\d{3}-\d{4}-[A-Z]{3,}-[A-Z]{3,}-[A-Z]{2,}-\d{3}-\d{6,7}-[A-Z]` (the JCB-DWG pattern observed in the corpus), OR shorter project-specific pattern as fallback |
| `drawing_title` | Longest text cluster in the title block zone that isn't matched by the other fields |
| `discipline` | Match against known the client project codes: TM, SW, SG, EL, LI, ST, WS, IR, TL, SE, SF, IF — extracted from the drawing-number pattern. Civil/structural/MEP/electrical/etc. — full discipline name from a lookup table |
| `revision` | Single letter or digit near a "Rev" label |
| `scale` | Pattern `1:\d+`, `NTS`, `N.T.S.`, `NOT TO SCALE` |
| `date` | Date pattern (DD/MM/YY, DD-MM-YYYY, DD.MM.YYYY) near "Date" label |
| `drafter` | Text near "Drawn", "Drafted", "By" labels |
| `checked_by` | Text near "Checked", "Reviewed" labels |
| `project_name` | Text near top of title block (project header), e.g. "the client project" |
| `sheet_number` | Pattern `Sheet \d+ of \d+`, `\d+/\d+`, `Sheet \d+` |

If `drawing_number` is not extractable, fall back to the file basename (without extension) and log a warning to the audit row.

**Step 3 — Drawing-zone text classification by font size.**

Extract all text objects (with size + bbox) from the drawing zone. Classify each:

- `size >= 4.0` pts → **notes_and_labels** — KEEP
- `2.0 <= size < 4.0` pts → **dimensions** — KEEP with prefix `"DIM: "`
- `size < 2.0` pts → **cad_tags** — FILTER OUT (this is the soup)

Then apply pattern filtering on the kept text (even large text can be a CAD tag):

- Drop strings matching pure CAD tag patterns:
  - All caps + digits + hyphens, no spaces, length 4-15: `DE10-PU-01`, `TM-2024`, `S7`, `R10`
  - Coordinate pairs: `\d+\.\d+\s*,\s*\d+\.\d+` (e.g., `1234.56, 7890.12`)
  - Single letters or digits isolated (text cluster of length 1-2)
- Drop strings repeating the same token more than 3 times in a row (e.g., `PACKAGE C PACKAGE C PACKAGE C ...`)

**Step 4 — Cross-reference extraction.**

Scan drawing-zone text for match-line and continuation references. Patterns:

- `SEE DWG\s+([A-Z0-9-]+)`
- `CONT\.?\s*ON\s+([A-Z0-9-]+)`
- `REF(?:ER)?\.?\s*(?:TO\s+)?(?:SHEET|DWG|DRAWING)\s+(?:NO\.?\s*)?([A-Z0-9-]+)`
- `MATCH\s*LINE.*?SHEET\s+(?:NO\.?\s*)?([A-Z0-9-]+)`

Each match becomes `{"ref_type": "<continuation|match_line|reference>", "target_drawing": "<extracted id>", "raw": "<full match string>"}`.

The collection of cross-refs across all drawings forms a **sheet connectivity graph** (deferred to a follow-up consumer; v1 just emits the cross_refs field).

**Step 5 — Output structure (per drawing PDF).**

```json
{
  "drawing_number": "AB-CDE-001-0000-XYZ-DWG-TM-200-0000001-A",
  "drawing_title": "Traffic Management Layout — Package C",
  "discipline": "TM",
  "discipline_full": "Traffic Management",
  "revision": "A",
  "scale": "1:500",
  "date": "2024-03-12",
  "drafter": "JCB",
  "checked_by": "ABC",
  "project_name": "the client project",
  "sheet_number": "1 of 4",
  "notes": [
    "All dimensions in mm unless noted otherwise",
    "Refer to specification section 02-100 for materials",
    "Proposed detachable bollards at 2.00m interval"
  ],
  "dimensions": [
    "DIM: 12.000",
    "DIM: 18.000",
    "DIM: 10.500"
  ],
  "cross_refs": [
    {"ref_type": "match_line", "target_drawing": "TM-200-1000006", "raw": "MATCH LINE: FOR REFERENCE REFER TO SHEET NO: 02"},
    {"ref_type": "continuation", "target_drawing": "TM-200-1000004", "raw": "CONT. ON SHEET 04"}
  ],
  "raw_chunk": "<the RAG-indexable text — see Chunking Strategy below>",
  "errors": []
}
```

`errors` collects non-fatal extraction issues (e.g., `"drawing_number_fallback_to_filename"`, `"title_block_zone_text_sparse"`) so the audit log surfaces quality drift.

### Chunking strategy

For RAG indexing, each drawing produces **ONE primary chunk** (`raw_chunk`):

```
{drawing_number} — {drawing_title} ({discipline_full}, Rev {revision})
Scale: {scale} | Date: {date} | Project: {project_name} | Sheet: {sheet_number}

Notes:
- {note 1}
- {note 2}
- {note 3}

References:
- {ref_type}: {target_drawing} ({raw})
```

Dimensions are NOT included in `raw_chunk` (they'd dilute the semantic signal). They're stored in the structured output for QTO consumers but not vector-indexed in v1.

This is what the Drive indexer chunks + embeds. ONE chunk per drawing, semantically dense, queryable by sheet number / discipline / project / note content.

### Multi-page drawings

Some drawings are multi-page (drawing set with cover sheet + multiple sheets in one PDF). Process each page through Steps 1–5 independently, then combine:

- Take the title-block fields from page 1 by default; if page N has a different `drawing_number`, treat it as a separate drawing (split the PDF logically).
- Concatenate notes across pages with page markers: `[Sheet 1] Note text...`, `[Sheet 2] Note text...`.
- Cross-refs accumulate across all pages.
- `raw_chunk` is built from the combined structured output.

If the PDF is logically multiple drawings (different `drawing_number` per page), emit ONE chunk per drawing with the right page assignment.

### Edge cases

| Case | Handling |
|---|---|
| Title block not found / drawing number unextractable | Use file basename as `drawing_number`, append error `"drawing_number_fallback_to_filename"` |
| Rotated drawings (landscape PDF) | pdfplumber handles in its native frame; if `page.rotation` is set, parse in the rotated frame |
| Multi-page drawing PDFs | Per Multi-page section above |
| Password-protected PDF | Catch pdfplumber's exception, log error, return `{errors: ["password_protected"]}` with no chunk |
| Scanned drawings (no text layer) | pdfplumber returns no chars. Fall back to OCR via existing `ocr.py` block: render at 300 DPI, run Tesseract `image_to_data` to get per-word bboxes + text. Reuse Steps 1–5 with bbox-only (no font-size info). Audit-log the fallback. Note: font-size-based dimension classification doesn't work on OCR'd output — treat all OCR text as notes |
| Empty PDF / corrupt | Catch the parse exception, return `{errors: [str(exc)]}` with no chunk |
| Arabic / mixed-language content (the client project) | pdfplumber handles Unicode text correctly. For scanned Arabic drawings, OCR fallback needs `RAG_OCR_LANG=eng+ara` — already documented as the Drive-indexer prerequisite. If `ara.traineddata` is missing locally, Arabic OCR degrades to English-only output (already flagged) |

### Implementation requirements

1. **Replace `app/blocks/drawing_qto.py` body.** Keep the class name `DrawingQTOBlock` and the async `process(input_data, params)` interface so the existing block registry + script wiring stays intact. Keep the existing geometry-extraction code paths if any are working (`fitz.get_drawings()` for areas / line lengths); the v1 change is purely about adding the new text-extraction logic. The output schema gains the new fields above.

2. **Uncomment + install pdfplumber** in `requirements.txt` (already listed as `# pdfplumber>=0.11.0     # PDF table extraction`). Run `pip install pdfplumber` in the venv. Add a `pip freeze` diff to the PR.

3. **Tests at `tests/test_drawing_qto.py`.** Fixture: a synthetic traffic-management sheet generated in the test (`tests/_synthetic_fixtures.build_drawing_detail_pdf`), carrying an invented long-form drawing number such as `QZ-SWK-100-0000-ABC-DWG-TM-200-0000101-A`. No client drawing is committed. Assertions:
   - `drawing_number` extracted (not the filename fallback)
   - `discipline` equals `"TM"`
   - `discipline_full` equals `"Traffic Management"`
   - `notes` is a list with 1 ≤ len ≤ 50 and total word count > 10
   - `raw_chunk` does NOT contain pure CAD-tag patterns (`DE\d+-[A-Z]+-\d+`, isolated `S\d+`/`R\d+`)
   - `raw_chunk` does NOT contain repeated tokens (no `PACKAGE C PACKAGE C` runs)
   - `cross_refs` parses at least one entry from the known match-line text on that fixture

   Run via `pytest tests/test_drawing_qto.py -v`.

4. **Audit-row extension.** When the Drive indexer calls `DrawingQTOBlock.process`, the per-doc audit row in `data/logs/drive_indexer_audit*.jsonl` gains:
   - `drawing_number` (string or null)
   - `discipline` (string or null)
   - `n_notes` (int)
   - `n_cross_refs` (int)
   - `cad_tags_filtered` (int — count of strings dropped by Step 3)
   - `extraction_path` (`"pdfplumber"` or `"ocr_fallback"`)
   - `errors` (array — same field as in output)

5. **Block-result shape** stays backward-compatible with whatever existing callers consume. The new structured fields go under `result["drawing"]` namespace, the chunk-ready text is at `result["text"]` so the Drive indexer's `_extract_pdf` substitute path stays the same.

## Validation Plan

After implementation, run the new `DrawingQTOBlock` against 5 specific drawings from the pilot batch (paths from `data/logs/drive_indexer_audit_drawings_pilot.jsonl`). Span at least 4 disciplines.

For each drawing, report:
- `drawing_number` extracted: yes / no / fallback
- `discipline` detected: <value>
- `n_notes` word count (should be ≥ 10 and ≤ 500)
- `cad_tags_filtered`: count removed
- `n_cross_refs`: count
- `raw_chunk`: first 200 chars preview

**Hard gate:** All 5 drawings must pass the `tests/test_drawing_qto.py` assertions. The Drive indexer and any batch run stay parked until the operator signs off on the validation report.

## Open Questions

- **None explicit.** Operator's spec is prescriptive. Surfacing two implicit questions in case clarification is wanted before plan-writing:
  1. Multi-page handling — should pages with different `drawing_number` produce separate audit rows (separate doc_ids), or stay as one composite chunk? Default in this spec: separate doc_ids per logical drawing.
  2. Scanned-drawing fallback path — should v1 include the OCR-bbox fallback, or is it acceptable to mark scanned drawings as `errors: ["no_text_layer"]` and skip them, deferring OCR-based parsing to v2? Default in this spec: include OCR fallback (degraded but useful).

## File / Path Index

- Spec: `docs/superpowers/specs/2026-06-11-drawing-reader-design.md` (this file)
- Plan: `docs/superpowers/plans/2026-06-11-drawing-reader.md` (to be written next)
- Code (modify): `app/blocks/drawing_qto.py`
- Code (modify): `requirements.txt` (uncomment pdfplumber)
- Test (new): `tests/test_drawing_qto.py`
- Test fixture (generated): `tests/_synthetic_fixtures.py` (synthetic drawing sheets built per run)
- Audit log (existing, extend rows): `data/logs/drive_indexer_audit*.jsonl`
- Validation output (new): `data/logs/drawing_reader_validation_<date>.md`
