# Drawing Reader Phase 1.5 — Title / Match-Line Fixes

**Parent spec:** `2026-06-11-drawing-reader-design.md` (Phase 1, now landed)
**Scope:** Three bug fixes surfaced by 5-drawing validation. No new features.
**Out of scope:** Issue #4 (zero dimensions on 3/5 sheets) and #5 (key-plan label contamination of notes) — deferred to Phase 1.6.

---

## Bug 1 — `drawing_title` returns drawing-number-with-typo

**Observed (SG):** `drawing_title: "QZ-ENG-310-0000-EXC-DWG-SG-200-100100A0"` — the drawing number itself, with a clustering artifact appending the revision letter.
**Observed (EL):** Same pattern.

**Root cause:** The title-block extractor picks "longest text cluster" as the title. The longest cluster is often the drawing number (a long contiguous text run) or a near-duplicate with bad word-boundary clustering.

**Fix:**
1. When selecting `drawing_title` from title-block clusters, exclude any cluster whose normalized form matches the extracted `drawing_number` or contains a substring of length ≥ 12 in common with it. Normalize by removing non-alphanumerics.
2. If no remaining cluster qualifies, set `drawing_title = None` and append `"drawing_title_not_found"` to `errors`. **Do not** fabricate a title from drawing-number remnants.

**Acceptance:** SG and EL no longer return `drawing_title` matching their drawing number. TL (`SECTIONAL ELEVATION`) and others with real titles still pass through unchanged.

---

## Bug 2 — Truncated EXC regex match accepted as final answer (WS case)

**Observed (WS):** `drawing_number: "QZ-ENG-310-EXC"`, `discipline: None`, `revision: "EXC"`.

**Root cause:** The drawing-number regex has a primary pattern (long EXC) and a fallback short pattern (`[A-Z]{2,}-[A-Z]{2,}-\d+-\w+`). The short fallback matched `QZ-ENG-310-EXC` from a random text fragment in the title block, and the code accepted it without re-running the full fallback chain. The actual WS drawing number is `QZ-ENG-310-0000-EXC-DWG-WS-600-0000001-C` per filename.

**Fix:**
1. Reject any drawing-number match that does NOT contain `EXC-DWG-` as a substring on this corpus. If the long regex didn't match and the short regex didn't include `EXC-DWG-`, the title-block search has failed for the drawing-number field — trigger the operator's title-block fallback chain (right 20% → full page).
2. If even the full-page scan can't find a `EXC-DWG-` pattern, fall back to filename-derived drawing number per the existing spec rule and append `"drawing_number_fallback_to_filename"` to errors. Do NOT return a half-matched substring.
3. Re-derive `discipline` and `revision` from the corrected drawing number. The discipline parser was getting `None` because the truncated number had no discipline code; the revision parser was returning `"EXC"` because it grabbed the literal string from the truncated tail.

**Acceptance:** WS returns `drawing_number: "QZ-ENG-310-0000-EXC-DWG-WS-600-0000001"` (or `-C` revision if the parser captures it), `discipline: "WS"`, `discipline_full: "Water Supply"`, `revision: "C"`.

---

## Bug 3 — Cross-ref regex misses real matches AND over-matches noise

**Observed (TM):** Drawing literally contains `MATCH LINE : FOR REFERENCE REFER TO SHEET NO : 02` text (confirmed in the earlier dumb-pilot chunk preview) but `n_cross_refs: 0`.
**Observed (SG):** `n_cross_refs: 1755` — clearly over-matching every label fragment.

**Root cause:**
- The `MATCH\s*LINE.*?SHEET\s+(?:NO\.?\s*)?([A-Z0-9-]+)` regex's `.*?` is non-greedy but doesn't tolerate the colon-with-spaces between `LINE` and `FOR REFERENCE`. Run it case-insensitively against the actual TM text — it likely fails on `LINE :` (space before colon) or on the word `FOR` between LINE and SHEET.
- The `REF(?:ER)?\.?\s*(?:TO\s+)?(?:SHEET|DWG|DRAWING)\s+(?:NO\.?\s*)?([A-Z0-9-]+)` regex matches any `REF...SHEET...<some token>` and the SG drawing has hundreds of label strings that fit that loose shape.

**Fix:**
1. Make all four cross-ref regexes case-insensitive, multiline, and tolerant of arbitrary whitespace (including `\s+:?\s+`) between tokens.
2. Tighten the captured `target_drawing` group: only match drawing-number-shaped tokens (`[A-Z0-9]+(?:-[A-Z0-9]+){2,}` OR plain `\d{2,3}` for short sheet refs like `SHEET NO 02`). Reject 1-character matches or matches that don't look like sheet identifiers.
3. Deduplicate cross-refs by `(ref_type, target_drawing)` tuple before returning. SG's 1755 is almost certainly many repeats of the same handful of refs.
4. Add a sanity cap: if after dedup the count is > 100 for a single sheet, append `"cross_refs_count_suspect_over_100"` to `errors` and trim to the first 100 (alphabetical). This is a guardrail against future regex regressions.

**Acceptance:**
- TM returns ≥ 1 cross_ref whose `raw` contains `MATCH LINE` and `SHEET`.
- SG returns ≤ 50 unique cross_refs (after dedup).
- WS, EL, TL counts stay roughly the same (0, 1, 0).

---

## Implementation Constraints

- Modify only `app/blocks/drawing_qto.py`. No spec docs to update except this fix-list.
- All 7 existing tests in `tests/test_drawing_qto.py` must continue to pass.
- Add THREE new tests in `tests/test_drawing_qto.py`:
  - `test_drawing_title_not_equal_to_drawing_number` — applies to TM fixture; assert `drawing["drawing_title"]` does not contain `drawing["drawing_number"]` as a substring (after non-alphanumeric normalization).
  - `test_cross_refs_match_line_detected` — applies to TM fixture; assert at least one cross_ref has `raw` containing `MATCH LINE` (case-insensitive).
  - `test_cross_refs_deduplicated` — applies to TM fixture; assert no two entries in `cross_refs` share the same `(ref_type, target_drawing)` pair.

- Re-run `scripts/_validate_drawing_reader.py` against all 5 pilot drawings after fixes land. Print the updated table.

## Reporting back

- Diff summary (which functions changed, ~lines)
- All 10 tests (7 existing + 3 new) pytest output
- Updated 5-row validation table
- For each of the 5 drawings: before/after for `drawing_title`, `drawing_number`, `discipline`, `n_cross_refs`, first cross_ref `raw` if any

STOP after validation. Operator signs off before Phase 1.6.
