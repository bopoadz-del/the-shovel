"""Drawing QTO Block - Quantity Take-Off from DXF and PDF construction drawings.

DWG is not read (owner ruling): a DWG input is refused with
``app.core.cad_formats.DWG_NOT_SUPPORTED``.

V1.2 swaps the text-extraction path from pdfplumber to PyMuPDF (fitz).
pdfplumber was a 80-180s/drawing bottleneck on dense CAD PDFs; fitz
returns the same span-level data in a fraction of the time. The
font-size buckets, candidate filters, drawing-number regex,
cross-ref extraction, multi-page combine, fallback chains, and the
chunk-builder are unchanged.

Coordinate-system note: pdfplumber returned y0 in PDF user-space
(0 = bottom of page, increasing upward). PyMuPDF returns y0 top-down
(0 = top of page, increasing downward). Every comparison that referred
to "bottom 15%" or sorted top-down has been flipped accordingly.
"""

import json
import logging
import math
import os
import re
from typing import Any

from app.core.cad_formats import dwg_not_supported, is_dwg
from app.core.universal_base import UniversalBlock

_logger = logging.getLogger(__name__)

# A page whose text layer is shorter than this is treated as text-free (CAD
# plot with text as curves, or a raster scan) and eligible for the OCR
# fallback. A real plan's text layer with room labels carries hundreds of
# characters per drawing; plots render titles as curves and yield ~0.
_OCR_TEXT_THRESHOLD = int(os.getenv("QTO_OCR_TEXT_THRESHOLD", "80"))

# F26c: skip the vector-geometry pass on pages whose content stream exceeds
# this (dense CAD plots: ~5 MB / ~110k objects per A1 page measured live;
# normal floor plans are ~257 KB). Rooms and OCR still run on such pages.
_GEOMETRY_MAX_CONTENT_BYTES = int(os.getenv("QTO_GEOMETRY_MAX_CONTENT_BYTES", "2000000"))


# --- Discipline -------------------------------------------------------------
# A drawing's discipline is read from its title block's DISCIPLINE label. A
# drawing-number code (a two-letter segment) means something only within one
# project's numbering scheme, so the code carries no table of them: when a
# code has to be named, the mapping comes from the open project's profile --
# the project fact below, a JSON object {"<code>": "<discipline>"} recorded
# from the project's own drawing register. With no label and no mapping the
# discipline is reported as unknown, never guessed.
DISCIPLINE_CODES_FACT = "drawing_discipline_codes"
DISCIPLINE_UNKNOWN = "unknown"
_DN_SPLIT_RE = re.compile(r"[-_./\s]+")


def _project_discipline_codes(project_id: str | None) -> dict[str, str]:
    """The open project's drawing discipline codes, ``{CODE: name}``.

    Read from the project profile (``DISCIPLINE_CODES_FACT``). Empty when no
    project is open, the project records no mapping, or the stored value is
    not a JSON object of code -> name.
    """
    if not project_id:
        return {}
    try:
        from app.core import projects as projects_store

        fact = projects_store.get_fact(str(project_id), DISCIPLINE_CODES_FACT)
    except Exception:
        _logger.warning("project discipline codes unreadable for %s", project_id,
                        exc_info=True)
        return {}
    if not fact:
        return {}
    try:
        raw = json.loads(fact.get("value") or "")
    except (TypeError, ValueError):
        _logger.warning("project %s fact %s is not JSON", project_id,
                        DISCIPLINE_CODES_FACT)
        return {}
    if not isinstance(raw, dict):
        return {}
    codes: dict[str, str] = {}
    for k, v in raw.items():
        code, name = str(k).strip().upper(), str(v).strip()
        if code and name:
            codes[code] = name
    return codes


def _resolve_discipline(labelled: str | None, drawing_number: str | None,
                        codes: dict[str, str]) -> tuple[str | None, str | None, str]:
    """``(discipline, discipline_full, source)`` for one drawing.

    1. The title block's DISCIPLINE label, as written; a project code mapping
       names it when the label carries a code.
    2. Else a segment of the drawing number that the project's mapping names.
    3. Else unknown: no code is guessed.
    """
    if labelled:
        return labelled, codes.get(labelled.strip().upper(), labelled), "title_block"
    if codes and drawing_number:
        for seg in _DN_SPLIT_RE.split(drawing_number.upper()):
            if seg in codes:
                return seg, codes[seg], "project_profile"
    return None, None, DISCIPLINE_UNKNOWN

# Long hyphenated drawing-number shape (any originator code).
# Two token orders both appear in the wild:
#   AB-CDE-001-0000-KLM-DWG-QA-200-0000001-A   (zone, then originator)
#   AB-CDE-001-KLM-0000-DWG-QB-600-0000001-C   (tokens 4-5 swapped)
# Accept both by alternation. Shorter fallback covers project-specific
# schemes that don't use the full two-group prefix.
_DWG_NUMBER_FULL = re.compile(
    r"[A-Z]{2,}-[A-Z]{2,}-\d{3}-"
    r"(?:\d{4}-[A-Z]{3,}|[A-Z]{3,}-\d{4})-"
    r"[A-Z]{3,}-[A-Z]{2,}-\d{3}-\d{6,7}(?:-[A-Z0-9]+)?"
)
_DWG_NUMBER_SHORT = re.compile(r"[A-Z]{2,}-[A-Z]{2,}-\d{2,}-[A-Z0-9]+")


_DN_SEGMENT_RE = re.compile(r"[A-Z0-9]+")


def _is_full_drawing_number(s: str | None) -> bool:
    """True iff ``s`` has the SHAPE of a complete long-form drawing number.

    Structural, never an originator's code: nine or more hyphen-separated
    segments, each a run of letters/digits, with at least two numeric and at
    least three alphabetic segments (project/package letters, originator,
    document type, discipline). A half-match cut from a title-block fragment
    ("AB-CDE-001-KLM") has too few segments; a title has spaces.

    Used by both the title-block band-preference pick (in ``_process_page``)
    and the rescue path (in ``_extract_drawing_text``).
    """
    if not s:
        return False
    segs = s.strip().upper().split("-")
    if len(segs) < 9 or not all(_DN_SEGMENT_RE.fullmatch(x) for x in segs):
        return False
    return sum(x.isdigit() for x in segs) >= 2 and sum(x.isalpha() for x in segs) >= 3


def _revision_from_number_tail(dn: str | None) -> str | None:
    """The revision carried as the last hyphenated token, or None.

    A revision is a short code (``A``, ``C2``, ``P01``). A pure number is a
    sheet sequence, and a three-letter word is a code token left by a
    truncated number (originator, document type), never a revision.
    """
    tail = (dn or "").rsplit("-", 1)[-1]
    if not (1 <= len(tail) <= 3 and re.fullmatch(r"[A-Z0-9]+", tail)):
        return None
    if tail.isdigit() or (len(tail) == 3 and tail.isalpha()):
        return None
    return tail


# Title-block field labels. Generic drawing vocabulary only -- never a firm,
# person or place name. A value is found by STRUCTURE: inline after the label
# and a separator, in the cell to the label's right, or in the cell directly
# below it. Order matters: earlier alternatives win at the same position.
_TB_LABEL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("drawing_number",
     r"(?:DRAWING|DRG|DWG)\.?\s*(?:NO|NUMBER|NUM)\.?"),
    ("project_name", r"PROJECT(?:\s+(?:NAME|TITLE))?"),
    ("drawing_title", r"(?:DRAWING\s+|SHEET\s+)?TITLE"),
    ("drafter", r"DRAWN(?:\s+BY)?|DRAFTED(?:\s+BY)?|DRAFTER|DRAFTSMAN"),
    ("checked_by", r"CHECKED(?:\s+BY)?"),
    ("approved_by", r"APPROVED(?:\s+BY)?"),
    ("designed_by", r"DESIGNED(?:\s+BY)?"),
    ("scale", r"SCALE"),
    ("revision", r"REV(?:ISION)?\.?"),
    ("date", r"DATE"),
    ("sheet_number", r"SHEET(?:\s+(?:NO|NUMBER))?\.?"),
    ("discipline", r"DISCIPLINE"),
    ("consultant", r"CONSULTANT|CLIENT|CONTRACTOR|EMPLOYER|ENGINEER"),
)
_TB_LABEL_RE = re.compile(
    r"(?<![A-Z0-9])(?:"
    + "|".join(f"(?P<{f}>{p})" for f, p in _TB_LABEL_PATTERNS)
    + r")(?![A-Z0-9])"
)
_TB_SEP_RE = re.compile(r"\s*[:\-=]\s*")
_SCALE_RE = re.compile(r"(1\s*:\s*\d{1,5}|N\.?T\.?S\.?|NOT\s*TO\s*SCALE)", re.IGNORECASE)
_DATE_RE = re.compile(r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b")
_SHEET_VALUE_RE = re.compile(r"^(\d+(?:\s*(?:OF|/)\s*\d+)?)\b", re.IGNORECASE)
_CODE_VALUE_RE = re.compile(r"[A-Z0-9]+(?:[-_./][A-Z0-9]+)+|[A-Z]*\d[A-Z0-9]*")
# Fields whose value has a checkable shape may follow the label inline with
# no separator ("SHEET 3 OF 7"); free-text fields need a separator or a cell
# of their own, so "PROJECT SYSTEM ..." is never read as a project name.
_TB_SHAPED_FIELDS = frozenset({"drawing_number", "scale", "revision", "date", "sheet_number"})
# Fields whose value may wrap onto following lines of the same cell.
_TB_WRAPPING_FIELDS = frozenset({"drawing_title", "project_name"})


def _norm_tb_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").upper()).strip()


def _tb_label_at_start(text: str):
    """The field label a title-block cell opens with, or None."""
    m = _TB_LABEL_RE.match(text.strip().upper())
    if not m:
        return None
    return m


def _validate_tb_value(field: str, value: str) -> str | None:
    """The value cut to its field's shape, or None when it has no such shape."""
    v = (value or "").strip().strip(":-=").strip()
    if not v:
        return None
    vu = v.upper()
    if field == "drawing_number":
        m = _CODE_VALUE_RE.search(vu.split(" ")[0]) if vu else None
        return m.group(0) if m and any(ch.isdigit() for ch in m.group(0)) else None
    if field == "scale":
        m = _SCALE_RE.search(v)
        return m.group(1).strip() if m else None
    if field == "date":
        m = _DATE_RE.search(v)
        return m.group(1) if m else None
    if field == "revision":
        tok = vu.split(" ")[0]
        return tok if re.fullmatch(r"[A-Z0-9]{1,3}", tok) else None
    if field == "sheet_number":
        m = _SHEET_VALUE_RE.match(v)
        return m.group(1).strip() if m else None
    if _TB_LABEL_RE.match(vu) or len(v) > 160:
        return None
    return v


def _strip_doubled_letter_prefix(dn: str) -> str:
    """Strip stray leading characters that fall outside a clean hyphenated
    drawing-number prefix.

    Bug observed in pilot: one sheet returned ``AAB-CDE-001-...`` because
    the source text run was something like ``XAAB-CDE-001-...`` and the
    regex `[A-Z]{2,}-[A-Z]{2,}-...` legitimately accepted ``XAAB`` (or in
    a leading position, ``AAB``). A negative-lookbehind in the regex does
    not help: at string start there's no preceding char, so
    ``XAAB-CDE-...`` produces ``AAB-...`` and ``AAB-CDE-...`` produces
    ``AAB-...`` again.

    Strategy: peel one leading char at a time as long as the remainder
    still matches the same full-or-short pattern. The pattern's minimum
    first token is two letters, so this stops once the first token shrinks
    to that length. Single-pass over the
    string; cheap and pattern-aware.
    """
    if not dn:
        return dn
    candidate = dn
    while len(candidate) > 2 and candidate[0].isalpha():
        peeled = candidate[1:]
        # Only peel while the remainder STILL parses as a drawing
        # number with the same suffix. Use fullmatch to make sure we're
        # not accidentally shrinking past the prefix.
        if (_DWG_NUMBER_FULL.fullmatch(peeled) or
                _DWG_NUMBER_SHORT.fullmatch(peeled)):
            candidate = peeled
            continue
        break
    return candidate


# Phase 1.7: reject drawing-title candidates whose final char is a single
# lowercase letter directly after an uppercase run (e.g. "KEY PLANg" —
# a fitz-only artifact where a subscript glyph bled into the title span).
# Legitimate mixed-case titles like "Section A-A" and "CONCRETE ENCASEMENT"
# do not match this shape.
_TRAILING_LOWERCASE_ARTIFACT_RE = re.compile(r"[A-Z]{2,}[a-z]$")


def _has_trailing_lowercase_artifact(text: str) -> bool:
    """True if a candidate ends in ``[A-Z]{2,}[a-z]`` — caught by
    Phase 1.7 filter. Operates on the trimmed last token to ignore
    trailing punctuation."""
    if not text:
        return False
    tail = text.strip().split()[-1] if text.strip() else ""
    # Strip trailing punctuation so e.g. ``KEY PLANg.`` still trips
    while tail and not tail[-1].isalnum():
        tail = tail[:-1]
    return bool(_TRAILING_LOWERCASE_ARTIFACT_RE.search(tail))


# Phase 1.8: Levenshtein dedup for repetitive legend / schedule tables
# (observed on one pilot sheet: 161 notes vs 2-4 for other disciplines of similar
# size). Many near-identical notes survive the existing CAD-tag /
# pure-numeric filters — e.g. five "ISSUED FOR CONSTRUCTION
# (CONDITIONAL) NN DD/MM/YY AJ" revision-history rows. We drop any note
# whose Levenshtein distance from an already-accepted note is < 5.
try:
    # Prefer the C-extension if it's available; otherwise fall back to a
    # hand-rolled DP. The lists are small (<= a few hundred notes per
    # drawing, each <= 200 chars), so either is fast enough.
    from Levenshtein import distance as _lev_distance  # type: ignore
    _LEV_IMPL = "Levenshtein"
except ImportError:
    _LEV_IMPL = "hand_rolled"

    def _lev_distance(a: str, b: str) -> int:  # type: ignore
        """Classic DP Levenshtein. O(len(a) * len(b)) time, O(min) space."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        # Make `a` the shorter string for the O(min) row buffer.
        if len(a) > len(b):
            a, b = b, a
        prev = list(range(len(a) + 1))
        for i, cb in enumerate(b, 1):
            curr = [i] + [0] * len(a)
            for j, ca in enumerate(a, 1):
                cost = 0 if ca == cb else 1
                curr[j] = min(
                    curr[j - 1] + 1,        # insert
                    prev[j] + 1,            # delete
                    prev[j - 1] + cost,     # substitute
                )
            prev = curr
        return prev[-1]


def _note_near_duplicate(
    candidate: str, accepted: list[str], max_distance: int = 5
) -> bool:
    """True if ``candidate`` is within ``max_distance`` Levenshtein
    edit-distance of ANY string already in ``accepted``.

    Early-out: if ``abs(len(candidate) - len(existing)) > max_distance``
    the distance is necessarily greater, so skip the DP entirely. Cheap
    O(1) length check protects us against the worst case where the
    candidate is much longer/shorter than every accepted entry.
    """
    if not candidate:
        return False
    cand_len = len(candidate)
    for existing in accepted:
        if abs(cand_len - len(existing)) > max_distance:
            continue
        if _lev_distance(candidate, existing) < max_distance:
            return True
    return False


def _to_metres_factor(doc_units: int) -> float:
    """Map ezdxf unit codes (INSUNITS) to a multiplier that converts to metres.

    Reference ezdxf.units constants:
        0 = Unitless, 1 = Inches, 2 = Feet, 4 = Millimeters,
        5 = Centimeters, 6 = Meters
    Anything else falls back to 0.001 (assume mm), preserving prior behaviour.
    """
    mapping = {
        1: 0.0254,   # inches  -> m
        2: 0.3048,   # feet    -> m
        4: 0.001,    # mm      -> m
        5: 0.01,     # cm      -> m
        6: 1.0,      # m       -> m
    }
    try:
        return mapping.get(int(doc_units), 0.001)
    except (TypeError, ValueError):
        return 0.001


class DrawingQTOBlock(UniversalBlock):
    name = "drawing_qto"
    version = "1.0.0"
    description = "Extract measurements, areas, and volumes from DXF and PDF construction drawings"
    layer = 3
    tags = ["domain", "construction", "drawing", "qto", "dxf", "quantities"]
    requires = []

    default_config = {
        "unit_scale": 1.0,       # multiplier if drawing units ≠ mm
        "area_layer_filter": [],  # empty = all layers
        "min_area_m2": 0.01,
    }

    ui_schema = {
        "input": {
            "type": "file",
            "accept": [".dxf", ".pdf"],
            "placeholder": "Upload DXF or PDF drawing...",
        },
        "output": {
            "type": "table",
            "fields": [
                {"name": "measurements", "type": "list", "label": "Linear Measurements"},
                {"name": "areas", "type": "list", "unit": "m²", "label": "Areas"},
                {"name": "estimated_volumes", "type": "list", "unit": "m³", "label": "Estimated Volumes (area × assumed height)"},
                {"name": "total_area_m2", "type": "number", "unit": "m²", "label": "Total Area"},
            ],
        },
        "quick_actions": [
            {"icon": "", "label": "Full QTO", "prompt": "Extract all quantities from this drawing"},
            {"icon": "", "label": "Measurements", "prompt": "List all linear measurements"},
            {"icon": "", "label": "Floor Areas", "prompt": "Calculate floor areas by room"},
        ],
    }

    async def process(self, input_data: Any, params: dict = None) -> dict:
        params = params or {}
        data = input_data if isinstance(input_data, dict) else {}

        # Support string path input directly, or InputAdapter {"text": "/path/to/file.dxf"}
        if isinstance(input_data, str) and not data:
            file_path = input_data
        else:
            file_path = data.get("file_path") or params.get("file_path") or data.get("text") or data.get("input") or ""
        if not file_path:
            return {"status": "error", "error": "No file_path provided. Requires a DXF or PDF file path."}
        # DWG is refused by format, before the file is looked for.
        if is_dwg(file_path):
            return dwg_not_supported()
        if not os.path.exists(file_path):
            return {"status": "error", "error": f"File not found: {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()

        # --- PDF input: extract vector drawings via PyMuPDF -----------------
        # PDF drawings carry their geometry as vector paths in the page
        # content stream. We can pull lines, rectangles, and closed shapes
        # straight out — coordinates come back in PDF points (1pt = 1/72")
        # at the page's scale, NOT real-world metres. The caller usually
        # knows the title-block scale (e.g. 1:100) and can pass
        # `pdf_scale_factor` to convert from page-units to metres.
        if ext == ".pdf":
            # Run the legacy geometry extractor for backward compat
            # (measurements, areas, estimated_volumes), then layer the new
            # text-based structured drawing fields on top.
            geom = self._extract_from_pdf(file_path, params)
            text_result = self._extract_drawing_text(
                file_path,
                project_id=params.get("project_id") or data.get("project_id"),
            )
            # Merge: legacy keys first, new fields supplement.
            merged = dict(geom) if isinstance(geom, dict) else {}
            merged.update({
                "text": text_result.get("text", ""),
                "drawing": text_result.get("drawing", {}),
                "errors": text_result.get("errors", []),
            })
            # The text extractor legitimately finds nothing on a text-free
            # plot (text rendered as curves / raster scan); its status must
            # not MASK a successful geometry or OCR take-off. Only when the
            # geometry pass itself failed does the text status decide.
            if merged.get("status") != "success":
                merged["status"] = text_result.get("status", "error")
            return merged

        if ext != ".dxf":
            return {"status": "error", "error": f"Unsupported format: {ext}. Use .dxf or .pdf"}

        try:
            import ezdxf
        except ImportError:
            return {"status": "error", "error": "ezdxf not installed. Run: pip install ezdxf"}

        # open_plaintext transparently decrypts when DATA_ENCRYPTION_KEY is set
        # on the server (uploads go through file_crypto.write_document); no-op
        # for plaintext files.
        from app.core.file_crypto import open_plaintext
        try:
            with open_plaintext(file_path) as plain_path:
                doc = ezdxf.readfile(plain_path)
        except Exception as e:
            return {"status": "error", "error": f"DXF read error: {e}"}

        scale = float(params.get("unit_scale", self.config.get("unit_scale", 1.0)))
        layer_filter = params.get("area_layer_filter", self.config.get("area_layer_filter", []))
        min_area = float(params.get("min_area_m2", self.config.get("min_area_m2", 0.01)))

        # Honour the DXF's declared units instead of assuming millimetres.
        to_metres_factor = _to_metres_factor(doc.units)
        unit_factor = scale * to_metres_factor

        msp = doc.modelspace()
        measurements, bulge_segments_count, len_diag = self._extract_measurements(
            msp, unit_factor
        )
        areas, hatch_hole_fallback, area_diag = self._extract_areas(
            msp, unit_factor, layer_filter, min_area
        )
        volumes = self._estimate_volumes(areas, params)
        layers = list({e.dxf.layer for e in msp if hasattr(e.dxf, "layer")})

        total_area = sum(a["area_m2"] for a in areas)
        total_length = sum(m["length_m"] for m in measurements)

        response = {
            "status": "success",
            "measurements": measurements,
            "areas": areas,
            "estimated_volumes": volumes,
            "total_area_m2": round(total_area, 3),
            "total_length_m": round(total_length, 3),
            "entity_count": len(list(msp)),
            "layers": layers[:50],
            "drawing_units": str(doc.units),
            "input_units": doc.units,
            "to_metres_factor": to_metres_factor,
            "bulge_segments_count": bulge_segments_count,
            "geometry_engine": area_diag.get("geometry_engine", "shapely"),
            # Surface silent-failure counters so a corrupt or partially-
            # readable DXF doesn't ship as a confidently-low quantity total.
            "bulge_fallbacks": len_diag.get("bulge_fallbacks", 0),
            "entities_skipped": (
                len_diag.get("entities_skipped", 0)
                + area_diag.get("entities_skipped", 0)
            ),
            "hatches_skipped": area_diag.get("hatches_skipped", 0),
            "entities_skipped_reasons": (
                len_diag.get("entities_skipped_reasons", [])
                + area_diag.get("entities_skipped_reasons", [])
            ),
            "polyline_area_note": (
                "Arc-bounded polygon area approximated as chord polygon area; "
                "difference < 5% for typical bulges"
            ),
        }
        if hatch_hole_fallback:
            response["hatch_hole_handling"] = "may include holes as positive area"
        return response

    def _extract_from_pdf(self, file_path: str, params: dict) -> dict:
        """Extract vector geometry from a PDF drawing via PyMuPDF.

        PDF drawings carry their geometry as ``page.get_drawings()`` items —
        each item has a ``type`` (``'l'`` line, ``'re'`` rect, ``'c'`` curve)
        and a ``rect`` bbox plus an ``items`` path. We translate those into
        the same shape ``drawing_qto`` produces for DXF (measurements +
        areas + estimated_volumes), with coordinates in **PDF points** by
        default. Pass ``pdf_scale_factor`` to convert to metres (e.g.
        ``0.000352778`` to go from 1pt to mm-of-paper, then multiply by the
        drawing's plot scale).
        """
        try:
            import fitz
        except ImportError:
            return {"status": "error", "error": "PyMuPDF (fitz) not installed."}
        from app.core.file_crypto import open_plaintext

        scale = float(params.get("pdf_scale_factor", 1.0))
        max_pages = int(params.get("max_pages", self.config.get("max_pages", 20)))
        min_length = float(params.get("min_length_units", 0.5))  # in input units
        ocr_fallback = bool(params.get("ocr_fallback", True))

        measurements: list[dict] = []
        areas: list[dict] = []
        rooms: list[dict] = []
        pages_inspected = 0
        page_dims: list[dict] = []
        pages_ocr_attempted = 0
        pages_ocr_yielded = 0
        pages_geometry_skipped = 0

        try:
            with open_plaintext(file_path) as plain_path:
                doc = fitz.open(plain_path)
                pages_inspected = min(len(doc), max_pages)
                for pi in range(pages_inspected):
                    page = doc[pi]
                    page_dims.append({
                        "page": pi + 1,
                        "width_pt": page.rect.width,
                        "height_pt": page.rect.height,
                    })
                    # Room labels carry their own dimensions, which is the
                    # only quantity a scale-less PDF can yield honestly.
                    page_text = page.get_text() or ""
                    page_rooms = self._extract_rooms(page_text)
                    # Text-free plot fallback (drawing-reader, 2026-08-15):
                    # CAD plots with text rendered as curves and raster scans
                    # have an empty/near-empty text layer -- previously they
                    # returned 0 rooms with no attempt. When the layer is
                    # thin and yielded nothing, OCR the rendered page (the
                    # pixel-bounded renderer from doc_index -- F26) and run
                    # the same room extraction over the OCR text. The room
                    # pattern's structure (name + metres-to-2dp pair) filters
                    # OCR noise; OCR-sourced rooms are marked so a reviewer
                    # knows to verify against the sheet.
                    if (
                        ocr_fallback
                        and not page_rooms
                        and len(page_text.strip()) < _OCR_TEXT_THRESHOLD
                    ):
                        pages_ocr_attempted += 1
                        from app.core.doc_index import _ocr_pdf_page
                        ocr_text = _ocr_pdf_page(page)
                        if ocr_text:
                            page_rooms = self._extract_rooms(ocr_text)
                            if page_rooms:
                                pages_ocr_yielded += 1
                                for room in page_rooms:
                                    room["source"] = "ocr"
                    for room in page_rooms:
                        room.setdefault("source", "text_layer")
                        room["page"] = pi + 1
                        rooms.append(room)
                    # F26c: geometry extraction is bounded by content-stream
                    # size. A dense A1 CAD plot measured ~5 MB of stream and
                    # ~110,000 drawing objects per page (vs ~257 KB / 4,631 on
                    # a normal floor plan); get_drawings() materialises a dict
                    # per object and is the same OOM class that dropped the
                    # box twice on 2026-08-15. Rooms/OCR above still ran; only
                    # the vector-geometry pass is skipped, and the skip is
                    # REPORTED, never silent.
                    try:
                        content_bytes = len(page.read_contents() or b"")
                    except Exception:
                        content_bytes = 0
                    if content_bytes > _GEOMETRY_MAX_CONTENT_BYTES:
                        pages_geometry_skipped += 1
                        continue
                    drawings = page.get_drawings() or []
                    for d in drawings:
                        # `items` is a list of path commands: ("l", p1, p2)
                        # for lines, ("re", rect) for rectangles, ("c", ...)
                        # for cubic Béziers, ("qu", quad) for quads.
                        for item in d.get("items") or []:
                            kind = item[0]
                            if kind == "l" and len(item) >= 3:
                                p1, p2 = item[1], item[2]
                                length_pt = math.hypot(p2.x - p1.x, p2.y - p1.y)
                                if length_pt < min_length:
                                    continue
                                measurements.append({
                                    "type": "line",
                                    "page": pi + 1,
                                    "length_pt": round(length_pt, 3),
                                    "length_scaled": round(length_pt * scale, 6),
                                    "start": [round(p1.x, 2), round(p1.y, 2)],
                                    "end":   [round(p2.x, 2), round(p2.y, 2)],
                                })
                            elif kind == "re" and len(item) >= 2:
                                r = item[1]
                                w, h = abs(r.width), abs(r.height)
                                if w < min_length and h < min_length:
                                    continue
                                area = w * h
                                areas.append({
                                    "type": "rect",
                                    "page": pi + 1,
                                    "width_pt": round(w, 3),
                                    "height_pt": round(h, 3),
                                    "area_pt2": round(area, 3),
                                    "area_scaled": round(area * scale * scale, 6),
                                })
        except Exception as e:
            return {"status": "error", "error": f"PDF drawing read error: {e}"}

        total_area = sum(a["area_pt2"] for a in areas)
        total_length = sum(m["length_pt"] for m in measurements)
        return {
            "status": "success",
            "source_format": "pdf",
            "pages_inspected": pages_inspected,
            "page_dimensions": page_dims,
            "measurements_count": len(measurements),
            "measurements": measurements[:200],   # cap for response size
            "areas_count": len(areas),
            "areas": areas[:200],
            # Text-layer take-off. Independent of pdf_scale_factor: these are
            # the dimensions the drawing STATES, already in metres, so they
            # stand even when the geometry has no usable scale.
            "rooms_count": len(rooms),
            "rooms": rooms[:200],
            "pages_geometry_skipped": pages_geometry_skipped,
            "ocr_fallback": {
                "enabled": ocr_fallback,
                "pages_attempted": pages_ocr_attempted,
                "pages_yielded": pages_ocr_yielded,
                "note": ("rooms marked source=ocr were read from a rendered "
                         "page; verify dimensions against the sheet")
                        if pages_ocr_yielded else "",
            },
            "net_room_area_m2": round(sum(r["area_m2"] for r in rooms), 2),
            "room_perimeter_m": round(sum(r["perimeter_m"] for r in rooms), 2),
            "totals": {
                "length_pt": round(total_length, 3),
                "length_scaled": round(total_length * scale, 6),
                "area_pt2": round(total_area, 3),
                "area_scaled": round(total_area * scale * scale, 6),
            },
            "pdf_scale_factor": scale,
            "scale_note": (
                "PDF drawings carry no intrinsic scale — coordinates are in "
                "PDF points (1pt = 1/72\"). Pass `pdf_scale_factor` to convert "
                "to your target unit (e.g. for a 1:100 plotted drawing in mm, "
                "use 0.000352778 * 100 = 0.0352778 pt → m)."
            ),
        }

    # ====================================================================
    # Room take-off from the drawing's TEXT layer
    # ====================================================================
    # Architectural plans label each room with its size -- "BEDROOM 1 4.00 X
    # 3.60". Geometry extraction returns thousands of unscaled line segments
    # and no areas, so before this the interior take-off (floor tiling,
    # skirting, plaster, paint) had no quantity to stand on even though the
    # drawing stated it outright.
    #
    # Metres to two decimals is what makes this safe to parse: product sizes
    # are whole millimetres ("600x600mm" tiles, "900x2100mm" doors), so
    # requiring a decimal point excludes them without a keyword blacklist.
    _ROOM_RE = re.compile(
        r"([A-Za-z][A-Za-z0-9 &/.\-]{1,30}?)\s*"
        r"(\d{1,2}\.\d{2})\s*[Xx]\s*(\d{1,2}\.\d{2})"
    )
    # A room is at least a cupboard and at most a hall. Outside this band the
    # match is a grid reference, a level, or a site dimension.
    _ROOM_MIN_M = 0.6
    _ROOM_MAX_M = 60.0

    def _extract_rooms(self, text: str) -> list[dict]:
        """Rooms with area and perimeter, read from the drawing's text layer.

        Area drives floor finishes; perimeter drives skirting and the wall
        area for plaster and paint. Both are returned per room so an interior
        bill can be built without re-deriving them.
        """
        rooms: list[dict] = []
        if not text:
            return rooms
        flat = " ".join(str(text).split())
        for m in self._ROOM_RE.finditer(flat):
            name = m.group(1).strip(" -.&/")
            try:
                w = float(m.group(2))
                d = float(m.group(3))
            except ValueError:
                continue
            if not (self._ROOM_MIN_M <= w <= self._ROOM_MAX_M):
                continue
            if not (self._ROOM_MIN_M <= d <= self._ROOM_MAX_M):
                continue
            if not name:
                continue
            rooms.append({
                "name": name,
                "width_m": round(w, 2),
                "depth_m": round(d, 2),
                "area_m2": round(w * d, 2),
                "perimeter_m": round(2 * (w + d), 2),
            })
        return rooms

    # ====================================================================
    # V1.2 -- PyMuPDF (fitz) based text extraction for CAD drawings
    # ====================================================================
    # See docs/superpowers/specs/2026-06-11-drawing-reader-design.md.
    # Coordinate orientation: fitz returns y0 in TOP-DOWN page space
    # (y0=0 at top, y0=page.rect.height at bottom). All "bottom 15%" zones
    # are y0 > page.rect.height * 0.85 — the OPPOSITE of pdfplumber.
    # The right-20% fallback (x0 > 0.80*width) is direction-agnostic and
    # unchanged. Title-block clustering sorts top-down (smaller y0 first).

    @staticmethod
    def _chars_from_fitz(page) -> list[dict]:
        """Pull span-level "char" records from a fitz Page.

        Each fitz span carries its own text (including internal spaces),
        font size, font name, and bbox — the same shape the pdfplumber
        path consumed at char granularity. We emit ONE record per span:
        the downstream line clustering already glues runs by proximity
        and the regex/cad-tag filters operate on assembled text, so
        per-span (vs per-char) granularity is strictly faster and avoids
        the per-char ligature ambiguities that occasionally bit
        pdfplumber.

        Returned dicts mirror the pdfplumber char schema so all downstream
        logic (line clustering, font-size buckets, drawing-number regex,
        cross-ref regex) works unchanged:
            {"text", "x0", "y0", "x1", "y1", "size", "fontname"}
        Coordinates are in fitz's TOP-DOWN space (y0=0 at page top).
        """
        out: list[dict] = []
        try:
            d = page.get_text("dict") or {}
        except Exception:
            return out
        for block in d.get("blocks", []) or []:
            # Image blocks have no "lines" key — skip.
            for line in block.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    text = span.get("text", "")
                    if text == "":
                        continue
                    bbox = span.get("bbox", (0.0, 0.0, 0.0, 0.0))
                    out.append({
                        "text": text,
                        "x0": float(bbox[0]),
                        "y0": float(bbox[1]),
                        "x1": float(bbox[2]),
                        "y1": float(bbox[3]),
                        "size": float(span.get("size", 0.0)),
                        "fontname": span.get("font", ""),
                    })
        return out

    def _extract_drawing_text(self, file_path: str,
                              project_id: str | None = None) -> dict:
        """Top-level text-extraction orchestrator: returns
        ``{"text", "drawing", "errors", "status"}``.

        ``project_id`` is the open project; its profile may name the
        drawing-number discipline codes (``DISCIPLINE_CODES_FACT``)."""
        errors: list[str] = []
        try:
            import fitz
        except ImportError:
            return {
                "status": "error",
                "text": "",
                "drawing": {},
                "errors": ["pymupdf_not_installed"],
            }
        from app.core.file_crypto import open_plaintext

        page_full_raw_texts: list[str] = []
        try:
            with open_plaintext(file_path) as plain_path:
                try:
                    doc = fitz.open(plain_path)
                except Exception as exc:
                    msg = str(exc).lower()
                    if "password" in msg or "encrypt" in msg:
                        return {
                            "status": "error",
                            "text": "",
                            "drawing": {},
                            "errors": ["password_protected"],
                        }
                    return {
                        "status": "error",
                        "text": "",
                        "drawing": {},
                        "errors": [f"pdf_open_failed: {exc}"],
                    }

                try:
                    if doc.needs_pass:
                        return {
                            "status": "error",
                            "text": "",
                            "drawing": {},
                            "errors": ["password_protected"],
                        }
                    if doc.page_count == 0:
                        return {
                            "status": "error",
                            "text": "",
                            "drawing": {},
                            "errors": ["no_pages"],
                        }

                    page_results = []
                    total_chars = 0
                    for page in doc:
                        chars = self._chars_from_fitz(page)
                        total_chars += len(chars)
                        # Save raw full-page text for drawing-number fallback
                        # rescue (Bug 2: when title-block extractor returned a
                        # half-match like "AB-CDE-001-KLM" we re-scan the full
                        # page for a complete drawing number).
                        page_full_raw_texts.append(
                            "".join(c["text"] for c in chars)
                        )
                        page_results.append(self._process_page(page, chars, errors))

                    if total_chars == 0:
                        # Scanned drawing / no text layer. OCR fallback is
                        # deferred to a follow-up task per the spec.
                        return {
                            "status": "error",
                            "text": "",
                            "drawing": {},
                            "errors": errors + ["no_text_layer_pymupdf"],
                        }
                finally:
                    doc.close()
        except Exception as exc:
            return {
                "status": "error",
                "text": "",
                "drawing": {},
                "errors": errors + [f"text_extract_failed: {exc}"],
            }

        # Multi-page combine. Take page-1 title block; if page N differs,
        # collapse-to-one-chunk for v1 and flag in errors.
        primary = page_results[0]
        for pr in page_results[1:]:
            if (pr["title_block"].get("drawing_number") and
                primary["title_block"].get("drawing_number") and
                pr["title_block"]["drawing_number"] !=
                    primary["title_block"]["drawing_number"]):
                errors.append("multi_drawing_pdf_collapsed_to_one_chunk")
                break

        # Aggregate notes/dimensions/cross_refs across pages
        all_notes: list[str] = []
        all_dims: list[str] = []
        all_refs: list[dict] = []
        cad_filtered = 0
        dedup_dropped_total = 0
        # Dedup cross_refs across pages too, by (ref_type, target_drawing).
        seen_refs: set = set()
        for i, pr in enumerate(page_results, 1):
            if len(page_results) > 1:
                all_notes.extend(f"[Sheet {i}] {n}" for n in pr["notes"])
                all_dims.extend(f"[Sheet {i}] {d}" for d in pr["dimensions"])
            else:
                all_notes.extend(pr["notes"])
                all_dims.extend(pr["dimensions"])
            for r in pr["cross_refs"]:
                key = (r.get("ref_type"), r.get("target_drawing"))
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                all_refs.append(r)
            cad_filtered += pr["cad_tags_filtered_count"]
            dedup_dropped_total += pr.get("notes_dedup_dropped_count", 0)

        # Guardrail cap: if dedup yielded >100 unique cross_refs we've almost
        # certainly regressed the regex; trim alphabetically and flag.
        if len(all_refs) > 100:
            errors.append("cross_refs_count_suspect_over_100")
            all_refs = sorted(
                all_refs,
                key=lambda r: (r.get("target_drawing") or "",
                               r.get("ref_type") or ""),
            )[:100]

        tb = dict(primary["title_block"])
        # Boilerplate by repetition: when the PDF holds different drawings,
        # title-block text present on every sheet (consultant, project
        # header) is not any one sheet's title.
        page_dns = {pr["title_block"].get("drawing_number") for pr in page_results}
        if len(page_results) > 1 and len(page_dns) > 1 and None not in page_dns:
            repeated = set.intersection(*(pr["_tb_texts"] for pr in page_results))
            if tb.get("_title_candidates") and tb.get("drawing_title") and (
                _norm_tb_text(tb["drawing_title"]) in repeated
            ):
                tb["drawing_title"] = next(
                    (t for t in tb["_title_candidates"]
                     if _norm_tb_text(t) not in repeated),
                    tb["drawing_title"],
                )
        dn_labelled = bool(tb.pop("_dn_labelled", False))
        tb.pop("_title_candidates", None)
        # Bug 2: reject drawing-number matches that aren't a complete long
        # drawing number. The short fallback regex sometimes grabs a
        # half-match ("AB-CDE-001-KLM") from a random title-block fragment.
        # Completeness is judged by shape (``_is_full_drawing_number``), in
        # either token order and for any originator code; if the current
        # value is incomplete, re-scan the full page raw text.
        current_dn = tb.get("drawing_number")

        # A number read from its DRAWING NO label is taken as written: the
        # label, not the shape, says it is the drawing number.
        if current_dn and not dn_labelled and not _is_full_drawing_number(current_dn):
            rescued = None
            for raw in page_full_raw_texts:
                m = _DWG_NUMBER_FULL.search(raw)
                if m and _is_full_drawing_number(m.group(0)):
                    # Phase 1.7: strip leading doubled-letter artifacts
                    # (e.g. ``XAB-CDE-...`` -> ``AB-CDE-...``).
                    rescued = _strip_doubled_letter_prefix(m.group(0))
                    break
            # A rescued number replaces the half-match; with none, drop the
            # half-match so the filename fallback below fires.
            tb["drawing_number"] = rescued
            tb["revision"] = None
        if not tb.get("drawing_number"):
            tb["drawing_number"] = os.path.splitext(
                os.path.basename(file_path)
            )[0]
            errors.append("drawing_number_fallback_to_filename")
        # Discipline: the DISCIPLINE label, else the open project's own code
        # mapping applied to the final drawing number, else unknown.
        tb["discipline"], tb["discipline_full"], tb["discipline_source"] = (
            _resolve_discipline(
                tb.get("discipline"),
                tb.get("drawing_number"),
                _project_discipline_codes(project_id),
            )
        )
        # Re-derive the revision from the (possibly rescued or
        # filename-fallback) drawing_number so all paths agree.
        if not tb.get("revision") and tb.get("drawing_number"):
            rev = _revision_from_number_tail(tb["drawing_number"])
            if rev:
                tb["revision"] = rev
        # Phase 1.5 fallback: many drawing filenames carry the revision as a
        # trailing letter (e.g. ...-0000001-A.pdf). When title-block parse
        # and drawing-number-tail extraction both miss it, look at the
        # filename. Single uppercase letter immediately before the .pdf
        # extension wins. Numeric tails like "04" / "05" are NOT accepted
        # here because they are sheet-sequence indices, not revisions.
        if not tb.get("revision"):
            stem = os.path.splitext(os.path.basename(file_path))[0]
            m = re.search(r"-([A-Z])$", stem)
            if m:
                tb["revision"] = m.group(1)
                errors.append("revision_fallback_to_filename")

        # Bug 1: reject drawing_title that's actually the drawing_number with
        # a clustering artifact. The title-block extractor picks "longest
        # cluster" which often grabs the drawing number with a typo or trailing
        # revision letter glued on. Normalize both (uppercase + strip
        # non-alphanumerics) and reject any title that contains a 12+ char
        # substring of the normalized drawing_number.
        title = tb.get("drawing_title")
        dn = tb.get("drawing_number") or ""
        if title and dn:
            norm_title = re.sub(r"[^A-Z0-9]", "", title.upper())
            norm_dn = re.sub(r"[^A-Z0-9]", "", dn.upper())
            collision = False
            if norm_title and norm_dn:
                if norm_title == norm_dn:
                    collision = True
                elif len(norm_dn) >= 12:
                    for i in range(len(norm_dn) - 11):
                        if norm_dn[i:i + 12] in norm_title:
                            collision = True
                            break
            if collision:
                tb["drawing_title"] = None
                errors.append("drawing_title_not_found")

        drawing = {
            **tb,
            "notes": all_notes,
            "dimensions": all_dims,
            "cross_refs": all_refs,
            "cad_tags_filtered_count": cad_filtered,
            "notes_dedup_dropped_count": dedup_dropped_total,
            "n_pages": len(page_results),
        }
        raw_chunk = self._build_raw_chunk(drawing)
        return {
            "status": "success",
            "text": raw_chunk,
            "drawing": drawing,
            "errors": errors,
        }

    # --- per-page pipeline --------------------------------------------------
    def _process_page(self, page, chars: list[dict], errors: list[str]) -> dict:
        """Steps 1-5 of the spec for a single page."""
        title_block_chars, drawing_zone_chars = self._split_page_chars(
            page, chars
        )

        # Capture the drawing number from the ORIGINAL bottom-15% band
        # BEFORE the richness fallback below widens ``title_block_chars``
        # to the right-20% zone or the full page. On one pilot sheet,
        # bottom-15% has only 3 spans (lines < 5), tripping the right-20%
        # fallback. Right-20% raw char order then puts a referenced
        # drawing number ahead of the title-block one, so a plain
        # ``_DWG_NUMBER_FULL.search()`` returns the wrong number. The
        # bottom-band match is the title-block's own number — prefer it
        # when it parses as a full drawing number. Safe because the band IS the
        # title-block region by spatial definition; widening was only
        # needed to harvest title/scale/date labels.
        band_raw = "".join(c["text"] for c in title_block_chars)
        m_band = _DWG_NUMBER_FULL.search(band_raw)
        band_dn = (
            _strip_doubled_letter_prefix(m_band.group(0)) if m_band else None
        )

        # --- Title-block fallback chain ------------------------------------
        # Use clustered line count as a "richness" signal -- below 5 lines
        # we fall back to the right-20% zone (landscape title blocks), then
        # to the full page.
        tb_lines = self._lines_from_chars(title_block_chars)
        if len(tb_lines) < 5:
            # Right-20% fallback is x-axis only; unchanged across the
            # pdfplumber -> fitz swap. ``page.rect.width`` on fitz.
            right_chars = [c for c in chars if c["x0"] >= page.rect.width * 0.80]
            right_lines = self._lines_from_chars(right_chars)
            if len(right_lines) >= 5:
                title_block_chars = right_chars
                tb_lines = right_lines
                # The right-hand column IS the title block: its text is
                # metadata, never a note or a dimension.
                drawing_zone_chars = [
                    c for c in drawing_zone_chars
                    if c["x0"] < page.rect.width * 0.80
                ]
            else:
                # full-page scan fallback
                title_block_chars = chars
                tb_lines = self._lines_from_chars(chars)
                errors.append("title_block_zone_fallback_full_page")

        title_block = self._extract_title_block(title_block_chars, page)

        # If the bottom-15% band yielded a valid full drawing number, prefer
        # it over the (possibly contaminated) widened-zone match. Clear the
        # revision so the downstream re-derive in ``_extract_drawing_text``
        # re-fills it from the corrected number — that re-derive only fires
        # when the field is missing, so an explicit reset is required, not
        # just an overwrite of ``drawing_number``.
        if (
            band_dn
            and _is_full_drawing_number(band_dn)
            and title_block.get("drawing_number") != band_dn
        ):
            title_block["drawing_number"] = band_dn
            title_block["revision"] = None
            errors.append("drawing_number_picked_from_bottom_band")

        # --- Drawing-zone classification -----------------------------------
        notes, dimensions, filtered_count, dedup_dropped = (
            self._classify_drawing_zone(drawing_zone_chars or chars)
        )

        # --- Cross-refs ----------------------------------------------------
        # Run on the *raw* char order of the page -- a single Tj operator's
        # chars are contiguous in page.chars even when the label is
        # rotated, which spatial reconstruction would scatter.
        raw_text = "".join(c["text"] for c in chars)
        cross_refs = self._extract_cross_refs(raw_text)

        return {
            "title_block": title_block,
            "_tb_texts": {
                _norm_tb_text(L["text"]) for L in tb_lines if L["text"].strip()
            },
            "notes": notes,
            "dimensions": dimensions,
            "cross_refs": cross_refs,
            "cad_tags_filtered_count": filtered_count,
            "notes_dedup_dropped_count": dedup_dropped,
        }

    # --- Step 1: page region split -----------------------------------------
    @staticmethod
    def _split_page_chars(page, chars: list[dict]) -> tuple[list[dict], list[dict]]:
        """Bottom 15% of page height -> title-block zone, rest -> drawing
        zone. fitz y0=0 is the page TOP, so "bottom 15%" is the high-y0
        band: y0 > height * 0.85. (Coordinate-flip vs the pdfplumber
        path; see module docstring.)"""
        # Phase 1.7 (fitz coords flip): bottom band is y0 > 0.85*height,
        # not y0 < 0.15*height.
        height = page.rect.height
        threshold = height * 0.85
        tb, dz = [], []
        for c in chars:
            if c["y0"] > threshold:
                tb.append(c)
            else:
                dz.append(c)
        return tb, dz

    # --- helpers: reconstruct lines from chars -----------------------------
    @staticmethod
    def _lines_from_chars(
        chars: list[dict], y_tol: float = 2.0, x_gap: float = 30.0
    ) -> list[dict]:
        """Cluster chars into reading-order lines.

        Returns a list of ``{"y": <y0>, "size": <avg>, "text": <str>}``
        records. Same line = chars within ``y_tol`` of the same y0
        baseline; same word/line continuation = adjacent chars within
        ``x_gap`` horizontally. This is intentionally lossy on rotated
        labels (those come out scrambled) -- spatial reconstruction is
        for the title block and dimension/notes blocks, not for
        rotated callouts (those go through raw-char-order cross-ref
        scanning instead).
        """
        if not chars:
            return []
        # Bucket by y0 rounded to tolerance
        buckets: dict[float, list[dict]] = {}
        for c in chars:
            key = round(c["y0"] / y_tol) * y_tol
            buckets.setdefault(key, []).append(c)

        lines: list[dict] = []
        for y, cs in buckets.items():
            cs_sorted = sorted(cs, key=lambda c: c["x0"])
            # Split into runs separated by big x gaps
            run: list[dict] = []
            last_x1 = None
            for c in cs_sorted:
                if last_x1 is not None and c["x0"] - last_x1 > x_gap:
                    if run:
                        lines.append(_line_from_run(y, run))
                    run = []
                run.append(c)
                last_x1 = c["x1"]
            if run:
                lines.append(_line_from_run(y, run))
        # Sort lines top-down for readability. fitz: y0=0 at the top of
        # the page, so ascending y0 IS top-down. (Inverse of the
        # pdfplumber path, which sorted by -y.)
        lines.sort(key=lambda L: L["y"])
        return lines

    # --- Step 2: title-block structured extraction -------------------------
    def _title_block_fields(
        self, tb_chars: list[dict]
    ) -> tuple[dict[str, str], set[str]]:
        """Read labelled title-block fields by structure.

        Cells are the title-block text runs. A cell that opens with a field
        label (``_TB_LABEL_PATTERNS``) is a label; its value is the text after
        the label and a separator, else the nearest non-label cell to its
        right on the same row, else the nearest non-label cell directly below
        it (wrapping onto further lines for titles). No name is consulted.

        Returns ``(fields, meta_texts)``: the values found, and the
        normalised text of every label and value cell, which is title-block
        metadata and never a drawing-title candidate.
        """
        cells = [
            c for c in self._lines_from_chars(tb_chars, y_tol=2.0, x_gap=8.0)
            if c["text"].strip()
        ]
        is_label = [bool(_tb_label_at_start(c["text"])) for c in cells]
        fields: dict[str, str] = {}
        meta: set[str] = set()

        def _below(prev: dict, lx0: float, lx1: float, wrap: bool) -> int | None:
            best, best_dy = None, None
            for j, d in enumerate(cells):
                if is_label[j]:
                    continue
                dy = d["y"] - prev["y"]
                if dy <= 0.5:
                    continue
                size = max(prev["size"], d["size"])
                if dy > (2.0 if wrap else 2.5) * size + 4.0:
                    continue
                if wrap:
                    if abs(d["x0"] - lx0) > 10.0:
                        continue
                    if not 0.75 <= (d["size"] or 1.0) / (prev["size"] or 1.0) <= 1.33:
                        continue
                elif d["x0"] > lx1 + 10.0 or d["x1"] < lx0 - 10.0:
                    continue
                if best_dy is None or dy < best_dy:
                    best, best_dy = j, dy
            return best

        def _right(c: dict, lx1: float) -> int | None:
            best, best_dx = None, None
            for j, d in enumerate(cells):
                if is_label[j] or abs(d["y"] - c["y"]) > 2.0:
                    continue
                dx = d["x0"] - lx1
                if dx < 0 or dx > max(60.0, 6.0 * c["size"]):
                    continue
                if best_dx is None or dx < best_dx:
                    best, best_dx = j, dx
            return best

        for i, c in enumerate(cells):
            if not is_label[i]:
                continue
            text = c["text"].strip()
            meta.add(_norm_tb_text(text))
            ms = list(_TB_LABEL_RE.finditer(text.upper()))
            width = max(c["x1"] - c["x0"], 1.0)
            for k, m in enumerate(ms):
                field = m.lastgroup
                end = ms[k + 1].start() if k + 1 < len(ms) else len(text)
                rest = text[m.end():end]
                sep = _TB_SEP_RE.match(rest)
                if sep:
                    inline = rest[sep.end():]
                elif field in _TB_SHAPED_FIELDS or not rest.strip():
                    inline = rest
                else:
                    # A free-text label word followed by more words with no
                    # separator ("PROJECT SYSTEM ...") is not a field.
                    continue
                if field in fields:
                    continue
                value = _validate_tb_value(field, inline) if inline.strip() else None
                if value is None and not inline.strip() and k == len(ms) - 1:
                    lx0 = c["x0"] + width * m.start() / max(len(text), 1)
                    lx1 = c["x1"]
                    j = _right(c, lx1)
                    if j is None:
                        j = _below(c, lx0, lx1, wrap=False)
                    if j is not None:
                        value = _validate_tb_value(field, cells[j]["text"])
                        if value is not None:
                            meta.add(_norm_tb_text(cells[j]["text"]))
                            if field in _TB_WRAPPING_FIELDS:
                                parts, prev = [value], cells[j]
                                while True:
                                    n = _below(prev, prev["x0"], prev["x1"], wrap=True)
                                    if n is None:
                                        break
                                    parts.append(cells[n]["text"].strip())
                                    meta.add(_norm_tb_text(cells[n]["text"]))
                                    prev = cells[n]
                                value = " ".join(parts)
                if value is not None:
                    fields[field] = value
        return fields, meta

    def _extract_title_block(
        self, tb_chars: list[dict], page, boilerplate: frozenset = frozenset()
    ) -> dict:
        """Extract drawing_number, title, discipline, revision, scale,
        date, drafter, checked_by, approved_by, project_name, sheet_number
        from the title-block char set.

        Fields are found by structure (``_title_block_fields``): generic
        field labels and the position of each value relative to its label.
        The drawing number may also come from its shape in raw char order (a
        single rotated Tj operator can land contiguously in the content
        stream even when its bounding boxes scatter). When no TITLE label is
        present the title is the largest remaining title-block text that is
        not metadata, preferring text inside the title-block region of the
        page and skipping ``boilerplate`` (text repeated on every sheet)."""
        result: dict[str, Any] = {
            "drawing_number": None,
            "drawing_title": None,
            "discipline": None,
            "discipline_full": None,
            "revision": None,
            "scale": None,
            "date": None,
            "drafter": None,
            "checked_by": None,
            "approved_by": None,
            "project_name": None,
            "sheet_number": None,
        }
        if not tb_chars:
            return result

        fields, meta_texts = self._title_block_fields(tb_chars)

        # --- Drawing number: full shape, else its label, else short shape --
        raw = "".join(c["text"] for c in tb_chars)
        m_full = _DWG_NUMBER_FULL.search(raw)
        labelled_dn = fields.get("drawing_number")
        dn = None
        if labelled_dn and (not m_full or _is_full_drawing_number(labelled_dn)):
            dn = labelled_dn
            result["_dn_labelled"] = True
        else:
            m = m_full or _DWG_NUMBER_SHORT.search(raw)
            if m:
                # Phase 1.7: strip leading doubled-letter artifacts
                # (e.g. ``XAB-CDE-...`` -> ``AB-CDE-...``). Source text runs
                # occasionally start one char inside an earlier token and the
                # ``[A-Z]{2,}`` head accepts that as a valid prefix. The
                # strip is pattern-aware so legitimate prefixes survive.
                dn = _strip_doubled_letter_prefix(m.group(0))
                # Last hyphenated token of the long pattern is the revision
                tail = dn.rsplit("-", 1)[-1]
                if 1 <= len(tail) <= 3 and re.fullmatch(r"[A-Z0-9]+", tail):
                    result["revision"] = tail
        if dn:
            result["drawing_number"] = dn
        if not result["revision"] and fields.get("revision"):
            result["revision"] = fields["revision"]

        # People, project and discipline: only ever the value of their label.
        # (A discipline code is named later, from the open project's profile.)
        for f in ("drafter", "checked_by", "approved_by", "project_name",
                  "discipline"):
            if fields.get(f):
                result[f] = fields[f]

        # --- Cluster title-block into lines for unlabelled fallbacks -------
        lines = self._lines_from_chars(tb_chars, y_tol=2.0, x_gap=50.0)
        line_texts = [L["text"] for L in lines if L["text"].strip()]
        all_text = " \n".join(line_texts)

        # Scale: labelled, else 1:NNN, NTS, N.T.S., NOT TO SCALE anywhere
        if fields.get("scale"):
            result["scale"] = fields["scale"]
        else:
            sm = _SCALE_RE.search(all_text)
            if sm:
                result["scale"] = sm.group(1).strip()

        # Date: labelled, else DD/MM/YY etc. anywhere
        if fields.get("date"):
            result["date"] = fields["date"]
        else:
            dm = _DATE_RE.search(all_text)
            if dm:
                result["date"] = dm.group(1)

        # Sheet number: labelled, else Sheet N of M, or N/M near "SHEET"
        if fields.get("sheet_number"):
            result["sheet_number"] = fields["sheet_number"]
        else:
            shm = re.search(
                r"(?:SHEET|SH\.?)\s*[:\-]?\s*(\d+(?:\s*(?:OF|/)\s*\d+)?)",
                all_text,
                re.IGNORECASE,
            )
            if shm:
                result["sheet_number"] = shm.group(1).strip()

        if fields.get("drawing_title"):
            result["drawing_title"] = fields["drawing_title"][:200]
            return result

        # Drawing title by position: the title-block region of the page is
        # the bottom band and the right-hand column. When the reader had to
        # widen to the whole page, text inside that region is preferred over
        # text out in the drawing area (place labels on a key plan).
        rect = getattr(page, "rect", None)

        def _in_region(L: dict) -> bool:
            if rect is None:
                return True
            return (L["y"] > rect.height * 0.85
                    or L.get("x0", 0.0) >= rect.width * 0.80)

        value_texts = [t for t in meta_texts if len(t) >= 4]
        candidates = sorted(
            (L for L in lines if L["text"].strip()),
            key=lambda L: (not _in_region(L), -L["size"]),
        )
        titles: list[str] = []
        for L in candidates:
            t = L["text"].strip()
            if not t or len(t) < 4:
                continue
            tu = t.upper()
            if result["drawing_number"] and result["drawing_number"] in tu:
                continue
            # Title-block metadata: a label cell, or a labelled field's value.
            tn = _norm_tb_text(t)
            if _tb_label_at_start(t) or any(v in tn for v in value_texts):
                continue
            # Bug 1.5b: reject cross-ref callouts as title candidates.
            # On detail sheets the longest cluster was the MATCH LINE
            # text. Skip anything that looks like a sheet-to-sheet ref.
            if re.search(
                r"\bMATCH\s*LINE\b|"
                r"\bCONT(?:INUED|D|\.)?\s*ON\b|"
                r"\bSEE\s+DWG\b|"
                r"\bREF(?:ER|\.)?[^\n]{0,40}?\b(?:SHEET|DWG|DRAWING)\b",
                tu,
            ):
                continue
            # Phase 1.6: reject pure-numeric and scale-shaped candidates.
            # On one pilot sheet the title-block selection picked "1800" — a chainage
            # station number. The user wanted "1:1800" as scale, but
            # that lives in a different field; for drawing_title we just
            # refuse all numeric-shaped strings. Phase 1.7: ``+`` joins
            # chainage stations (``0+124.138``) — extend the class.
            t_compact = re.sub(r"\s+", "", t)
            if re.fullmatch(r"[\d.,/:\-+]+", t_compact):
                continue
            # Reject scale labels (1:N or 1: N etc.) that escaped the
            # numeric check above due to embedded spaces.
            if re.fullmatch(r"1\s*:\s*\d+", t):
                continue
            # Phase 1.7: reject candidates with trailing lowercase
            # artifacts (e.g. ``KEY PLANg`` — a subscript glyph that bled
            # into the span text). Catches ``[A-Z]{2,}[a-z]`` at the end
            # of the trimmed last token, leaves ``Section A-A`` and
            # ``CONCRETE ENCASEMENT`` alone.
            if _has_trailing_lowercase_artifact(t):
                continue
            # Generic drawing furniture (web/postal address, survey datum,
            # notes heading): drawing vocabulary, not names.
            if any(k in tu for k in (
                "WWW.", "P.O. BOX", "DATUM", "GEODETIC",
                "PROJECT SYSTEM", "ZONE:", "NOTES",
            )):
                continue
            # Phase 1.7: reject CAD Xref filepath strings (one pilot sheet
            # leaks the underlying ``Xref ..\..\<folder>\<file>`` debug
            # label as a high-font cluster under fitz — pdfplumber never
            # surfaced it). Match "XREF " prefix or a backslash anywhere.
            if tu.startswith("XREF ") or "\\" in t:
                continue
            titles.append(t[:200])
        # Text repeated on every sheet of a set is boilerplate (consultant,
        # project header); the caller re-picks once all pages are read.
        result["_title_candidates"] = titles
        result["drawing_title"] = next(
            (t for t in titles if _norm_tb_text(t) not in boilerplate),
            titles[0] if titles else None,
        )
        return result

    # --- Step 3: drawing-zone font-size classification ---------------------
    def _classify_drawing_zone(
        self, dz_chars: list[dict]
    ) -> tuple[list[str], list[str], int, int]:
        """Classify drawing-zone text clusters by font size, then pattern-
        filter the kept text. Returns
        ``(notes, dimensions, filtered_count, dedup_dropped_count)``.

        Phase 1.8 adds Levenshtein-based note dedup: when a candidate is
        within edit-distance 5 of an already-accepted note, it's dropped
        and the dedup counter is bumped. This kills the repetitive
        legend / schedule / revision-history overproduction observed on
        one pilot sheet (161 notes vs 2-4 for other disciplines of comparable
        size).
        """
        if not dz_chars:
            return [], [], 0, 0

        # Build clusters: chars within 1px vertically + 5px horizontally are
        # one word; words on same y line within 30px gap are one line.
        lines = self._lines_from_chars(dz_chars, y_tol=1.5, x_gap=30.0)

        notes: list[str] = []
        dimensions: list[str] = []
        filtered = 0
        dedup_dropped = 0

        for L in lines:
            text = L["text"].strip()
            if not text:
                continue
            size = L["size"]

            # Size-based bucketing
            if size < 2.0:
                filtered += 1
                continue
            target_bucket = "notes" if size >= 4.0 else "dimensions"

            # Pattern filters apply to all kept clusters
            if _is_cad_tag(text):
                filtered += 1
                continue
            if _is_coordinate_pair(text):
                filtered += 1
                continue
            if len(text) <= 2:
                filtered += 1
                continue
            if _has_repeated_run(text, 4):
                filtered += 1
                continue

            if target_bucket == "notes":
                # Phase 1.8: drop near-identical duplicates AFTER all
                # other filters. Edit-distance < 5 = duplicate.
                if _note_near_duplicate(text, notes, max_distance=5):
                    dedup_dropped += 1
                    continue
                notes.append(text)
            else:
                dimensions.append(f"DIM: {text}")

        return notes, dimensions, filtered, dedup_dropped

    # --- Step 4: cross-ref extraction --------------------------------------
    # Sheet-identifier shape: either a long hyphenated number
    # (3+ tokens) OR a short sheet number (2-4 digits like "02", "10", "1234").
    # Loose `[A-Z0-9-]+` over-matched on one pilot sheet (1755 hits) so we lock this down.
    _SHEET_ID_RE = re.compile(
        r"(?:[A-Z0-9]+(?:-[A-Z0-9]+){2,}|\d{2,4})"
    )

    @classmethod
    def _extract_cross_refs(cls, raw_text: str) -> list[dict]:
        """Scan raw page text for match-line / continuation / reference
        callouts. Returns one dict per (ref_type, target_drawing) tuple
        after dedup. Caps at 100 entries per page (alphabetical) with a
        guardrail error if exceeded."""
        refs: list[dict] = []
        # Tolerant patterns: allow arbitrary whitespace and optional colons
        # between tokens, and capture a strict sheet-id shape only.
        sheet = r"(?P<target>[A-Z0-9]+(?:-[A-Z0-9]+){2,}|\d{2,4})"
        patterns: list[tuple[str, str]] = [
            ("match_line",
             r"MATCH\s*LINE\b[\s:.,\-]*"
             r"(?:FOR\s+REFERENCE\s+)?"
             r"(?:REFER(?:ENCE)?\s+(?:TO\s+)?)?"
             r"SHEET\s*(?:NO\.?)?\s*[:.\-]?\s*" + sheet),
            ("continuation",
             r"CONT(?:INUED|D|\.)?\.?\s*ON\s*[:.\-]?\s*" + sheet),
            ("reference",
             r"SEE\s+DWG\.?\s*[:.\-]?\s*" + sheet),
            ("reference",
             r"REF(?:ER|\.)?\.?\s*(?:TO\s+)?"
             r"(?:SHEET|DWG|DRAWING)\s+(?:NO\.?\s*)?[:.\-]?\s*" + sheet),
        ]
        # Dedup by (ref_type, target_drawing) — repeated identical match-line
        # callouts collapse to one entry.
        dedup: dict[tuple[str, str], dict] = {}
        for ref_type, pat in patterns:
            for m in re.finditer(pat, raw_text, re.IGNORECASE | re.MULTILINE):
                target = (m.group("target") or "").strip().upper()
                if not target or len(target) < 2:
                    continue
                key = (ref_type, target)
                if key in dedup:
                    continue
                dedup[key] = {
                    "ref_type": ref_type,
                    "target_drawing": target,
                    "raw": m.group(0).strip(),
                }
        refs = list(dedup.values())
        return refs

    # --- Step 6: raw chunk builder -----------------------------------------
    @staticmethod
    def _build_raw_chunk(drawing: dict) -> str:
        """Assemble the RAG-indexable chunk per the spec's template."""
        lines: list[str] = []
        header = drawing.get("drawing_number") or "(unknown)"
        title = drawing.get("drawing_title")
        disc_full = drawing.get("discipline_full") or drawing.get("discipline") or ""
        rev = drawing.get("revision") or ""
        head_bits = [header]
        if title:
            head_bits.append(f"-- {title}")
        meta = []
        if disc_full:
            meta.append(disc_full)
        if rev:
            meta.append(f"Rev {rev}")
        if meta:
            head_bits.append(f"({', '.join(meta)})")
        lines.append(" ".join(head_bits))

        meta_line_bits = []
        if drawing.get("scale"):
            meta_line_bits.append(f"Scale: {drawing['scale']}")
        if drawing.get("date"):
            meta_line_bits.append(f"Date: {drawing['date']}")
        if drawing.get("project_name"):
            meta_line_bits.append(f"Project: {drawing['project_name']}")
        if drawing.get("sheet_number"):
            meta_line_bits.append(f"Sheet: {drawing['sheet_number']}")
        if meta_line_bits:
            lines.append(" | ".join(meta_line_bits))

        notes = drawing.get("notes") or []
        if notes:
            lines.append("")
            lines.append("Notes:")
            for n in notes:
                lines.append(f"- {n}")

        refs = drawing.get("cross_refs") or []
        if refs:
            lines.append("")
            lines.append("References:")
            for r in refs:
                lines.append(
                    f"- {r['ref_type']}: {r['target_drawing']} ({r['raw']})"
                )
        return "\n".join(lines)

    # ====================================================================

    def _extract_measurements(self, msp, unit_factor: float) -> tuple[list[dict], int, dict[str, Any]]:
        """``unit_factor`` converts raw drawing units straight to metres.

        Returns (measurements, bulge_segments_count, diagnostics) so the
        caller can know how many LWPOLYLINE segments were arc-faced vs
        straight chords AND which entities/bulge-arcs were silently dropped.

        ``diagnostics`` is::

            {
                "bulge_fallbacks": int,   # bulge_to_arc threw; arc was collapsed to chord
                "entities_skipped": int,
                "entities_skipped_reasons": [
                    {"etype": str, "layer": str, "error": str},
                    ...
                ],
            }
        """
        results = []
        bulge_segments_count = 0
        bulge_fallbacks = 0
        entities_skipped = 0
        entities_skipped_reasons: list[dict[str, Any]] = []
        # Import bulge_to_arc lazily; only LWPOLYLINE with non-zero bulge needs it.
        try:
            from ezdxf.math import bulge_to_arc
        except Exception:
            bulge_to_arc = None
        for entity in msp:
            etype = entity.dxftype()
            try:
                if etype == "LINE":
                    start = entity.dxf.start
                    end = entity.dxf.end
                    length = math.dist(
                        (start.x, start.y, start.z),
                        (end.x, end.y, end.z)
                    ) * unit_factor
                    results.append({
                        "type": "line",
                        "length_m": round(length, 4),
                        "layer": entity.dxf.layer,
                        "start": [round(start.x * unit_factor, 3), round(start.y * unit_factor, 3)],
                        "end": [round(end.x * unit_factor, 3), round(end.y * unit_factor, 3)],
                    })
                elif etype == "CIRCLE":
                    radius = entity.dxf.radius * unit_factor
                    circumference = 2 * math.pi * radius
                    results.append({
                        "type": "circle",
                        "radius_m": round(radius, 4),
                        "circumference_m": round(circumference, 4),
                        "length_m": round(circumference, 4),
                        "layer": entity.dxf.layer,
                    })
                elif etype == "ARC":
                    radius = entity.dxf.radius * unit_factor
                    start_angle = math.radians(entity.dxf.start_angle)
                    end_angle = math.radians(entity.dxf.end_angle)
                    if end_angle < start_angle:
                        end_angle += 2 * math.pi
                    arc_length = radius * (end_angle - start_angle)
                    results.append({
                        "type": "arc",
                        "radius_m": round(radius, 4),
                        "arc_length_m": round(arc_length, 4),
                        "length_m": round(arc_length, 4),
                        "layer": entity.dxf.layer,
                    })
                elif etype == "LWPOLYLINE":
                    # get_points() returns (x, y, start_w, end_w, bulge) tuples.
                    pts = list(entity.get_points())
                    length = 0.0
                    entity_bulge_segs = 0

                    def seg_len(p_a, p_b):
                        # p_a is the start vertex (carries the bulge to p_b).
                        nonlocal entity_bulge_segs, bulge_fallbacks
                        bulge = p_a[4] if len(p_a) >= 5 else 0.0
                        if bulge_to_arc is not None and abs(bulge) >= 1e-9:
                            try:
                                _center, _start_a, _end_a, radius = bulge_to_arc(
                                    (p_a[0], p_a[1]), (p_b[0], p_b[1]), bulge
                                )
                                entity_bulge_segs += 1
                                return radius * abs(_end_a - _start_a)
                            except Exception as exc:
                                # bulge math blew up — fall back to chord, but
                                # record it so the caller knows polyline length
                                # is an under-estimate by some unknown amount.
                                _logger.warning(
                                    "drawing_qto: bulge_to_arc failed on layer=%s "
                                    "(bulge=%s); falling back to chord: %s",
                                    getattr(entity.dxf, "layer", "?"),
                                    bulge,
                                    exc,
                                )
                                bulge_fallbacks += 1
                        return math.dist((p_a[0], p_a[1]), (p_b[0], p_b[1]))

                    for i in range(len(pts) - 1):
                        length += seg_len(pts[i], pts[i + 1])
                    if entity.is_closed and len(pts) > 1:
                        length += seg_len(pts[-1], pts[0])
                    length = length * unit_factor
                    bulge_segments_count += entity_bulge_segs
                    results.append({
                        "type": "polyline",
                        "length_m": round(length, 4),
                        "closed": entity.is_closed,
                        "vertex_count": len(pts),
                        "bulge_segments": entity_bulge_segs,
                        "layer": entity.dxf.layer,
                    })
                elif etype == "POLYLINE":
                    pts = list(entity.points())
                    # POLYLINE flag bit 8 = is_3d_polyline (also exposed as
                    # `is_3d_polyline` attribute on ezdxf objects).
                    is_3d = bool(getattr(entity, "is_3d_polyline", False))
                    if not is_3d:
                        try:
                            is_3d = bool(int(getattr(entity.dxf, "flags", 0)) & 8)
                        except Exception:
                            is_3d = False

                    def _pt_xyz(p):
                        # Vertex coords may be Vec3 or tuple — be defensive.
                        x = getattr(p, "x", None)
                        if x is None:
                            x = p[0]
                            y = p[1]
                            z = p[2] if len(p) > 2 else 0.0
                        else:
                            y = p.y
                            z = getattr(p, "z", 0.0)
                        return (x, y, z)

                    length = 0.0
                    if is_3d:
                        coords = [_pt_xyz(p) for p in pts]
                        for i in range(len(coords) - 1):
                            length += math.dist(coords[i], coords[i + 1])
                        if entity.is_closed and len(coords) > 1:
                            length += math.dist(coords[-1], coords[0])
                    else:
                        for i in range(len(pts) - 1):
                            length += math.dist(
                                (pts[i][0], pts[i][1]),
                                (pts[i + 1][0], pts[i + 1][1])
                            )
                        if entity.is_closed and len(pts) > 1:
                            length += math.dist(
                                (pts[-1][0], pts[-1][1]),
                                (pts[0][0], pts[0][1])
                            )
                    length = length * unit_factor
                    results.append({
                        "type": "polyline_3d" if is_3d else "polyline",
                        "length_m": round(length, 4),
                        "closed": entity.is_closed,
                        "vertex_count": len(pts),
                        "is_3d": is_3d,
                        "layer": entity.dxf.layer,
                    })
                elif etype == "DIMENSION":
                    if hasattr(entity.dxf, "actual_measurement"):
                        val = entity.dxf.actual_measurement * unit_factor
                        results.append({
                            "type": "dimension",
                            "length_m": round(val, 4),
                            "layer": entity.dxf.layer,
                            "text": getattr(entity.dxf, "text", ""),
                        })
            except Exception as exc:
                layer = getattr(getattr(entity, "dxf", None), "layer", "?")
                _logger.warning(
                    "drawing_qto: skipped %s on layer=%s during length extract: %s",
                    etype, layer, exc,
                )
                entities_skipped += 1
                if len(entities_skipped_reasons) < 50:
                    # Cap reasons list — a corrupt DXF can produce thousands of
                    # identical errors; the counter remains accurate.
                    entities_skipped_reasons.append({
                        "etype": etype,
                        "layer": str(layer),
                        "error": str(exc),
                    })
                continue
        diagnostics = {
            "bulge_fallbacks": bulge_fallbacks,
            "entities_skipped": entities_skipped,
            "entities_skipped_reasons": entities_skipped_reasons,
        }
        return results, bulge_segments_count, diagnostics

    def _extract_areas(
        self, msp, unit_factor: float, layer_filter: list[str], min_area: float
    ) -> tuple[list[dict], bool, dict[str, Any]]:
        """``unit_factor`` converts raw drawing units straight to metres.

        Returns (areas, hatch_hole_fallback, diagnostics). hatch_hole_fallback
        is True if any HATCH path lacked readable path_type_flags so the
        caller is warned that holes may have been added as positive area.

        ``diagnostics`` is::

            {
                "entities_skipped": int,
                "hatches_skipped": int,
                "entities_skipped_reasons": [
                    {"etype": str, "layer": str, "error": str},
                    ...
                ],
            }
        """
        results = []
        hatch_hole_fallback = False
        entities_skipped = 0
        hatches_skipped = 0
        entities_skipped_reasons: list[dict[str, Any]] = []
        try:
            from shapely.geometry import Polygon
            use_shapely = True
        except ImportError:
            use_shapely = False

        for entity in msp:
            etype = entity.dxftype()
            layer = getattr(entity.dxf, "layer", "0")
            if layer_filter and layer not in layer_filter:
                continue
            try:
                if etype == "CIRCLE":
                    r = entity.dxf.radius * unit_factor
                    area = math.pi * r * r
                    if area >= min_area:
                        results.append({
                            "type": "circle",
                            "area_m2": round(area, 4),
                            "perimeter_m": round(2 * math.pi * r, 4),
                            "layer": layer,
                        })
                elif etype in ("LWPOLYLINE", "POLYLINE") and entity.is_closed:
                    pts = list(entity.get_points() if etype == "LWPOLYLINE" else entity.points())
                    coords = [(p[0] * unit_factor, p[1] * unit_factor) for p in pts]
                    if use_shapely and len(coords) >= 3:
                        poly = Polygon(coords)
                        area = poly.area
                        perim = poly.length
                    else:
                        area = abs(_shoelace(coords))
                        perim = sum(
                            math.dist(coords[i], coords[(i + 1) % len(coords)])
                            for i in range(len(coords))
                        )
                    if area >= min_area:
                        results.append({
                            "type": "polyline_area",
                            "area_m2": round(area, 4),
                            "perimeter_m": round(perim, 4),
                            "vertex_count": len(pts),
                            "layer": layer,
                        })
                elif etype == "HATCH":
                    if hasattr(entity, "paths"):
                        # Aggregate one entry per HATCH entity: external/outermost
                        # boundary paths add area, internal islands subtract it.
                        # If we can't read path_type_flags, fall back to summing
                        # |shoelace| per path (legacy behaviour) and flag it.
                        net_area = 0.0
                        per_path_legacy_area = 0.0
                        flags_readable = True
                        path_count = 0
                        for path in entity.paths:
                            if not (hasattr(path, "vertices") and len(path.vertices) >= 3):
                                continue
                            path_count += 1
                            coords = [
                                (v[0] * unit_factor, v[1] * unit_factor)
                                for v in path.vertices
                            ]
                            a = abs(_shoelace(coords))
                            per_path_legacy_area += a
                            ptf = getattr(path, "path_type_flags", None)
                            if ptf is None:
                                flags_readable = False
                                continue
                            try:
                                ptf_int = int(ptf)
                            except Exception:
                                flags_readable = False
                                continue
                            # Bit 1 = external boundary, Bit 4 = outermost.
                            # Either marks an outer (additive) contour; otherwise
                            # treat as a hole/island to subtract.
                            if ptf_int & 1 or ptf_int & 4:
                                net_area += a
                            else:
                                net_area -= a
                        if path_count == 0:
                            continue
                        if flags_readable:
                            area_value = max(net_area, 0.0)
                        else:
                            hatch_hole_fallback = True
                            area_value = per_path_legacy_area
                        if area_value >= min_area:
                            results.append({
                                "type": "hatch_area",
                                "area_m2": round(area_value, 4),
                                "path_count": path_count,
                                "hole_handling": (
                                    "outer_minus_holes" if flags_readable
                                    else "may_include_holes_as_positive_area"
                                ),
                                "layer": layer,
                            })
            except Exception as exc:
                _logger.warning(
                    "drawing_qto: skipped %s on layer=%s during area extract: %s",
                    etype, layer, exc,
                )
                entities_skipped += 1
                if etype == "HATCH":
                    hatches_skipped += 1
                if len(entities_skipped_reasons) < 50:
                    entities_skipped_reasons.append({
                        "etype": etype,
                        "layer": str(layer),
                        "error": str(exc),
                    })
                continue
        diagnostics = {
            "entities_skipped": entities_skipped,
            "hatches_skipped": hatches_skipped,
            "entities_skipped_reasons": entities_skipped_reasons,
            # Without shapely the polygon area math switches to a simpler
            # algorithm — quantities are less accurate for complex/holed
            # shapes. Surface it instead of degrading silently.
            "geometry_engine": "shapely" if use_shapely else "fallback",
        }
        return (
            sorted(results, key=lambda x: x["area_m2"], reverse=True),
            hatch_hole_fallback,
            diagnostics,
        )

    def _estimate_volumes(self, areas: list[dict], params: dict) -> list[dict]:
        # Default ceiling height comes from app.core.construction_constants
        # so all blocks share the same domain assumption. Caller overrides
        # via params["height_m"] for project-specific data.
        from app.core.construction_constants import DEFAULT_CEILING_HEIGHT_M
        height = float(params.get("height_m", DEFAULT_CEILING_HEIGHT_M))
        volumes = []
        for a in areas:
            if a["area_m2"] > 1.0:
                volumes.append({
                    "type": f"{a['type']}_volume",
                    "area_m2": a["area_m2"],
                    "height_m": height,
                    "assumed_height_m": height,
                    "method": "area_x_height_assumption",
                    "volume_m3": round(a["area_m2"] * height, 4),
                    "layer": a.get("layer", ""),
                })
        return volumes


def _line_from_run(y: float, run: list[dict]) -> dict:
    """Build a line record from a list of chars (fitz spans) that share
    a y baseline.

    Under the pdfplumber path the run was per-character so a naive join
    was correct. Under fitz the run is per-span; if two spans on the
    same line don't already end/start with whitespace, glue them with a
    space so word boundaries survive ("KEY PLAN" + "Road" = "KEY PLAN
    Road", not "KEY PLANRoad"). A char/span that already ends in
    whitespace or starts in whitespace is left alone.
    """
    if not run:
        return {"y": y, "size": 0.0, "text": "", "x0": 0.0, "x1": 0.0}
    parts: list[str] = []
    for i, c in enumerate(run):
        if i == 0:
            parts.append(c["text"])
            continue
        prev_text = run[i - 1]["text"]
        cur_text = c["text"]
        # Only insert a separator when neither side already has one AND
        # there's a visible x gap between the spans. Single-char items
        # (pdfplumber back-compat) never have a gap >= 0.5 between
        # adjacent chars, so this is a no-op on that path.
        gap = c["x0"] - run[i - 1]["x1"]
        needs_space = (
            gap >= 0.5
            and prev_text
            and cur_text
            and not prev_text[-1].isspace()
            and not cur_text[0].isspace()
        )
        if needs_space:
            parts.append(" ")
        parts.append(cur_text)
    text = "".join(parts)
    sizes = [c["size"] for c in run if c.get("size")]
    avg_size = sum(sizes) / len(sizes) if sizes else 0.0
    return {"y": y, "size": avg_size, "text": text,
            "x0": float(run[0]["x0"]), "x1": float(max(c["x1"] for c in run))}


# Pure CAD-tag patterns (all-caps + digits + hyphens, 4-15 chars, no spaces)
_CAD_TAG_RE = re.compile(r"^[A-Z0-9]{1,8}(?:-[A-Z0-9]{1,8}){1,4}$")
_COORD_PAIR_RE = re.compile(r"^\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+\s*$")


def _is_cad_tag(text: str) -> bool:
    t = text.strip()
    if not t or " " in t:
        return False
    if not (4 <= len(t) <= 15):
        return False
    # All-caps + digits + hyphens, must have at least one digit AND a hyphen
    if "-" not in t or not any(ch.isdigit() for ch in t):
        return False
    return bool(_CAD_TAG_RE.match(t))


def _is_coordinate_pair(text: str) -> bool:
    return bool(_COORD_PAIR_RE.match(text.strip()))


def _has_repeated_run(text: str, n: int) -> bool:
    """True if any single token repeats >= n times consecutively."""
    tokens = text.split()
    if len(tokens) < n:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= n:
                return True
        else:
            run = 1
    return False


def _shoelace(coords: list[tuple[float, float]]) -> float:
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1]
        area -= coords[j][0] * coords[i][1]
    return area / 2.0
