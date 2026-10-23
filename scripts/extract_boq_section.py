"""OCR pipeline for scanned (image-only) construction Bills of Quantities.

Many client BOQs arrive as scanned PDFs whose body pages carry no text layer,
so the platform's normal xlsx/csv ingest and the lightweight server-side text
extractors cannot read them. The 2GB production box also cannot afford to OCR a
370-page document on the fly. This module is the offline pre-processor that
turns such a scanned BOQ into structured, ingestion-ready line items.

What it does
------------
1. DISCOVERY: render every page at a low DPI and OCR only the top header band to
   find the pages belonging to a requested section (a "Bill" in CESMM BOQs),
   matching a ``--section`` keyword regex and excluding ``--exclude``. The match
   is done on the running page header ("BILL NO. 12 WASTE WATER NETWORK") so that
   passing mentions of the keyword inside other bills' line-item text do not
   produce false positives.
2. EXTRACTION: render the located pages at a high DPI and OCR them in two passes
   per page -- a full-page pass for the description column, and a cropped
   numeric-band pass (Tesseract ``--psm 6``) for the Unit / Quantity / Unit-Price
   (Rate) / Total-Price (Amount) columns. The numeric band is OCR'd separately
   because the table's vertical rules make a single page-wide pass drop the inner
   numeric cells. Rows are reconstructed by vertical (y) alignment.
3. VALIDATION (QA gate): every priced row carries built-in redundancy
   (quantity * unit_price == total_price). ``validate_rows`` checks this
   arithmetically and flags any row whose product does not reconcile, so that
   only arithmetically sound rows are trusted / ingested and OCR misreads are
   caught instead of silently corrupting quantities or values.

It is memory-safe: each page pixmap is rendered, OCR'd and freed before the next
page, so peak memory stays to a single rasterised page.

Output: structured JSON to stdout (section pages, per-row five-field records,
QA summary, per-unit and per-diameter length totals, section price total) and,
optionally, an ingestion-ready .xlsx via ``--xlsx``.

Example
-------
    python scripts/extract_boq_section.py \
        --pdf "G:/.../QZ-ENG-...-000008-B_Bill of Quantities (Unpriced).pdf" \
        --section "waste water|foul|sewer" --exclude "storm|surface" \
        --xlsx out.xlsx
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# --- Column geometry -------------------------------------------------------
# Boundaries expressed as a fraction of page pixel width. Calibrated on the the client project
# Infrastructure Package 1 BOQ (CESMM4 layout: Item | Description | CESMM ref |
# Unit | Qty | Rate | Amount). The numeric columns sit in the right portion of
# the page; the description occupies the left portion.
NUMERIC_BAND_START = 0.50  # left edge of the numeric band (fraction of width)
COL_REF_MAX = 0.585        # CESMM reference column ends here
COL_UNIT_MAX = 0.635       # Unit column ends here
COL_QTY_MAX = 0.720        # Quantity column ends here
COL_RATE_MAX = 0.825       # Unit-price (Rate) column ends here; Amount is beyond

HEADER_BAND_FRAC = 0.22    # top fraction of a page scanned during discovery

# Units of measure understood by the parser (upper-cased comparison).
UNIT_TOKENS = {
    "M", "M2", "M3", "M?", "NR", "NO", "NO.", "SUM", "T", "HA", "KG",
    "ITEM", "LM", "L.M", "LS", "MM",
}
LENGTH_UNITS = {"M", "LM", "L.M"}

# Lines that are running totals / collection carries, never priced line items.
SKIP_DESC_RE = re.compile(
    r"carried\s+to\s+collection|brought\s+forward|carried\s+forward|"
    r"\bc/?f\b|\bb/?f\b|to\s+collection|to\s+summary|^collection$|"
    r"^\s*total\b|sub[\s-]?total|grand\s+total",
    re.IGNORECASE,
)

NUM_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")
# Diameter must come from the pipe "nominal bore" text, NOT from depth-of-cover
# phrases such as "min 1000" or "exceeds 4000mm" found in ancillary items.
BORE_RE = re.compile(r"bore[:\s]*\(?(\d{2,4})\s*mm", re.IGNORECASE)
DIAM_RE = re.compile(r"(\d{2,4})\s*mm\s*dia", re.IGNORECASE)
# Depth band / excavation depth context, e.g. "Depth not exceeding 1.5m",
# "Depth 1.5 to 2m", "Depth of cover min 1000 to max ...".
DEPTH_RE = re.compile(
    r"depth(?:\s+of\s+cover)?\s+"
    r"(not\s+exceeding\s+[\d.]+\s*m|exceeding\s+[\d.]+\s*m|"
    r"[\d.]+\s*(?:to|-)\s*[\d.]+\s*m|min\s+\d+\s+to\s+max[\w\s]*?\d*\s*m?)",
    re.IGNORECASE,
)


def _depth_label(text):
    m = DEPTH_RE.search(text or "")
    if not m:
        return ""
    label = re.sub(r"\s+", " ", "Depth " + m.group(1)).strip().lower()
    # Normalise "4.0" -> "4" so the BOQ's inconsistent "4 to 4.5m" and
    # "4.0 to 4.5m" wordings collapse into one depth band.
    label = re.sub(r"(\d)\.0(?=\D|$)", r"\1", label)
    return label


def cesmm_class(ref):
    """Map an OCR'd CESMM item reference to its class letter.

    CESMM4 references are a class letter followed by digits, e.g. "I 112"
    (Class I = Pipework - Pipes), "L 591" (Class L = Supports/Protection,
    i.e. bedding and surround ancillaries), "J" (fittings/valves),
    "K" (manholes). Tesseract frequently reads the leading "I" as "1" or "|",
    so a leading digit / pipe character is treated as Class I.
    """
    s = ref.strip().lstrip("|").strip()
    if not s:
        return ""
    c = s[0]
    if c in "1I|" or c.isdigit():
        return "I"
    if c.isalpha():
        return c.upper()
    return ""


@dataclass
class LineItem:
    """A single extracted BOQ line item with the five reportable fields."""

    section: str
    page: int                       # 1-based human page number
    item_ref: str                   # CESMM reference, e.g. "I 112"
    cesmm_class: str                # CESMM class letter (I pipes, L ancillaries...)
    description: str
    quantity: Optional[float]
    unit: str
    unit_price: Optional[float]     # Rate column
    total_price: Optional[float]    # Amount column
    diameter_mm: Optional[int] = None
    depth_band: str = ""            # excavation / cover depth band from description
    is_pipe_length: bool = False    # True only for Class I pipe-laying length items
    qa_row_ok: Optional[bool] = None
    qa_note: str = ""


# --- Low level OCR helpers -------------------------------------------------

def _render(page: "fitz.Page", dpi: int) -> Image.Image:
    """Rasterise a single PDF page to a PIL image. Caller must let it be freed."""
    pm = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pm.tobytes("png")))
    return img


def _num(token: str) -> Optional[float]:
    token = token.strip()
    if not NUM_RE.match(token):
        return None
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _lines_from_dataframe(df):
    """Group an image_to_data dataframe into text lines with geometry.

    Returns a list of dicts: {top, left, right, tokens:[(left,text)], text}.
    """
    df = df[df.conf > 10].dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    lines = []
    for _, g in df.groupby(["block_num", "par_num", "line_num"]):
        g = g.sort_values("left")
        tokens = [(int(r.left), str(r.text)) for r in g.itertuples()]
        if not tokens:
            continue
        lines.append(
            {
                "top": float(g["top"].mean()),
                "left": int(g["left"].min()),
                "right": int((g["left"] + g["width"]).max()),
                "tokens": tokens,
                "text": " ".join(t for _, t in tokens),
            }
        )
    lines.sort(key=lambda x: x["top"])
    return lines


def _ocr_header(page: "fitz.Page", dpi: int) -> str:
    """OCR only the top header band of a page (cheap), upper-cased single line."""
    img = _render(page, dpi)
    w, h = img.size
    band = img.crop((0, 0, w, int(h * HEADER_BAND_FRAC)))
    txt = pytesseract.image_to_string(band)
    return " ".join(txt.split()).upper()


# --- Discovery -------------------------------------------------------------

BILL_MARKER_RE = re.compile(r"BILL\s*NO|WORK\s+ITEMS", re.IGNORECASE)


def _contiguous_groups(pages):
    """Split a sorted list of page indices into contiguous runs."""
    groups = []
    for p in pages:
        if groups and p == groups[-1][-1] + 1:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def find_section_pages(doc, section_re, exclude_re, dpi=130, verbose=True,
                       require_bill=True):
    """Return the 0-based page indices whose header matches the section.

    Matching is performed on the page header band only, so line-item text deeper
    in unrelated bills does not create false positives. When ``require_bill`` is
    set (default), a page must also carry a bill marker ("BILL NO" / "WORK
    ITEMS") in its header, which drops preamble, specification and summary pages
    that merely mention the keyword.

    Note: a broad keyword can legitimately match more than one bill (e.g.
    "waste water" matches both the Waste Water Network bill and a Waste Water
    Pump Station bill). The caller is told about each contiguous page-group so
    that distinct bills are visible rather than silently merged. Use ``--pages``
    or a more specific ``--section`` to target a single bill.
    """
    pages = []
    for i in range(doc.page_count):
        header = _ocr_header(doc[i], dpi)
        if not section_re.search(header):
            continue
        if exclude_re and exclude_re.search(header):
            continue
        if require_bill and not BILL_MARKER_RE.search(header):
            continue
        pages.append(i)
        if verbose:
            sys.stderr.write("  discovery matched page %d\n" % (i + 1))
        if verbose and i % 25 == 0:
            sys.stderr.flush()
    return pages


# --- Extraction ------------------------------------------------------------

def _classify_numeric_row(tokens, width):
    """Assign band tokens to (ref, unit, qty, rate, amount) using x-fractions."""
    ref, unit, qty, rate, amount = "", "", None, None, None
    for left, text in tokens:
        frac = left / width
        up = text.upper().strip(".")
        if frac < COL_REF_MAX:
            ref = (ref + " " + text).strip()
        elif frac < COL_UNIT_MAX:
            if up in UNIT_TOKENS or up.isalpha():
                unit = text
            else:
                v = _num(text)
                if v is not None and qty is None:
                    qty = v
        elif frac < COL_QTY_MAX:
            v = _num(text)
            if v is not None:
                qty = v if qty is None else qty
            elif up in UNIT_TOKENS:
                unit = text
        elif frac < COL_RATE_MAX:
            v = _num(text)
            if v is not None:
                rate = v
        else:
            v = _num(text)
            if v is not None:
                amount = v
    return ref, unit, qty, rate, amount


def _extract_page(page, page_no, section_name, dpi):
    """Extract line items from one page. Returns list[LineItem]."""
    img = _render(page, dpi)
    width, height = img.size
    tol = height * 0.012  # vertical alignment tolerance

    # Pass 1: full page -> description-side lines (left of the numeric band).
    full_df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
    all_lines = _lines_from_dataframe(full_df)
    desc_lines = [ln for ln in all_lines if ln["left"] < NUMERIC_BAND_START * width]

    # Pass 2: numeric band -> rows (psm 6 keeps the bordered cells intact).
    band = img.crop((int(NUMERIC_BAND_START * width), 0, width, height))
    band_df = pytesseract.image_to_data(
        band, config="--psm 6", output_type=pytesseract.Output.DATAFRAME
    )
    band_lines = _lines_from_dataframe(band_df)
    # Shift band token x back into full-page coordinates.
    x_off = int(NUMERIC_BAND_START * width)
    for ln in band_lines:
        ln["tokens"] = [(l + x_off, t) for l, t in ln["tokens"]]

    # Ignore the running page header and the page footer (date / page number).
    top_limit = height * HEADER_BAND_FRAC * 0.6
    bot_limit = height * 0.93

    # Context carried down the page (material header and depth sub-header).
    material_ctx = ""
    depth_ctx = ""
    items = []
    for bl in band_lines:
        if bl["top"] < top_limit or bl["top"] > bot_limit:
            continue
        ref, unit, qty, rate, amount = _classify_numeric_row(bl["tokens"], width)
        # A genuine line item has at least a quantity or a unit-priced amount,
        # and is not a collection / carried-forward total.
        has_values = (qty is not None) or (amount is not None and unit)
        if not has_values:
            continue

        # Find the nearest description line on the same row.
        leaf = ""
        best = None
        for dl in desc_lines:
            d = abs(dl["top"] - bl["top"])
            if d <= tol and (best is None or d < best):
                best = d
                leaf = dl["text"]

        # Update carried context from description lines above this row.
        for dl in desc_lines:
            if dl["top"] >= bl["top"] - tol:
                continue
            t = dl["text"]
            if re.search(r"\bdepth\b", t, re.IGNORECASE):
                depth_ctx = t
            if re.search(r"pipe|culvert|duct|main|gravity|rising|manhole|chamber|fitting",
                         t, re.IGNORECASE) and "nominal bore" not in t.lower():
                material_ctx = t

        if SKIP_DESC_RE.search(leaf) or SKIP_DESC_RE.search(ref):
            continue

        parts = [p for p in (material_ctx, depth_ctx, leaf) if p]
        description = " | ".join(parts) if parts else leaf
        dm = BORE_RE.search(leaf) or DIAM_RE.search(leaf) or BORE_RE.search(description)
        diameter = int(dm.group(1)) if dm else None

        klass = cesmm_class(ref)
        is_pipe = (
            klass == "I"
            and unit
            and unit.upper() in LENGTH_UNITS
            and qty is not None
        )
        items.append(
            LineItem(
                section=section_name,
                page=page_no,
                item_ref=ref,
                cesmm_class=klass,
                description=description,
                quantity=qty,
                unit=unit,
                unit_price=rate,
                total_price=amount,
                diameter_mm=diameter,
                depth_band=_depth_label(depth_ctx) or _depth_label(description),
                is_pipe_length=bool(is_pipe),
            )
        )
    return items


# --- QA gate ---------------------------------------------------------------

def validate_rows(items, tol_frac=0.005, tol_abs=1.0):
    """Arithmetic QA gate: quantity * unit_price must equal total_price.

    Mutates each item's ``qa_row_ok`` / ``qa_note`` and returns a summary dict.
    Rows lacking the three numbers (e.g. unpriced file, or sum-unit items) are
    marked ok=None (not checkable) rather than failed.
    """
    total = len(items)
    passed = failed = uncheckable = 0
    failures = []
    for it in items:
        if it.quantity is None or it.unit_price is None or it.total_price is None:
            it.qa_row_ok = None
            it.qa_note = "not checkable (missing qty/rate/amount)"
            uncheckable += 1
            continue
        expected = it.quantity * it.unit_price
        tol = max(tol_abs, tol_frac * abs(it.total_price))
        if abs(expected - it.total_price) <= tol:
            it.qa_row_ok = True
            passed += 1
        else:
            it.qa_row_ok = False
            it.qa_note = "qty*rate=%.2f != amount=%.2f" % (expected, it.total_price)
            failed += 1
            failures.append(
                {
                    "page": it.page,
                    "item_ref": it.item_ref,
                    "description": it.description,
                    "quantity": it.quantity,
                    "unit_price": it.unit_price,
                    "total_price": it.total_price,
                    "expected_amount": round(expected, 2),
                }
            )
    return {
        "total_rows": total,
        "rows_passing": passed,
        "rows_failing": failed,
        "rows_uncheckable": uncheckable,
        "failures": failures,
    }


# --- Aggregation -----------------------------------------------------------

def summarise(items):
    """Pipe-laying length and section price totals.

    The network pipe length is summed ONLY over Class I pipe-laying items
    (``is_pipe_length``). Class L ancillary items (pipe bedding, surround and
    protection) are also measured in metres and mirror the pipe runs, so adding
    them would double-count the network length; their length is reported
    separately as an excluded figure. Length and price totals use QA-passing
    rows only; values from rows that failed the arithmetic gate are reported
    separately as provisional.
    """
    pipe_by_diam = {}
    pipe_by_depth = {}
    pipe_total = 0.0
    pipe_price = 0.0
    ancillary_length = 0.0
    section_price = 0.0
    section_price_provisional = 0.0
    for it in items:
        if it.is_pipe_length and it.qa_row_ok is not False:
            pipe_total += it.quantity
            dkey = "%dmm" % it.diameter_mm if it.diameter_mm else "unspecified"
            pipe_by_diam[dkey] = pipe_by_diam.get(dkey, 0.0) + it.quantity
            depth = it.depth_band or "unspecified"
            pipe_by_depth[depth] = pipe_by_depth.get(depth, 0.0) + it.quantity
            if it.total_price is not None:
                pipe_price += it.total_price
        elif (it.cesmm_class == "L" and it.unit and it.unit.upper() in LENGTH_UNITS
              and it.quantity is not None):
            ancillary_length += it.quantity
        if it.total_price is not None:
            if it.qa_row_ok is False:
                section_price_provisional += it.total_price
            else:
                section_price += it.total_price

    def _diam_key(k):
        return (k == "unspecified", int(k[:-2]) if k[:-2].isdigit() else 0)

    def _depth_key(k):
        nums = re.findall(r"[\d.]+", k)
        return (k == "unspecified", float(nums[0]) if nums else 0.0)

    return {
        "pipe_length_total_m": round(pipe_total, 2),
        "pipe_length_by_diameter_m": {
            k: round(pipe_by_diam[k], 2) for k in sorted(pipe_by_diam, key=_diam_key)
        },
        "pipe_length_by_depth_m": {
            k: round(pipe_by_depth[k], 2) for k in sorted(pipe_by_depth, key=_depth_key)
        },
        "pipe_laying_price": round(pipe_price, 2),
        "ancillary_length_excluded_m": round(ancillary_length, 2),
        "section_total_price": round(section_price, 2),
        "section_price_in_failed_rows": round(section_price_provisional, 2),
    }


def write_xlsx(items, path):
    """Write the ingestion-ready workbook (one row per item, five fields + QA)."""
    import pandas as pd

    rows = []
    for it in items:
        rows.append(
            {
                "Section": it.section,
                "Page": it.page,
                "Item Ref": it.item_ref,
                "CESMM Class": it.cesmm_class,
                "Description": it.description,
                "Diameter (mm)": it.diameter_mm,
                "Depth": it.depth_band,
                "Quantity": it.quantity,
                "Unit": it.unit,
                "Unit Price": it.unit_price,
                "Total Price": it.total_price,
                "Is Pipe Length": it.is_pipe_length,
                "qa_row_ok": it.qa_row_ok,
                "qa_note": it.qa_note,
            }
        )
    pd.DataFrame(rows).to_excel(path, index=False)


# --- Driver ----------------------------------------------------------------

def extract(pdf, section, exclude=None, disc_dpi=130, ocr_dpi=300,
            pages=None, verbose=True):
    """Run discovery + extraction + QA on one PDF; return a result dict."""
    section_re = re.compile(section, re.IGNORECASE)
    exclude_re = re.compile(exclude, re.IGNORECASE) if exclude else None
    doc = fitz.open(pdf)

    if pages:
        page_idx = [p - 1 for p in pages]
    else:
        if verbose:
            sys.stderr.write("Discovery: locating section pages...\n")
        page_idx = find_section_pages(doc, section_re, exclude_re, disc_dpi, verbose)

    section_name = section.split("|")[0].strip().title()
    items = []
    for i in page_idx:
        if verbose:
            sys.stderr.write("  OCR page %d\n" % (i + 1))
            sys.stderr.flush()
        items.extend(_extract_page(doc[i], i + 1, section_name, ocr_dpi))

    qa = validate_rows(items)
    agg = summarise(items)
    groups = [[i + 1 for i in g] for g in _contiguous_groups(sorted(page_idx))]
    doc.close()
    return {
        "pdf": pdf,
        "section": section,
        "exclude": exclude,
        "section_pages_human": [i + 1 for i in page_idx],
        "section_page_groups": groups,
        "ocr_dpi": ocr_dpi,
        "qa_summary": qa,
        "totals": agg,
        "items": [asdict(it) for it in items],
    }, items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, help="Path to the scanned BOQ PDF")
    ap.add_argument("--section", required=True,
                    help="Section keyword regex, e.g. 'waste water|foul|sewer'")
    ap.add_argument("--exclude", default=None,
                    help="Exclusion keyword regex, e.g. 'storm|surface'")
    ap.add_argument("--dpi", type=int, default=300, help="OCR DPI for line items")
    ap.add_argument("--discovery-dpi", type=int, default=130)
    ap.add_argument("--pages", default=None,
                    help="Optional explicit 1-based page range a-b to skip discovery")
    ap.add_argument("--xlsx", default=None, help="Write ingestion-ready xlsx here")
    ap.add_argument("--json-out", default=None, help="Write full JSON result here")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    pages = None
    if args.pages:
        a, b = args.pages.split("-")
        pages = list(range(int(a), int(b) + 1))

    result, items = extract(
        args.pdf, args.section, args.exclude,
        disc_dpi=args.discovery_dpi, ocr_dpi=args.dpi,
        pages=pages, verbose=not args.quiet,
    )

    if args.xlsx:
        write_xlsx(items, args.xlsx)
        result["xlsx"] = args.xlsx
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    out = json.dumps(result, indent=2, ensure_ascii=True)
    sys.stdout.write(out.encode("ascii", "replace").decode() + "\n")

    qa = result["qa_summary"]
    t = result["totals"]
    sys.stderr.write("\n=== QA SUMMARY ===\n")
    sys.stderr.write(
        "%d rows: %d pass, %d FAIL, %d uncheckable\n"
        % (qa["total_rows"], qa["rows_passing"], qa["rows_failing"],
           qa["rows_uncheckable"])
    )
    sys.stderr.write("Pipe-laying length (m): %s\n" % t["pipe_length_total_m"])
    sys.stderr.write("Ancillary length excluded (m): %s\n" % t["ancillary_length_excluded_m"])
    sys.stderr.write("Section total price (SAR): %s\n" % t["section_total_price"])
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
