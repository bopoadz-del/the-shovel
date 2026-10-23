"""Extract the scanned example developer "Heights" RW Bill of Quantities.

Target: BOQ-RW.pdf (28 pages, image-only / no text layer). Fronds Apartments
A & B (Main Works), Package 2 [N004-060-01].

Layout of the bill pages (identical blank tender form on every page):

    Item No. | Description | Qty | Unit | Rate | Amount

Note the column order is Qty BEFORE Unit (the opposite of the CESMM4 BOQs this
platform has handled before), so column geometry is calibrated fresh here and
the CESMM class / pipe-diameter machinery is deliberately NOT reused.

Reality of this file (established by inspection, see the result .md):
  * Pages 1-6 are "Preambles & Method of Measurement" (P/n footers) -- skipped.
  * Pages 7-28 are the priced-bill pages, but this is a TENDER issue: the Rate
    and Amount columns are BLANK on almost every row (contractor to fill).
  * A small number of rows carry pre-filled "Supply rate = AED .../unit" values
    (the tile and door supply-rate lines on pages 15, 24, 28). Those are the
    only genuinely priced rows and the only rows the qty*rate=amount QA can
    check. Everything else is extracted with Rate/Amount blank and qa_row_ok
    left as None (uncheckable), NOT failed.
  * There is no Collection / Summary / grand-total page; every "Total Carried
    To Collection" row is blank. So page-carry reconciliation is not applicable.

Currency: AED (confirmed from the "Supply rate = AED 45/m2" / "AED 986.24/no"
text on the priced rows, not assumed from the project location).

Method: render each page at 300 dpi, OCR with pytesseract image_to_data to get
per-token bounding boxes, assign every token to a column by its x-fraction,
cluster value tokens into rows by y, and attach the multi-line description from
the description column. Each pixmap is freed before the next page (memory-safe).

Output: an .xlsx of per-item rows and a JSON dump; a companion .md QA report is
written by hand from these results.
"""

from __future__ import annotations

import io
import json
import os
import re
import sys

import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image

PDF = os.environ.get("BOQ_PDF", "/data/example/boq_rw.pdf")

# Bill (line-item) pages are 1-based 7..28; preambles are 1..6.
FIRST_BILL_PAGE = 7

# Column boundaries as a fraction of page pixel width, calibrated on the priced
# pages (15/24/28) where known qty/rate/amount tokens fix each column:
#   Qty ~0.57 | Unit ~0.64 | Rate ~0.745 | Amount ~0.87 (centres).
ITEM_MAX = 0.11   # Item No. letter column ends here
DESC_MAX = 0.53   # Description column ends here
QTY_MAX = 0.61    # Qty column ends here
UNIT_MAX = 0.69   # Unit column ends here
RATE_MAX = 0.80   # Rate column ends here; Amount is beyond, to the right border

BODY_TOP = 0.165  # below the running header + column-header band
BODY_BOT = 0.925  # above the page footer (HQS ref / Page x/y/z)

DPI = 300
CONF_MIN = 25

NUM_RE = re.compile(r"^[\d,]+(?:\.\d+)?$")
UNIT_RE = re.compile(r"^(m|m2|m3|no|no\.|nr|item|sum|kg|t|ha|%|lm)$", re.IGNORECASE)
# Item reference in the far-left column: a letter, optionally with a digit
# suffix (A, B, ... E1, E2, E3, H1, H2, H3, F1 ...).
ITEMREF_RE = re.compile(r"^[A-Z]\d?$")

SKIP_DESC_RE = re.compile(
    r"carried\s+to\s+collection|to\s+collection|brought\s+forward|"
    r"carried\s+forward|^collection$|^\s*total\b|sub[\s-]?total|grand\s+total",
    re.IGNORECASE,
)

# Bill section titles as they appear in the running page header.
BILL_RE = re.compile(r"BILL\s*NO\.?\s*(\d+)\s*[-–]?\s*(.*)", re.IGNORECASE)

# Tokens that belong to the ARENCO "TENDER DOCUMENTS / 29 OCT 2007" stamp, used
# to fence off the stamp so its date digits are not mistaken for rate/amount.
STAMP_WORD_RE = re.compile(r"ARENCO|TENDER|DOCUMENT|OCT", re.IGNORECASE)


def _num(tok):
    tok = tok.strip()
    if not NUM_RE.match(tok):
        return None
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return None


def _render_gray_and_rgb(page, dpi):
    pm = page.get_pixmap(dpi=dpi)
    img = Image.open(io.BytesIO(pm.tobytes("png")))
    return img


def _page_header_title(img):
    """OCR the top band and return (bill_no, bill_title) if a BILL NO header."""
    w, h = img.size
    band = img.crop((0, 0, w, int(0.14 * h)))
    txt = " ".join(pytesseract.image_to_string(band).split())
    m = BILL_RE.search(txt)
    if not m:
        return None, txt
    no = m.group(1)
    title = m.group(2)
    # Trim trailing OCR noise / logo words.
    title = re.split(r"\bNAKHEEL\b", title, flags=re.IGNORECASE)[0].strip(" -–.")
    return "BILL NO. %s" % no, title


NUMERIC_BAND_START = 0.50  # left edge of the numeric second-pass band


def _df_tokens(img, conf_min, x_off_px=0):
    w_full = img.size[0] + x_off_px  # full-page width for x-fraction reference
    df = pytesseract.image_to_data(
        img, config="--psm 6" if x_off_px else "",
        output_type=pytesseract.Output.DATAFRAME)
    df = df[df.conf > conf_min].dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    return df


def _tokens(img):
    """Full-page pass (descriptions, item refs) + numeric-band pass (psm 6).

    The bordered numeric cells are dropped by a single page-wide pass, so the
    Qty/Unit/Rate/Amount band is OCR'd again in isolation with psm 6 and its
    tokens are mapped back into full-page x-fractions and merged in.
    """
    w, h = img.size
    toks = []
    for r in _df_tokens(img, CONF_MIN).itertuples():
        t = str(r.text).strip()
        toks.append({"t": t, "xc": (r.left + r.width / 2) / w,
                     "yc": (r.top + r.height / 2) / h, "src": "full"})

    x_off = int(NUMERIC_BAND_START * w)
    band = img.crop((x_off, 0, w, h))
    for r in _df_tokens(band, 10, x_off_px=x_off).itertuples():
        t = str(r.text).strip()
        toks.append({"t": t, "xc": (r.left + x_off + r.width / 2) / w,
                     "yc": (r.top + r.height / 2) / h, "src": "band"})

    # Item-No. column pass (psm 6, low conf): the single italic letters are
    # faint and the page-wide pass drops them.
    lx = int(ITEM_MAX * w)
    lband = img.crop((0, 0, lx, h))
    for r in _df_tokens(lband, 0, x_off_px=1).itertuples():
        t = str(r.text).strip()
        toks.append({"t": t, "xc": (r.left + r.width / 2) / w,
                     "yc": (r.top + r.height / 2) / h, "src": "item"})
    return toks


def _stamp_top(toks):
    """y-fraction of the top of the ARENCO stamp, or 1.0 if none found.

    The stamp sits bottom-right; anything at or below this y and to the right of
    the rate column is stamp noise (its '29'/'2007'/'OCT' date), never a value.
    """
    ys = [tk["yc"] for tk in toks if STAMP_WORD_RE.search(tk["t"]) and tk["xc"] > 0.6]
    return min(ys) if ys else 1.0


def _column(xc):
    if xc < ITEM_MAX:
        return "item"
    if xc < DESC_MAX:
        return "desc"
    if xc < QTY_MAX:
        return "qty"
    if xc < UNIT_MAX:
        return "unit"
    if xc < RATE_MAX:
        return "rate"
    return "amount"


def extract_page(page, page_no):
    img = _render_gray_and_rgb(page, DPI)
    bill, title = _page_header_title(img)
    toks = _tokens(img)
    stamp_y = _stamp_top(toks)

    body = [tk for tk in toks if BODY_TOP <= tk["yc"] <= BODY_BOT]

    # Fence off stamp date digits: drop numeric tokens in the rate/amount band
    # that fall inside the stamp box (at/below its top, right of the rate col).
    def is_stamp(tk):
        return tk["yc"] >= stamp_y - 0.005 and tk["xc"] > 0.60 and _num(tk["t"]) is not None

    body = [tk for tk in body if not is_stamp(tk)]

    line_h = 0.010  # ~ one text line as a fraction of page height
    row_tol = 0.012

    # Value tokens anchor rows: a real line item carries a qty (and/or a
    # unit / rate / amount). Section sub-headings carry none and are folded into
    # the following item's description instead.
    def money_ok(tk):
        # A genuine Rate/Amount in this priced tender is always a money figure
        # with a decimal (e.g. 45.00, 986.24, 57,375.00). Bare integers in these
        # columns are ARENCO stamp date fragments (29 / 2007 / 2097) -> reject.
        t = tk["t"]
        return _num(t) is not None and ("." in t or "," in t)

    def is_value(tk):
        col = _column(tk["xc"])
        if col == "qty" and _num(tk["t"]) is not None:
            return True
        if col == "unit" and UNIT_RE.match(tk["t"]):
            return True
        if col in ("rate", "amount") and money_ok(tk):
            return True
        return False

    value_toks = [tk for tk in body if is_value(tk)]
    value_toks.sort(key=lambda tk: tk["yc"])

    # Cluster value tokens into rows by y proximity.
    clusters = []
    for tk in value_toks:
        if clusters and tk["yc"] - clusters[-1]["ymax"] <= row_tol:
            clusters[-1]["toks"].append(tk)
            clusters[-1]["ymax"] = max(clusters[-1]["ymax"], tk["yc"])
        else:
            clusters.append({"toks": [tk], "yc": tk["yc"], "ymax": tk["yc"]})
    for c in clusters:
        c["yc"] = float(np.mean([tk["yc"] for tk in c["toks"]]))

    item_toks = [tk for tk in body if _column(tk["xc"]) == "item"
                 and ITEMREF_RE.match(tk["t"].upper())]
    desc_toks = [tk for tk in body if _column(tk["xc"]) == "desc"]

    rows = []
    for i, c in enumerate(clusters):
        y = c["yc"]
        y_next = clusters[i + 1]["yc"] if i + 1 < len(clusters) else BODY_BOT

        qty = unit = rate = amount = None
        for tk in c["toks"]:
            col = _column(tk["xc"])
            v = _num(tk["t"])
            if col == "qty" and v is not None and qty is None:
                qty = v
            elif col == "unit" and UNIT_RE.match(tk["t"]) and not unit:
                unit = tk["t"]
            elif col == "rate" and money_ok(tk) and rate is None:
                rate = v
            elif col == "amount" and money_ok(tk) and amount is None:
                amount = v

        # Item ref: nearest item-column letter on this row.
        ref = ""
        best = None
        for tk in item_toks:
            d = abs(tk["yc"] - y)
            if d <= row_tol and (best is None or d < best):
                best = d
                ref = tk["t"].upper()

        # Description: desc-column text from just above this row down to just
        # above the next row, in reading order (folds in any sub-heading above).
        lo = y - line_h * 1.5 if i == 0 else (clusters[i - 1]["yc"] + y) / 2
        hi = (y + y_next) / 2 if i + 1 < len(clusters) else y_next
        dts = [tk for tk in desc_toks if lo <= tk["yc"] < hi]
        # Group into physical text lines (bin by y) before reading left-to-right,
        # so multi-line descriptions do not interleave into word salad.
        dts.sort(key=lambda tk: tk["yc"])
        lines = []
        for tk in dts:
            if lines and tk["yc"] - lines[-1][0] <= 0.006:
                lines[-1][1].append(tk)
            else:
                lines.append([tk["yc"], [tk]])
        parts = []
        for _, lts in lines:
            lts.sort(key=lambda tk: tk["xc"])
            parts.append(" ".join(tk["t"] for tk in lts))
        desc = re.sub(r"\s+", " ", " ".join(parts)).strip()

        if SKIP_DESC_RE.search(desc) or SKIP_DESC_RE.search(ref):
            continue
        # Rows with neither a quantity nor a value are not line items.
        if qty is None and rate is None and amount is None and unit is None:
            continue

        rows.append({
            "bill": bill or "",
            "bill_title": title if bill else "",
            "page": page_no,
            "item_ref": ref,
            "description": desc,
            "quantity": qty,
            "unit": unit or "",
            "unit_price": rate,
            "total_price": amount,
        })

    del img
    return rows, (bill, title)


def qa_rows(rows, tol_frac=0.005, tol_abs=1.0):
    priced = checked = passed = failed = 0
    failures = []
    for r in rows:
        q, rt, am = r["quantity"], r["unit_price"], r["total_price"]
        if rt is not None and am is not None:
            priced += 1
        if q is None or rt is None or am is None:
            r["qa_row_ok"] = None
            r["qa_note"] = "unpriced / not checkable (missing qty, rate or amount)"
            continue
        checked += 1
        expected = q * rt
        tol = max(tol_abs, tol_frac * abs(am))
        if abs(expected - am) <= tol:
            r["qa_row_ok"] = True
            r["qa_note"] = "ok: %.2f * %.2f = %.2f" % (q, rt, am)
            passed += 1
        else:
            r["qa_row_ok"] = False
            r["qa_note"] = "qty*rate=%.2f != amount=%.2f" % (expected, am)
            failed += 1
            failures.append({"page": r["page"], "item_ref": r["item_ref"],
                             "quantity": q, "unit_price": rt, "total_price": am,
                             "expected": round(expected, 2),
                             "description": r["description"][:80]})
    return {"total_rows": len(rows), "priced_rows": priced,
            "rows_checked": checked, "rows_pass": passed, "rows_fail": failed,
            "rows_uncheckable": len(rows) - checked, "failures": failures}


def main():
    doc = fitz.open(PDF)
    all_rows = []
    bills = []
    for i in range(doc.page_count):
        page_no = i + 1
        if page_no < FIRST_BILL_PAGE:
            continue
        sys.stderr.write("OCR page %d\n" % page_no)
        sys.stderr.flush()
        rows, (bill, title) = extract_page(doc[i], page_no)
        all_rows.extend(rows)
        bills.append({"page": page_no, "bill": bill, "title": title,
                      "rows": len(rows)})
    doc.close()

    # Forward-fill the bill/section header onto pages where its OCR failed
    # (bills run contiguously through the document).
    last_bill = last_title = ""
    for b in bills:
        if b["bill"]:
            last_bill, last_title = b["bill"], b["title"]
        else:
            b["bill"], b["title"] = last_bill, last_title
    page_bill = {b["page"]: (b["bill"], b["title"]) for b in bills}
    for r in all_rows:
        if not r["bill"]:
            r["bill"], r["bill_title"] = page_bill.get(r["page"], ("", ""))

    qa = qa_rows(all_rows)
    result = {"pdf": PDF, "currency": "AED", "qa": qa,
              "pages": bills, "items": all_rows}

    out_dir = os.path.join(os.environ.get("FORK_SCRATCH_DIR", "/data/example/scratch"), "wetransfer_boq")
    with open(out_dir + "/boq_rw_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=True)

    # xlsx
    import pandas as pd
    df = pd.DataFrame([{
        "Bill/Section": (r["bill"] + (" - " + r["bill_title"] if r["bill_title"] else "")).strip(" -"),
        "Page": r["page"],
        "Item Ref": r["item_ref"],
        "Description": r["description"],
        "Quantity": r["quantity"],
        "Unit": r["unit"],
        "Unit Price": r["unit_price"],
        "Total Price": r["total_price"],
        "qa_row_ok": r["qa_row_ok"],
        "qa_note": r["qa_note"],
    } for r in all_rows])
    df.to_excel(out_dir + "/boq_rw_exampledev_items.xlsx", index=False)

    sys.stderr.write(json.dumps(qa, indent=2).encode("ascii", "replace").decode() + "\n")
    # Echo priced rows for verification.
    for r in all_rows:
        if r["unit_price"] is not None or r["total_price"] is not None:
            line = ("PRICED p%d %s | q=%s u=%s rate=%s amt=%s | %s | %s"
                    % (r["page"], r["item_ref"], r["quantity"], r["unit"],
                       r["unit_price"], r["total_price"], r["qa_row_ok"],
                       r["description"][:60]))
            sys.stderr.write(line.encode("ascii", "replace").decode() + "\n")


if __name__ == "__main__":
    main()
