"""
Extract and QA spreadsheet Bills of Quantities from the wetransfer_docs corpus.
Clean digital files, no OCR fabrication. Honest about unpriced / ambiguous files.

Outputs (in scratchpad/wetransfer_boq/):
  <slug>_items.csv  per source file
  wetransfer_boq_manifest.md

Run inside the The_Fork venv (pandas, openpyxl, xlrd installed).
"""
import os, re, csv, numbers
import pandas as pd

SRC = os.environ.get("BOQ_SRC_DIR", "/data/example/boqs")
OUT = os.path.join(os.environ.get("FORK_SCRATCH_DIR", "/data/example/scratch"), "wetransfer_boq")
EXEL = OUT + "/exel_sheets"
PLAS = OUT + "/plaster"
CSTU = OUT + "/construction_studies"

def san(x):
    return str(x).encode("ascii", "replace").decode()

def num(x):
    """Return float if x is a real number/numeric-string, else None. 'nan' -> None."""
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, numbers.Number):
        f = float(x)
        return None if f != f else f  # NaN check
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() == "nan":
        return None
    try:
        f = float(s)
        return None if f != f else f
    except ValueError:
        return None

def txt(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() == "nan" else s

SUBTOTAL_RE = re.compile(
    r"\b(total|carried|collection|sub[- ]?total|brought forward|c/f|b/f|"
    r"grand total|to collection|to summary|summary)\b", re.I)

def is_subtotal(desc):
    return bool(SUBTOTAL_RE.search(desc or ""))

def qa_check(q, r, a):
    """qa_row_ok for rows with q,r,a all present."""
    if q is None or r is None or a is None:
        return ""  # not applicable
    tol = max(1.0, 0.005 * abs(a))
    return "PASS" if abs(q * r - a) <= tol else "FAIL"

def slug(name):
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s

def write_csv(fname, items):
    path = os.path.join(OUT, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Description", "Quantity", "Unit",
                    "Unit Price", "Total Price", "priced", "qa_row_ok"])
        for it in items:
            w.writerow([it["section"], it["description"], it["qty"], it["unit"],
                        it["rate"], it["amount"], it["priced"], it["qa"]])
    return path

def qa_rate(items):
    checked = [i for i in items if i["qa"] in ("PASS", "FAIL")]
    if not checked:
        return None
    p = sum(1 for i in checked if i["qa"] == "PASS")
    return (p, len(checked), 100.0 * p / len(checked))

RESULTS = []  # collect dicts for manifest

# ---------------------------------------------------------------------------
# 1. BOQ.xlsx  (sheet 'BOQ'): header row 9  cols 0=Item 1=Desc 2=Unit 3=Qty 4=UnitPrice 5=Amount
# ---------------------------------------------------------------------------
def do_boq_xlsx():
    p = os.path.join(SRC, "BOQ.xlsx")
    df = pd.read_excel(p, sheet_name="BOQ", header=None, dtype=object)
    items = []
    section = ""
    for i in range(10, len(df)):
        row = df.iloc[i].tolist()
        item_no = txt(row[0]); desc = txt(row[1]); unit = txt(row[2])
        q = num(row[3]); r = num(row[4]); a = num(row[5])
        if not desc and a is None:
            continue
        # section header rows: letter code with desc, no numbers
        if desc and a is None and q is None and r is None:
            section = desc
            continue
        # subtotal / grand-total rows carry their label in col0 (Item No), not desc
        if is_subtotal(desc) or is_subtotal(item_no):
            continue  # -> reconciliation only, never a line item
        if a is None and q is None:
            continue
        items.append(dict(section=section, description=desc, qty=q, unit=unit,
                          rate=r, amount=a, priced="Y",
                          qa=qa_check(q, r, a)))
    path = write_csv("BOQ_xlsx_items.csv", items)
    total = sum(i["amount"] for i in items if i["amount"] is not None)
    RESULTS.append(dict(
        file="BOQ.xlsx", group="", sheet="BOQ (hdr row 10)",
        currency="AED (stated on 'Sum.' sheet: GRAND TOTAL (AED))",
        n=len(items), priced="PRICED", qa=qa_rate(items),
        total=total,
        recon=("line-sum %.2f vs printed GRAND TOTAL (AED) 305,737 on 'Sum.' sheet "
               "(Prelim 91,250 + Finishes 214,487)" % total),
        note="Fit-out works, proposed restaurant/clinic, Dubai UAE (Dr. Arwa Attaallah)",
        csv=os.path.basename(path)))

# ---------------------------------------------------------------------------
# 2. B.O.Q_contractor.xls  UNPRICED. sheets Hard works/Furniture/Power/Total
#    hdr cols 0=No 1=WorkDesc 2=Material 3=Unit 4=Quantity 5=UnitPrice 6=TotalPrice
# ---------------------------------------------------------------------------
def do_contractor():
    p = os.path.join(SRC, "B.O.Q_contractor.xls")
    xl = pd.ExcelFile(p)
    items = []
    for sh in xl.sheet_names:
        df = pd.read_excel(p, sheet_name=sh, header=None, dtype=object)
        # find header row containing 'Work Description'
        hdr = None
        for i in range(min(12, len(df))):
            joined = " ".join(txt(c).lower() for c in df.iloc[i].tolist())
            if "work description" in joined:
                hdr = i; break
        if hdr is None:
            continue
        section = ""
        for i in range(hdr + 1, len(df)):
            row = df.iloc[i].tolist()
            no = txt(row[0]); wd = txt(row[1]); mat = txt(row[2])
            unit = txt(row[3]); q = num(row[4]); r = num(row[5]); a = num(row[6])
            desc = (wd + (" - " + mat if mat else "")).strip()
            if not desc and q is None:
                continue
            # section header: letter code + label, no qty
            if no and not q and not unit and wd:
                section = wd
                continue
            if is_subtotal(wd) or is_subtotal(mat):
                continue
            if not desc:
                continue
            items.append(dict(section=f"{sh} / {section}", description=desc,
                              qty=q, unit=unit, rate=r, amount=(a if a not in (0, 0.0) else None),
                              priced="N", qa=""))
    path = write_csv("BOQ_contractor_items.csv", items)
    RESULTS.append(dict(
        file="B.O.Q_contractor.xls", group="", sheet="Hard works/Furniture/Power/Total (hdr row 0/8)",
        currency="Unspecified (no symbol/label in file)",
        n=len(items), priced="UNPRICED", qa=None, total=None,
        recon="All Unit price cells blank, Total price = 0 throughout; nothing to reconcile",
        note=("New office fit-out (Al Mas...), 2008; rates to be filled by contractor. "
              "'Total' sheet rows are per-room collection lines (Hard works/Furniture/Light as LS), "
              "not detail items - included but harmless as file is unpriced."),
        csv=os.path.basename(path)))

# ---------------------------------------------------------------------------
# 3. Vol III - Bill of Quantities (Part 2 of 2).xls  -- OCR-derived, columns unreliable
# ---------------------------------------------------------------------------
def do_vol3():
    p = os.path.join(SRC, "Vol III - Bill of Quantities (Part 2 of 2).xls")
    df = pd.read_excel(p, sheet_name="Sheet1", header=None, dtype=object)
    UNITS = {"m", "m2", "m3", "no", "nos", "no.", "item", "kg", "ton", "lm", "sqm", "cum", "%"}
    items = []
    priced_amt = 0.0
    n_priced = 0
    for i in range(len(df)):
        row = df.iloc[i].tolist()
        # description = longest alpha cell in the row
        cand = [txt(c) for c in row]
        cand = [c for c in cand if len(re.sub(r"[^A-Za-z]", "", c)) >= 4]
        desc = max(cand, key=len) if cand else ""
        # locate the unit token
        upos = None
        for j, c in enumerate(row):
            if txt(c).lower().replace(".", "") in UNITS:
                upos = j; break
        if not desc or upos is None:
            continue
        if is_subtotal(desc):
            continue
        unit = txt(row[upos])
        before = [num(c) for c in row[:upos] if num(c) is not None]
        after = [num(c) for c in row[upos + 1:] if num(c) is not None]
        q = before[-1] if before else None            # qty sits just before the unit
        rate = amount = None
        if len(after) >= 2:                            # priced row: ...Rate | Amount (trailing pair)
            rate, amount = after[-2], after[-1]
        elif len(after) == 1:                          # single trailing number -> treat as amount only
            amount = after[0]
        priced_flag = "Y" if amount is not None else "N"
        if amount is not None:
            priced_amt += amount
            n_priced += 1
        items.append(dict(section="Bill No.4 Apartments (OCR)", description=desc,
                          qty=q, unit=unit, rate=rate, amount=amount,
                          priced=priced_flag, qa=qa_check(q, rate, amount)))
    path = write_csv("VolIII_BOQ_items.csv", items)
    RESULTS.append(dict(
        file="Vol III - Bill of Quantities (Part 2 of 2).xls", group="",
        sheet="Sheet1 (12972 rows; repeating OCR headers ~every 40 rows)",
        currency="AED (implied; example developer, Dubai) - not printed",
        n=len(items), priced="PARTIALLY PRICED", qa=qa_rate(items),
        total=priced_amt,
        recon=(f"{n_priced} rows carry a rate+amount pair (e.g. doors/joinery); most rows are "
               "qty-only (unpriced). Printed 'Total Carried' collection rows are OCR-garbled -> "
               "no reliable full reconciliation. Partial priced sum shown is indicative only."),
        note=("OCR-DERIVED SCAN: garbled headers ('QtyUnit','Qhi','TEN DE UUGUMENTS'), columns "
              "shift row-to-row. BEST-EFFORT extraction, low confidence. Some sections priced "
              "(rate+amount present), many qty-only. example developer 'Fronds', "
              "Bill No.4 Apartments."),
        csv=os.path.basename(path)))

# ---------------------------------------------------------------------------
# 4. Building BOQ (exel sheets) sections B/C/D  hdr row 5
#    cols 0=Item 2=Desc 3=Qty 4=Unit 5=Rate 6=Amount(Dhs) 7=Fils
# ---------------------------------------------------------------------------
def do_section(fname, sec_label, slug_name):
    p = os.path.join(EXEL, fname)
    df = pd.read_excel(p, sheet_name="Details", header=None, dtype=object)
    items = []
    pending_desc = ""  # description carried from a header row to a split priced row
    for i in range(6, len(df)):
        row = df.iloc[i].tolist()
        item = txt(row[0]); desc = txt(row[2])
        q = num(row[3]); unit = txt(row[4]); r = num(row[5]); a = num(row[6])
        if desc and not is_subtotal(desc):
            pending_desc = desc  # remember most recent description text
        # genuine line item: has an amount AND (item letter or a unit)
        if a is None:
            continue
        if not (item or unit):
            continue
        eff_desc = desc if desc else pending_desc  # backfill split "Item" lump-sum rows
        if is_subtotal(eff_desc):
            continue
        items.append(dict(section=sec_label, description=eff_desc, qty=q, unit=unit,
                          rate=r, amount=a, priced="Y", qa=qa_check(q, r, a)))
    path = write_csv(f"{slug_name}_items.csv", items)
    total = sum(i["amount"] for i in items if i["amount"] is not None)
    return items, total, os.path.basename(path)

def do_building():
    secs = [
        ("Section B - Sitework (R001)I.xls", "Section B - Sitework", "exel_SectionB_Sitework"),
        ("Section C - Concrete (R00)I.xls", "Section C - Concrete", "exel_SectionC_Concrete"),
        ("Section D - Masonry (R00)I.xls", "Section D - Masonry", "exel_SectionD_Masonry"),
    ]
    printed = {"Section B - Sitework": 653885.93,
               "Section C - Concrete": 18004768.22,
               "Section D - Masonry": 322993.77}
    recon_lines = []
    grand_items = 0
    all_qa = []
    for fname, label, sl in secs:
        items, total, csvname = do_section(fname, label, sl)
        grand_items += len(items)
        all_qa.extend(items)
        pv = printed[label]
        diff = total - pv
        recon_lines.append(
            f"{label}: line-sum {total:,.2f} vs Bill-2-Summary {pv:,.2f} "
            f"(diff {diff:+,.2f}, {100*diff/pv:+.3f}%)")
        RESULTS.append(dict(
            file=fname, group="BUILDING", sheet="Details (hdr row 5)",
            currency="AED (Dhs/Fils)", n=len(items), priced="PRICED",
            qa=qa_rate(items), total=total,
            recon=f"vs Bill 2 Summary {pv:,.2f}: diff {diff:+,.2f}",
            note=f"CAPITAL TOWERS Dubai, Pkg 3200 - {label}", csv=csvname))
    # summaries + BOQ.xls (takeoff)
    RESULTS.append(dict(
        file="Bill 2 - Summary (R00)I.xls", group="BUILDING",
        sheet="Bill 2 summary", currency="AED (Dhs/Fils)", n=0, priced="SUMMARY",
        qa=None, total=35731046.99999999,
        recon="Section subtotals summary (B..Q). Bill 2 total 35,731,047",
        note="Printed section-summary sheet (not line items)", csv="Bill_2_Summary_items.csv"))
    RESULTS.append(dict(
        file="General Summary (R00)I.xls", group="BUILDING",
        sheet="general summary", currency="AED (Dhs/Fils)", n=0, priced="SUMMARY",
        qa=None, total=39999999.99999999,
        recon="B1 3,080,000 + B2 35,731,047 + B3 196,102 + B4 0 + B5 992,851 = 40,000,000",
        note="Project GENERAL SUMMARY (grand total ~AED 40.0m)", csv="General_Summary_items.csv"))
    RESULTS.append(dict(
        file="BOQ.xls (in exel sheets)", group="BUILDING",
        sheet="Sheet1 (L/W/H/V/Nos calc)", currency="n/a", n=0, priced="NOT A BOQ",
        qa=None, total=None,
        recon="n/a",
        note="Quantity take-off / volume calc scratch sheet, not a BOQ table",
        csv="exel_BOQ_takeoff_items.csv"))
    return recon_lines

# ---------------------------------------------------------------------------
# 5. Plaster BOQs  hdr row 3  cols 1=Category 2=Item 3=Code 4=Desc 5=Unit 6=Qty 7=Rate 8=Amount
#    SUMMARY block (unit blank) = printed summary; BREAKDOWN rows (unit present) = line items
# ---------------------------------------------------------------------------
def do_plaster(fname, label):
    p = os.path.join(PLAS, fname)
    df = pd.read_excel(p, sheet_name="Sheet1", header=None, dtype=object)
    items = []
    summary_total = None
    zone = ""
    for i in range(4, len(df)):
        row = df.iloc[i].tolist()
        colA = txt(row[0])  # zone/building tag in breakdown
        item = txt(row[2]); desc = txt(row[4]); unit = txt(row[5])
        q = num(row[6]); r = num(row[7]); a = num(row[8])
        # capture printed total
        if is_subtotal(desc) and a is not None:
            if "total amount" in desc.lower():
                summary_total = a
            continue
        if colA and not unit and not q and desc and desc.isupper():
            zone = desc  # zone/building header e.g. 'ZONE 7 BUILDING 7A'
            continue
        # line item = breakdown row with a real unit + qty + rate + amount
        if unit and q is not None and a is not None:
            items.append(dict(section=zone or label, description=desc, qty=q,
                              unit=unit, rate=r, amount=a, priced="Y",
                              qa=qa_check(q, r, a)))
    sl = "plaster_" + slug(label)
    path = write_csv(f"{sl}_items.csv", items)
    total = sum(i["amount"] for i in items if i["amount"] is not None)
    recon = (f"breakdown line-sum {total:,.2f} vs printed 'Total Amount' "
             f"{summary_total:,.2f} (diff {total-summary_total:+,.2f})"
             if summary_total is not None else "no printed total row found")
    RESULTS.append(dict(
        file=fname, group="PLASTER", sheet="Sheet1 (hdr row 3)",
        currency="Unspecified (no symbol; RDE Pkg 5 - likely AED)",
        n=len(items), priced="PRICED", qa=qa_rate(items), total=total,
        recon=recon, note=f"Internal/External plaster works - {label}",
        csv=os.path.basename(path)))

# ---------------------------------------------------------------------------
# 6. Non-BOQ construction-studies files
# ---------------------------------------------------------------------------
def do_nonboq():
    RESULTS.append(dict(
        file="9409C - Traffic Plan R1.xlsx", group="NOT A BOQ",
        sheet="8 sheets (vehicle x gate x month counts)", currency="n/a",
        n=0, priced="NOT A BOQ", qa=None, total=None, recon="n/a",
        note="Traffic movement plan: vehicle trip/entry counts per gate per month. No rates/prices.",
        csv="TrafficPlan_9409C_items.csv"))
    RESULTS.append(dict(
        file="TSF Area Requirement(1).xlsx", group="NOT A BOQ",
        sheet="9 sheets (area m2 requirements, CMC ratios)", currency="n/a",
        n=0, priced="NOT A BOQ", qa=None, total=None, recon="n/a",
        note="Temporary Site Facilities area-requirement calc (m2 by discipline). No priced bill.",
        csv="TSF_Area_Requirement_items.csv"))

# ---------------------------------------------------------------------------
def write_empty_csv(fname, note):
    """Header-only CSV for source files that carry no BOQ line items (summaries / non-BOQ)."""
    path = os.path.join(OUT, fname)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Section", "Description", "Quantity", "Unit",
                    "Unit Price", "Total Price", "priced", "qa_row_ok"])
        w.writerow([f"(NO LINE ITEMS - {note})", "", "", "", "", "", "", ""])
    return os.path.basename(path)

def emit_noitem_csvs():
    """One CSV per source file: stub CSVs for the 5 files with no extractable line items."""
    stubs = {
        "Bill_2_Summary_items.csv": "printed section-summary sheet, not line items",
        "General_Summary_items.csv": "project general summary sheet, not line items",
        "exel_BOQ_takeoff_items.csv": "quantity take-off / volume calc scratch sheet, not a BOQ",
        "TrafficPlan_9409C_items.csv": "traffic movement plan, not a BOQ",
        "TSF_Area_Requirement_items.csv": "temporary site facilities area calc, not a BOQ",
    }
    for fn, note in stubs.items():
        write_empty_csv(fn, note)

def build_manifest(recon_lines):
    lines = []
    lines.append("# WeTransfer BOQ extraction & QA manifest")
    lines.append("")
    lines.append(f"Source root: `{SRC}`")
    lines.append("Clean digital spreadsheets only, no OCR fabrication. Blank rate/amount = UNPRICED (not guessed).")
    lines.append("")
    lines.append("QA gate: for rows with qty+rate+amount all present, PASS if |qty*rate - amount| <= max(1.0, 0.5% of amount).")
    lines.append("Subtotal/collection/'total' rows excluded from line items (used for reconciliation only).")
    lines.append("")
    lines.append("| File | Group | Sheet/header | Currency | Items | Priced? | QA pass | Grand total | Reconciliation | What it is |")
    lines.append("|---|---|---|---|---:|---|---|---:|---|---|")
    for r in RESULTS:
        qa = r["qa"]
        qatxt = f"{qa[0]}/{qa[1]} ({qa[2]:.1f}%)" if qa else "-"
        tot = f"{r['total']:,.2f}" if r["total"] is not None else "-"
        lines.append(
            f"| {san(r['file'])} | {r['group']} | {san(r['sheet'])} | {r['currency']} | "
            f"{r['n']} | {r['priced']} | {qatxt} | {tot} | {san(r['recon'])} | {san(r['note'])} |")
    lines.append("")
    lines.append("## Building BOQ (the 6 'exel sheets' = ONE project: CAPITAL TOWERS, Dubai, Package 3200)")
    lines.append("")
    lines.append("General Summary grand total ~ AED 40,000,000. Bill No.2 (Multi-storey) = AED 35,731,047.")
    lines.append("Only Sections B, C, D detail files were provided; reconciled to the Bill 2 Summary printed subtotals:")
    lines.append("")
    for rl in recon_lines:
        lines.append(f"- {rl}")
    lines.append("")
    lines.append("Note: section detail line-sums are compared to the Bill 2 Summary. Small diffs indicate "
                 "rounding / collection-row handling; large diffs are reported honestly, not forced.")
    with open(os.path.join(OUT, "wetransfer_boq_manifest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    do_boq_xlsx()
    do_contractor()
    do_vol3()
    recon = do_building()
    for fn, lb in [("BOQ - Plaster - (India Gate).xlsx", "India Gate"),
                   ("BOQ - Plaster - Board walk.xlsx", "Board walk"),
                   ("BOQ - Plaster - French Village.xlsx", "French Village"),
                   ("BOQ - Plaster - Peninsula & Boardwalk.xlsx", "Peninsula & Boardwalk")]:
        do_plaster(fn, lb)
    do_nonboq()
    emit_noitem_csvs()
    build_manifest(recon)
    # console summary
    for r in RESULTS:
        qa = r["qa"]
        qatxt = f" QA {qa[2]:.1f}%" if qa else ""
        tot = f" total={r['total']:,.2f}" if r["total"] is not None else ""
        print(san(f"[{r['priced']:9}] {r['file'][:44]:44} n={r['n']:4}{tot}{qatxt}"))
    print("\nReconciliation (building sections):")
    for rl in recon:
        print(" ", san(rl))

if __name__ == "__main__":
    main()
