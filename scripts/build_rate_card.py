"""Build a typed BOQ rate-card from the session's QA'd priced BOQs.

Normalizes every priced line-item artifact into one table, classifies each line
into a work category, and reports median unit rates per
(asset_type, work_category, unit, currency). Currencies are NOT converted - each
rate-card cell is per-currency, so no FX assumption is baked in.

Only REAL priced rows are used (rate>0, qty>0). Synthetic/estimated rows (the
ROSHN training price) and unpriced BOQs are deliberately excluded so the card
reflects observed market rates only.
"""
import os, re, glob, statistics as st
import pandas as pd

SCR = os.environ.get("FORK_SCRATCH_DIR", "/data/example/scratch")
OUT = os.path.join(SCR, "rate_card")
os.makedirs(OUT, exist_ok=True)

# (path, asset_type, currency, kind)  kind: 'items' = has Description/Unit/Unit Price/Total Price ; 'raw' = section-sheet xlsx needing header detect
SOURCES = [
    (f"{SCR}/waste_water_boq_items.xlsx", "Infrastructure", "SAR", "items"),
    (f"{SCR}/infra1_demolition_boq_items.xlsx", "Infrastructure", "SAR", "items"),
    (f"{SCR}/demo_contract_demolition_boq_items.xlsx", "Infrastructure", "SAR", "items"),
    (f"{SCR}/boq_batch/kenya_dc.xlsx", "Data Center", "USD", "raw"),
    (f"{SCR}/boq_batch/gcc_farmhouse.xlsx", "Villas", "SAR", "raw"),
    # medieval_modern GCC cost-BOQ doubles as the operator's general GCC building
    # rate book (full-trade SAR market rates), so expose it under Buildings too.
    (f"{SCR}/boq_batch/gcc_farmhouse.xlsx", "Buildings/Towers", "SAR", "raw"),
    (f"{SCR}/boq_batch/tower1_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/BOQ_xlsx_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/exel_SectionB_Sitework_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/exel_SectionC_Concrete_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/exel_SectionD_Masonry_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/plaster_Board_walk_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/plaster_French_Village_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/plaster_India_Gate_items.csv", "Buildings/Towers", "AED", "items"),
    (f"{SCR}/wetransfer_boq/plaster_Peninsula_Boardwalk_items.csv", "Buildings/Towers", "AED", "items"),
]

CATS = [
    ("Earthworks/Excavation", r"excavat|earthwork|backfill|\bfill\b|grading|clearing|disposal|dewater|trench"),
    ("Demolition", r"demolit|dismantl|\bbreak|removal|salvage|strip"),
    ("Piling/Foundations", r"pil(e|ing)|bored|caisson|footing|foundation"),
    ("Concrete", r"concrete|\brcc\b|blinding|screed|\bslab|grade \d|c\d{2}\b|pour"),
    ("Reinforcement", r"reinforc|rebar|steel bar|\bmesh|bending"),
    ("Formwork", r"formwork|shutter|falsework"),
    ("Structural Steel", r"structural steel|steelwork|\bfabricat|purlin|truss|bracing"),
    ("Masonry/Blockwork", r"block|brick|masonry|thermal ?stone"),
    ("Pipework/Drainage", r"\bpipe|sewer|foul|drainage|manhole|culvert|gully|gravity|rising main|\bgrp\b|vitrified"),
    ("Roads/Paving", r"asphalt|paving|\broad|kerb|pavement|interlock|sub-?base|road base"),
    ("Windows/Doors/Facade", r"window|\bdoor|aluminum|aluminium|glazing|glass|curtain wall|facade|cladding|shopfront"),
    ("Finishes", r"plaster|render|paint|\btile|floor(ing)?|ceiling|gypsum|skirting|coving|wallpaper|screed"),
    ("Waterproofing/Insulation", r"waterproof|membrane|insulat|damp proof|tanking"),
    ("Electrical (MEP)", r"cable|switchgear|transformer|busduct|electric|lighting|\bpanel|\bmv\b|\bhv\b|\blv\b|earthing|containment"),
    ("Mechanical/HVAC (MEP)", r"chiller|\bpump|hvac|cooling|\bcrah|\bcdu\b|ducting|\bahu\b|ventilat|refrigerant|mechanical"),
    ("Fire Protection", r"fire (extinguish|blanket|protec|alarm|fighting)|sprinkler|\bfe-\d"),
    ("Sanitary/Accessories", r"toilet|sanitary|\bwc\b|mirror|towel|holder|\bbin\b|basin|accessor|\btray\b"),
    ("Landscape/Softscape", r"\bplant|\btree|palm|softscape|landscape|irrigat|shrub|turf|topsoil"),
    ("Preliminaries/General", r"prelim|mobiliz|insurance|provisional|general item|attendance|supervision|overhead"),
]

def norm_unit(u):
    u = str(u).strip().lower().replace("²", "2").replace("³", "3").replace(" ", "")
    m = {"sqm": "m2", "sq.m": "m2", "m^2": "m2", "cum": "m3", "cu.m": "m3", "m^3": "m3",
         "l.m": "m", "lm": "m", "l.m.": "m", "rm": "m", "r.m": "m", "lin.m": "m", "linearmeter": "m",
         "nos": "no", "nos.": "no", "nr": "no", "no.": "no", "each": "no", "ea": "no",
         "pcs": "no", "pce": "no", "pc": "no",
         "l.s": "ls", "l.s.": "ls", "lumpsum": "ls", "sum": "ls", "item": "ls", "lot": "ls"}
    return m.get(u, u) or "?"

def num(x):
    if x is None: return None
    s = str(x).replace(",", "")
    for c in ("USD", "SAR", "AED", "OMR", "VND", "$"): s = s.replace(c, "")
    s = s.strip()
    try:
        v = float(s)
        return v if v == v else None  # drop NaN
    except Exception:
        return None

def categorize(desc):
    d = str(desc).lower()
    for name, pat in CATS:
        if re.search(pat, d):
            return name
    return "Other/Uncategorized"

def _map_cols(cols):
    idx = {}
    for i, c in enumerate(cols):
        cl = str(c).strip().lower()
        if "description" in cl and "desc" not in idx: idx["desc"] = c
        elif cl in ("unit", "uom"): idx["unit"] = c
        elif cl.startswith("quantity") or cl == "qty": idx.setdefault("qty", c)
        elif "unit price" in cl or cl.startswith("rate"): idx.setdefault("rate", c)
        elif "total price" in cl or cl.startswith("amount") or cl.startswith("total"): idx.setdefault("amt", c)
    return idx

def load_items(path):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    idx = _map_cols(df.columns)
    rows = []
    if not ({"desc", "qty"} <= set(idx)): return rows
    for _, r in df.iterrows():
        desc = r.get(idx["desc"]); qty = num(r.get(idx.get("qty")))
        rate = num(r.get(idx.get("rate"))) if "rate" in idx else None
        amt = num(r.get(idx.get("amt"))) if "amt" in idx else None
        unit = str(r.get(idx.get("unit", ""), "")).strip() if "unit" in idx else ""
        if (rate is None or rate <= 0) and amt and qty: rate = amt / qty
        if rate and rate > 0 and qty and qty > 0 and str(desc).strip() and str(desc).strip().lower() != "nan":
            rows.append((str(desc).strip(), norm_unit(unit), qty, round(rate, 2)))
    return rows

def load_raw(path):
    rows = []
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        df = pd.read_excel(path, sheet_name=sh, header=None)
        hr = None; idx = {}
        for rr in range(min(8, len(df))):
            row = [str(c).strip().lower() for c in df.iloc[rr].tolist()]
            j = " ".join(row)
            if ("quantity" in j or "qty" in j) and "rate" in j and "amount" in j:
                for i, c in enumerate(row):
                    if c.startswith("desc"): idx["d"] = i
                    elif c == "unit" or c == "uom": idx["u"] = i
                    elif c.startswith("quantity") or c == "qty": idx.setdefault("q", i)
                    elif c.startswith("rate"): idx.setdefault("r", i)
                    elif c.startswith("amount"): idx.setdefault("a", i)
                if {"d", "q", "r"} <= set(idx): hr = rr; break
        if hr is None: continue
        for rr in range(hr + 1, len(df)):
            desc = df.iloc[rr, idx["d"]]; qty = num(df.iloc[rr, idx["q"]]); rate = num(df.iloc[rr, idx["r"]])
            unit = str(df.iloc[rr, idx.get("u", idx["d"])]).strip() if "u" in idx else "?"
            if rate and rate > 0 and qty and qty > 0 and str(desc).strip() and str(desc).strip().lower() != "nan":
                rows.append((str(desc).strip(), norm_unit(unit), qty, round(rate, 2)))
    return rows

# ---- build unified table ----
records = []
print("Loaded per source:")
for path, asset, ccy, kind in SOURCES:
    if not os.path.exists(path):
        print(f"  MISSING {os.path.basename(path)}"); continue
    rows = load_raw(path) if kind == "raw" else load_items(path)
    for desc, unit, qty, rate in rows:
        records.append({"asset": asset, "ccy": ccy, "cat": categorize(desc),
                        "unit": unit.lower(), "rate": rate, "desc": desc})
    print(f"  {len(rows):4} rows  [{asset}/{ccy}]  {os.path.basename(path)}")

df = pd.DataFrame(records)
print(f"\nTOTAL priced line items in rate-card: {len(df)}")
print("\nBy asset type:", df['asset'].value_counts().to_dict())

# ---- aggregate ----
def agg(g):
    r = sorted(g["rate"].tolist())
    return pd.Series({"n": len(r), "median": round(st.median(r), 2),
                      "p25": round(r[len(r)//4], 2), "p75": round(r[(3*len(r))//4], 2),
                      "min": round(min(r), 2), "max": round(max(r), 2)})

card = (df.groupby(["asset", "cat", "unit", "ccy"]).apply(agg).reset_index())
card = card[card["n"] >= 2].sort_values(["asset", "cat", "n"], ascending=[True, True, False])
card.to_excel(os.path.join(OUT, "rate_card.xlsx"), index=False)

# ---- markdown ----
def safe(s): return str(s).encode("ascii", "replace").decode()
lines = ["# Typed BOQ Rate-Card", "",
         f"Built from {len(df)} QA'd priced line items across the session's BOQ corpus. "
         "Rates are per-currency (no FX conversion). Cells with n>=2 comparable items only. "
         "Synthetic/estimated (ROSHN training) and unpriced BOQs are excluded.", ""]
for asset in sorted(df["asset"].unique()):
    sub = card[card["asset"] == asset]
    lines.append(f"## {asset}")
    lines.append("| Work category | Unit | Ccy | n | median | p25-p75 | min-max |")
    lines.append("|---|---|---|--:|--:|--:|--:|")
    for _, r in sub.iterrows():
        lines.append(f"| {safe(r['cat'])} | {safe(r['unit'])} | {r['ccy']} | {int(r['n'])} | "
                     f"{r['median']:,.2f} | {r['p25']:,.0f}-{r['p75']:,.0f} | {r['min']:,.0f}-{r['max']:,.0f} |")
    lines.append("")
open(os.path.join(OUT, "rate_card.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"\nsaved -> rate_card/rate_card.xlsx  +  rate_card.md  ({len(card)} rate cells)")
print("\n=== sample: Infrastructure + Buildings cells ===")
print(safe(card[card['asset'].isin(['Infrastructure','Buildings/Towers'])].head(22).to_string(index=False)))
