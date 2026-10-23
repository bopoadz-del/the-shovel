"""Price an UNPRICED BOQ from the typed rate-card (scripts/build_rate_card.py output).

For each unpriced line item: classify -> look up the rate-card median rate for
(asset_type, work_category, unit, currency), apply amount = qty * rate. Falls
back category-wide, then asset-wide, and flags anything with no comparable rate
instead of inventing one. Every priced line records its rate_source and the
sample size (n) behind the rate, so confidence is visible - this is the
platform's "price an unpriced BOQ" capability, kept honest.

Usage: python scripts/price_boq.py <unpriced.csv|xlsx> <asset_type> <currency> [out.xlsx]
"""
import os, re, sys, statistics as st
import pandas as pd

SCR = os.environ.get("FORK_SCRATCH_DIR", "/data/example/scratch")
CARD = os.path.join(SCR, "rate_card", "rate_card.xlsx")

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
    m = {"sqm": "m2", "sq.m": "m2", "m^2": "m2", "mq": "m2", "mq.": "m2", "mc": "m3",
         "cum": "m3", "cu.m": "m3", "m^3": "m3",
         "l.m": "m", "lm": "m", "ml": "m", "ml.": "m", "rm": "m", "r.m": "m", "lin.m": "m", "linearmeter": "m",
         "nos": "no", "nr": "no", "no.": "no", "each": "no", "ea": "no", "pcs": "no", "pce": "no", "pc": "no",
         "l.s": "ls", "lumpsum": "ls", "sum": "ls", "item": "ls", "lot": "ls"}
    return m.get(u, u) or "?"

def num(x):
    if x is None: return None
    s = re.sub(r"[,\sA-Za-z$]", "", str(x))
    try:
        v = float(s); return v if v == v else None
    except Exception:
        return None

def categorize(desc):
    d = str(desc).lower()
    for name, pat in CATS:
        if re.search(pat, d): return name
    return "Other/Uncategorized"

def build_lookup(asset, ccy):
    c = pd.read_excel(CARD)
    c = c[(c["asset"] == asset) & (c["ccy"] == ccy)]
    exact, bycat, byasset = {}, {}, []
    for _, r in c.iterrows():
        exact[(r["cat"], r["unit"])] = (r["median"], int(r["n"]))
        bycat.setdefault(r["cat"], []).append((r["median"], int(r["n"])))
        byasset.append((r["median"], int(r["n"])))
    return exact, bycat, byasset

def lookup(cat, unit, exact, bycat, byasset):
    if (cat, unit) in exact:
        m, n = exact[(cat, unit)]; return m, n, "exact (cat+unit)"
    if cat in bycat:
        vals = bycat[cat]; m = st.median([v for v, _ in vals]); n = sum(n for _, n in vals)
        return round(m, 2), n, "fallback (category median)"
    if byasset:
        m = st.median([v for v, _ in byasset]); return round(m, 2), sum(n for _, n in byasset), "weak (asset median)"
    return None, 0, "NO RATE"

def main():
    src, asset, ccy = sys.argv[1], sys.argv[2], sys.argv[3]
    out = sys.argv[4] if len(sys.argv) > 4 else os.path.join(SCR, "rate_card", "priced_output.xlsx")
    df = pd.read_csv(src) if src.endswith(".csv") else pd.read_excel(src)
    dcol = next(c for c in df.columns if "desc" in str(c).lower())
    qcol = next(c for c in df.columns if str(c).lower().startswith(("quantity", "qty")))
    ucol = next((c for c in df.columns if str(c).lower() in ("unit", "uom")), None)
    exact, bycat, byasset = build_lookup(asset, ccy)
    rows = []; total = 0.0; priced = 0; exact_n = 0
    for _, r in df.iterrows():
        desc = r.get(dcol); qty = num(r.get(qcol))
        unit = norm_unit(r.get(ucol)) if ucol else "?"
        if qty is None or qty <= 0 or not str(desc).strip() or str(desc).strip().lower() == "nan":
            continue
        cat = categorize(desc)
        rate, n, src_tag = lookup(cat, unit, exact, bycat, byasset)
        if rate is None:
            rows.append([desc, unit, qty, cat, None, None, src_tag, 0]); continue
        amt = round(qty * rate, 2); total += amt; priced += 1
        if src_tag.startswith("exact"): exact_n += 1
        rows.append([desc, unit, qty, cat, rate, amt, src_tag, n])
    out_df = pd.DataFrame(rows, columns=["Description", "Unit", "Quantity", "Work Category",
                                         f"Unit Price ({ccy}, EST)", f"Total Price ({ccy}, EST)", "Rate Source", "n behind rate"])
    out_df.to_excel(out, index=False)
    def safe(s): return str(s).encode("ascii", "replace").decode()
    print(f"Priced {priced}/{len(out_df)} line items from the {asset}/{ccy} rate-card "
          f"({exact_n} exact cat+unit match, {priced-exact_n} fallback).")
    print(f"ESTIMATED TOTAL: {ccy} {total:,.0f}")
    unmatched = out_df[out_df['Rate Source'] == 'NO RATE']
    print(f"No-rate (flagged, not priced): {len(unmatched)}")
    print(f"saved -> {out}")
    print("\n=== sample priced lines ===")
    print(safe(out_df.head(12).to_string(index=False, max_colwidth=34)))

if __name__ == "__main__":
    main()
