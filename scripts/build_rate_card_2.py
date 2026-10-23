"""
build_rate_card_2.py -- INDEPENDENT second rate-card build + adversarial reconcile vs card #1.

Independently re-ingests the session's QA'd priced BOQs, applies an ORIGINAL
finer (CESMM-flavoured) work-category taxonomy and an original unit normaliser,
aggregates per (asset_type, work_category, unit, currency), then reconciles the
result against card #1 as a cross-check.

Written from first principles. Does NOT read scripts/build_rate_card.py.
Only reads card #1's OUTPUT xlsx (allowed) for reconciliation.
No emojis anywhere.
"""
import os, sys, re
import numpy as np
import pandas as pd

SCR = os.environ.get("FORK_SCRATCH_DIR", "/data/example/scratch")
OUT = SCR + "/rate_card_2"
os.makedirs(OUT, exist_ok=True)


def sp(s):
    return str(s).encode("ascii", "replace").decode()


# ---------------------------------------------------------------------------
# 1. numeric coercion
# ---------------------------------------------------------------------------
def num(x):
    if x is None:
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d\.\-]", "", s.replace(",", ""))
    if s in ("", "-", ".", "--"):
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 2. ORIGINAL unit normaliser (finer / explicit)
#    Currencies never converted. Weight tonnes unified. Italian mq/mc/ml handled.
# ---------------------------------------------------------------------------
def norm_unit(u):
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None
    s = str(u).strip().lower()
    s = s.replace(".", "").replace(" ", "")
    s = s.replace("²", "2").replace("³", "3")  # superscripts
    if s in ("", "nan", "none"):
        return None
    area = {"m2", "sqm", "sm", "sq", "mq", "m^2", "sqmt", "sqmtr", "squm"}
    vol = {"m3", "cum", "mc", "cbm", "m^3", "cm", "cumt"}
    length = {"m", "ml", "lm", "rm", "rmt", "lin", "linm", "linmtr", "runm", "rft", "mtr", "mt2"}
    count = {"no", "nos", "nr", "nrs", "each", "ea", "pcs", "pc", "unit", "units",
             "point", "pt", "pts", "set", "sets", "ttem"}
    lump = {"ls", "lump", "lumpsum", "lot", "sum", "item", "sundry", "allow", "prov"}
    weight = {"t", "ton", "tons", "tonne", "tonnes", "mt", "ton(s)", "metricton"}
    # order: resolve collisions deliberately.
    if s in weight:
        return "t"
    if s in area:
        return "m2"
    if s in vol:
        return "m3"
    if s in length:
        return "m"
    if s in count:
        return "no"
    if s in lump:
        return "ls"
    if s in {"ha", "hectare", "hectares"}:
        return "ha"
    if s in {"kg", "kgs", "kilo"}:
        return "kg"
    if s in {"month", "months", "mo", "mon"}:
        return "month"
    if s in {"bay", "bays"}:
        return "bay"
    if s in {"day", "days", "wk", "week", "weeks", "hr", "hour", "hours"}:
        return "time"
    if s in {"l", "ltr", "litre", "liter"}:
        return "litre"
    return s  # unknown -> keep raw token, surfaced later


# ---------------------------------------------------------------------------
# 3. ORIGINAL work-category taxonomy (finer than card #1) + rollup to card #1
#    buckets for the cross-check. First keyword match wins.
# ---------------------------------------------------------------------------
# (fine_category, [keywords])  -- order = priority (specific first)
RULES = [
    ("Fire Protection", ["fire alarm", "firefighting", "fire fighting", "sprinkler",
                          "fire hydrant", "fm200", "fm-200", "suppression", "fire pump",
                          "smoke detect", "fire ", "fire-"]),
    ("Plumbing & Sanitary", ["sanitary", "water closet", "wc ", "wash basin", "washbasin",
                             "wash hand", "lavatory", "urinal", "faucet", "mixer tap",
                             "shower", "bathtub", "bath tub", "toilet", "cistern",
                             "water heater", "geyser", "plumb", "water supply", "cold water",
                             "hot water", "floor drain"]),
    ("Drainage & Pipework", ["manhole", "sewer", "culvert", "storm", "gully", "drainage",
                             "drain ", "ductile iron pipe", "hdpe pipe", "pvc pipe",
                             "upvc", "grp pipe", "rcp", "pipe", "pipeline", "inspection chamber",
                             "catch basin", "bedding", "trench"]),
    ("Mechanical / HVAC (MEP)", ["hvac", "chiller", "cooling tower", "crac", "crah", "ahu",
                                 "fcu", "fan coil", "air handling", "ventilation", "air condition",
                                 "ductwork", "duct ", "exhaust fan", "extract fan", "supply fan",
                                 "vrf", "vrv", "split unit", "condenser", "pump ", "cold plunge",
                                 "sauna", "spa "]),
    ("Electrical (MEP)", ["transformer", "switchgear", "switch gear", "substation", "busbar",
                          "bus bar", "generator", "genset", "ups", "power distribution",
                          "distribution board", "db board", "mcc ", "lv panel", "mv panel",
                          "cable tray", "cabling", "cable", "wiring", "conduit", "earthing",
                          "lightning", "light fitting", "lighting", "luminaire", "socket",
                          "electric", "kv ", "kva", "power "]),
    ("Low Voltage / Data / Security", ["structured cabling", "data hall", "server rack",
                                       "fiber", "fibre", "cctv", "access control", "security system",
                                       "bms", "building management", "telecom", "network switch",
                                       "containment", "patch panel", "elv"]),
    ("Roofing", ["roofing", "roof sheet", "roof tile", "roof deck", "shingle", "parapet",
                 "roof ", "roof-"]),
    ("Windows, Doors & Facade", ["curtain wall", "curtainwall", "facade", "faade", "glazing",
                                 "glazed", "aluminium", "aluminum", "glass ", "window", "door",
                                 "louver", "louvre", "joinery", "wardrobe", "kitchen cabinet",
                                 "shopfront", "shutter door", "rolling shutter", "handrail",
                                 "balustrade", "railing", "ironmongery"]),
    ("Waterproofing & Insulation", ["waterproof", "tanking", "damp proof", "damp-proof", "dpc",
                                    "dpm", "membrane", "bituminous", "bitumen", "thermal insulation",
                                    "insulation", "protection board", "geotextile", "geomembrane"]),
    ("Plaster, Render & Screed", ["plaster", "render", "screed", "skim", "cement sand bed",
                                  "cement bed"]),
    ("Painting & Decoration", ["painting", "paint ", "emulsion", "enamel", "primer", "epoxy coat",
                               "protective coating", "decorat", "wall paper", "wallpaper"]),
    ("Finishes (floor/wall/ceiling)", ["tiling", "tile", "ceramic", "porcelain", "marble",
                                       "granite", "terrazzo", "vinyl", "carpet", "flooring",
                                       "false ceiling", "suspended ceiling", "gypsum board",
                                       "gypsum", "drywall", "dry wall", "cladding", "wall panel",
                                       "skirting", "cornice", "epoxy floor", "raised floor",
                                       "access floor", "ceiling", "finish"]),
    ("Masonry & Blockwork", ["blockwork", "block work", "concrete block", "hollow block",
                             "solid block", "aac block", "thermal block", "masonry", "brick",
                             "block ", "partition wall", "partition"]),
    ("Reinforcement & Rebar", ["reinforcement", "reinforcing", "rebar", "re-bar", "steel bar",
                               "high yield", "mild steel bar", "bar bending", "wire mesh",
                               "mesh reinforc", "brc"]),
    ("Formwork", ["formwork", "form work", "shuttering", "shutter", "falsework", "mould"]),
    ("Concrete (in-situ)", ["blinding", "concrete", "rcc", "pcc", "ready mix", "readymix",
                            "ready-mix", "raft", "pile cap", "grade slab", "slab", "in-situ",
                            "insitu", "in situ", "lean concrete", "screeding concrete", "grout"]),
    ("Piling & Foundations", ["bored pile", "sheet pile", "micropile", "cfa pile", "piling",
                              "pile ", "pile,", "raft foundation", "footing", "foundation",
                              "pile cap"]),
    ("Structural Steel & Metalwork", ["structural steel", "steel structure", "steel column",
                                      "steel beam", "steel truss", "metal deck", "purlin",
                                      "steelwork", "steel work", "fabricated steel", "ms plate",
                                      "steel section", "grating", "metalwork"]),
    ("Roads, Paving & Kerbs", ["asphalt", "wearing course", "binder course", "base course",
                               "sub-base", "subbase", "aggregate base", "road ", "roadway",
                               "pavement", "paver", "paving", "interlock", "kerb", "curb ",
                               "sidewalk", "footpath", "tarmac", "carriageway"]),
    ("Landscape & Softscape", ["landscap", "soft landscape", "hardscape", "planting", "plant ",
                               "trees", "tree ", "palm", "shrub", "turf", "grass", "lawn",
                               "irrigation", "garden", "pergola", "gazebo", "water feature"]),
    ("Earthworks & Excavation", ["excavat", "earthwork", "earth work", "backfill", "back fill",
                                 "filling", "cut and fill", "cut/fill", "grading", "formation",
                                 "compaction", "compact", "dewatering", "topsoil", "anti-termite",
                                 "termite", "disposal of surplus", "spoil", "embankment", "hardcore"]),
    ("Demolition & Site Clearance", ["demolit", "demolish", "breakout", "break out", "break-out",
                                     "dismantl", "removal of", "remove existing", "remove the exist",
                                     "site clearance", "site clearing", "clearance", "grubbing",
                                     "strip out", "chipping"]),
    ("Commissioning & Testing", ["commission", "testing and commission", "integrated system test",
                                 "t&c", "pre-commission", "witness test", "snagging"]),
    ("Design, Engineering & PM (soft cost)", ["design ", "detailed design", "engineering ",
                                              "consultan", "project management", "supervision",
                                              "permit", "authority fee", "statutory fee", "legal fee",
                                              "insurance", "performance bond", "contingency",
                                              "import", "customs", "logistics", "freight", "soft cost",
                                              "geotechnical", "survey", "environmental impact"]),
    ("Preliminaries & General", ["preliminar", "prelim", "mobiliz", "mobilis", "demobil",
                                 "general requirement", "site office", "hoarding", "fencing",
                                 "temporary", "site facilit", "overhead", "profit", "provisional sum",
                                 "attendance", "security guard", "welfare"]),
    ("FF&E / Fixtures & Equipment", ["furniture", "ff&e", "ffe ", "fixtures and", "appliance",
                                     "signage", "kitchen equipment", "white good", "millwork"]),
]

# fine_category -> card #1 comparable bucket (rollup)
ROLLUP = {
    "Fire Protection": "Fire Protection",
    "Plumbing & Sanitary": "Sanitary/Accessories",
    "Drainage & Pipework": "Pipework/Drainage",
    "Mechanical / HVAC (MEP)": "Mechanical/HVAC (MEP)",
    "Electrical (MEP)": "Electrical (MEP)",
    "Low Voltage / Data / Security": "Other/Uncategorized",
    "Roofing": "Other/Uncategorized",
    "Windows, Doors & Facade": "Windows/Doors/Facade",
    "Waterproofing & Insulation": "Waterproofing/Insulation",
    "Plaster, Render & Screed": "Finishes",
    "Painting & Decoration": "Finishes",
    "Finishes (floor/wall/ceiling)": "Finishes",
    "Masonry & Blockwork": "Masonry/Blockwork",
    "Reinforcement & Rebar": "Concrete",
    "Formwork": "Concrete",
    "Concrete (in-situ)": "Concrete",
    "Piling & Foundations": "Piling/Foundations",
    "Structural Steel & Metalwork": "Structural Steel",
    "Roads, Paving & Kerbs": "Roads/Paving",
    "Landscape & Softscape": "Landscape/Softscape",
    "Earthworks & Excavation": "Earthworks/Excavation",
    "Demolition & Site Clearance": "Demolition",
    "Commissioning & Testing": "Other/Uncategorized",
    "Design, Engineering & PM (soft cost)": "Preliminaries/General",
    "Preliminaries & General": "Preliminaries/General",
    "FF&E / Fixtures & Equipment": "Other/Uncategorized",
    "Other / Uncategorized": "Other/Uncategorized",
}


def classify(desc, section):
    t = (" " + str(desc) + " || " + str(section) + " ").lower()
    for cat, kws in RULES:
        for kw in kws:
            if kw in t:
                return cat
    return "Other / Uncategorized"


# ---------------------------------------------------------------------------
# 4. Ingestion. Returns list of dict rows: asset, ccy, desc, section, qty, rate, unit_raw
# ---------------------------------------------------------------------------
records = []          # unified priced items
source_counts = {}    # per-source ingested count
unknown_units = {}    # normalised token -> count for tokens we could not map


def qa_ok(v):
    """Explicit-fail drop only; NaN kept (many valid rows carry NaN qa)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return True
    s = str(v).strip().lower()
    if s in ("false", "0", "0.0", "fail", "no", "n"):
        return False
    return True


def add(asset, ccy, desc, section, qty, up, tp, qaval, src):
    if not qa_ok(qaval):
        return 0
    q = num(qty)
    upv = num(up)
    tpv = num(tp)
    rate = upv if (upv is not None and upv > 0) else (
        tpv / q if (tpv is not None and tpv > 0 and q is not None and q > 0) else np.nan)
    if not (q is not None and q > 0):
        return 0
    if not (rate is not None and rate > 0):
        return 0
    records.append(dict(asset=asset, ccy=ccy, desc=desc, section=section,
                        qty=q, rate=rate, src=src))
    source_counts[src] = source_counts.get(src, 0) + 1
    return 1


def unit_of(u):
    nu = norm_unit(u)
    if nu is None:
        nu = "ls"   # blank/NaN unit -> treat as lump-sum line
    if nu not in {"m2", "m3", "m", "no", "ls", "t", "ha", "kg", "month", "bay", "time", "litre"}:
        unknown_units[nu] = unknown_units.get(nu, 0) + 1
    return nu


# --- item-style files (already parsed to Description/Quantity/Unit/Unit Price/Total Price)
def ingest_items(path, asset, ccy, src):
    df = pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)
    cols = {c.lower().strip(): c for c in df.columns}
    dcol = cols.get("description")
    ucol = cols.get("unit")
    qcol = cols.get("quantity")
    upcol = cols.get("unit price")
    tpcol = cols.get("total price")
    scol = cols.get("section") or cols.get("bill/section") or cols.get("bill / section")
    qacol = next((cols[k] for k in cols if "qa_row" in k), None)
    n = 0
    for _, r in df.iterrows():
        got = add(asset, ccy, r.get(dcol), r.get(scol) if scol else "",
                  r.get(qcol), r.get(upcol) if upcol else np.nan,
                  r.get(tpcol) if tpcol else np.nan,
                  r.get(qacol) if qacol else np.nan, src)
        if got:
            records[-1]["unit"] = unit_of(r.get(ucol))
            n += 1
    return n


# --- raw section-sheet workbooks (kenya_dc, gcc_farmhouse)
def ingest_raw(path, sheets, asset_list, ccy, src):
    """asset_list: list of asset_types to emit each row under (gcc -> two)."""
    total = 0
    for s in sheets:
        df = pd.read_excel(path, sheet_name=s, header=None)
        hdr = None
        for i in range(min(10, len(df))):
            row = [str(x).strip().lower() for x in df.iloc[i].tolist()]
            if "quantity" in row and "unit" in row and any(str(x).startswith("rate") for x in row):
                hdr = i
                break
        if hdr is None:
            continue
        header = [str(x).strip() for x in df.iloc[hdr].tolist()]
        idx = {h.lower(): j for j, h in enumerate(header)}
        j_desc = idx.get("description")
        j_unit = idx.get("unit")
        j_qty = idx.get("quantity")
        j_rate = next((idx[h] for h in idx if h.startswith("rate")), None)
        j_amt = next((idx[h] for h in idx if h.startswith("amount")), None)
        sub = df.iloc[hdr + 1:]
        for _, r in sub.iterrows():
            desc = r.iloc[j_desc] if j_desc is not None else ""
            ds = str(desc).strip().lower()
            if ds in ("", "nan") or ds.startswith("subtotal") or ds.startswith("total") \
               or ds.startswith("bill ") or "subtotal" in ds:
                continue
            for asset in asset_list:
                got = add(asset, ccy, desc, s,
                          r.iloc[j_qty] if j_qty is not None else np.nan,
                          r.iloc[j_rate] if j_rate is not None else np.nan,
                          r.iloc[j_amt] if j_amt is not None else np.nan,
                          np.nan, src if len(asset_list) == 1 else src + " (" + asset + ")")
                if got:
                    records[-1]["unit"] = unit_of(r.iloc[j_unit] if j_unit is not None else None)
                    total += 1
    return total


# ---- run ingestion over the declared source set ----
INFRA = "Infrastructure"
BLD = "Buildings/Towers"
VIL = "Villas"
DC = "Data Center"

ingest_items(SCR + "/waste_water_boq_items.xlsx", INFRA, "SAR", "waste_water")
ingest_items(SCR + "/infra1_demolition_boq_items.xlsx", INFRA, "SAR", "demolition")
ingest_items(SCR + "/demo_contract_demolition_boq_items.xlsx", INFRA, "SAR", "demo_contract_demolition")
ingest_items(SCR + "/boq_batch/tower1_items.csv", BLD, "AED", "tower1")
ingest_items(SCR + "/wetransfer_boq/BOQ_xlsx_items.csv", BLD, "AED", "wt_BOQ_xlsx")
ingest_items(SCR + "/wetransfer_boq/exel_SectionB_Sitework_items.csv", BLD, "AED", "wt_SectionB_Sitework")
ingest_items(SCR + "/wetransfer_boq/exel_SectionC_Concrete_items.csv", BLD, "AED", "wt_SectionC_Concrete")
ingest_items(SCR + "/wetransfer_boq/exel_SectionD_Masonry_items.csv", BLD, "AED", "wt_SectionD_Masonry")
for pf in ["Board_walk", "French_Village", "India_Gate", "Peninsula_Boardwalk"]:
    ingest_items(SCR + "/wetransfer_boq/plaster_%s_items.csv" % pf, BLD, "AED", "wt_plaster_" + pf)

kenya_sheets = [s for s in pd.ExcelFile(SCR + "/boq_batch/kenya_dc.xlsx").sheet_names if s[0].isdigit()]
ingest_raw(SCR + "/boq_batch/kenya_dc.xlsx", kenya_sheets, [DC], "USD", "kenya_dc")
ingest_raw(SCR + "/boq_batch/gcc_farmhouse.xlsx", ["BOQ_Detail"], [VIL, BLD], "SAR", "gcc_farmhouse")

df = pd.DataFrame(records)
df["fine_cat"] = df.apply(lambda r: classify(r["desc"], r["section"]), axis=1)
df["rollup_cat"] = df["fine_cat"].map(ROLLUP)

print("TOTAL priced items ingested:", len(df))
print("Distinct source ledger:")
for k in sorted(source_counts):
    print("  %-32s %d" % (sp(k), source_counts[k]))
if unknown_units:
    print("UNKNOWN units (kept, not in canonical set):", {sp(k): v for k, v in unknown_units.items()})


# ---------------------------------------------------------------------------
# 5. Aggregation helpers
# ---------------------------------------------------------------------------
def trimmed_mean(a):
    a = np.sort(np.asarray(a, dtype=float))
    if len(a) < 3:
        return float(np.mean(a))
    k = int(np.floor(len(a) * 0.10))
    core = a[k:len(a) - k] if k > 0 else a
    if len(core) == 0:
        core = a
    return float(np.mean(core))


def aggregate(frame, catcol):
    rows = []
    for (asset, cat, unit, ccy), g in frame.groupby([ "asset", catcol, "unit", "ccy"]):
        r = g["rate"].values
        if len(r) < 2:
            continue
        rows.append(dict(asset=asset, category=cat, unit=unit, ccy=ccy, n=len(r),
                         median=round(float(np.median(r)), 2),
                         mean=round(float(np.mean(r)), 2),
                         trimmed_mean=round(trimmed_mean(r), 2),
                         p25=round(float(np.percentile(r, 25)), 2),
                         p75=round(float(np.percentile(r, 75)), 2),
                         min=round(float(np.min(r)), 2),
                         max=round(float(np.max(r)), 2)))
    out = pd.DataFrame(rows).sort_values(["asset", "category", "unit", "ccy"]).reset_index(drop=True)
    return out


fine = aggregate(df, "fine_cat")       # card #2 primary (finer taxonomy)
roll = aggregate(df, "rollup_cat")     # rollup for reconciliation
print("Card #2 fine cells (n>=2):", len(fine))
print("Card #2 rollup cells (n>=2):", len(roll))

# also a coarse (asset,unit,ccy) view for range-level comparison
coarse_rows = []
for (asset, unit, ccy), g in df.groupby(["asset", "unit", "ccy"]):
    r = g["rate"].values
    coarse_rows.append(dict(asset=asset, unit=unit, ccy=ccy, n=len(r),
                            median=round(float(np.median(r)), 2),
                            min=round(float(np.min(r)), 2),
                            max=round(float(np.max(r)), 2)))
coarse = pd.DataFrame(coarse_rows).sort_values(["asset", "unit", "ccy"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Load card #1 and normalise its units with the SAME normaliser
# ---------------------------------------------------------------------------
c1 = pd.read_excel(SCR + "/rate_card/rate_card.xlsx")
c1 = c1.rename(columns={"cat": "category"})
c1["unit_norm"] = c1["unit"].map(lambda u: norm_unit(u) if pd.notna(u) else None)


# ---------------------------------------------------------------------------
# 7. Reconciliation: join on (asset, category, unit_norm, ccy)
# ---------------------------------------------------------------------------
roll2 = roll.copy()
roll2["unit_norm"] = roll2["unit"]  # already normalised
key = ["asset", "category", "unit_norm", "ccy"]

m = pd.merge(
    c1[key + ["n", "median", "min", "max"]].rename(
        columns={"n": "n_c1", "median": "med_c1", "min": "min_c1", "max": "max_c1"}),
    roll2[key + ["n", "median", "min", "max"]].rename(
        columns={"n": "n_c2", "median": "med_c2", "min": "min_c2", "max": "max_c2"}),
    on=key, how="outer", indicator=True)

both = m[m["_merge"] == "both"].copy()
both["pct_diff"] = (both["med_c2"] - both["med_c1"]) / both["med_c1"] * 100.0
both["abs_pct"] = both["pct_diff"].abs()
both["min_n"] = both[["n_c1", "n_c2"]].min(axis=1)
both["confidence"] = np.where(both["min_n"] >= 5, "solid", "thin(n<5)")
both = both.sort_values("abs_pct", ascending=False).reset_index(drop=True)

diverge = both[both["abs_pct"] > 25].copy()
only_c1 = m[m["_merge"] == "left_only"].copy()
only_c2 = m[m["_merge"] == "right_only"].copy()

print("Comparable (matched) cells:", len(both))
print("Divergences >25%:", len(diverge), " of which solid(min_n>=5):",
      int((diverge["min_n"] >= 5).sum()))
print("Cells only in card #1:", len(only_c1), " only in card #2:", len(only_c2))

# Other/Uncategorized reclamation metric
c1_other = int(c1[c1["category"] == "Other/Uncategorized"]["n"].sum())
c2_other = int(roll[roll["category"] == "Other/Uncategorized"]["n"].sum())
print("Card #1 Other/Uncategorized items (cells n>=2):", c1_other)
print("Card #2 Other/Uncategorized items (rollup cells n>=2):", c2_other)


# ---------------------------------------------------------------------------
# 8. Write xlsx
# ---------------------------------------------------------------------------
xls_path = OUT + "/rate_card_2.xlsx"
with pd.ExcelWriter(xls_path, engine="openpyxl") as xw:
    fine.to_excel(xw, sheet_name="card2_fine", index=False)
    roll.to_excel(xw, sheet_name="card2_rollup", index=False)
    coarse.to_excel(xw, sheet_name="card2_coarse_by_unit", index=False)
    both.drop(columns=["_merge"]).to_excel(xw, sheet_name="reconcile_matched", index=False)
    diverge.drop(columns=["_merge"]).to_excel(xw, sheet_name="reconcile_diverge_gt25", index=False)
    only_c1.drop(columns=["n_c2", "med_c2", "min_c2", "max_c2", "_merge"]).to_excel(
        xw, sheet_name="only_in_card1", index=False)
    only_c2.drop(columns=["n_c1", "med_c1", "min_c1", "max_c1", "_merge"]).to_excel(
        xw, sheet_name="only_in_card2", index=False)
    pd.DataFrame([{"source": k, "n": v} for k, v in sorted(source_counts.items())]).to_excel(
        xw, sheet_name="source_ledger", index=False)
print("wrote", sp(xls_path))

# stash frames for the reporter step
df.to_pickle(OUT + "/_records.pkl")
fine.to_pickle(OUT + "/_fine.pkl")
roll.to_pickle(OUT + "/_roll.pkl")
both.to_pickle(OUT + "/_both.pkl")
diverge.to_pickle(OUT + "/_diverge.pkl")
only_c1.to_pickle(OUT + "/_only_c1.pkl")
only_c2.to_pickle(OUT + "/_only_c2.pkl")
pd.DataFrame([{"source": k, "n": v} for k, v in sorted(source_counts.items())]).to_pickle(OUT + "/_ledger.pkl")
import json
meta = dict(total=len(df), fine_cells=len(fine), roll_cells=len(roll),
            matched=len(both), diverge=len(diverge),
            diverge_solid=int((diverge["min_n"] >= 5).sum()),
            only_c1=len(only_c1), only_c2=len(only_c2),
            c1_other=c1_other, c2_other=c2_other,
            unknown_units={sp(k): v for k, v in unknown_units.items()})
open(OUT + "/_meta.json", "w").write(json.dumps(meta, indent=2))
print("META", json.dumps(meta))
