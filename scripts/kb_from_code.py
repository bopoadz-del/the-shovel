#!/usr/bin/env python3
"""Regenerate the knowledge documents whose content lives only in code.

Writes docs/knowledge/aci_318_metric_design_values.md (the construction
calculator's own ACI 318-19 SI outputs over a grid of standard inputs) and
docs/knowledge/boq_rate_card_reference.md (app/lib/data/boq_rate_card.json).
Nothing is typed by hand: rerun after a calculator or rate-card change.

    PYTHONPATH=. python scripts/kb_from_code.py
"""
import json
from pathlib import Path

from app.lib import construction_formulas as cf

C = cf.CALCULATORS
OUT = Path("docs/knowledge")


def call(name, **kw):
    r = C[name](**kw)
    return r.get("result", r) if isinstance(r, dict) else r


def num(d, *keys):
    for k in keys:
        if isinstance(d, dict) and d.get(k) is not None:
            return d[k]
    for k, v in (d or {}).items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


lines = ["# ACI 318-19 design values (SI / metric) — as used in KSA practice", "",
         "Metric (SI) design values computed by the platform's construction calculator from ",
         "ACI 318-19 and ACI 209R/308. ACI 318 in SI units is the basis the Saudi Building ",
         "Code concrete provisions (SBC 304) adapt; confirm the SBC edition in force for a ",
         "permit. Every value in these tables is the calculator's own output for the stated inputs.", ""]

# Modulus of elasticity and rupture
lines += ["## Modulus of elasticity and modulus of rupture (normal-weight concrete)", "",
          "Ec = 4700·√f'c (ACI 318-19 Eq. 19.2.2.1b, SI); fr = 0.62·√f'c (Eq. 19.2.3.1).", "",
          "| f'c (MPa, cylinder) | Ec (MPa) | fr (MPa) |", "|---|---|---|"]
for fc in (20, 25, 28, 30, 32, 35, 40, 45, 50, 60):
    ec = call("modulus_of_elasticity_concrete", fck_n_mm2=fc, code="aci")
    fr = call("modulus_of_rupture", fck_n_mm2=fc, code="aci")
    e = num(ec, "ec_mpa", "Ec_mpa", "modulus_mpa", "ec_n_mm2")
    r = num(fr, "fr_mpa", "modulus_of_rupture_mpa", "modulus_of_rupture_n_mm2")
    lines.append(f"| {fc} | {e:,.0f} | {r:.2f} |")
lines.append("")

# Minimum slab thickness
lines += ["## Minimum thickness of non-prestressed one-way solid slabs (ACI 318-19 Table 7.3.1.1)", "",
          "For slabs not supporting or attached to partitions likely to be damaged by large ",
          "deflections. fy = 420 MPa (for other fy multiply by 0.4 + fy/700).", "",
          "| Span (m) | Simply supported (mm) | One end continuous (mm) | Both ends continuous (mm) | Cantilever (mm) |",
          "|---|---|---|---|---|"]
supports = ("simply_supported", "one_end_continuous", "both_ends_continuous", "cantilever")
for span in (3.0, 3.6, 4.0, 4.8, 5.0, 6.0, 7.2, 8.0):
    row = []
    for s in supports:
        try:
            v = num(call("slab_thickness_min", span_mm=span * 1000, support_condition=s, code="aci"),
                    "min_thickness_mm")
            row.append(f"{v:.0f}")
        except Exception:  # noqa: BLE001 - a support the calculator does not model
            row.append("—")
    lines.append(f"| {span:g} | " + " | ".join(row) + " |")
lines.append("")

# Lap lengths
lines += ["## Tension lap splice length, Class B (ACI 318-19 §25.4.2 / §25.5.2)", "",
          "fy = 420 MPa, confinement term (cb+Ktr)/db = 1.5, uncoated bottom bars. The bar-size factor "
          "psi_s is taken as 1.0 for all sizes, so values for bars of 19 mm and smaller are conservative "
          "(ACI permits 0.8).", "",
          "| Bar Ø (mm) | f'c 25 MPa (mm) | f'c 30 MPa (mm) | f'c 35 MPa (mm) | f'c 40 MPa (mm) |",
          "|---|---|---|---|---|"]
for db in (10, 12, 16, 20, 25, 32):
    row = []
    for fc in (25, 30, 35, 40):
        r = call("rebar_lap_length", bar_diameter_mm=db, fy_mpa=420, fc_mpa=fc, code="aci")
        row.append(f"{num(r, 'lap_length_mm', 'class_b_lap_mm', 'lap_mm'):.0f}")
    lines.append(f"| {db} | " + " | ".join(row) + " |")
lines.append("")

# Beam shear (concrete contribution)
lines += ["## Concrete shear strength of beams, φVc (ACI 318-19 §22.5, φ = 0.75)", "",
          "Simplified Vc = 0.17·λ·√f'c·bw·d (members with at least minimum shear reinforcement); "
          "longitudinal steel ratio ρl = 1%.", "",
          "| b × d (mm) | f'c 25 MPa (kN) | f'c 30 MPa (kN) | f'c 35 MPa (kN) | f'c 40 MPa (kN) |",
          "|---|---|---|---|---|"]
for b, d in ((250, 450), (300, 500), (300, 600), (400, 700), (500, 900)):
    row = []
    for fc in (25, 30, 35, 40):
        r = call("rc_beam_shear_capacity", width_mm=b, eff_depth_mm=d, fc_mpa=fc, code="aci")
        row.append(f"{num(r, 'phi_vc_kn', 'design_shear_kn', 'vc_kn'):.1f}")
    lines.append(f"| {b} × {d} | " + " | ".join(row) + " |")
lines.append("")

# Shrinkage and curing
lines += ["## Drying shrinkage and strength gain (ACI 209R)", "",
          "Hyperbolic shrinkage, ultimate 780 microstrain, time constant 35 days.", "",
          "| Age (days) | Shrinkage (microstrain) |", "|---|---|"]
for t in (7, 14, 28, 56, 90, 180, 365):
    r = call("concrete_shrinkage", time_days=t)
    lines.append(f"| {t} | {num(r, 'shrinkage_microstrain', 'strain_microstrain'):.0f} |")
lines += ["", "| Target fraction of 28-day strength | Curing time (days) |", "|---|---|"]
for frac in (0.5, 0.6, 0.7, 0.75, 0.9):
    r = call("concrete_curing_time", target_strength_fraction=frac)
    lines.append(f"| {frac:.0%} | {num(r, 'days', 'curing_days', 'time_days'):.1f} |")
lines.append("")
(OUT / "aci_318_metric_design_values.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

# BOQ rate card
d = json.load(open("app/lib/data/boq_rate_card.json", encoding="utf-8"))
rl = ["# BOQ rate card — observed unit rates by asset type (median and interquartile range)", "",
      "The platform's rate card: observed unit rates by trade category, asset type and currency. ",
      f"Source: {d['_meta'].get('source', '')}.", "",
      "n = number of observed BOQ lines; p25/p75 = interquartile range. Rates are indicative ",
      "for estimating; always prefer a project's own priced BOQ.", ""]
for asset, by_cur in d["cards"].items():
    for cur, rows in by_cur.items():
        rl += [f"## {asset} — {cur}", "", "| Category | Unit | n | Median | p25 | p75 | Source |",
               "|---|---|---|---|---|---|---|"]
        for r in rows:
            rl.append(f"| {r.get('cat', '')} | {r.get('unit', '')} | {r.get('n', '')} | "
                      f"{r.get('median', '')} | {r.get('p25', '')} | {r.get('p75', '')} | {r.get('source', '')} |")
        rl.append("")
(OUT / "boq_rate_card_reference.md").write_text("\n".join(rl) + "\n", encoding="utf-8")
print("written")
