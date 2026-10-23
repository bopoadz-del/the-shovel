"""Construction-domain constants and lookup tables.

All construction blocks that used hardcoded magic numbers should import
their defaults from here. The values are intentionally **defaults** —
callers can (and should) override via params for project-specific data.

Lookup tables (grade strengths, standard purposes) encode real domain
knowledge so the spec analyzer can answer "what is C30?" with
"30 MPa characteristic cylinder strength" instead of treating "C30"
as an opaque string.
"""

from typing import Dict, Optional


# ─────────────────────────────────────────────────────────────────────────
# DEFAULT NUMERIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────

# Quantity heuristics — used as fallbacks when real geometric data is absent.
DEFAULT_CEILING_HEIGHT_M = 3.0          # ceiling height assumption for vol = area * h
DEFAULT_SLAB_THICKNESS_M = 0.15         # concrete slab thickness fallback
DEFAULT_REBAR_RATIO_KG_PER_M3 = 120     # rebar-per-concrete-volume rule of thumb

# Cash-flow / cost defaults (when no project-specific data is supplied).
DEFAULT_OVERHEAD_PCT = 0.10              # 10% project overhead
DEFAULT_PROFIT_PCT = 0.08                # 8% profit
DEFAULT_CONTINGENCY_PCT = 0.05           # 5% contingency
DEFAULT_RETENTION_PCT = 0.05             # 5% retention typical
DEFAULT_ADVANCE_PAYMENT_PCT = 0.10       # 10% advance / mobilization
DEFAULT_PAYMENT_DELAY_DAYS = 30          # net-30 typical

# Scenario multipliers for risk-adjusted forecasts (caller can override).
DEFAULT_OPTIMISTIC_MULT = 1.10
DEFAULT_PESSIMISTIC_MULT = 0.85

# Working-day assumptions — these should ultimately come from the P6
# CALENDAR table; constants here are the fallback when calendar parsing
# isn't available.
DEFAULT_HOURS_PER_WORKDAY = 8.0
DEFAULT_WORKDAYS_PER_WEEK = 5
DEFAULT_WORKDAYS_PER_MONTH = 21
DEFAULT_WORKDAYS_PER_YEAR = 250

# Carbon defaults — embodied CO2-equivalent rules of thumb (kgCO2e per unit).
# These vary substantially by mix design; treat as ROM-only.
DEFAULT_CONCRETE_KGCO2_PER_M3 = 250
DEFAULT_STEEL_KGCO2_PER_KG = 2.3


# ─────────────────────────────────────────────────────────────────────────
# GRADE → STRENGTH LOOKUP
# ─────────────────────────────────────────────────────────────────────────

# Concrete grades — characteristic compressive strength in MPa.
# Sources: EN 206 (C-classes), BS 8500 (UK), ACI 318 (US f'c values).
# Format: grade label → {"fck_mpa": <cylinder>, "fcu_mpa": <cube>, "system": "EN|US|UK"}
# For EN classes "C25/30", fck=25 (cylinder), fcu=30 (cube).
CONCRETE_GRADES: Dict[str, Dict] = {
    # EN 206 cylinder/cube classes
    "C12/15": {"fck_mpa": 12, "fcu_mpa": 15, "system": "EN"},
    "C16/20": {"fck_mpa": 16, "fcu_mpa": 20, "system": "EN"},
    "C20/25": {"fck_mpa": 20, "fcu_mpa": 25, "system": "EN"},
    "C25/30": {"fck_mpa": 25, "fcu_mpa": 30, "system": "EN"},
    "C30/37": {"fck_mpa": 30, "fcu_mpa": 37, "system": "EN"},
    "C35/45": {"fck_mpa": 35, "fcu_mpa": 45, "system": "EN"},
    "C40/50": {"fck_mpa": 40, "fcu_mpa": 50, "system": "EN"},
    "C45/55": {"fck_mpa": 45, "fcu_mpa": 55, "system": "EN"},
    "C50/60": {"fck_mpa": 50, "fcu_mpa": 60, "system": "EN"},
    "C55/67": {"fck_mpa": 55, "fcu_mpa": 67, "system": "EN"},
    "C60/75": {"fck_mpa": 60, "fcu_mpa": 75, "system": "EN"},
    "C70/85": {"fck_mpa": 70, "fcu_mpa": 85, "system": "EN"},
    "C80/95": {"fck_mpa": 80, "fcu_mpa": 95, "system": "EN"},
    # Short forms commonly used in BOQs / drawings — same as the cylinder side.
    "C20": {"fck_mpa": 20, "fcu_mpa": 25, "system": "EN"},
    "C25": {"fck_mpa": 25, "fcu_mpa": 30, "system": "EN"},
    "C30": {"fck_mpa": 30, "fcu_mpa": 37, "system": "EN"},
    "C35": {"fck_mpa": 35, "fcu_mpa": 45, "system": "EN"},
    "C40": {"fck_mpa": 40, "fcu_mpa": 50, "system": "EN"},
    "C45": {"fck_mpa": 45, "fcu_mpa": 55, "system": "EN"},
    "C50": {"fck_mpa": 50, "fcu_mpa": 60, "system": "EN"},
    "C60": {"fck_mpa": 60, "fcu_mpa": 75, "system": "EN"},
    # ACI / US: f'c in psi commonly stated as 3000, 4000, 5000 etc.
    "3000PSI": {"fck_mpa": 20.7, "fcu_mpa": None, "system": "US"},
    "4000PSI": {"fck_mpa": 27.6, "fcu_mpa": None, "system": "US"},
    "5000PSI": {"fck_mpa": 34.5, "fcu_mpa": None, "system": "US"},
    "6000PSI": {"fck_mpa": 41.4, "fcu_mpa": None, "system": "US"},
    "8000PSI": {"fck_mpa": 55.2, "fcu_mpa": None, "system": "US"},
    # Indian Standard (IS 456) — M-grade is cube strength in MPa.
    "M20": {"fck_mpa": 16, "fcu_mpa": 20, "system": "IS"},
    "M25": {"fck_mpa": 20, "fcu_mpa": 25, "system": "IS"},
    "M30": {"fck_mpa": 24, "fcu_mpa": 30, "system": "IS"},
    "M35": {"fck_mpa": 28, "fcu_mpa": 35, "system": "IS"},
    "M40": {"fck_mpa": 32, "fcu_mpa": 40, "system": "IS"},
    "M45": {"fck_mpa": 36, "fcu_mpa": 45, "system": "IS"},
    "M50": {"fck_mpa": 40, "fcu_mpa": 50, "system": "IS"},
}


# Rebar grades — characteristic yield strength in MPa.
REBAR_GRADES: Dict[str, Dict] = {
    # US (ASTM A615)
    "Grade 40":  {"fy_mpa": 280, "system": "US",  "standard": "ASTM A615"},
    "Grade 60":  {"fy_mpa": 415, "system": "US",  "standard": "ASTM A615"},
    "Grade 75":  {"fy_mpa": 520, "system": "US",  "standard": "ASTM A615"},
    "Grade 80":  {"fy_mpa": 550, "system": "US",  "standard": "ASTM A615"},
    "Grade 100": {"fy_mpa": 690, "system": "US",  "standard": "ASTM A1035"},
    # UK / EN (BS 4449)
    "B500A":     {"fy_mpa": 500, "system": "EN",  "standard": "BS 4449"},
    "B500B":     {"fy_mpa": 500, "system": "EN",  "standard": "BS 4449"},
    "B500C":     {"fy_mpa": 500, "system": "EN",  "standard": "BS 4449"},
    # India (IS 1786)
    "Fe415":     {"fy_mpa": 415, "system": "IS",  "standard": "IS 1786"},
    "Fe500":     {"fy_mpa": 500, "system": "IS",  "standard": "IS 1786"},
    "Fe550":     {"fy_mpa": 550, "system": "IS",  "standard": "IS 1786"},
}


# Structural steel grades — yield strength MPa.
STRUCTURAL_STEEL_GRADES: Dict[str, Dict] = {
    # ASTM
    "A36":     {"fy_mpa": 250, "fu_mpa": 400, "system": "US",  "standard": "ASTM A36"},
    "A572-50": {"fy_mpa": 345, "fu_mpa": 450, "system": "US",  "standard": "ASTM A572"},
    "A992":    {"fy_mpa": 345, "fu_mpa": 450, "system": "US",  "standard": "ASTM A992"},
    "A500-B":  {"fy_mpa": 290, "fu_mpa": 400, "system": "US",  "standard": "ASTM A500"},
    # EN 10025
    "S235":    {"fy_mpa": 235, "fu_mpa": 360, "system": "EN",  "standard": "EN 10025"},
    "S275":    {"fy_mpa": 275, "fu_mpa": 410, "system": "EN",  "standard": "EN 10025"},
    "S355":    {"fy_mpa": 355, "fu_mpa": 470, "system": "EN",  "standard": "EN 10025"},
    "S420":    {"fy_mpa": 420, "fu_mpa": 520, "system": "EN",  "standard": "EN 10025"},
    "S460":    {"fy_mpa": 460, "fu_mpa": 540, "system": "EN",  "standard": "EN 10025"},
}


# ─────────────────────────────────────────────────────────────────────────
# STANDARD → PURPOSE LOOKUP
# ─────────────────────────────────────────────────────────────────────────

# What each named standard actually covers. Used by spec_analyzer to give
# context beyond "we found a reference to ASTM A615".
STANDARDS_PURPOSE: Dict[str, Dict] = {
    # ASTM
    "ASTM A36":   {"covers": "Carbon structural steel", "system": "US", "category": "structural_steel"},
    "ASTM A615": {"covers": "Deformed and plain carbon-steel bars for concrete reinforcement", "system": "US", "category": "rebar"},
    "ASTM A706": {"covers": "Low-alloy steel deformed bars for concrete reinforcement (weldable)", "system": "US", "category": "rebar"},
    "ASTM A992": {"covers": "Structural steel shapes (wide-flange / W-shapes)", "system": "US", "category": "structural_steel"},
    "ASTM A325": {"covers": "Structural bolts, heat-treated", "system": "US", "category": "fasteners"},
    "ASTM A490": {"covers": "Structural bolts, high-strength heat-treated", "system": "US", "category": "fasteners"},
    "ASTM A500": {"covers": "Cold-formed welded structural tubing", "system": "US", "category": "structural_steel"},
    "ASTM C150": {"covers": "Portland cement", "system": "US", "category": "cement"},
    "ASTM C33":  {"covers": "Concrete aggregates", "system": "US", "category": "aggregates"},
    "ASTM C94":  {"covers": "Ready-mixed concrete", "system": "US", "category": "concrete"},
    # ACI
    "ACI 318":   {"covers": "Building code requirements for structural concrete", "system": "US", "category": "concrete_design"},
    "ACI 301":   {"covers": "Specifications for structural concrete", "system": "US", "category": "concrete"},
    "ACI 350":   {"covers": "Concrete environmental engineering structures (water-retaining)", "system": "US", "category": "concrete_design"},
    # BS / EN
    "BS 8500":   {"covers": "Concrete (complementary to BS EN 206)", "system": "UK", "category": "concrete"},
    "BS 4449":   {"covers": "Steel for the reinforcement of concrete (weldable)", "system": "UK", "category": "rebar"},
    "BS EN 206": {"covers": "Concrete specification, performance, production and conformity", "system": "EN", "category": "concrete"},
    "BS EN 1992": {"covers": "Eurocode 2 — Design of concrete structures", "system": "EN", "category": "concrete_design"},
    "BS EN 1993": {"covers": "Eurocode 3 — Design of steel structures", "system": "EN", "category": "structural_steel_design"},
    "BS EN 10025": {"covers": "Hot-rolled products of structural steels", "system": "EN", "category": "structural_steel"},
    # Indian Standards
    "IS 456":    {"covers": "Plain and reinforced concrete code of practice", "system": "IS", "category": "concrete_design"},
    "IS 1786":   {"covers": "High-strength deformed steel bars and wires for concrete reinforcement", "system": "IS", "category": "rebar"},
    "IS 800":    {"covers": "General construction in steel code of practice", "system": "IS", "category": "structural_steel_design"},
    # Saudi / SASO
    "SASO 14":   {"covers": "Building code (Saudi Arabia)", "system": "SA", "category": "building_code"},
    # International / Other
    "IBC":       {"covers": "International Building Code", "system": "INTL", "category": "building_code"},
    "FIDIC":     {"covers": "Standard contract conditions for construction", "system": "INTL", "category": "contract"},
    "NEC":       {"covers": "New Engineering Contract suite", "system": "UK", "category": "contract"},
}


def lookup_grade(label: str) -> Optional[Dict]:
    """Return the grade info dict for a label (case-insensitive trimmed match).

    Searches concrete, rebar, then structural steel tables in that order.
    Returns None if the label isn't recognised.
    """
    if not label:
        return None
    key = label.strip().upper().replace(" ", "")
    # Try direct match in each table.
    for table, kind in (
        (CONCRETE_GRADES, "concrete"),
        (REBAR_GRADES, "rebar"),
        (STRUCTURAL_STEEL_GRADES, "structural_steel"),
    ):
        for grade_label, info in table.items():
            normalised = grade_label.upper().replace(" ", "")
            if normalised == key:
                return {**info, "label": grade_label, "kind": kind}
    return None


def lookup_standard(reference: str) -> Optional[Dict]:
    """Return the purpose dict for a standard reference string.

    Tries direct lookup first; then permissive matches (strips trailing
    year suffixes like ``-22``, ``:2019``). Returns None if unrecognised.
    """
    if not reference:
        return None
    key = reference.strip().upper()
    # Direct hit?
    if key in STANDARDS_PURPOSE:
        return {**STANDARDS_PURPOSE[key], "reference": key}
    # Strip trailing year / revision suffixes.
    import re
    cleaned = re.sub(r"[-:](\d{2,4}[A-Z]*)$", "", key).strip()
    if cleaned in STANDARDS_PURPOSE:
        return {**STANDARDS_PURPOSE[cleaned], "reference": cleaned}
    # Try as "ASTM XXX" where the user might have passed just "A615".
    for prefix in ("ASTM ", "BS ", "ACI ", "IS ", "EN ", "BS EN "):
        candidate = f"{prefix}{cleaned}"
        if candidate in STANDARDS_PURPOSE:
            return {**STANDARDS_PURPOSE[candidate], "reference": candidate}
    return None
