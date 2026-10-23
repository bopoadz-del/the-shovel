"""BOQ unit-of-measurement inference.

Determines the CORRECT unit of measurement for a BOQ line item from its work
type (the description), independent of what a scanned PDF / OCR produced for the
printed unit. Grounded in the standard methods of measurement -- CESMM4
(civil/infrastructure) and POMI (buildings); the shipped units-of-measurement
reference lives in docs/knowledge/.

Used at ingestion (boq_processor) to (a) FILL a blank/unreadable unit from the
inferred work-type unit and (b) FLAG a stated unit that contradicts the work
type (e.g. an 'excavation' item read as 'm' when earthworks is measured in m3).
It never silently overwrites a priced line's stated unit -- conflicts are kept
and flagged for review; only blank units are filled.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Ordered rules: (unit, [regex keywords]). FIRST match wins, so more-specific /
# overriding operations precede the bulk material they act on: formwork before
# concrete, pipe fittings before pipe, mesh before reinforcement.
_RULES = [
    # 1. area operations that override the material they act on
    ("m2", [r"formwork", r"form work", r"shutter", r"falsework"]),
    # 2. fabric/mesh reinforcement is area, not weight
    ("m2", [r"fabric reinforc", r"mesh reinforc", r"reinforc\w* mesh", r"welded fabric", r"\bbrc\b"]),
    # 3. pipe ancillaries (bedding/surround/protection) are volume -- BEFORE 'pipe'->m
    ("m3", [r"\bbedding", r"\bsurround", r"haunch", r"granular fill", r"pipe protection"]),
    # 4. enumerated fittings / structures -- before 'pipe' -> m
    ("nr", [r"manhole", r"inspection chamber", r"\bchamber\b", r"catch ?pit", r"\bgully",
            r"headwall", r"\bvalve", r"hydrant", r"\bfitting", r"\bbend\b", r"\bjunction",
            r"\btee\b", r"coupling", r"penstock", r"sluice", r"washout", r"air ?valve"]),
    # 5. linear items
    ("m", [r"\bpipe", r"pipe ?work", r"gravity sewer", r"\bsewer", r"\bfoul", r"waste ?water",
           r"rising main", r"\bculvert", r"\bduct\b", r"ducting", r"\bcable", r"\bconduit",
           r"trunking", r"containment", r"\bkerb", r"\bedging", r"\bchannel", r"\bcoping",
           r"skirting", r"handrail", r"balustrade", r"\bdpc\b", r"damp proof course",
           r"string course", r"expansion joint"]),
    # 6. area finishes / coverings
    ("m2", [r"\bplaster", r"render", r"\bskim", r"\bpaint", r"decorat", r"\btiling", r"\btile\b",
            r"floor(ing| finish| screed)", r"\bscreed", r"\bceiling", r"gypsum", r"partition",
            r"blockwork", r"brickwork", r"masonry", r"cladding", r"waterproof", r"tanking",
            r"\bmembrane", r"insulat", r"surfacing", r"asphalt", r"bituminous", r"wearing course",
            r"interlock", r"\bpaving", r"geotextile", r"site clearance", r"topsoil"]),
    # 7. weight -- SPECIFIC to steel (won't catch 'reinforced concrete')
    ("t", [r"\brebar\b", r"\breinforcement\b", r"reinforcing (bar|steel)", r"high[- ]yield",
           r"structural steel", r"steel ?work", r"\brsj\b", r"purlin", r"mild steel bar"]),
    # 8. volume (bulk material / works)
    ("m3", [r"excavat", r"earth ?work", r"back ?fill", r"\bfilling\b", r"imported fill",
            r"selected fill", r"\bfill material", r"disposal", r"cart away", r"muck away",
            r"remove surplus", r"\bconcrete", r"\brcc\b", r"blinding", r"grade [cm]",
            r"\bc\d{2}\b", r"sub-?base", r"road ?base", r"hardcore", r"granular"]),
    # 9. enumerated (buildings / MEP / accessories)
    ("nr", [r"\bdoor", r"\bwindow", r"ironmonger", r"sanitary", r"\bwc\b", r"\bbasin", r"\bsink",
            r"\bmirror", r"accessor", r"\bfixture", r"\bsocket", r"\bswitch", r"\bpanel\b",
            r"transformer", r"\bpump\b", r"chiller", r"\bahu\b", r"\bcrah", r"manhole cover",
            r"\blift\b", r"escalator", r"\btree\b", r"\bshrub", r"waste bin", r"extinguisher"]),
    # 10. lump sum
    ("sum", [r"preliminar", r"mobili[sz]", r"insurance", r"provisional sum", r"commission",
             r"\bday ?work", r"contingency", r"allow for", r"general item"]),
]

_COMPILED = [(u, [re.compile(k, re.I) for k in ks]) for u, ks in _RULES]

_CANON = {
    "lm": "m", "l.m": "m", "rm": "m", "r.m": "m", "m.": "m", "mq": "m2", "sqm": "m2",
    "sq.m": "m2", "cum": "m3", "cu.m": "m3", "mc": "m3", "nos": "nr", "no": "nr",
    "no.": "nr", "ea": "nr", "each": "nr", "pcs": "nr", "tonne": "t", "ton": "t",
    "ls": "sum", "item": "sum", "lot": "sum",
}


def infer_unit(description: str) -> Optional[str]:
    """Best-guess standard unit for a BOQ item from its description, else None."""
    d = (description or "").strip()
    if not d:
        return None
    for unit, pats in _COMPILED:
        for p in pats:
            if p.search(d):
                return unit
    return None


def canon_unit(u: str) -> str:
    """Normalise a unit token (mq->m2, cum->m3, nos->nr, tonne->t, ls->sum ...)."""
    u = (u or "").strip().lower().replace("²", "2").replace("³", "3").replace(" ", "")
    return _CANON.get(u, u)


def reconcile_unit(parsed_unit: str, description: str) -> Tuple[str, str, bool, Optional[str]]:
    """Reconcile a parsed/OCR unit against the inferred work-type unit.

    Returns ``(unit, source, suspect, expected)``:
      * ``source`` is ``"inferred"`` when a blank/unreadable unit was filled from
        the work type, else ``"parsed"``.
      * ``suspect`` is True when a STATED unit contradicts the inferred unit --
        the stated unit is KEPT (never overwrite priced data) and flagged, with
        ``expected`` holding the work-type unit.
    """
    inferred = infer_unit(description)
    p = (parsed_unit or "").strip()
    if not p or p.lower() == "nan":
        return (inferred, "inferred", False, None) if inferred else ("", "parsed", False, None)
    if inferred and canon_unit(p) != canon_unit(inferred):
        return p, "parsed", True, inferred
    return p, "parsed", False, None
