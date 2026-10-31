"""Formulas from app.lib.construction_formulas_reference_tables owned by the qaqc hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.agents.hats.qaqc.results import qaqc_calculator
from app.lib.formula_registry import Param, formula


# Typical clash tolerance by BIM LOD (mm). Project BEP governs — indicative only.
_LOD_CLASH_MM = {100: None, 200: 50.0, 300: 25.0, 350: 12.0, 400: 6.0, 500: 3.0}

@formula(
    owner='qaqc',
    display_name='BIM clash tolerance',
    description="Clash tolerance for a model's level of development.",
    inputs={
        'lod': Param('count', 100, 500, label='level of development'),
    },
    outputs={'lod': '-', 'clash_tolerance_mm': 'mm'},
)
@qaqc_calculator
def bim_clash_tolerance(lod: int = 350) -> dict:
    """Typical hard-clash tolerance for a BIM level of development. This is a
    BEP convention, NOT a standard constant — confirm in the project BEP."""
    if int(lod) not in _LOD_CLASH_MM:
        return {
            "error": (
                "The level of development must be 100, 200, 300, 350, 400 or 500, "
                f"not {lod}."
            ),
        }
    tol = _LOD_CLASH_MM.get(int(lod))
    return {
        "kind": "reference_table",
        "lod": int(lod),
        "clash_tolerance_mm": tol,
        "standard": "BIM BEP convention (indicative)",
        "note": (f"LOD {lod}: typical hard-clash tolerance "
                 f"{'undefined' if tol is None else f'±{tol} mm'}. "
                 "Project BEP governs; not a standard constant."),
    }

@formula(
    owner='qaqc',
    display_name='Laser scan accuracy at range',
    description="A scanner's stated ranging accuracy scaled to a working range, for comparison with the tolerance.",
    inputs={
        'range_m': Param('m', 0, 2000, label='distance'),
        'accuracy_at_10m_mm': Param('mm', 0, 100, label='stated accuracy'),
    },
    outputs={'range_m': 'm', 'estimated_accuracy_mm': 'mm'},
)
@qaqc_calculator
def laser_scan_accuracy(range_m: float, accuracy_at_10m_mm: float = 2.0) -> dict:
    """Scale a scanner's stated ranging accuracy to a working range (linear
    approximation). Pass the accuracy from the scanner datasheet; the value is
    vendor/instrument-specific, not a derived constant."""
    r = float(range_m)
    acc = float(accuracy_at_10m_mm) * (r / 10.0)
    return {
        "kind": "reference_table",
        "range_m": r,
        "estimated_accuracy_mm": round(acc, 2),
        "standard": "scanner datasheet (vendor-specific)",
        "note": (f"{accuracy_at_10m_mm} mm at 10 m scaled to {r} m ~= {acc:.2f} mm "
                 "(linear approx.; use the instrument datasheet)."),
    }


from app.agents.hats.qaqc.defaults import install_default_disclosure

install_default_disclosure()
