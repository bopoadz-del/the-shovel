"""Formulas from app.lib.construction_formulas_reference_tables owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


# Indicative embodied-carbon factors (kgCO2e per m3 of concrete), ICE/EPD family.
# Real projects use the supplier's EPD — these are typical GGBS-free CEM I values.
_CONCRETE_ECO2_KGM3 = {"c20": 260.0, "c25": 290.0, "c30": 320.0,
                       "c35": 350.0, "c40": 390.0}

@formula(
    owner='design',
    display_name='Embodied carbon of concrete',
    description='Embodied carbon of a concrete volume from an emission factor per cubic metre.',
    inputs={
        'volume_m3': Param('m3', 0, 1e7, label='volume'),
        'grade': Param('-', label='concrete grade', grade='concrete'),
        'embodied_kgco2e_m3': Param('kgCO2e/m3', 0, 2000, label='embodied carbon per m3'),
    },
    outputs={'total_kgco2e': 'kgCO2e', 'total_tco2e': 'tCO2e', 'factor_kgco2e_m3': 'kgCO2e/m3'},
)
def carbon_footprint_concrete(
    volume_m3: float,
    grade: str = "c30",
    embodied_kgco2e_m3: float = None,
) -> dict:
    """Embodied CO2e of a concrete pour = volume * factor. Factor from a typical
    ICE/EPD table by grade, or pass ``embodied_kgco2e_m3`` from the supplier EPD."""
    if float(volume_m3) < 0:
        return {"error": "volume_m3 must be >= 0."}
    g = (grade or "c30").strip().lower()
    factor = float(embodied_kgco2e_m3) if embodied_kgco2e_m3 is not None \
        else _CONCRETE_ECO2_KGM3.get(g, 320.0)
    total = float(volume_m3) * factor
    return {
        "kind": "reference_table",
        "total_kgco2e": round(total, 1),
        "total_tco2e": round(total / 1000.0, 3),
        "factor_kgco2e_m3": factor,
        "standard": "ICE / EPD embodied-carbon (indicative)",
        "note": (f"{volume_m3} m3 * {factor} kgCO2e/m3 = {total:.0f} kgCO2e. "
                 "Use the supplier EPD for a project figure; table is indicative."),
    }
