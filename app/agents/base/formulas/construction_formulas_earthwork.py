"""Formulas from app.lib.construction_formulas_earthwork owned by the base package (every hat).

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='base',
    display_name='Excavation volume',
    description='Bank (in-situ) excavation volume of a rectangular pit or trench, and the loose volume after bulking.',
    inputs={
        'length_m': Param('m', 0, 10000, label='length'),
        'width_m': Param('m', 0, 10000, label='width'),
        'depth_m': Param('m', 0, 500, label='depth'),
        'bulking_factor': Param('fraction', 0, 1, label='bulking'),
    },
    outputs={'bank_volume_m3': 'm3', 'loose_volume_m3': 'm3', 'bulked_volume_m3': 'm3', 'bulking_factor': '-'},
)
def excavation_volume(
    length_m: float,
    width_m: float,
    depth_m: float,
    bulking_factor: float = 0.25,
) -> dict:
    """Bank (in-situ) excavation volume L*W*D, and the loose (bulked) volume for
    haulage = bank*(1+bulking)."""
    if min(float(length_m), float(width_m), float(depth_m)) < 0:
        return {"error": "excavation length_m, width_m and depth_m must be >= 0."}
    if float(bulking_factor) < 0:
        return {"error": "bulking_factor must be >= 0."}
    bank = float(length_m) * float(width_m) * float(depth_m)
    loose = bank * (1.0 + bulking_factor)
    bank_r = round(bank, 3)
    loose_r = round(loose, 3)
    return {
        "bank_volume_m3": bank_r,
        "loose_volume_m3": loose_r,
        "bulked_volume_m3": loose_r,
        "bulking_factor": bulking_factor,
        "standard": "geometry / soil bulking",
        "note": (f"Bank = {length_m}*{width_m}*{depth_m} = {bank:.2f} m3; "
                 f"loose = bank*(1+{bulking_factor}) = {loose:.2f} m3 (haulage)."),
    }

@formula(
    owner='base',
    display_name='Backfill volume',
    description='Backfill needed around a structure: the void left after the structure, and the loose volume to import allowing for swell.',
    inputs={
        'excavation_bank_m3': Param('m3', 0, 1e8, label='excavated bank volume'),
        'structure_volume_m3': Param('m3', 0, 1e8, label='structure volume'),
        'swell_factor': Param('fraction', 0, 1, label='swell'),
    },
    outputs={'void_volume_m3': 'm3', 'loose_backfill_needed_m3': 'm3', 'swell_factor': '-'},
)
def backfill_volume(
    excavation_bank_m3: float,
    structure_volume_m3: float,
    swell_factor: float = 0.20,
) -> dict:
    """Backfill needed to fill the void around a structure. Void (compacted) =
    excavation - structure; loose material to import = void*(1+swell)."""
    if float(excavation_bank_m3) < 0 or float(structure_volume_m3) < 0:
        return {"error": "excavation_bank_m3 and structure_volume_m3 must be >= 0."}
    if float(swell_factor) < 0:
        return {"error": "swell_factor must be >= 0."}
    void = float(excavation_bank_m3) - float(structure_volume_m3)
    void = max(void, 0.0)
    loose_needed = void * (1.0 + swell_factor)
    return {
        "void_volume_m3": round(void, 3),
        "loose_backfill_needed_m3": round(loose_needed, 3),
        "swell_factor": swell_factor,
        "standard": "geometry / soil swell",
        "note": (f"Void = {excavation_bank_m3} - {structure_volume_m3} = "
                 f"{void:.2f} m3; loose to import = void*(1+{swell_factor}) = "
                 f"{loose_needed:.2f} m3."),
    }
