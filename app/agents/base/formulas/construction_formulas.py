"""Formulas from app.lib.construction_formulas owned by the base package (every hat).

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Dict


@formula(
    owner='base',
    display_name='Diaphragm wall panel volume',
    description='Concrete volume of diaphragm wall panels, with an allowance for tremie overbreak.',
    inputs={
        'panel_length': Param('m', 0, 100, label='panel length'),
        'wall_thickness': Param('m', 0, 5, label='wall thickness'),
        'excavation_depth': Param('m', 0, 200, label='excavation depth'),
        'panel_count': Param('nr', 1, 10000, label='number of panels'),
    },
    outputs={'panel_count': '-', 'volume_per_panel_m3': 'm3', 'total_volume_m3': 'm3', 'volume_with_waste_m3': 'm3', 'waste_factor': '-'},
)
def diaphragm_wall_panel_volume(
    panel_length: float, wall_thickness: float,
    excavation_depth: float, panel_count: int = 1,
) -> Dict[str, float]:
    """Concrete volume for diaphragm wall panels with 10% tremie waste."""
    if panel_length <= 0 or wall_thickness <= 0 or excavation_depth <= 0:
        return {"error": "panel length, wall thickness and excavation depth must be > 0."}
    if panel_count <= 0:
        return {"error": "number of panels must be > 0."}
    vol = panel_length * wall_thickness * excavation_depth
    total = vol * panel_count
    return {
        "panel_count": panel_count,
        "volume_per_panel_m3": round(vol, 2),
        "total_volume_m3": round(total, 2),
        "volume_with_waste_m3": round(total * 1.10, 2),
        "waste_factor": 1.10,
    }
