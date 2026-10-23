"""Formulas from app.lib.construction_formulas_earthwork owned by the quantities hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='quantities',
    display_name='Cut and fill balance',
    description='Site earthwork balance: surplus to export or deficit to import, from cut and fill volumes.',
    inputs={
        'cut_volume_m3': Param('m3', 0, 1e8, label='cut volume'),
        'fill_volume_m3': Param('m3', 0, 1e8, label='fill volume'),
        'bulking_factor': Param('fraction', 0, 1, label='bulking'),
    },
    outputs={'balance_bank_m3': 'm3', 'haul_loose_m3': 'm3'},
)
def cut_fill_balance(
    cut_volume_m3: float,
    fill_volume_m3: float,
    bulking_factor: float = 0.25,
) -> dict:
    """Site earthwork balance in bank m3. balance = cut - fill; positive = surplus
    to export, negative = deficit to import. Loose export/import shown for haulage."""
    if float(cut_volume_m3) < 0 or float(fill_volume_m3) < 0:
        return {"error": "cut_volume_m3 and fill_volume_m3 must be >= 0."}
    if float(bulking_factor) < 0:
        return {"error": "bulking_factor must be >= 0."}
    cut = float(cut_volume_m3)
    fill = float(fill_volume_m3)
    balance = cut - fill
    status = "export_surplus" if balance > 0 else ("import_deficit" if balance < 0 else "balanced")
    loose = abs(balance) * (1.0 + bulking_factor)
    return {
        "balance_bank_m3": round(balance, 3),
        "status": status,
        "haul_loose_m3": round(loose, 3),
        "standard": "earthwork mass balance",
        "note": (f"Balance = cut - fill = {cut} - {fill} = {balance:.1f} m3 "
                 f"({status}); loose to haul = |bal|*(1+{bulking_factor}) = {loose:.1f} m3."),
    }
