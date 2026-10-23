"""Formulas from app.lib.construction_formulas owned by more than one owner.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from typing import Any
import re


def _is_aci_code(code: Any) -> bool:
    """"ACI", "ACI 318-19", "aci318", "ACI 318-19 (SI)" all name the ACI form.

    The previous exact-token match knew five spellings and none of them was
    the one the model writes ("ACI 318-19"), so that spelling silently fell
    to the metric-technical form -- 280,624 kg/cm2 under an ACI label.
    """
    return re.sub(r"[^a-z0-9]", "", str(code or "").lower()).startswith("aci")
