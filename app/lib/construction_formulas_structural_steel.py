"""app.lib.construction_formulas_structural_steel: its formulas moved to their owners' packages (F-DRIVER
Phase A). Kept so existing imports keep working; removed with the
legacy paths at F-DRIVER step 16."""
from __future__ import annotations

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.hats.design.formulas.construction_formulas_structural_steel import (  # noqa: F401 -- moved
    ADDITIONAL_CALCULATORS,
    _ACI,
    _EC,
    _norm_code,
    bolt_shear_capacity,
    steel_tension_capacity,
    weld_capacity,
)
