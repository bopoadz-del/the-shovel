"""Reference-table calculators (additive library, gap-fill).

These are NOT derived physics — they return values from a cited factor/threshold
table or classify a caller-supplied spec. Each result carries ``kind =
"reference_table"`` and a note stating the source and its project/vendor caveat,
so a plausible-looking number is never mistaken for a derivation.
"""
from __future__ import annotations

from app.lib.formula_registry import formula

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.hats.commercial.formulas.construction_formulas_reference_tables import (  # noqa: F401 -- moved
    _LEED_LEVELS,
    leed_points_estimate,
)
from app.agents.hats.design.formulas.construction_formulas_reference_tables import (  # noqa: F401 -- moved
    _CONCRETE_ECO2_KGM3,
    carbon_footprint_concrete,
)
from app.agents.hats.qaqc.formulas.construction_formulas_reference_tables import (  # noqa: F401 -- moved
    _LOD_CLASH_MM,
    bim_clash_tolerance,
    laser_scan_accuracy,
)

ADDITIONAL_CALCULATORS = {
    "carbon_footprint_concrete": carbon_footprint_concrete,
    "leed_points_estimate": leed_points_estimate,
    "bim_clash_tolerance": bim_clash_tolerance,
    "laser_scan_accuracy": laser_scan_accuracy,
}
