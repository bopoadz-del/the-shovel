"""Earthwork / geotech calculators (additive library, drop-catalog gap-fill).

Geometry + soil-state factors (bulking/swell, compaction) and an infinite-slope
factor of safety. Code-agnostic. Factors are parameters; arithmetic in ``note``.
Volume states: BANK (in situ), LOOSE (excavated, bulked), COMPACTED (placed).
"""
from __future__ import annotations

from app.lib.formula_registry import formula

import math

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.base.formulas.construction_formulas_earthwork import (  # noqa: F401 -- moved
    backfill_volume,
    excavation_volume,
)
from app.agents.hats.design.formulas.construction_formulas_earthwork import (  # noqa: F401 -- moved
    slope_fos_simple,
)
from app.agents.hats.qaqc.formulas.construction_formulas_earthwork import (  # noqa: F401 -- moved
    compaction_control,
)
from app.agents.hats.quantities.formulas.construction_formulas_earthwork import (  # noqa: F401 -- moved
    cut_fill_balance,
)

ADDITIONAL_CALCULATORS = {
    "excavation_volume": excavation_volume,
    "backfill_volume": backfill_volume,
    "cut_fill_balance": cut_fill_balance,
    "compaction_control": compaction_control,
    "slope_fos_simple": slope_fos_simple,
}
