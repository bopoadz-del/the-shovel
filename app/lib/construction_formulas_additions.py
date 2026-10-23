"""
The Fork -- Additional Construction Formulas

New calculators added post-live-chat-test to cover gaps found
in the 13-query battery. These supplement the existing
construction_formulas.py CALCULATORS registry.

Each formula returns a dict with deterministic values.
No fabrication -- all values traceable to standards.
"""
from __future__ import annotations

from app.lib.formula_registry import formula

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.hats.commercial.formulas.construction_formulas_additions import (  # noqa: F401 -- moved
    calculate_interim_payment,
)
from app.agents.hats.safety.formulas.construction_formulas_additions import (  # noqa: F401 -- moved
    guardrail_top_rail_height,
)

ADDITIONAL_CALCULATORS = {
    "guardrail_top_rail_height": guardrail_top_rail_height,
    "calculate_interim_payment": calculate_interim_payment,
}
