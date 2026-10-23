"""app.lib.construction_formulas_structural_rc: its formulas moved to their owners' packages (F-DRIVER
Phase A). Kept so existing imports keep working; removed with the
legacy paths at F-DRIVER step 16."""
from __future__ import annotations

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.hats.design.formulas.construction_formulas_structural_rc import (  # noqa: F401 -- moved
    ADDITIONAL_CALCULATORS,
    _ACI,
    _ACI_ASK_RE,
    _ACI_RATIOS,
    _BOTH_ENDS_RE,
    _EC,
    _EC_ASK_RE,
    _EC_RATIOS,
    _FY_RE,
    _NOT_SLAB_THICKNESS_RE,
    _ONE_END_RE,
    _ONE_WAY_SLAB_RE,
    _SPAN_GROUP_PAIRS,
    _SPAN_NUM,
    _SPAN_RE,
    _SPAN_UNIT,
    _THICKNESS_RE,
    _format_mm,
    _norm_code,
    _span_mm_from_ask,
    _support_from_ask,
    answer_states_slab_thickness_result,
    format_slab_thickness_answer,
    looks_like_slab_thickness_min_ask,
    rc_beam_moment_capacity,
    rc_beam_shear_capacity,
    rebar_lap_length,
    slab_thickness_min,
    slab_thickness_params_from_ask,
)
