"""A reader sees a plain name for these calculators.

The registry id and the source filename are identifiers for code. The
display name is what an answer credits.
"""
from __future__ import annotations

from app.lib import formula_registry
from app.lib.source_labels import calculator_label, formula_display_name

#: Registry id -> the words a reader is shown.
PLAIN = {
    "beam_deflection_ss_udl": "Beam deflection under a uniform load",
    "beam_deflection_cantilever_udl": "Cantilever deflection under a uniform load",
    "beam_deflection_cantilever_point_load": "Cantilever deflection under a point load",
    "beam_deflection_ss_point_load_midspan": "Beam deflection under a central point load",
    "post_tensioning_force": "Post-tensioning force",
    "slope_fos_simple": "Slope factor of safety",
    "modulus_of_elasticity_concrete": "Concrete modulus of elasticity",
    "live_load_reduction": "Live load reduction",
    "cost_buildup_rebar": "Reinforcement cost build-up",
}


def test_these_calculators_credit_a_plain_display_name():
    for name, plain in PLAIN.items():
        spec = formula_registry.get(name)
        assert spec is not None, name
        stem = spec.fn.__module__.rsplit(".", 1)[-1]
        assert spec.display_name == plain
        assert formula_display_name(name) == plain
        label = calculator_label(name)
        assert label.startswith(plain)
        assert name not in label
        assert stem not in label
        assert ".py" not in label
        assert "_" not in spec.display_name
