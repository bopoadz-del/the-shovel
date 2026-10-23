"""The precast erection check is its own registry entry.

It is the member's weight plus rigging, times the dynamic allowance, as a
share of the rated capacity at the stated radius. That is not the crane-lift
calculator (chart capacity minus deductions, compared with a utilisation limit).
"""
from __future__ import annotations

import pytest

from app.lib import formula_registry


_FUNCTION_WORDS = frozenset({
    "a", "an", "and", "as", "at", "for", "of", "the", "to", "with",
})


def _words(text: str) -> set[str]:
    return {
        word for word in "".join(ch if ch.isalnum() else " " for ch in text.casefold()).split()
        if word and word not in _FUNCTION_WORDS
    }


def test_precast_erection_is_declared_apart_from_the_crane_lift_calculator():
    precast = formula_registry.get("precast_beam_erection_check")
    crane = formula_registry.get("crane_lift_capacity")
    assert precast is not None and crane is not None

    shown = f"{precast.display_name} {precast.description}"
    assert "_" not in precast.display_name
    assert "." not in precast.display_name
    assert precast.display_name != precast.name
    # The crane calculator's own name, and the words that name it, do not
    # describe this entry.
    assert crane.display_name.casefold() not in shown.casefold()
    crane_words = _words(crane.display_name)
    assert not crane_words & _words(shown)

    labels = []
    for key, param in precast.params.items():
        assert (param.label or "").strip(), key
        labels.append(param.label)
    crane_labels = {(p.label or "").casefold() for p in crane.params.values()}
    assert not crane_labels & {label.casefold() for label in labels}


def test_precast_erection_demand_still_applies_the_dynamic_allowance():
    precast = formula_registry.get("precast_beam_erection_check")
    result = precast.fn(
        beam_weight_t=10,
        beam_length_m=8,
        crane_capacity_t=20,
        lift_radius_m=10,
        rigging_weight_t=0,
        dynamic_factor=1.25,
    )
    assert result["effective_lift_weight_t"] == pytest.approx(12.5, abs=0.01)
    assert result["utilization_pct"] == pytest.approx(62.5, abs=0.1)
