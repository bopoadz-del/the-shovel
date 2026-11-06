"""Registered parameter labels in the answer-exit rewriter.

A registered parameter reads as its declared label even when its name has
no wordy segments. These tests pin the mechanism, never a specific key:

* the label-aware helper returns the registry's label for a declared
  parameter and the name-derived words for an undeclared one;
* the identifier rewriter replaces a registered parameter token with its
  label, keeps a non-wordy code key as written, and marks step names;
* the registry maps cache their build (the second call is free).
"""

from __future__ import annotations

import pytest

from app.agents.answer_exit import _plain_identifiers
from app.lib import source_labels
from app.lib.source_labels import _registry_maps, parameter_label_words


def _fresh_maps():
    """Reset the cached maps so each test builds them from the live registry."""
    source_labels._display_by_id = None
    source_labels._param_units = None
    source_labels._known_units = None
    source_labels._param_labels = {}
    return _registry_maps()


@pytest.mark.parametrize(
    "key,unit,expected",
    [
        ("load_factor_kn_m", "kN/m", "load factor"),
        ("area_factor_m2", "m2", "area factor"),
    ],
)
def test_parameter_label_words_falls_back_to_name_words(key, unit, expected):
    assert parameter_label_words(key, unit) == expected


# The declared labels, read from the live registry at import -- no key is
# written into this file.
_fresh_maps()
_LABELLED = sorted(source_labels._param_labels.items())
assert len(_LABELLED) >= 2, "the registry declares fewer than two labelled parameters"


@pytest.mark.parametrize("key,label", _LABELLED[:2])
def test_parameter_label_words_returns_the_declared_label(key, label):
    assert parameter_label_words(key) == label


@pytest.mark.parametrize("key", ["a_b", "x_y"])
def test_plain_identifiers_keeps_a_non_wordy_code_key(key):
    out = _plain_identifiers(f"{key}=72")
    assert f"{key}=72" in out


def test_plain_identifiers_rewrites_a_registered_parameter():
    _fresh_maps()
    labels = sorted(source_labels._param_labels.items())
    assert labels
    key, label = labels[0]
    out = _plain_identifiers(f"- {key}: 72")
    assert key not in out
    assert label in out


@pytest.mark.parametrize("step", ["use_takeoff_tool", "run_risk_review"])
def test_plain_identifiers_marks_a_step_name(step):
    out = _plain_identifiers(f"Call {step} for it.", steps={step})
    assert "\x03" in out


@pytest.mark.parametrize("calls", [1, 2])
def test_registry_maps_cache_their_build(monkeypatch, calls):
    _fresh_maps()
    from app.lib import formula_registry

    counter = {"n": 0}
    real_specs = formula_registry.all_specs

    def counted():
        counter["n"] += 1
        return real_specs()

    monkeypatch.setattr(formula_registry, "all_specs", counted)
    source_labels._display_by_id = None
    source_labels._param_units = None
    source_labels._known_units = None
    source_labels._param_labels = {}
    for _ in range(calls):
        _registry_maps()
    assert counter["n"] == 1
