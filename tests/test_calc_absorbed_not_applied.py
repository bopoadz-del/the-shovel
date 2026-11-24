"""A figure a ``**kwargs`` calculator absorbs without declaring is named.

``_partition_bound_params`` hands every unbound key to a calculator whose
signature takes ``**kwargs``. Before this, a typed figure such a calculator
never reads vanished without a word: the result came back as if the user had
not given it. The envelope now lists it under ``not_applied`` and says so in
the note. Every input here is built by the test; no product answer is copied.
"""
from __future__ import annotations

from typing import Any, Dict

from app.lib import construction_formulas as cf


def _absorbing(length_m: float, **_kwargs: Any) -> Dict[str, Any]:
    return {"length_m": length_m}


def _strict(length_m: float) -> Dict[str, Any]:
    return {"length_m": length_m}


# ── the helper: which figures count as not applied ──────────────────────────

def test_undeclared_caller_figure_is_listed_for_a_kwargs_calculator():
    bound = {"length_m": 4.0, "colour_code": 7.0}
    rows = cf._absorbed_not_applied(
        "no_such_registry_name", _absorbing, bound, {"length_m": 4.0, "colour_code": 7.0})
    assert rows == [{"parameter": "colour_code", "value": 7.0}]

    bound = {"length_m": 2.5, "depth_mm": 300.0, "grade": 40.0}
    rows = cf._absorbed_not_applied(
        "no_such_registry_name", _absorbing, bound,
        {"length_m": 2.5, "depth_mm": 300.0, "grade": 40.0})
    assert rows == [{"parameter": "depth_mm", "value": 300.0},
                    {"parameter": "grade", "value": 40.0}]


def test_a_calculator_without_kwargs_lists_nothing():
    assert cf._absorbed_not_applied(
        "no_such_registry_name", _strict, {"length_m": 4.0, "colour_code": 7.0},
        {"length_m": 4.0, "colour_code": 7.0}) == []
    assert cf._absorbed_not_applied(
        "no_such_registry_name", _strict, {"length_m": 9.0},
        {"length_m": 9.0, "depth_mm": 12.0}) == []


def test_a_figure_read_from_the_ask_and_not_sent_by_the_caller_is_not_listed():
    # Bound by the ask extractors, absent from the caller's own figures.
    assert cf._absorbed_not_applied(
        "no_such_registry_name", _absorbing, {"length_m": 4.0, "colour_code": 7.0},
        {"length_m": 4.0}) == []
    assert cf._absorbed_not_applied(
        "no_such_registry_name", _absorbing, {"length_m": 6.0, "depth_mm": 50.0},
        {"length_m": 6.0}) == []


def test_a_figure_the_registry_declares_is_not_listed(monkeypatch):
    from app.lib import formula_registry

    spec = object()
    monkeypatch.setattr(formula_registry, "get", lambda name: spec if name == "declared" else None)
    monkeypatch.setattr(formula_registry, "parameters",
                        lambda s: ["colour_code"] if s is spec else [])
    figures = {"length_m": 4.0, "colour_code": 7.0}
    assert cf._absorbed_not_applied("declared", _absorbing, dict(figures), figures) == []
    assert cf._absorbed_not_applied("undeclared", _absorbing, dict(figures), figures) == [
        {"parameter": "colour_code", "value": 7.0}]


# ── through run_calculation, on the one registered **kwargs calculator ─────

def test_slip_form_names_a_typed_slump_it_does_not_take():
    for slump in (80, 110):
        env = cf.run_calculation("concrete_mix_slip_form", {"slump_mm": slump})
        assert env["status"] == "success"
        assert env["not_applied"] == [{"parameter": "slump_mm", "value": slump}]
        assert f"slump_mm={slump}" in env["note"]
        assert "Not applied" in env["note"]


def test_slip_form_with_no_figure_has_no_not_applied_line():
    for params in ({}, None):
        env = cf.run_calculation("concrete_mix_slip_form", params)
        assert env["status"] == "success"
        assert "not_applied" not in env
        assert "Not applied" not in env["note"]
