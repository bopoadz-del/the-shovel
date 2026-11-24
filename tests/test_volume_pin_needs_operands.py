"""The concrete-volume pin takes an unnamed call only when a size is there.

``resolve_concrete_volume_calc`` pins ``concrete_volume`` for a call that
named no calculator (or the earthwork alias) when the ask has a concrete word
in it. A word alone is not a volume ask: "slab concrete ... how many crews"
types no element size, and pinning it asked the user for a length the
question never needed, before the ask's own calculator name was read.

Every input is built in this file. Two inputs per decision.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

from app.lib.construction_formulas import run_calculation  # noqa: E402
from app.lib.construction_formulas_quantities import (  # noqa: E402
    resolve_concrete_volume_calc,
)


# Decision 1: an unnamed call with a concrete word is pinned only when the
# ask types an element size.
@pytest.mark.parametrize("ask", [
    "Slab concrete pour of 300 m3 with 4 crews, how many days?",
    "Footing concrete with waste factor 5 %, 120 m3 placed by 3 crews, how many days?",
])
def test_unnamed_concrete_word_without_size_is_not_pinned(ask):
    calc, _ = resolve_concrete_volume_calc(None, {}, ask)
    assert calc is None


@pytest.mark.parametrize("ask, dims", [
    ("Slab concrete 8 m x 6 m x 0.2 m, how many m3?", (8.0, 6.0, 0.2)),
    ("Raft concrete 20 m x 15 m x 1.2 m, volume?", (20.0, 15.0, 1.2)),
])
def test_unnamed_concrete_word_with_typed_size_is_pinned(ask, dims):
    calc, out = resolve_concrete_volume_calc(None, {}, ask)
    assert calc == "concrete_volume"
    assert (out["length_m"], out["width_m"], out["thickness_m"]) == dims


# Decision 2: a size the CALL carries counts as the operand too -- but only
# a size input of the calculator, and only when it is a figure.
@pytest.mark.parametrize("params, ask", [
    ({"length_m": 12, "width_m": 4, "thickness_m": 0.3},
     "Raft concrete with the documented waste factor"),
    ({"width_m": 3}, "Slab concrete, include waste"),
])
def test_size_in_call_params_pins(params, ask):
    calc, _ = resolve_concrete_volume_calc(None, params, ask)
    assert calc == "concrete_volume"


@pytest.mark.parametrize("params", [{"thickness": 0.25}, {"length": 9}])
def test_bare_stem_size_in_call_params_pins(params):
    calc, _ = resolve_concrete_volume_calc(
        None, params, "Footing concrete, include waste",
    )
    assert calc == "concrete_volume"


@pytest.mark.parametrize("params", [
    {"waste_factor": 0.05, "quantity": 3, "crew": 6},
    {"rate_per_m3": 450, "pours": 2},
])
def test_non_size_params_do_not_pin(params):
    calc, _ = resolve_concrete_volume_calc(
        None, params, "Footing concrete, include waste",
    )
    assert calc is None


@pytest.mark.parametrize("params", [
    {"length_m": 0, "width_m": ""},
    {"thickness_m": None, "length": 0.0},
])
def test_zero_size_param_does_not_pin(params):
    calc, _ = resolve_concrete_volume_calc(
        None, params, "Slab concrete cycle per floor",
    )
    assert calc is None


# Decision 3: a call that names concrete_volume keeps it with or without a
# size; the earthwork alias is released when nothing is sized.
@pytest.mark.parametrize("ask", [
    "Slab concrete, how much?", "Pile cap concrete, include waste",
])
def test_named_concrete_volume_stays_without_size(ask):
    calc, _ = resolve_concrete_volume_calc("concrete_volume", {}, ask)
    assert calc == "concrete_volume"


@pytest.mark.parametrize("ask", [
    "Slab concrete, how much?", "Footing concrete total?",
])
def test_earthwork_alias_without_size_is_not_pinned(ask):
    calc, _ = resolve_concrete_volume_calc("excavation_volume", {}, ask)
    assert calc == "excavation_volume"


# Decision 4 (through the real entry point): an ask with a concrete word and
# no element size is not asked for a slab length; an ask that writes another
# calculator's name reaches that calculator; a sized ask still reaches the
# volume.
@pytest.mark.parametrize("ask", [
    "Slab concrete pour of 300 m3 with 4 crews, how many days?",
    "Raft concrete 450 m3, pour rate 30 m3/h per crew, 2 crews: pour duration?",
])
def test_unsized_concrete_ask_is_not_asked_for_a_length(ask):
    r = run_calculation(None, {"text": ask})
    assert "length_m" not in (r.get("missing") or [])


@pytest.mark.parametrize("ask", [
    "Live load reduction for a slab: tributary area 60 m2, "
    "base live load 5 kN/m2, how much can I reduce?",
    "Live load reduction on a concrete footing: tributary area 90 m2, "
    "base live load 4 kN/m2?",
])
def test_named_ask_with_slab_word_reaches_its_calculator(ask):
    r = run_calculation(None, {"text": ask})
    assert r.get("calculation") == "live_load_reduction"
    assert "length_m" not in (r.get("missing") or [])


@pytest.mark.parametrize("ask", [
    "Slab concrete 8 m x 6 m x 0.2 m, how many m3?",
    "Footing concrete 2 m x 2 m x 0.5 m, how many m3?",
])
def test_sized_slab_ask_still_reaches_concrete_volume(ask):
    r = run_calculation(None, {"text": ask})
    assert r.get("status") == "success"
    assert r.get("calculation") == "concrete_volume"
