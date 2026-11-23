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

os.environ.setdefault("RAG_EMBEDDING_MODEL", "fake")

from app.lib.construction_formulas import run_calculation  # noqa: E402
from app.lib.construction_formulas_quantities import (  # noqa: E402
    resolve_concrete_volume_calc,
)


# Decision 1: an unnamed call with a concrete word is pinned only when the
# ask types an element size.
def test_unnamed_concrete_word_without_size_is_not_pinned():
    calc, _ = resolve_concrete_volume_calc(
        None, {}, "Slab concrete pour of 300 m3 with 4 crews, how many days?",
    )
    assert calc is None


def test_unnamed_concrete_word_with_typed_size_is_pinned():
    calc, out = resolve_concrete_volume_calc(
        None, {}, "Slab concrete 8 m x 6 m x 0.2 m, how many m3?",
    )
    assert calc == "concrete_volume"
    assert (out["length_m"], out["width_m"], out["thickness_m"]) == (8.0, 6.0, 0.2)


# Decision 2: a size the CALL carries counts as the operand too -- but only
# a size input of the calculator, and only when it is a figure.
def test_size_in_call_params_pins():
    calc, _ = resolve_concrete_volume_calc(
        None, {"length_m": 12, "width_m": 4, "thickness_m": 0.3},
        "Raft concrete with the documented waste factor",
    )
    assert calc == "concrete_volume"


def test_bare_stem_size_in_call_params_pins():
    calc, _ = resolve_concrete_volume_calc(
        None, {"thickness": 0.25}, "Footing concrete, include waste",
    )
    assert calc == "concrete_volume"


def test_non_size_params_do_not_pin():
    calc, _ = resolve_concrete_volume_calc(
        None, {"waste_factor": 0.05, "quantity": 3, "crew": 6},
        "Footing concrete, include waste",
    )
    assert calc is None


def test_zero_size_param_does_not_pin():
    calc, _ = resolve_concrete_volume_calc(
        None, {"length_m": 0, "width_m": ""}, "Slab concrete cycle per floor",
    )
    assert calc is None


# Decision 3: a call that names concrete_volume keeps it with or without a
# size; the earthwork alias is released when nothing is sized.
def test_named_concrete_volume_stays_without_size():
    calc, _ = resolve_concrete_volume_calc(
        "concrete_volume", {}, "Slab concrete, how much?",
    )
    assert calc == "concrete_volume"


def test_earthwork_alias_without_size_is_not_pinned():
    calc, _ = resolve_concrete_volume_calc(
        "excavation_volume", {}, "Slab concrete, how much?",
    )
    assert calc == "excavation_volume"


# Decision 4 (through the real entry point): an ask that writes another
# calculator's name and a concrete word reaches that calculator, not a
# request for a slab length.
def test_named_ask_with_slab_word_reaches_its_calculator():
    r = run_calculation(None, {
        "text": "Live load reduction for a slab: tributary area 60 m2, "
                "base live load 5 kN/m2, how much can I reduce?",
    })
    assert r.get("calculation") == "live_load_reduction"
    assert "length_m" not in (r.get("missing") or [])


def test_sized_slab_ask_still_reaches_concrete_volume():
    r = run_calculation(None, {"text": "Slab concrete 8 m x 6 m x 0.2 m, how many m3?"})
    assert r.get("status") == "success"
    assert r.get("calculation") == "concrete_volume"
