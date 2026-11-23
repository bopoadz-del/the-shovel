"""A word written right before a figure names the input it is for.

"live 4.0 kN/m2, dead 5.0 kN/m2" carries two figures in one unit. The unit
alone cannot place them, and "live" is neither a whole label nor a symbol,
so the typed live load was reported missing although the user wrote it.
The word before a figure now names an input when it is a word of exactly
one open input's labels whose unit measures what the figure's unit
measures.

Every input here is built in this file; no value is taken from a product
answer. Each decision is checked on two inputs.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import CALCULATORS, extract_calculation_params_from_text


def _crane(crane_capacity_t: float, hook_weight_t: float) -> float:
    return crane_capacity_t + hook_weight_t


def _cover(top_cover_mm: float, bottom_cover_mm: float) -> float:
    return top_cover_mm + bottom_cover_mm


def _loads(load_at_rest_kn: float, load_in_motion_kn: float, other_kn: float) -> float:
    return load_at_rest_kn + load_in_motion_kn + other_kn


def _members(steel_t: float, precast_beam_t: float) -> float:
    return steel_t + precast_beam_t


# Decision 1: a label word before the figure binds it, among figures that
# share one unit.
@pytest.mark.parametrize("text, live", [
    ("4 storeys, 45 m2 tributary per column, live 3.5 kN/m2, dead 6.0 kN/m2. "
     "Reduce the live load (KLL 4).", 3.5),
    ("Interior column, tributary 80 m2, finishes 1.5 kN/m2, live 2.5 kN/m2, "
     "KLL 4. Reduced live load?", 2.5),
])
def test_label_word_binds_the_typed_figure(text, live):
    found = extract_calculation_params_from_text(CALCULATORS["live_load_reduction"], text)
    assert found.get("base_live_load_kn_m2") == pytest.approx(live)


# Decision 2: a word two inputs of one kind share names neither.
@pytest.mark.parametrize("text, expected", [
    ("cover 40 mm", {}),
    ("top 40 mm and bottom 55 mm", {"top_cover_mm": 40.0, "bottom_cover_mm": 55.0}),
])
def test_word_shared_by_two_inputs_names_neither(text, expected):
    found = extract_calculation_params_from_text(_cover, text)
    assert {k: found[k] for k in ("top_cover_mm", "bottom_cover_mm") if k in found} == expected


# Decision 3: the figure's unit must measure what the input measures.
@pytest.mark.parametrize("text, crane", [
    ("crane hired for 6 months, hook 1.5 t", None),
    ("crane 25 t, hook 1.5 t", 25.0),
])
def test_word_binds_only_a_figure_of_the_inputs_kind(text, crane):
    found = extract_calculation_params_from_text(_crane, text)
    assert found.get("crane_capacity_t") == crane
    assert found.get("hook_weight_t") == pytest.approx(1.5)


# Decision 4: an input named for two figures binds neither.
@pytest.mark.parametrize("text, live", [
    ("live 4.0 kN/m2 on the floors, live 1.5 kN/m2 on the roof, "
     "tributary 60 m2, dead 5.0 kN/m2", None),
    ("live 4.0 kN/m2 on the floors, tributary 60 m2, dead 5.0 kN/m2", 4.0),
])
def test_input_named_twice_binds_neither(text, live):
    found = extract_calculation_params_from_text(CALCULATORS["live_load_reduction"], text)
    assert found.get("base_live_load_kn_m2") == live


# Decision 5: a word shorter than three letters names nothing.
@pytest.mark.parametrize("text, at_rest", [
    ("at 5 kN, other 7 kN", None),
    ("rest 5 kN, other 7 kN", 5.0),
])
def test_short_word_names_nothing(text, at_rest):
    found = extract_calculation_params_from_text(_loads, text)
    assert found.get("load_at_rest_kn") == at_rest
    assert found.get("other_kn") == pytest.approx(7.0)


# Decision 6: a figure another rule already read is not read again, and an
# input already read keeps its figure.
@pytest.mark.parametrize("text, expected", [
    ("beam 14 t steel", {"steel_t": 14.0}),
    ("beam 9 t, 14 t steel", {"steel_t": 14.0, "precast_beam_t": 9.0}),
])
def test_a_figure_already_read_is_not_read_again(text, expected):
    found = extract_calculation_params_from_text(_members, text)
    assert {k: found[k] for k in ("steel_t", "precast_beam_t") if k in found} == expected


@pytest.mark.parametrize("text, live", [
    ("unreduced live load 4.0 kN/m2, live 3.0 kN/m2 elsewhere, tributary 60 m2", 4.0),
    ("unreduced live load 2.5 kN/m2, live 5.0 kN/m2 elsewhere, tributary 60 m2", 2.5),
])
def test_an_input_already_read_keeps_its_figure(text, live):
    found = extract_calculation_params_from_text(CALCULATORS["live_load_reduction"], text)
    assert found.get("base_live_load_kn_m2") == pytest.approx(live)
