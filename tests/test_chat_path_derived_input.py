"""A calculator input the ask states as a product of its own figures.

Chain class "multi-step: derived input" (exam 6). The ask sizes an element
(L x W x T), gives a figure per cubic metre ("25 kN/m3") and may say how
many times the element repeats ("5 floors, each slab ..."). The calculator
wants the product (a weight in kN). Before this, the input read found no
figure written in kN and the turn stopped on "missing seismic weight".

Stage boundary: driven through the real ``Agent.chat`` with the model
stubbed (tests/_chat_path_harness.py), so routing, the input read and the
derivation below are deterministic code. Every expected figure is worked by
hand from the ask in the comment beside it.

The asks write "R = 5": a one-letter label is read only with "=", ":",
"is" or "of" after it (extract_calculation_params_from_text), so "R 5"
is not read and the calculator's default R applies. That is a separate
input-read rule, not this derivation, and is not changed here.
"""
from __future__ import annotations

import pytest

from app.lib import construction_formulas as cf
from tests._chat_path_harness import install_model_stub, run_turns


def _shear_ask(floors: int, dims: str, gamma: int, sds: float, r: int, ie: float) -> str:
    return (
        f"{floors} floors, each slab {dims} m, concrete {gamma} kN/m3, nothing else "
        f"counts as weight. ASCE 7 equivalent lateral force, SDS {sds}, R = {r}, "
        f"Ie {ie}. Base shear?"
    )


def _envelope(out):
    for call in out.get("tool_calls") or []:
        if call.get("name") == "construction_calc":
            return call.get("result") or {}
    return {}


@pytest.mark.parametrize("floors, dims, gamma, sds, r, ie, weight, shear", [
    # 5 x (30 x 20 x 0.25 = 150 m3) x 25 = 18,750 kN; Cs 0.9 x 1.0 / 5 = 0.18; V 3,375 kN
    (5, "30 x 20 x 0.25", 25, 0.9, 5, 1.0, 18_750.0, 3_375.0),
    # 4 x (20 x 15 x 0.2 = 60 m3) x 24 = 5,760 kN; Cs 0.6 x 1.25 / 6 = 0.125; V 720 kN
    (4, "20 x 15 x 0.2", 24, 0.6, 6, 1.25, 5_760.0, 720.0),
])
async def test_weight_written_as_slabs_times_unit_weight_reaches_base_shear(
    monkeypatch, floors, dims, gamma, sds, r, ie, weight, shear,
):
    install_model_stub(monkeypatch)
    (out,) = await run_turns([_shear_ask(floors, dims, gamma, sds, r, ie)])
    env = _envelope(out)
    assert env.get("status") == "success", env
    assert env.get("calculation") == "seismic_base_shear", env
    assert env["result"]["base_shear_kn"] == pytest.approx(shear), env
    (row,) = env["derived_inputs"]
    assert row["parameter"] == "seismic_weight_kn"
    assert row["value"] == pytest.approx(weight)
    assert row["unit"] == "kN"


# --- the derivation itself: what it takes, and when it takes nothing -------

def _fn():
    return cf._build_calculator_registry()["seismic_base_shear"]


_MISSING = ["seismic_weight_kn"]


@pytest.mark.parametrize("ask, weight", [
    ("slab 30 x 20 x 0.25 m, concrete 25 kN/m3", 3_750.0),   # 150 m3 x 25, no repeat
    ("deck 10 x 8 x 0.5 m in 24 kN/m³ concrete", 960.0),      # 40 m3 x 24, no repeat
    ("3 levels, each slab 10 x 10 x 0.2 m, 25 kN/m3", 1_500.0),  # 3 x 20 m3 x 25
    ("2 decks each 12000 x 5000 x 300 mm, 24 kN/m3", 864.0),     # 2 x 18 m3 x 24 (mm -> m)
])
def test_derives_the_product_in_the_missing_input_unit(ask, weight):
    params, rows = cf._derive_missing_by_unit(_fn(), {"sds": 1.0}, _MISSING, ask)
    assert params["seismic_weight_kn"] == pytest.approx(weight)
    assert params["sds"] == 1.0
    assert [r["parameter"] for r in rows] == ["seismic_weight_kn"]


@pytest.mark.parametrize("missing, ask", [
    # nothing missing / no ask words
    ([], "slab 30 x 20 x 0.25 m, concrete 25 kN/m3"),
    (_MISSING, ""),
    # no per-m3 figure / two of them
    (_MISSING, "slab 30 x 20 x 0.25 m, concrete weight unknown"),
    (_MISSING, "slab 30 x 20 x 0.25 m, 25 kN/m3 or 24 kN/m3"),
    # no size chain / two of them
    (_MISSING, "a slab of 25 kN/m3 concrete"),
    (_MISSING, "slab 30 x 20 x 0.25 m and 10 x 10 x 0.2 m, 25 kN/m3"),
    # two repeat counts: which one multiplies is not written
    (_MISSING, "5 floors, each slab 30 x 20 x 0.25 m; 2 bays each; 25 kN/m3"),
    # the per-m3 unit is not the missing input's unit
    (_MISSING, "slab 30 x 20 x 0.25 m, concrete 2400 kg/m3"),
    (["sds"], "slab 30 x 20 x 0.25 m, concrete 25 kN/m3"),
])
def test_derives_nothing_unless_one_reading(missing, ask):
    params, rows = cf._derive_missing_by_unit(_fn(), {"sds": 1.0}, missing, ask)
    assert rows == []
    assert params == {"sds": 1.0}


def test_two_missing_inputs_in_that_unit_take_nothing(monkeypatch):
    monkeypatch.setattr(cf, "_declared_units", lambda fn: {"a_kn": "kN", "b_kn": "kN"})
    params, rows = cf._derive_missing_by_unit(
        _fn(), {}, ["a_kn", "b_kn"], "slab 30 x 20 x 0.25 m, 25 kN/m3")
    assert (params, rows) == ({}, [])
    params, rows = cf._derive_missing_by_unit(
        _fn(), {}, ["a_kn"], "slab 30 x 20 x 0.25 m, 25 kN/m3")
    assert params == {"a_kn": pytest.approx(3_750.0)}


def test_run_without_a_derivable_weight_still_asks_for_it():
    # The ask gives no unit weight: the weight is asked, no value assumed.
    token = cf.set_current_ask("5 floors, each slab 30 x 20 x 0.25 m. SDS 0.9, R 5, Ie 1.0")
    try:
        env = cf.run_calculation("seismic_base_shear", {"sds": 0.9, "r": 5, "ie": 1.0})
    finally:
        cf._CURRENT_ASK.reset(token)
    assert env.get("status") != "success", env
    assert "derived_inputs" not in env
    token = cf.set_current_ask("5 floors, each slab 30 x 20 x 0.25 m, 25 kN/m3. SDS 0.9, R 5, Ie 1.0")
    try:
        env = cf.run_calculation("seismic_base_shear", {"sds": 0.9, "r": 5, "ie": 1.0})
    finally:
        cf._CURRENT_ASK.reset(token)
    assert env["result"]["base_shear_kn"] == pytest.approx(3_375.0), env
