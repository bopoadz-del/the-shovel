"""Chain scenarios driven from the user message, not from the calculator.

The calculator-layer tests (tests/test_formula_mixed_chains_*.py) call
``run_calculation`` with inputs already named. Those prove the arithmetic
and nothing above it. This file drives the same kind of ask through the
real ``Agent.chat`` with the model stubbed (tests/_chat_path_harness.py),
and asserts what the deterministic code does with it:

  routing     - WHICH calculator is reached,
  input read  - WHICH named inputs it receives (value, unit, role),
  result      - the figure the calculator returns for those inputs.

Every expected figure is a hand value worked from the ask in the comment
beside it, never copied from a product answer. Each decision has two
inputs, so a rule that keys on one literal number cannot pass both.
"""
from __future__ import annotations

import math

import pytest

from tests._chat_path_harness import (
    call_arguments,
    install_model_stub,
    reached,
    run_turns,
)


# --- Grout for post-tensioning ducts: duct bore and pressure read ------------

def _duct_ask(bore_mm: int, pressure: float) -> str:
    return (
        f"6 ducts of {bore_mm} mm, 45 m each, grout pressure {pressure} N/mm2; "
        "my pump gauge is in bar. Grout needed with 10 % waste, in m3?"
    )


@pytest.mark.parametrize("bore_mm, pressure, litres_per_m", [
    # pi * (0.100 / 2)^2 m2 * 1000 L/m3 = 7.854 L per metre of duct
    (100, 1.2, 7.854),
    # pi * (0.080 / 2)^2 m2 * 1000 L/m3 = 5.027 L per metre of duct
    (80, 0.8, 5.027),
])
async def test_duct_grout_ask_reaches_grout_calculator_with_read_inputs(
    monkeypatch, bore_mm, pressure, litres_per_m,
):
    install_model_stub(monkeypatch)
    (out,) = await run_turns([_duct_ask(bore_mm, pressure)])
    name, result = reached(out)
    assert name == "grout_pressure_calc", out.get("tool_calls")
    # Input read: the bore and the pressure the user typed, in their units.
    assert result.get("duct_area_mm2") == pytest.approx(
        math.pi * (bore_mm / 2) ** 2, rel=1e-4,
    ), result
    assert result.get("pressure_n_mm2") == pytest.approx(pressure), result
    # Result: litres of grout per metre of duct.
    assert result.get("grout_volume_l_m") == pytest.approx(
        litres_per_m, abs=1e-3,
    ), result


# --- Raft volume inside a crew-sizing ask: the three dimensions read --------

def _raft_ask(length: int, width: int, depth: float, days: int) -> str:
    return (
        f"Raft {length} m x {width} m x {depth} m. "
        f"How many crews to pour it in {days} days?"
    )


@pytest.mark.parametrize("length, width, depth, days, net_m3", [
    (40, 20, 0.8, 10, 640.0),  # 40 * 20 * 0.8
    (30, 12, 0.5, 6, 180.0),   # 30 * 12 * 0.5
])
async def test_raft_crew_ask_reaches_volume_with_the_typed_dimensions(
    monkeypatch, length, width, depth, days, net_m3,
):
    install_model_stub(monkeypatch)
    (out,) = await run_turns([_raft_ask(length, width, depth, days)])
    name, result = reached(out)
    assert name == "concrete_volume", out.get("tool_calls")
    # The net volume is the product of the three dimensions the user typed.
    assert result.get("net_volume_m3") == pytest.approx(net_m3), result
    params = call_arguments(out)
    assert params.get("text"), params  # the ask itself reaches the calculator
