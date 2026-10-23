"""Calculator answers state the calculation, and only the calculation.

Synthetic questions only. Two mechanisms:

* A formula-style ask that names no registry calculator is not pre-dispatched
  with ``calculation=None``: that returned "Unknown calculation 'None'" plus
  an instruction to "answer from that result", and the model filled the gap
  with free-form variants that differed run to run.
* When construction_calc produced the result, a figure in the result's unit
  that neither the result nor the operator's numbers ground (an unrequested
  variant, a limit recalled from memory) is removed with its clause.
"""
from __future__ import annotations

import asyncio
import json

from app.agents import runtime


class _Agent:
    allowed_blocks = ("construction",)

    def __init__(self):
        self.calls = []

    async def _run_tool_call(self, tc):
        self.calls.append(tc)
        return {"ok": True, "result": {}}


def test_a_formula_ask_naming_no_calculator_is_not_told_to_answer_from_an_error():
    ask = "How many hours to lay 600 m of kerb with 3 crews each placing 25 m an hour?"
    assert runtime._formula_calculator_name_from_message(ask) is None

    class _Unknown(_Agent):
        async def _run_tool_call(self, tc):
            self.calls.append(tc)
            return {"ok": False, "result": {"error": "Unknown calculation 'None'."}}

    agent = _Unknown()
    messages = [{"role": "user", "content": ask}]

    asyncio.run(runtime._predispatch_formula_calc(agent, messages, None))

    bubble = messages[-1]["content"]
    assert "Answer from that result" not in bubble
    assert "no calculator result" in bubble and "single result" in bubble
    assert "Master Corpus" in bubble  # the anti-bleed instruction stays


def _calc_turn(ask: str, result: dict) -> list:
    return [
        {"role": "user", "content": ask},
        {"role": "user", "content": (
            f"{runtime._PREDISPATCH_PREFIX} construction_calc has ALREADY been run "
            f"from the operator facts in this turn. Authoritative draft:\n{json.dumps(result)}\n"
            "construction_calc has already been run. Answer from that result."
        )},
    ]


def test_a_calculator_answer_drops_figures_the_calculation_did_not_produce():
    ask = "What is the minimum depth of a simply supported timber joist spanning 3.6 m?"
    messages = _calc_turn(ask, {"calculation": "joist_depth_min", "min_depth_mm": 180.0,
                                "note": "L/20 = 180.0 mm"})
    answer = (
        "**Minimum depth: 180 mm.**\n\n"
        "L = 3.6 m = 3600 mm, so h = 3600 / 20 = 180 mm "
        "(well above the 150 mm floor most suppliers stock).\n"
        "If you allow a 10% margin, use 198 mm. Check deflection separately."
    )

    out = runtime._calc_figure_grounding_gate(answer, messages)

    assert "180 mm" in out and "3600 mm" in out  # result and unit rescale stay
    assert "150 mm" not in out and "198 mm" not in out
    assert "Check deflection separately." in out


def test_working_from_the_operators_numbers_is_kept():
    """Arithmetic on the operator's and the calculator's numbers is grounded."""
    ask = "How much concrete for 6 pads, each 2.0 m by 1.5 m by 0.6 m?"
    messages = _calc_turn(ask, {"calculation": "concrete_volume", "volume_m3": 10.8})
    answer = (
        "**Total: 10.8 m3.**\n"
        "Each pad: 2.0 x 1.5 x 0.6 = 1.8 m3; 6 pads x 1.8 m3 = 10.8 m3.\n"
        "With a typical 5% waste you would order 11.34 m3."
    )

    out = runtime._calc_figure_grounding_gate(answer, messages)

    assert "1.8 m3" in out and "10.8 m3" in out
    assert "11.34 m3" not in out  # an unrequested waste variant


def test_without_a_calculator_result_the_answer_is_untouched():
    ask = "What is the minimum depth of a simply supported timber joist spanning 3.6 m?"
    answer = "Minimum depth is about 180 mm (150 mm is the usual stock floor)."
    assert runtime._calc_figure_grounding_gate(answer, [{"role": "user", "content": ask}]) == answer
