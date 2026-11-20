"""Exam 8 -- chain defects, classes X1 to X9 (X8 and X9 are PRODUCT's).

The exam ordered on 15 Nov (SPEC_EXAM_PLAN_15NOV.md, row 8): two CI cases per
class here, one live case per class run by TESTER. Each case constructs its
own synthetic input and drives the general mechanism that owns the rule; no
case names a project, document, row or question from any corpus.

The classes (15 Nov orders, section 09:19Z and 18:03Z):

* X1 user inputs overridden -- values the user states in the request are the
  inputs; a document may fill an input the user did not give.
* X2 calculation refused as a document lookup -- an ask whose operands are
  all stated and that matches a registered calculator is a calculation.
* X3 self-contradicting answer -- a stated result must agree with the
  answer's own working.
* X4 a chain loses everything when one step fails -- every step that can be
  computed is shown, and the failing step is named with what it needs.
* X5 asks for an input the user already gave -- an input stated in the
  request is bound before anything is asked.
* X6 an unrequested default changed the headline -- a documented default is
  applied only where it belongs and is stated; an undocumented one never
  changes a figure.
* X7 refuses to commit to a figure it has just derived -- a figure computed
  from the user's own inputs is stated as the answer.

X8 (silent conflict) and X9 (precedence among document kinds) moved to
PRODUCT on 15 Nov and are not exam 8's here.
"""
from __future__ import annotations

import pytest

from app.agents import runtime
from app.agents.core.computed_answer import Call, answer_turn
from app.agents.core.tool_registry import Statement
from app.lib import construction_formulas as cf
from app.lib import construction_formulas_quantities as quantities
from app.lib.formula_registry import all_specs

_SPECS = {spec.name: spec for spec in all_specs()}
DELAY = _SPECS["delay_damages_daily"]
CONCRETE = _SPECS["concrete_volume"]
AVAILABLE = {"construction_calc"}


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def _numbered() -> Call:
    """A call that returned one figure with its own working."""
    return Call(
        tool="construction_calc",
        heading="Concrete volume",
        statements=(
            Statement(
                figure=210.0,
                what="volume",
                unit="m3",
                working="Net = 40 x 10 x 0.5 = 200; plus 5% = 210",
            ),
        ),
        asked={"length_m": 40.0, "width_m": 10.0, "thickness_m": 0.5},
    )


# --------------------------------------------------------------------- X1
# Values the user states are the inputs.


def test_x1a_the_typed_contract_amount_and_rate_are_the_operands():
    ask = (
        "Calculate delay damages per day: contract amount 37,500,000, "
        "rate 0.08% per day."
    )
    params = cf.extract_calculation_params_from_text(DELAY.fn, ask)
    assert params["contract_amount"] == pytest.approx(37_500_000.0)
    assert params["rate_percent"] == pytest.approx(0.08)
    # Every required operand came from the request: nothing is left to ask.
    assert cf._missing_required(DELAY.fn, params) == []


def test_x1b_the_typed_dimensions_are_the_operands():
    ask = "Concrete volume for a slab 14 m long, 6 m wide and 0.4 m thick."
    params = cf.extract_calculation_params_from_text(CONCRETE.fn, ask)
    assert params["length_m"] == pytest.approx(14.0)
    assert params["width_m"] == pytest.approx(6.0)
    assert params["thickness_m"] == pytest.approx(0.4)


# --------------------------------------------------------------------- X2
# A fully specified calculation is a calculation, not a document lookup.


def test_x2a_a_fully_specified_calculation_forces_the_calculator():
    ask = "Concrete volume for a slab 14 m long, 6 m wide and 0.4 m thick."
    assert runtime._forced_specific_tool(_user(ask), AVAILABLE) == "construction_calc"


def test_x2b_a_document_question_without_operands_is_not_stolen():
    # The same nouns, no operands: this is a lookup of the contract's values.
    ask = "What are the delay damages for late completion under this contract?"
    assert runtime._forced_specific_tool(_user(ask), AVAILABLE) is None


# --------------------------------------------------------------------- X3
# A stated result agrees with the answer's own working.


def test_x3a_a_stated_result_that_disagrees_with_its_working_is_found():
    assert runtime.derivation_mismatches("Total = 12 x 4 = 50") == [
        ("12 x 4", 50.0, 48.0)
    ]


def test_x3b_a_stated_result_that_agrees_with_its_working_is_not_flagged():
    assert runtime.derivation_mismatches("Total = 12 x 4 = 48") == []


# --------------------------------------------------------------------- X4
# One failed step never takes the computed steps with it.


def test_x4a_the_computed_step_is_shown_and_the_failed_one_is_named():
    failed = Call(
        tool="generate_wbs",
        heading="Schedule",
        error="contract_value_required",
        asked={"days": 7},
    )
    answer = answer_turn([_numbered(), failed], "en")
    assert "210" in answer
    assert "Schedule" in answer
    assert "did not return" in answer


def test_x4b_when_every_step_fails_every_step_is_named():
    first = Call(tool="generate_wbs", heading="Schedule", error="provider_unavailable",
                 asked={"days": 7})
    second = Call(tool="cash_flow_forecast", heading="Cash flow",
                  error="provider_unavailable", asked={"months": 6})
    answer = answer_turn([first, second], "en")
    assert "Schedule" in answer
    assert "Cash flow" in answer
    assert "did not return" in answer


# --------------------------------------------------------------------- X5
# An input the user gave is bound before anything is asked.


def test_x5a_a_fully_supplied_calculation_answers_without_asking():
    ask = (
        "Calculate delay damages per day: contract amount 37,500,000, "
        "rate 0.08% per day."
    )
    params = cf.extract_calculation_params_from_text(DELAY.fn, ask)
    assert cf._missing_required(DELAY.fn, params) == []
    result = cf.run_calculation("delay_damages_daily", params)
    assert result["status"] == "success"
    # 37,500,000 x 0.08% = 30,000 per day; the case computes its own figure.
    assert result["result"]["daily_amount"] == pytest.approx(30_000.0)


def test_x5b_only_the_absent_input_is_asked_for():
    ask = "Calculate delay damages per day: contract amount 37,500,000."
    token = cf.set_current_ask(ask)
    try:
        result = cf.run_calculation(
            "delay_damages_daily", {"contract_amount": 37_500_000.0}
        )
    finally:
        cf._CURRENT_ASK.reset(token)
    assert result["status"] == "needs_input"
    assert result["missing"] == ["rate_percent"]
    # The operand the request supplied is not asked for again.
    assert "contract_amount" not in result["missing"]


# --------------------------------------------------------------------- X6
# Only a documented default, in its own place, may change the headline.


def test_x6a_a_documented_waste_allowance_is_applied_and_stated():
    ask = "Concrete volume for a raft 40 x 10 x 0.5 m."
    computed = quantities.compose_concrete_volume_from_ask(ask)
    assert computed is not None
    # Net 200 m3; the documented 5% allowance gives 210 m3, and the line says so.
    assert computed["net_volume_m3"] == pytest.approx(200.0)
    assert computed["volume_m3"] == pytest.approx(210.0)
    assert "5%" in computed["line"]


def test_x6b_an_earthwork_volume_gets_no_concrete_allowance():
    ask = "Bank volume of a rectangular trench 9 m long, 2 m wide and 1.5 m deep."
    assert quantities.compose_concrete_volume_from_ask(ask) is None


# --------------------------------------------------------------------- X7
# A figure derived from the user's own inputs is stated as the answer.


def test_x7a_a_computed_figure_is_stated_on_the_first_line():
    answer = answer_turn([_numbered()], "en")
    opening = "\n".join([line for line in answer.splitlines() if line.strip()][:2])
    assert "210" in opening
    assert "unable to generate" not in answer.lower()


def test_x7b_two_computed_figures_are_both_stated():
    other = Call(
        tool="construction_calc",
        heading="Excavation volume",
        statements=(Statement(figure=27.0, what="volume", unit="m3"),),
        asked={"length_m": 9.0, "width_m": 2.0, "depth_m": 1.5},
    )
    answer = answer_turn([_numbered(), other], "en")
    assert "210" in answer
    assert "27" in answer
    assert "not answered" not in answer.lower()


# ------------------------------------------------------- the exam of the exam
# SPEC_EXAM_PLAN rule 3: one known-good input must pass and one known-bad
# input must fail, or the exam is broken.


def _audit(answer: str) -> list[str]:
    """What this exam would fail in one answer: a dropped computed figure or
    an arithmetic line that disagrees with itself."""
    problems: list[str] = []
    if "210" not in answer:
        problems.append("computed figure dropped")
    if runtime.derivation_mismatches(answer):
        problems.append("arithmetic disagrees with itself")
    return problems


def test_the_exam_passes_a_known_good_answer():
    good = answer_turn([_numbered()], "en")
    assert _audit(good) == []


def test_the_exam_fails_a_known_bad_answer():
    bad = "The volume is 200 m3.\n\nTotal = 12 x 4 = 50"
    assert _audit(bad) != []
