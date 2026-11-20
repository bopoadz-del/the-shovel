"""Exam 8 -- chain defects, classes X1 to X9 (X8 and X9 are PRODUCT's).

The exam ordered on 15 Nov (SPEC_EXAM_PLAN_15NOV.md, row 8): two CI cases per
class here, one live case per class run by TESTER. Each case constructs its
own input and drives the general mechanism that owns the rule; no case names
a project, document, row or question from any corpus.

The classes (15 Nov orders, sections 09:19Z and 18:03Z):

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


def _numbered(length: float, width: float, thickness: float) -> Call:
    """A call that returned one figure, with its own working built from the
    same three dimensions the case supplied."""
    net = length * width * thickness
    return Call(
        tool="construction_calc",
        heading="Concrete volume",
        statements=(
            Statement(
                figure=round(net * 1.05, 6),
                what="volume",
                unit="m3",
                working=(
                    f"Net = {length:g} x {width:g} x {thickness:g} = {net:g}; "
                    "plus 5% allowance"
                ),
            ),
        ),
        asked={"length_m": length, "width_m": width, "thickness_m": thickness},
    )


_SLABS = [(40.0, 10.0, 0.5), (12.5, 6.0, 0.3)]


def _concrete_ask(length: float, width: float, thickness: float) -> str:
    return (
        f"Concrete volume for a slab {length:g} m long, {width:g} m wide "
        f"and {thickness:g} m thick."
    )


# --------------------------------------------------------------------- X1
# Values the user states are the inputs.


@pytest.mark.parametrize("amount,rate", [(37_500_000.0, 0.08), (6_400_000.0, 0.25)])
def test_x1a_the_typed_contract_amount_and_rate_are_the_operands(amount, rate):
    ask = (
        f"Calculate delay damages per day: contract amount {amount:,.0f}, "
        f"rate {rate:g}% per day."
    )
    params = cf.extract_calculation_params_from_text(DELAY.fn, ask)
    assert params["contract_amount"] == pytest.approx(amount)
    assert params["rate_percent"] == pytest.approx(rate)
    # Every required operand came from the request: nothing is left to ask.
    assert cf._missing_required(DELAY.fn, params) == []


@pytest.mark.parametrize("dims", _SLABS)
def test_x1b_the_typed_dimensions_are_the_operands(dims):
    length, width, thickness = dims
    params = cf.extract_calculation_params_from_text(
        CONCRETE.fn, _concrete_ask(length, width, thickness)
    )
    assert params["length_m"] == pytest.approx(length)
    assert params["width_m"] == pytest.approx(width)
    assert params["thickness_m"] == pytest.approx(thickness)


# --------------------------------------------------------------------- X2
# A fully specified calculation is a calculation, not a document lookup.


@pytest.mark.parametrize(
    "ask",
    [
        "Concrete volume for a slab 14 m long, 6 m wide and 0.4 m thick.",
        "Excavation volume for a trench 9 m long, 2 m wide and 1.5 m deep.",
    ],
)
def test_x2a_a_fully_specified_calculation_forces_the_calculator(ask):
    assert runtime._forced_specific_tool(_user(ask), AVAILABLE) == "construction_calc"


@pytest.mark.parametrize(
    "ask",
    [
        "What are the delay damages for late completion under this contract?",
        "Who is the Engineer under this contract?",
    ],
)
def test_x2b_a_document_question_without_operands_is_not_stolen(ask):
    assert runtime._forced_specific_tool(_user(ask), AVAILABLE) is None


# --------------------------------------------------------------------- X3
# A stated result agrees with the answer's own working.


@pytest.mark.parametrize(
    "line,expression",
    [
        ("Total = 12 x 4 = 50", "12 x 4"),
        ("The run is 7 x 6 = 40 m", "7 x 6"),
    ],
)
def test_x3a_a_stated_result_that_disagrees_with_its_working_is_found(line, expression):
    found = runtime.derivation_mismatches(line)
    assert [(expr, stated) for expr, stated, _ in found] == [
        (expression, float(line.split("=")[-1].split()[0]))
    ]


@pytest.mark.parametrize(
    "line",
    ["Total = 12 x 4 = 48", "The run is 7 x 6 = 42 m"],
)
def test_x3b_a_stated_result_that_agrees_with_its_working_is_not_flagged(line):
    assert runtime.derivation_mismatches(line) == []


# --------------------------------------------------------------------- X4
# One failed step never takes the computed steps with it.


@pytest.mark.parametrize("dims", _SLABS)
def test_x4a_the_computed_step_is_shown_and_the_failed_one_is_named(dims):
    computed = _numbered(*dims)
    failed = Call(
        tool="generate_wbs",
        heading="Schedule",
        error="contract_value_required",
        asked={"days": 7},
    )
    answer = answer_turn([computed, failed], "en")
    figure = computed.statements[0].figure
    assert f"{figure:,.10g}" in answer.replace(",", "")
    assert "Schedule" in answer
    assert "did not return" in answer


@pytest.mark.parametrize(
    "tool,heading",
    [("generate_wbs", "Schedule"), ("cash_flow_forecast", "Cash flow")],
)
def test_x4b_when_every_step_fails_every_step_is_named(tool, heading):
    first = Call(tool="construction_calc", heading="Concrete volume",
                 error="provider_unavailable", asked={"length_m": 4.0})
    second = Call(tool=tool, heading=heading, error="provider_unavailable",
                  asked={"months": 6})
    answer = answer_turn([first, second], "en")
    assert "Concrete volume" in answer
    assert heading in answer
    assert "did not return" in answer


# --------------------------------------------------------------------- X5
# An input the user gave is bound before anything is asked.


@pytest.mark.parametrize("amount,rate", [(37_500_000.0, 0.08), (2_150_000.0, 0.5)])
def test_x5a_a_fully_supplied_calculation_answers_without_asking(amount, rate):
    ask = (
        f"Calculate delay damages per day: contract amount {amount:,.0f}, "
        f"rate {rate:g}% per day."
    )
    params = cf.extract_calculation_params_from_text(DELAY.fn, ask)
    assert cf._missing_required(DELAY.fn, params) == []
    result = cf.run_calculation("delay_damages_daily", params)
    assert result["status"] == "success"
    # The case computes its own expected figure from the operands it supplied.
    assert result["result"]["daily_amount"] == pytest.approx(amount * rate / 100.0)


@pytest.mark.parametrize("amount", [37_500_000.0, 2_150_000.0])
def test_x5b_only_the_absent_input_is_asked_for(amount):
    ask = f"Calculate delay damages per day: contract amount {amount:,.0f}."
    token = cf.set_current_ask(ask)
    try:
        result = cf.run_calculation(
            "delay_damages_daily", {"contract_amount": amount}
        )
    finally:
        cf._CURRENT_ASK.reset(token)
    assert result["status"] == "needs_input"
    assert result["missing"] == ["rate_percent"]
    # The operand the request supplied is not asked for again.
    assert "contract_amount" not in result["missing"]


# --------------------------------------------------------------------- X6
# Only a documented default, in its own place, may change the headline.


@pytest.mark.parametrize("dims", _SLABS)
def test_x6a_a_documented_waste_allowance_is_applied_and_stated(dims):
    length, width, thickness = dims
    net = length * width * thickness
    ask = f"Concrete volume for a raft {length:g} x {width:g} x {thickness:g} m."
    computed = quantities.compose_concrete_volume_from_ask(ask)
    assert computed is not None
    assert computed["net_volume_m3"] == pytest.approx(net)
    assert computed["volume_m3"] == pytest.approx(net * 1.05)
    assert "%" in computed["line"]


@pytest.mark.parametrize(
    "ask",
    [
        "Bank volume of a rectangular trench 9 m long, 2 m wide and 1.5 m deep.",
        "Bank volume of a rectangular trench 14.5 m long by 3.2 m wide by 1.75 m deep.",
    ],
)
def test_x6b_an_earthwork_volume_gets_no_concrete_allowance(ask):
    assert quantities.compose_concrete_volume_from_ask(ask) is None


# --------------------------------------------------------------------- X7
# A figure derived from the user's own inputs is stated as the answer.


@pytest.mark.parametrize("dims", _SLABS)
def test_x7a_a_computed_figure_is_stated_on_the_first_line(dims):
    computed = _numbered(*dims)
    answer = answer_turn([computed], "en")
    opening = "\n".join([line for line in answer.splitlines() if line.strip()][:2])
    assert f"{computed.statements[0].figure:,.10g}".replace(",", "") in opening.replace(",", "")
    assert "unable to generate" not in answer.lower()


@pytest.mark.parametrize("second_figure", [27.0, 96.5])
def test_x7b_two_computed_figures_are_both_stated(second_figure):
    first = _numbered(*_SLABS[0])
    other = Call(
        tool="construction_calc",
        heading="Excavation volume",
        statements=(Statement(figure=second_figure, what="volume", unit="m3"),),
        asked={"length_m": 9.0, "width_m": 2.0, "depth_m": 1.5},
    )
    answer = answer_turn([first, other], "en")
    compact = answer.replace(",", "")
    for figure in (first.statements[0].figure, second_figure):
        assert f"{figure:,.10g}" in compact
    assert "not answered" not in answer.lower()


# ------------------------------------------------------- the exam of the exam
# SPEC_EXAM_PLAN rule 3: one known-good input must pass and one known-bad
# input must fail, or the exam is broken.


def _audit(answer: str, expected_figure: float) -> list[str]:
    """What this exam would fail in one answer: a dropped computed figure or
    an arithmetic line that disagrees with itself."""
    problems: list[str] = []
    if f"{expected_figure:,.10g}" not in answer.replace(",", ""):
        problems.append("computed figure dropped")
    if runtime.derivation_mismatches(answer):
        problems.append("arithmetic disagrees with itself")
    return problems


@pytest.mark.parametrize("dims", _SLABS)
def test_the_exam_passes_a_known_good_answer(dims):
    computed = _numbered(*dims)
    good = answer_turn([computed], "en")
    assert _audit(good, computed.statements[0].figure) == []


@pytest.mark.parametrize(
    "bad_figure,bad_line",
    [(200.0, "Total = 12 x 4 = 50"), (61.5, "The run is 7 x 6 = 40 m")],
)
def test_the_exam_fails_a_known_bad_answer(bad_figure, bad_line):
    good_figure = 210.0
    bad = f"The volume is {bad_figure:g} m3.\n\n{bad_line}"
    assert _audit(bad, good_figure) != []
