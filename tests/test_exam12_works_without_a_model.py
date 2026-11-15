"""EXAM 12 -- works without a model (SPEC_EXAM_PLAN_15NOV, exam 12).

Three things are examined, each through a whole POST /v1/chat/stream turn:

  A. With NO model configured, a registered formula answers the question.
  B. With a model configured but unreachable (both providers fail, several
     failure kinds), a calculator result already computed survives.
  C. Otherwise the user gets the honest "model unavailable" message, with no
     plumbing (provider names, status codes, key words) in it.

Rules of the exam plan applied here:
  * Expected figures are hand calculations written in this file
    (length x width x depth; the concrete calculator's documented 5% waste),
    never copied from the product's output.
  * Each grader names the STAGE that failed, so a red run says where.
  * The graders are themselves tested: one invented turn that must pass and
    invented turns that must fail, each at its named stage.
  * The questions are invented; product code names none of them.
"""
from __future__ import annotations

import pytest

# The no-model fixture, the provider-failure fixture and the turn driver are
# shared with the BUILD 1 tests rather than rebuilt here.
from tests.test_a_provider_outage_is_survivable import providers  # noqa: F401 -- pytest fixture
from tests.test_calculations_survive_without_a_model import _ask, no_model  # noqa: F401 -- fixture

#: The one line that replaces the model's explanation.
NOTE = "the language model could not be reached"

#: Words that must never reach the user when the model is unavailable.
PLUMBING = ("api_key", "deepseek", "openrouter", "configured", "http 4", "http 5")

#: Signs that the raw tool record or a model-addressed instruction leaked.
RAW = ('"status"', "Answer from that result")


# -- the graders ---------------------------------------------------------------

def grade_calculation_turn(shown: str, errors: list[str], figure: str,
                           provider_calls: list[str] | None = None) -> str | None:
    """None when the turn passes; otherwise the first stage that failed.

    Stages in order: answered (no error event) -> figure (the hand-computed
    figure is shown) -> note_after_figure (the unavailable line is present and
    comes after the figure) -> no_raw_record -> offline (no provider call when
    no model is configured; skipped when ``provider_calls`` is None).
    """
    if errors:
        return "answered"
    if figure not in shown:
        return "figure"
    low = shown.lower()
    if NOTE not in low or low.index(NOTE) < shown.index(figure):
        return "note_after_figure"
    if any(r in shown for r in RAW):
        return "no_raw_record"
    if provider_calls:
        return "offline"
    return None


def grade_honest_turn(shown: str, errors: list[str], message: str) -> str | None:
    """None when the user got the honest unavailable message and no plumbing."""
    said = (shown + " " + " ".join(errors)).strip()
    if message not in said:
        return "honest_message"
    if any(p in said.lower() for p in PLUMBING):
        return "no_plumbing"
    return None


# -- the exam of the exam: each grader must pass a good turn and fail bad ones --

_GOOD_CALC = "21.00 m3 = 7 x 2.5 x 1.2\nThe explanation is unavailable because the language model could not be reached."


@pytest.mark.parametrize("shown,errors,calls,want", [
    (_GOOD_CALC, [], [], None),
    (_GOOD_CALC.replace("could not be reached", "could not be reached (again)"), [], None, None),
    ("", ["stream failed"], [], "answered"),
    ("The volume is about twenty cubic metres.\n" + NOTE, [], [], "figure"),
    (NOTE + "\n21.00 m3", [], [], "note_after_figure"),
    ("21.00 m3", [], [], "note_after_figure"),
    ('{"status": "success", "result": 21.00}\n' + NOTE, [], [], "no_raw_record"),
    (_GOOD_CALC, [], ["https://provider.invalid/v1/chat"], "offline"),
], ids=["good", "good-no-offline-check", "error-event", "no-figure", "note-first",
        "note-missing", "raw-record", "called-a-provider"])
def test_the_calculation_grader_passes_good_and_fails_bad_at_its_stage(shown, errors, calls, want):
    assert grade_calculation_turn(shown, errors, "21.00", calls) == want


_HONEST = "The assistant is unavailable right now. Please try again shortly."


@pytest.mark.parametrize("shown,errors,want", [
    (_HONEST, [], None),
    ("", [_HONEST], None),
    ("Something went wrong.", [], "honest_message"),
    (_HONEST + " (No DEEPSEEK_API_KEY configured)", [], "no_plumbing"),
], ids=["good-in-text", "good-in-error", "vague", "plumbing-leak"])
def test_the_honest_grader_passes_good_and_fails_bad_at_its_stage(shown, errors, want):
    assert grade_honest_turn(shown, errors, _HONEST) == want


# -- A: no model configured, a registered formula answers ------------------------

@pytest.mark.parametrize("message,figure", [
    ("Calculate the excavation volume for a trench 7 m long, 2.5 m wide and 1.2 m deep.",
     f"{7 * 2.5 * 1.2:.2f}"),
    # The concrete calculator documents a 5% waste allowance.
    ("Calculate the concrete volume for a slab 9 m long, 4 m wide and 0.3 m thick.",
     f"{9 * 4 * 0.3 * 1.05:g}"),
], ids=["A-excavation", "A-concrete"])
def test_exam12_a_registered_formula_answers_with_no_model(no_model, message, figure):  # noqa: F811
    shown, errors = _ask(message)
    stage = grade_calculation_turn(shown, errors, figure, no_model)
    assert stage is None, f"EXAM 12 A failed at stage '{stage}': {shown!r} {errors!r}"


# -- B: a model is configured but every provider fails --------------------------

@pytest.mark.parametrize("primary,fallback,message,figure", [
    ("refused", "500",
     "Calculate the excavation volume for a trench 6 m long, 2 m wide and 1.5 m deep.",
     f"{6 * 2 * 1.5:.2f}"),
    ("timeout", "402",
     "Calculate the concrete volume for a slab 8 m long, 5 m wide and 0.15 m thick.",
     f"{8 * 5 * 0.15 * 1.05:g}"),
], ids=["B-refused+500/excavation", "B-timeout+402/concrete"])
def test_exam12_a_calculator_result_survives_any_provider_failure(
        providers, primary, fallback, message, figure):  # noqa: F811
    providers(primary, fallback)
    shown, errors = _ask(message)
    stage = grade_calculation_turn(shown, errors, figure)
    assert stage is None, f"EXAM 12 B failed at stage '{stage}': {shown!r} {errors!r}"


# -- C: nothing to compute, no model: an honest message --------------------------

@pytest.mark.parametrize("message", [
    "Summarise the purpose of a site induction in two sentences.",
    "What should a good handover meeting between shifts cover?",
], ids=["C-induction", "C-handover"])
def test_exam12_otherwise_an_honest_unavailable_message(no_model, message):  # noqa: F811
    from app.routers.chat_watchdog import PROVIDER_FAILURE_MESSAGE

    shown, errors = _ask(message)
    stage = grade_honest_turn(shown, errors, PROVIDER_FAILURE_MESSAGE)
    assert stage is None, f"EXAM 12 C failed at stage '{stage}': {shown!r} {errors!r}"
