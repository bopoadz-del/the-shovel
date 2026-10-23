"""Six words must not produce a 204-activity programme.

Live SET5 A3, 6 runs out of 6 on dd4323e, identical every time:

    ask:    "How long will the foundations take?"
    answer: "Schedule built: 204 activities over 688 working days
             (44 on the critical path). Total effort is about 11,832 man-days."

Nothing was retrieved and nothing was asked. The keyword router scored the
turn confidently enough to reach ``generate_wbs``, ``lookup_question_hijack``
returned early on that confidence, and the schedule workflow did what it is
for -- it built a programme, out of nothing, and reported it as fact.

That is why the Hard rule about unnamed scope cannot fix this case: the model
is never asked to reason about the answer. The route has to be refused first.

The line drawn here is the narrowest one that holds: a duration question
carrying NO figures is asking about scope, and scope is a question. A duration
question that carries its own quantities and rates is a calculation and is
untouched (SET5 E18, T11 live). An explicit "build me a programme" is a
deliverable and is untouched.
"""
import pytest

from app.core import predefined_reasoning as pr


# ── the live defect: these must be taken off the schedule route ────────────

@pytest.mark.parametrize("msg", [
    "How long will the foundations take?",
    "how long will the foundations take",
    "How long will the works take?",
    "How long does the blockwork take?",
    "How many days will the piling take?",
    "How many weeks will the fit-out take?",
    "What is the duration of the substructure works?",
    "How long to complete the foundations?",
])
def test_a_bare_duration_question_is_a_hijack(msg):
    assert pr.is_bare_duration_question(msg) is True
    # High confidence is exactly the case that failed: the guard must fire
    # BEFORE the confidence cut-off, as the definition guard does.
    assert pr.lookup_question_hijack(msg, 0.9) is True


# ── a calculation carries its own numbers, and still computes ──────────────

@pytest.mark.parametrize("msg", [
    "How many working days for 6,300 m2 of blockwork with 12 masons each "
    "laying 14 m2 a day?",
    "How many days to lay 900 m2 of blockwork with 8 masons at 15 m2 a day?",
    "How many days will it take to pour 600 m3 of concrete?",
])
def test_a_question_with_its_own_figures_is_not_a_scope_question(msg):
    assert pr.is_bare_duration_question(msg) is False


# ── an explicit programme request is still a programme request ─────────────

@pytest.mark.parametrize("msg", [
    "Generate a 200-activity L3 schedule for a data centre.",
    "Build me a construction programme for the foundations.",
    "Create a work breakdown structure for the project.",
    "Produce a project schedule.",
])
def test_a_deliverable_request_is_never_a_bare_duration_question(msg):
    assert pr.is_bare_duration_question(msg) is False


def test_an_explicit_programme_request_still_routes(msg="Generate a 200-activity "
                                                        "L3 schedule for a data centre."):
    assert pr.lookup_question_hijack(msg, 0.9) is False


# ── it does not swallow every question with the word "long" in it ──────────

@pytest.mark.parametrize("msg", [
    "What is the Defects Notification Period under this contract?",
    "How long is the lap length for a 20 mm bar?",
    "What compaction is required under road pavement?",
    "",
])
def test_unrelated_questions_are_untouched(msg):
    assert pr.is_bare_duration_question(msg) is False
