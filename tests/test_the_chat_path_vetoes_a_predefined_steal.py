"""The chat path had its own predefined interception, with its own guards.

SET5 A3 was fixed in `lookup_question_hijack` and measured again on the next
tip: still 0/6, still "Schedule built: 204 activities over 688 working days
(44 on the critical path). Total effort is about 11,832 man-days." for the six
words "How long will the foundations take?".

The guard was never consulted. `lookup_question_hijack` protects the agents
router and the agent-swap decision in the runtime; `/v1/chat/stream` runs its
OWN predefined dispatch with its OWN three message-only vetoes, and a bare
duration question was not one of them. The turn emitted no `route` event at
all, which is what gave it away.

This is the third occurrence of one class. The runtime's note on the second
reads: "It never guarded THIS decision, which is the one that changes agents
-- so the same class kept happening one layer down." The four vetoes now sit
in a single predicate so the next one is added in a single place.
"""
import pytest

from app.routers import chat


# ── the live defect ────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt", [
    "How long will the foundations take?",
    "How long will the works take?",
    "How many days will the piling take?",
])
def test_a_bare_duration_question_is_vetoed(prompt):
    assert chat._message_vetoes_predefined(prompt) is True


# ── the three vetoes that were already there ───────────────────────────────

def test_a_contract_data_lookup_is_still_vetoed():
    assert chat._message_vetoes_predefined(
        "What is the Time for Completion for the whole of the Works?") is True


def test_a_wbs_export_is_still_vetoed():
    assert chat._message_vetoes_predefined(
        "Export the WBS to Excel.") is True


# ── a real deliverable request still reaches the workflow ──────────────────

@pytest.mark.parametrize("prompt", [
    "Generate a 200-activity L3 schedule for a data centre.",
    "Build me a construction programme for the foundations.",
])
def test_a_programme_request_is_not_vetoed(prompt):
    assert chat._message_vetoes_predefined(prompt) is False


def test_a_duration_calculation_is_not_vetoed():
    # Carries its own quantity and rate: there is something to compute with.
    assert chat._message_vetoes_predefined(
        "How many working days for 6,300 m2 of blockwork with 12 masons "
        "each laying 14 m2 a day?") is False


def test_an_empty_message_is_not_vetoed():
    assert chat._message_vetoes_predefined("") is False
