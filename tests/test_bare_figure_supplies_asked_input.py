"""A reply that is one figure with no label supplies what the previous reply
asked for; it replaces none of the figures the user stated.

Every thread here is invented. Each decision is pinned on two inputs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.amended_inputs import supplied_input_note
from app.agents.core.context import _build_messages

REPO = Path(__file__).resolve().parents[1]

DRAINAGE = ("Gully works measured to date 412,500, retention 4 %, advance recovery 9,000. "
            "What is payable this period?")
FORMWORK = "Core wall formwork hire 36 weeks, weekly rate 2,150. What is the hire cost?"
ASKED = "I need one more input before I can work this out. What amount was paid before?"


def _history(thread, reply=ASKED):
    out = []
    for ask in thread:
        out += [{"role": "user", "content": ask}, {"role": "assistant", "content": reply}]
    return out


def _agent():
    from app.agents.runtime import _parse_agent_file

    return _parse_agent_file(REPO / "app" / "agents" / "configs" / "project-assistant.md")


# --- the note ------------------------------------------------------------------

@pytest.mark.parametrize("thread, message, parts", [
    ([DRAINAGE], "198,000.", ["198,000", "measured to date: 412,500", "retention: 4 %",
                              "advance recovery: 9,000", DRAINAGE]),
    ([FORMWORK], "It was 3.", ["one figure, 3,", "weekly rate: 2,150", FORMWORK]),
])
def test_a_lone_figure_is_named_as_supplied_and_the_stated_figures_stand(thread, message, parts):
    note = supplied_input_note(message, thread)
    for part in parts:
        assert part in note
    assert "does not replace any figure the user stated" in note
    assert "previous reply asked" in note


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE], "Paid before was 198,000."),   # labelled: names its own input
    ([FORMWORK], "Make it 40 weeks at 2,000."),  # two figures
])
def test_a_figure_with_words_of_its_own_or_two_figures_is_not_a_lone_figure(thread, message):
    assert supplied_input_note(message, thread) is None


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE], "198,000?"),
    ([FORMWORK], "What about 3?"),
])
def test_a_lone_figure_asked_as_a_question_is_not_supplied(thread, message):
    assert supplied_input_note(message, thread) is None


@pytest.mark.parametrize("thread, message", [
    ([], "198,000."),
    (["Thanks, what else do you need?"], "3."),
])
def test_a_lone_figure_with_no_earlier_stated_question_carries_no_note(thread, message):
    assert supplied_input_note(message, thread) is None


@pytest.mark.parametrize("thread, message, now, before", [
    ([DRAINAGE, "Retention is 5 %."], "198,000.", "retention: 5 %", "retention: 4 %"),
    ([FORMWORK, "Weekly rate 2,400 then."], "3.", "weekly rate: 2,400", "weekly rate: 2,150"),
])
def test_an_amendment_after_the_question_carries_its_new_figure(thread, message, now, before):
    note = supplied_input_note(message, thread)
    assert f"\"{thread[0]}\"" in note
    assert now in note
    assert before not in note


@pytest.mark.parametrize("thread, message, question", [
    ([DRAINAGE, "198,000."], "And 2,000.", DRAINAGE),
    ([FORMWORK, "3."], "5.", FORMWORK),
])
def test_an_earlier_lone_figure_is_never_taken_for_the_question(thread, message, question):
    note = supplied_input_note(message, thread)
    assert f"\"{question}\"" in note


# --- the agent path ---------------------------------------------------------------

@pytest.mark.parametrize("thread, message, standing", [
    ([DRAINAGE], "198,000.", "measured to date: 412,500"),
    ([FORMWORK], "It was 3.", "weekly rate: 2,150"),
])
def test_the_turn_carries_the_note_just_before_the_reply(thread, message, standing):
    msgs = _build_messages(_agent(), message, _history(thread), None)
    assert msgs[-1] == {"role": "user", "content": message}
    assert msgs[-2]["role"] == "system"
    assert standing in msgs[-2]["content"]
    assert "does not replace any figure" in msgs[-2]["content"]


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE], "Paid before was 198,000, thanks."),
    ([FORMWORK], "Thanks, that is clear."),
])
def test_a_reply_that_is_not_a_lone_figure_carries_no_note(thread, message):
    msgs = _build_messages(_agent(), message, _history(thread), None)
    assert msgs[-1] == {"role": "user", "content": message}
    assert msgs[-2] == {"role": "assistant", "content": ASKED}


@pytest.mark.parametrize("thread, message, change", [
    ([DRAINAGE], "Retention is 5 %.", "retention: 4 % -> 5 %"),
    ([FORMWORK], "Weekly rate 2,400 then.", "weekly rate: 2,150 -> 2,400"),
])
def test_an_amendment_still_carries_the_amendment_note(thread, message, change):
    msgs = _build_messages(_agent(), message, _history(thread), None)
    assert change in msgs[-2]["content"]
    assert "does not replace any figure" not in msgs[-2]["content"]
