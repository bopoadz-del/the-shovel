"""A reply that is one figure with no label supplies what the previous reply
asked for; it replaces none of the figures the user stated.

Every thread here is invented. Each decision is pinned on two inputs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.amended_inputs import amendment_note, stated_figures, supplied_input_note
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


# --- grader findings on 99214ed: each pinned on two invented threads -------------

@pytest.mark.parametrize("question, message, kept", [
    ("Claim value 2,640,000, previous claim value 2,100,000, retention 6 %. What is due now?",
     "2,700,000.", ["claim value: 2,640,000", "previous claim value: 2,100,000", "retention: 6 %"]),
    ("Measured 830,000, retention 4 %, retention limit 7 % of the order sum 5,500,000.",
     "61.", ["measured: 830,000", "retention: 4 %", "retention limit: 7 %", "order sum: 5,500,000"]),
])
def test_a_label_inside_another_label_keeps_both_figures(question, message, kept):
    note = supplied_input_note(message, [question])
    for part in kept:
        assert part in note


@pytest.mark.parametrize("thread, message, given", [
    ([DRAINAGE, "198,000."], "12,000.", "198,000"),
    ([FORMWORK, "3."], "7.", "3"),
])
def test_an_earlier_lone_figure_stays_given_when_another_follows(thread, message, given):
    note = supplied_input_note(message, thread)
    assert f"already gave, in earlier replies and in this order: {given};" in note
    assert f"one figure, {message.rstrip('.')}," in note


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE], "198,000."),
    ([FORMWORK], "3."),
])
def test_no_earlier_lone_figure_names_none_given(thread, message):
    assert "already gave" not in supplied_input_note(message, thread)


@pytest.mark.parametrize("thread, message, written", [
    ([DRAINAGE], "-198,000.", "-198,000"),
    ([FORMWORK], "It was −3.", "-3"),
])
def test_a_lone_figure_keeps_its_minus_sign(thread, message, written):
    assert f"one figure, {written}," in supplied_input_note(message, thread)


@pytest.mark.parametrize("thread, message, written", [
    ([DRAINAGE], "Levels 3-5.", None),            # a range is not a sign, and two figures
    ([FORMWORK], "Item-3.", None),                # glued to a word: no figure at all
])
def test_a_dash_inside_words_or_ranges_is_not_a_sign(thread, message, written):
    assert supplied_input_note(message, thread) is written


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE, "What does clause 14 say about retention?"], "45."),
    ([FORMWORK, "Which item 3 covers the props?"], "6."),
    ([DRAINAGE, "Thanks, that is clear."], "45."),
])
def test_a_new_question_or_remark_after_the_worked_question_carries_no_note(thread, message):
    assert supplied_input_note(message, thread) is None


@pytest.mark.parametrize("question, message", [
    ("Read clause 14 again. Gross 300,000, retention 5 %.", "45."),
    ("Item 3 covers props: hire 12 weeks, weekly rate 900.", "2."),
])
def test_a_question_that_names_a_numbered_thing_still_lists_its_amounts(question, message):
    note = supplied_input_note(message, [question])
    assert "clause: 14" not in note and "item: 3" not in note
    assert "these stand: " in note


@pytest.mark.parametrize("thread, message", [
    ([DRAINAGE], "1,600,000 and 45,000."),
    ([FORMWORK], "3 or 4."),
])
def test_an_unlabelled_reply_with_two_figures_is_not_a_lone_figure(thread, message):
    assert supplied_input_note(message, thread) is None


@pytest.mark.parametrize("thread, message, written", [
    ([DRAINAGE], "5 %.", "5 %"),
    ([FORMWORK], "0.5.", "0.5"),
    ([DRAINAGE], "It was 12 percent.", "12 percent"),
])
def test_a_percent_or_decimal_lone_figure_is_written_whole(thread, message, written):
    assert f"one figure, {written}," in supplied_input_note(message, thread)


def test_the_agent_path_keeps_a_label_inside_another_label():
    question = "Claim value 2,640,000, previous claim value 2,100,000, retention 6 %. What is due now?"
    msgs = _build_messages(_agent(), "2,700,000.", _history([question]), None)
    assert "claim value: 2,640,000; previous claim value: 2,100,000" in msgs[-2]["content"]


@pytest.mark.parametrize("question, change, kept, note_has", [
    ("Claim value 2,640,000, previous claim value 2,100,000, retention 6 %. What is due now?",
     "Claim value 2,700,000.", "claim value: 2,640,000 -> 2,700,000", "previous claim value: 2,100,000"),
    ("Payroll 410,000, previous payroll 395,000. What is the increase?",
     "Payroll 420,000.", "payroll: 410,000 -> 420,000", "previous payroll: 395,000"),
])
def test_an_amendment_changes_the_label_it_names_not_the_longer_one(question, change, kept, note_has):
    assert kept in amendment_note(change, [question])
    note = supplied_input_note("45.", [question, change])
    assert note_has in note
    assert f"{kept.split(':')[0]}: {kept.split('-> ')[1]}" in note


@pytest.mark.parametrize("question, change, pair", [
    ("Variation -12,000, gross 300,000. What is due?", "Variation 12,000.", "variation: -12,000 -> 12,000"),
    ("Adjustment −4,500, measured 88,000.", "Adjustment 4,500.", "adjustment: -4,500 -> 4,500"),
])
def test_a_figure_whose_sign_changes_is_an_amendment(question, change, pair):
    assert pair in amendment_note(change, [question])


@pytest.mark.parametrize("text, value", [
    ("Grid A-12, retention 5 %.", 12.0),
    ("Panel P-14 weight 2,300.", 14.0),
])
def test_a_dash_glued_to_a_word_is_not_a_minus_sign(text, value):
    assert stated_figures(text)[0].value == value
    assert not stated_figures(text)[0].text.startswith("-")
