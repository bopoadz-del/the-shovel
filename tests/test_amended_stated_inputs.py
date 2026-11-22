"""A follow-up that only changes figures the user stated is worked again.

Every thread here is invented. Each decision is pinned on two inputs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.agents.amended_inputs import Amendment, amendment_note, amendments, stated_figures
from app.agents.core.context import _build_messages

REPO = Path(__file__).resolve().parents[1]

SLAB = "Slab 20 by 15, thickness 0.25, waste 5 %. Supplier quote 410 per m3. What does the concrete cost?"
VALUATION = ("Valuation to date 640,000, retention 7 %, earlier paid 210,000, "
             "materials on site 18,500. What is due now?")


def _read(text):
    return [(" ".join(f.label), f.value, f.percent) for f in stated_figures(text)]


# --- reading the figures a message states under its own words ---------------

@pytest.mark.parametrize("text, expected", [
    (SLAB, [("slab", 20.0, False), ("thickness", 0.25, False), ("waste", 5.0, True),
            ("supplier quote", 410.0, False)]),
    (VALUATION, [("valuation to date", 640000.0, False), ("retention", 7.0, True),
                 ("earlier paid", 210000.0, False), ("materials on site", 18500.0, False)]),
])
def test_each_figure_is_read_under_the_words_before_it(text, expected):
    assert _read(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Waste is 8 %, not 5.", [("waste", 8.0, True)]),
    ("Earlier paid was 190,000 (not 210,000).", [("earlier paid", 190000.0, False)]),
])
def test_a_figure_with_no_words_of_its_own_is_not_read(text, expected):
    assert _read(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("And the depth was 1,250.", [("depth", 1250.0, False)]),
    ("Then the rate is 3.75.", [("rate", 3.75, False)]),
])
def test_a_sentence_stop_ends_the_figure_not_its_digits(text, expected):
    assert _read(text) == expected


# --- deciding that a message only changes stated figures --------------------

@pytest.mark.parametrize("thread, message, changed", [
    ([SLAB], "Waste is 8 %, not 5.", [Amendment("waste", "5 %", "8 %")]),
    ([VALUATION], "And earlier paid was 190,000.", [Amendment("earlier paid", "210,000", "190,000")]),
])
def test_a_changed_stated_figure_is_an_amendment_of_the_question(thread, message, changed):
    question, got = amendments(message, thread)
    assert got == changed
    assert question == thread[-1]


@pytest.mark.parametrize("thread, first, second, changed", [
    ([SLAB], "Waste is 8 %.", "And the thickness is 0.3.",
     [Amendment("waste", "5 %", "8 %"), Amendment("thickness", "0.25", "0.3")]),
    ([VALUATION], "Retention is 10 %.", "And earlier paid was 190,000.",
     [Amendment("retention", "7 %", "10 %"), Amendment("earlier paid", "210,000", "190,000")]),
])
def test_a_second_follow_up_keeps_the_first_change(thread, first, second, changed):
    question, got = amendments(second, thread + [first])
    assert got == changed
    assert question == thread[-1]


@pytest.mark.parametrize("thread, first, second, changed", [
    ([SLAB], "Waste is 8 %.", "Waste is 6 %.", [Amendment("waste", "8 %", "6 %")]),
    ([VALUATION], "Retention is 10 %.", "Retention is 9 %.", [Amendment("retention", "10 %", "9 %")]),
])
def test_a_figure_changed_twice_runs_from_the_figure_before_to_the_latest(thread, first, second, changed):
    assert amendments(second, thread + [first])[1] == changed


@pytest.mark.parametrize("thread, first, back, changed", [
    ([SLAB], "Waste is 8 %.", "Waste is 5 % after all.", [Amendment("waste", "8 %", "5 %")]),
    ([VALUATION], "Retention is 10 %.", "Retention is 7 %.", [Amendment("retention", "10 %", "7 %")]),
])
def test_changing_a_figure_back_is_still_an_amendment(thread, first, back, changed):
    # The answer before it was worked on the changed figure.
    assert amendments(back, thread + [first])[1] == changed


@pytest.mark.parametrize("thread, message", [
    ([SLAB], "Waste is 5 %."),
    ([VALUATION], "Earlier paid was 210,000."),
])
def test_restating_the_same_figure_is_not_an_amendment(thread, message):
    assert amendments(message, thread) == ("", [])
    assert amendment_note(message, thread) is None


@pytest.mark.parametrize("thread, message", [
    ([SLAB], "What would it cost if waste is 8 %?"),
    ([VALUATION], "Is retention 10 % on this contract?"),
])
def test_a_message_with_its_own_question_is_not_an_amendment(thread, message):
    assert amendments(message, thread) == ("", [])


@pytest.mark.parametrize("thread, message", [
    ([SLAB], "Waste is 8 % and pump hire 1,200."),
    ([VALUATION], "Retention is 10 %, variations 40,000."),
])
def test_a_figure_under_a_label_never_stated_is_not_an_amendment(thread, message):
    assert amendments(message, thread) == ("", [])


@pytest.mark.parametrize("thread, message", [
    ([SLAB], "Thanks, that is clear."),
    ([], "Waste is 8 %."),
])
def test_no_figure_or_no_earlier_question_is_not_an_amendment(thread, message):
    assert amendments(message, thread) == ("", [])


@pytest.mark.parametrize("thread", [
    ["Waste is 8 %.", "Waste is 9 %."],
    ["Retention is 10 %.", "Retention is 9 %."],
])
def test_a_thread_of_nothing_but_amendments_has_a_question_of_its_own(thread):
    # The first message has nothing earlier to amend, so it is the question.
    question, got = amendments(thread[-1].replace("9", "7"), thread)
    assert question == thread[0]
    assert len(got) == 1


@pytest.mark.parametrize("thread, middle, message", [
    ([SLAB], "Thanks, and what about pump hire?", "Waste is 8 %."),
    ([VALUATION], "Can you also list the deductions?", "Retention is 10 %."),
])
def test_a_later_question_without_figures_becomes_the_question(thread, middle, message):
    # The latest earlier message that is not an amendment is the question;
    # it states no waste/retention, so the follow-up changes nothing it said.
    assert amendments(message, thread + [middle]) == ("", [])


@pytest.mark.parametrize("thread, middle, message", [
    ([SLAB], "Waste is 8 % and pump hire 1,200.", "Waste is 6 %."),
    ([VALUATION], "Retention is 10 %, variations 40,000.", "Retention is 9 %."),
])
def test_a_follow_up_that_is_not_an_amendment_breaks_the_chain(thread, middle, message):
    # The middle message brought a new input, so it is a question of its own,
    # and the latest message amends it, not the first.
    question, got = amendments(message, thread + [middle])
    assert question == middle
    assert len(got) == 1


# --- the note -----------------------------------------------------------------

@pytest.mark.parametrize("thread, message, parts", [
    ([SLAB], "Waste is 8 %, not 5.", ["waste: 5 % -> 8 %", SLAB]),
    ([VALUATION], "And earlier paid was 190,000.", ["earlier paid: 210,000 -> 190,000", VALUATION]),
])
def test_the_note_names_the_question_and_each_change(thread, message, parts):
    note = amendment_note(message, thread)
    for part in parts:
        assert part in note
    assert "out of date" in note


# --- the agent path -------------------------------------------------------------

def _agent():
    from app.agents.runtime import _parse_agent_file

    return _parse_agent_file(REPO / "app" / "agents" / "configs" / "project-assistant.md")


@pytest.mark.parametrize("thread, message, change", [
    ([SLAB], "Waste is 8 %, not 5.", "waste: 5 % -> 8 %"),
    ([VALUATION], "And earlier paid was 190,000.", "earlier paid: 210,000 -> 190,000"),
])
def test_the_turn_carries_the_note_just_before_the_message(thread, message, change):
    history = []
    for ask in thread:
        history += [{"role": "user", "content": ask}, {"role": "assistant", "content": "Worked answer 1,000."}]
    msgs = _build_messages(_agent(), message, history, None)
    assert msgs[-1] == {"role": "user", "content": message}
    assert msgs[-2]["role"] == "system"
    assert change in msgs[-2]["content"]


@pytest.mark.parametrize("thread, message", [
    ([SLAB], "What would it cost if waste is 8 %?"),
    ([VALUATION], "Earlier paid was 210,000."),
])
def test_a_turn_that_is_not_an_amendment_carries_no_note(thread, message):
    history = []
    for ask in thread:
        history += [{"role": "user", "content": ask}, {"role": "assistant", "content": "Worked answer 1,000."}]
    msgs = _build_messages(_agent(), message, history, None)
    assert msgs[-1] == {"role": "user", "content": message}
    assert msgs[-2] == {"role": "assistant", "content": "Worked answer 1,000."}


@pytest.mark.parametrize("project_id, context", [
    ("proj-alpha", "Project facts: alpha slab pour"),
    ("proj-beta", "Project facts: beta valuation"),
])
def test_a_project_turn_carries_its_project_context_and_a_projectless_one_does_not(
    monkeypatch, project_id, context,
):
    import app.core.project_memory as project_memory

    seen = []

    def fake_context(pid, message):
        seen.append(pid)
        return context

    monkeypatch.setattr(project_memory, "build_project_context", fake_context)
    agent = _agent()
    with_project = _build_messages(agent, SLAB, [], project_id)
    assert seen == [project_id]
    assert {"role": "system", "content": context} in with_project
    without = _build_messages(agent, SLAB, [], None)
    assert seen == [project_id]
    assert {"role": "system", "content": context} not in without
