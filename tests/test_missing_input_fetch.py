"""AN ANSWER THAT NAMES ITS MISSING INPUT GOES AND GETS IT (item 4).

F-E1-2, ``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``:

    E1 | FAIL | "do not state the specific monetary rate per calendar day".
    ...
    E1 and E2 together are the proof for F-E1-2. One question apart, same
    session, same corpus: E2 retrieved the Accepted Contract Amount to
    complete its arithmetic; E1 said the figure it needed was not available.
    It is not a retrieval limit. E1 identified its missing input and did not
    go and get it.

The battery questions here are the instrument's own strings, via the
sanitized catalog. The refusal texts are the recorded phrasing where the
pack records it (E1) and representative otherwise -- said plainly rather
than implied, because a fence written against a paraphrase proves nothing.

Each test names the mutation it kills.
"""

import json
from pathlib import Path

import pytest

from app.agents.missing_input import (
    enabled,
    fetch_query,
    fetched_context_supports,
    names_missing_input,
)

CATALOG = json.loads(
    (Path(__file__).parent / "fixtures" / "ui_phys" / "questions.json").read_text(
        encoding="utf-8"
    )
)
E1_ASK = CATALOG["cases"]["E1"]["ask"]

#: The recorded fragment, verbatim from the evidence pack, in a sentence.
E1_ANSWER = (
    "I can set out the method, but the retrieved excerpts do not state the "
    "specific monetary rate per calendar day for delay damages, so I cannot "
    "complete the calculation."
)


def test_the_question_is_the_instrument_s_own():
    assert E1_ASK == (
        "Calculate the delay damages per calendar day in SAR for the whole "
        "of the Works."
    )


# -- detection -------------------------------------------------------------


def test_e1_names_what_it_is_missing():
    """The recorded fragment, and the thing it names.

    Mutation killed: dropping the "do not state" cue, which is the exact
    wording the failure used.
    """
    got = names_missing_input(E1_ANSWER)
    assert got is not None
    assert "rate per calendar day" in got


def test_the_passive_form_is_read_too():
    """"X is not stated" puts the name BEFORE the cue. Mutation killed:
    matching only after the cue, which captures the place it was looked for
    ("in the retrieved excerpts") and never the thing."""
    got = names_missing_input(
        "The delay damages rate per calendar day is not stated in the "
        "retrieved excerpts."
    )
    assert got is not None
    assert "delay damages" in got and "calendar day" in got


@pytest.mark.parametrize("answer", [
    # G1's correct answer -- there is nothing to fetch.
    "Schedule 10 is Not Used.",
    # G5's correct refusal: the document is genuinely absent.
    "This document is not in the corpus, so I cannot summarise it.",
    # G3: names only "the value", one generic word.
    "The value is not stated in the Contract Data; 4.3.7 says a Parent "
    "Company Guarantee is required only if specified in the Contract Data.",
    # A clean answer.
    "The Accepted Contract Amount excluding VAT is SAR 8,640,000.00 per "
    "Contract Data 1.1.1.",
])
def test_an_answer_with_nothing_specific_to_fetch_triggers_nothing(answer):
    """A bounded fetch is cheap but not free, and an extra retrieval on a
    correct refusal is latency spent for nothing.

    Mutation killed: firing on any sentence containing "not".
    """
    assert names_missing_input(answer) is None


def test_a_single_generic_word_is_not_a_name():
    """E4 answered "I could not confirm this reference". Searching for
    "reference" returns the whole corpus, and the support guard below only
    needs its words to appear -- so a one-generic-word name would wave any
    junk through.

    Mutation killed: dropping the two-content-word floor.
    """
    assert names_missing_input(
        "I could not confirm this reference in the indexed project sources."
    ) is None


def test_a_name_carrying_a_number_survives_the_floor():
    """"Schedule 10" is two tokens but one of them is a digit; a clause
    number like 8.8.1 is one token. Both are searchable names.

    Mutation killed: requiring two ALPHABETIC words, which would refuse
    exactly the identifiers worth fetching.
    """
    assert names_missing_input(
        "The excerpts do not contain Schedule 10 of the contract volumes."
    ) is not None
    assert names_missing_input(
        "Sub-Clause 8.8.1 is not included in the retrieved excerpts."
    ) is not None
    # The load-bearing case: a bare identifier is TWO tokens and only one of
    # them is a word. Without the digit rescue these are refused, and they
    # are the most searchable names an answer can give.
    assert names_missing_input("The excerpts do not contain Schedule 10.") == (
        "Schedule 10"
    )
    assert names_missing_input("The excerpts do not state Bill 03.") == "Bill 03"
    assert names_missing_input("Clause 8.8 is not stated in the excerpts.") == (
        "Clause 8.8"
    )


def test_only_the_first_named_gap_is_returned():
    """The rule is ONE bounded attempt. Mutation killed: returning a list
    and searching each, which is the unbounded loop the rule forbids."""
    answer = (
        "The excerpts do not state the delay damages rate per calendar day, "
        "and they do not contain the Accepted Contract Amount either."
    )
    got = names_missing_input(answer)
    assert got is not None
    assert "calendar day" in got
    assert isinstance(got, str)


# -- the query -------------------------------------------------------------


def test_the_named_thing_leads_the_query():
    """Searching the original question again is what already failed.

    Mutation killed: building the query from the question alone, which
    reruns the search that produced the refusal.
    """
    missing = names_missing_input(E1_ANSWER)
    q = fetch_query(missing, E1_ASK)
    assert q.startswith(missing)


def test_the_question_narrows_the_query_without_repeating_it():
    """"rate per calendar day" alone matches half a construction corpus.
    Mutation killed: dropping the question terms entirely."""
    q = fetch_query("monetary rate per calendar day", E1_ASK)
    low = q.lower()
    assert "delay" in low or "damages" in low or "works" in low
    # Stopwords and words already in the name are not repeated.
    assert low.count("calendar") == 1
    for junk in (" the ", " for ", " what ", " which "):
        assert junk not in " " + low + " "


# -- the guard that stops a fetch making things worse ----------------------


def test_a_fetch_that_came_back_with_the_answer_is_accepted():
    missing = names_missing_input(E1_ANSWER)
    assert fetched_context_supports(
        "Sub-Clause 8.8.1 Delay Damages: 0.1% of the Contract Price per "
        "calendar day, capped at 10%.",
        missing,
    )


def test_a_fetch_that_came_back_with_something_else_is_refused():
    """THE guard. A search always returns something; injecting whatever came
    back and asking again is how a correct refusal becomes a confident wrong
    answer.

    Mutation killed: accepting any non-empty retrieval, which turns the
    bounded fetch into a fabrication engine.
    """
    missing = names_missing_input(E1_ANSWER)
    assert not fetched_context_supports(
        "The Contractor shall submit a programme within 28 days of the "
        "Commencement Date.",
        missing,
    )
    assert not fetched_context_supports("", missing)
    assert not fetched_context_supports("   \n  ", missing)


def test_one_matching_word_is_not_support_for_a_long_name():
    """"monetary rate per calendar day" is not satisfied by the word "rate".

    Mutation killed: requiring a single overlapping word.
    """
    assert not fetched_context_supports(
        "Unit rate for excavation is SAR 31.00 per m2.",
        "monetary rate per calendar day for delay damages",
    )


def test_a_two_word_name_needs_both_words():
    assert fetched_context_supports("Schedule 10: Not Used", "Schedule 10")
    assert not fetched_context_supports("Schedule 9: Health & Safety KPIs",
                                        "Schedule 10")


# -- the switch ------------------------------------------------------------


def test_the_feature_is_on_by_default_and_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("MISSING_INPUT_FETCH", raising=False)
    assert enabled()
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("MISSING_INPUT_FETCH", off)
        assert not enabled(), off
    monkeypatch.setenv("MISSING_INPUT_FETCH", "1")
    assert enabled()


# ── the bounded fetch, driven through the real agent method ──────────────

import asyncio  # noqa: E402

from app.agents.runtime import Agent  # noqa: E402


def _agent():
    return Agent(
        name="project-assistant",
        description="test",
        system_prompt="test",
        allowed_blocks=[],
    )


def _run(coro):
    return asyncio.run(coro)


def _rag_msg(text: str):
    return {
        "role": "system",
        "content": "AUTHORITATIVE REFERENCE CONTEXT\n"
                   "[doc_id=d1 chunk=1 score=0.700 class=project_corpus "
                   "src=S1_contract_data.md] " + text,
    }


class _Spy:
    """Records the calls the fetch makes, so 'bounded' can be asserted."""

    def __init__(self, rag_text=None, answer=None, llm_error=False):
        self.rag_text = rag_text
        self.answer = answer
        self.llm_error = llm_error
        self.rag_calls = []
        self.llm_calls = []

    def rag_inject(self, **kwargs):
        self.rag_calls.append(kwargs)
        if self.rag_text is None:
            return None, {}
        return _rag_msg(self.rag_text), {}

    async def call_llm(self, messages, api_key, **kwargs):
        self.llm_calls.append(messages)
        if self.llm_error:
            return {"status": "error", "error": "upstream down"}
        return {"status": "success",
                "choice": {"message": {"content": self.answer}}}


def _wire(monkeypatch, spy):
    import app.agents.runtime as rt

    monkeypatch.setattr(rt, "rag_inject", spy.rag_inject)
    monkeypatch.setattr(Agent, "_call_llm", lambda self, m, k, **kw: spy.call_llm(m, k, **kw))


REVISED = (
    "Delay damages for the whole of the Works are 0.2% of the Contract Price "
    "per calendar day (Sub-Clause 8.8.1), i.e. SAR 17,280.00 per day on an "
    "Accepted Contract Amount of SAR 8,640,000.00."
)


def test_the_named_gap_is_fetched_once_and_the_answer_is_revised(monkeypatch):
    """F-E1-2 closed: E1 named its missing input, the platform went and got
    it, and the answer completed.

    Mutation killed: not calling the fetch at all.
    """
    spy = _Spy(rag_text="Sub-Clause 8.8.1 Delay Damages: 0.2% of the Contract "
                        "Price per calendar day.", answer=REVISED)
    _wire(monkeypatch, spy)
    text, fetched = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [{"role": "user", "content": E1_ASK}],
        user_message=E1_ASK, project_id="p1", api_key="k", user_id="u",
    ))
    assert text == REVISED
    assert fetched and "calendar day" in fetched
    assert len(spy.rag_calls) == 1, "bounded means ONE retrieval"
    assert len(spy.llm_calls) == 1, "bounded means ONE re-ask"


def test_the_retrieval_searches_the_named_gap_not_the_question_again(monkeypatch):
    """Re-running the question is what produced the refusal.

    Mutation killed: passing user_message straight through as the query.
    """
    spy = _Spy(rag_text="0.2% of the Contract Price per calendar day", answer=REVISED)
    _wire(monkeypatch, spy)
    _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    q = spy.rag_calls[0]["user_message"]
    assert q != E1_ASK
    assert q.startswith(names_missing_input(E1_ANSWER))
    # And it must not be filed against the conversation: a targeted lookup is
    # not a turn, and it must not enter the next turn's follow-up context.
    assert spy.rag_calls[0]["conversation_id"] is None


def test_an_answer_that_names_nothing_costs_nothing(monkeypatch):
    spy = _Spy(rag_text="anything", answer="revised")
    _wire(monkeypatch, spy)
    original = "Schedule 10 is Not Used."
    text, fetched = _run(_agent()._fetch_named_missing_input(
        original, [], user_message="What does Schedule 10 contain?",
        project_id="p1", api_key="k", user_id="u",
    ))
    assert text == original and fetched is None
    assert spy.rag_calls == [] and spy.llm_calls == []


def test_a_fetch_that_found_something_else_leaves_the_refusal_standing(monkeypatch):
    """THE guard, through the real method. A search always returns
    something; re-asking on top of whatever came back is how a correct
    refusal becomes a confident wrong answer.

    Mutation killed: re-asking whenever the retrieval was non-empty.
    """
    spy = _Spy(rag_text="The Contractor shall submit a programme within 28 days.",
               answer="A WRONG ANSWER BUILT ON THE WRONG CHUNK")
    _wire(monkeypatch, spy)
    text, fetched = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER and fetched is None
    assert len(spy.rag_calls) == 1
    assert spy.llm_calls == [], "no re-ask on unsupported context"


def test_an_empty_retrieval_leaves_the_answer_alone(monkeypatch):
    spy = _Spy(rag_text=None, answer="should never be used")
    _wire(monkeypatch, spy)
    text, _ = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER
    assert spy.llm_calls == []


def test_a_failing_re_ask_leaves_the_answer_alone(monkeypatch):
    """The turn already has a usable answer. Mutation killed: returning the
    error text, which replaces a real refusal with a stack trace."""
    spy = _Spy(rag_text="Sub-Clause 8.8.1 Delay Damages: 0.2% of the "
                        "Contract Price per calendar day.",
               answer=None, llm_error=True)
    _wire(monkeypatch, spy)
    text, fetched = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER and fetched is None


def test_a_raising_retrieval_does_not_fail_the_turn(monkeypatch):
    import app.agents.runtime as rt

    def boom(**kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(rt, "rag_inject", boom)
    text, fetched = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER and fetched is None


def test_an_empty_re_ask_leaves_the_answer_alone(monkeypatch):
    spy = _Spy(rag_text="Sub-Clause 8.8.1 Delay Damages: 0.2% of the "
                        "Contract Price per calendar day.", answer="   ")
    _wire(monkeypatch, spy)
    text, _ = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER


def test_no_project_means_no_fetch(monkeypatch):
    spy = _Spy(rag_text="anything", answer="revised")
    _wire(monkeypatch, spy)
    text, _ = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id=None,
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER
    assert spy.rag_calls == []


def test_the_kill_switch_stops_the_whole_path(monkeypatch):
    monkeypatch.setenv("MISSING_INPUT_FETCH", "0")
    spy = _Spy(rag_text="Sub-Clause 8.8.1 Delay Damages: 0.2% of the "
                        "Contract Price per calendar day.", answer=REVISED)
    _wire(monkeypatch, spy)
    text, _ = _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    assert text == E1_ANSWER
    assert spy.rag_calls == []


def test_the_re_ask_forbids_estimating_the_missing_figure(monkeypatch):
    """If the fetched excerpts still fall short, a plausible substitute is
    worse than the gap the answer already reported.

    Mutation killed: a nudge that only says "answer in full".
    """
    spy = _Spy(rag_text="Sub-Clause 8.8.1 Delay Damages: 0.2% of the Contract "
                        "Price per calendar day.", answer=REVISED)
    _wire(monkeypatch, spy)
    _run(_agent()._fetch_named_missing_input(
        E1_ANSWER, [], user_message=E1_ASK, project_id="p1",
        api_key="k", user_id="u",
    ))
    nudge = spy.llm_calls[0][-1]["content"]
    assert "do NOT estimate" in nudge
    assert "say so plainly" in nudge
    assert names_missing_input(E1_ANSWER) in nudge


def test_the_fetch_runs_before_the_answer_is_finalised():
    """It must sit ahead of _postprocess_answer at both final-answer sites in
    chat(): the citation guard and the cost gate judge the FINAL text, and a
    revision made after them would be ungoverned.

    Mutation killed: calling the fetch after _postprocess_answer, or wiring
    only one of the two sites.
    """
    from tests import agent_source

    src = agent_source.text()  # runtime.py and app/agents/core/
    assert src.count("await self._fetch_named_missing_input(") == 2
    for block in src.split("await self._fetch_named_missing_input(")[1:]:
        head = block[: block.index("_postprocess_answer")]
        assert "final_text, _fetched_for" not in head.split("\n", 1)[0]
        assert head.count("_postprocess_answer") == 0
