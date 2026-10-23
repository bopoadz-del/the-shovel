"""The platform's own context must never come back out as an answer.

Live on 554e0b9, theshovel.ai / Master Corpus, fresh thread, asking for the
Delay Damages daily rate. The turn ran 564 seconds, routed to ``generate_wbs``,
and streamed the pre-dispatch payload to the user AS THE ANSWER. It opened:

    {"status": "success", "wbs_id": "wbs-ccc4ee6c", "project_type": "building",
     "brief": "AUTHORITATIVE REFERENCE CONTEXT - the material below was
     retrieved from the project corpus ...

and closed with the instruction the platform had written for the model
("Report the new durations ... Do not re-call generate_wbs"). In between: the
entire retrieval brief, per-excerpt ``[doc_id= chunk= score=]`` telemetry, and
the customer's absolute local paths -- ``X:\\Example Drive\\Master Folder\\the
project\\Contract Docs\\Contractor\\Contract docs UNSIGNED\\...``.

Nothing caught it. Every branch of ``_is_tool_call_obj`` recognises a tool
CALL; a tool RESULT matches none of them. These tests pin the result shape,
each of the four brief markers, and the excerpt telemetry -- at the sanitiser
chokepoint and on the streaming path, because the answer travels both.
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from tests._fixture_values import STUB_TOKEN_04

from app.agents.runtime import (
    Agent,
    _EMPTY_RESPONSE_FALLBACK,
    _TOOL_FORMAT_FALLBACK,
    _looks_like_internal_context_leak,
    _looks_like_internal_tool_json,
    _sanitize_final_text,
)


# The head of the real leaked answer, trimmed. Kept verbatim -- a paraphrase
# would stop being a regression fixture the first time the brief is reworded.
LEAKED_ANSWER = (
    '{"status": "success", "wbs_id": "wbs-ccc4ee6c", "project_type": '
    '"building", "brief": "AUTHORITATIVE REFERENCE CONTEXT \u2014 the material '
    "below was retrieved from the project corpus and curated knowledge base "
    'for THIS question. Treat it as ground truth and answer using it.\\n'
    "SCOPE OF ABSENCE \u2014 you are seeing a small sample of the corpus "
    'selected by search, NOT the whole of it.\\n'
    "(top 5 of 5 matches; cosine in [1.834, 3.434])\\n\\n"
    "[doc_id=d0c00002 chunk=11 score=2.199 src=AB-2023-101_Contract_Data.pdf] "
    "[source: X:\\\\Example Drive\\\\Master Folder\\\\the project\\\\Contract Docs\\\\"
    'Contractor\\\\Contract docs UNSIGNED\\\\AB-2023-101_Vol 1.0.pdf]"}'
    "\nReport the new durations and recomputed total/critical path from this "
    "result. Do not reuse a prior WBS table."
)


# -- the detector ---------------------------------------------------------


def test_the_live_leak_is_detected():
    assert _looks_like_internal_context_leak(LEAKED_ANSWER) is True


def test_the_old_tool_call_detector_could_not_have_caught_it():
    """Not a redundant assertion -- it is the reason this module exists.

    If a later change makes _looks_like_internal_tool_json catch tool RESULTS
    too, this test failing is the signal to re-read whether the new detector
    is still pulling its weight, not to delete the fixture.
    """
    assert _looks_like_internal_tool_json(LEAKED_ANSWER) is False


@pytest.mark.parametrize(
    "marker",
    [
        "AUTHORITATIVE REFERENCE CONTEXT",
        "SCOPE OF ABSENCE",
        "FILENAMES ARE EVIDENCE",
        "CONTRACT ATTRIBUTION \u2014",
        "PLATFORM PRE-DISPATCH",
    ],
)
def test_every_brief_marker_is_refused_even_in_plain_prose(marker):
    """The leak is dangerous in every shape, not only wrapped in JSON."""
    assert _looks_like_internal_context_leak(
        f"Sure, here is what I was told: {marker} ... and so on."
    ) is True


def test_excerpt_telemetry_alone_is_enough():
    """A cosine score is never something a user asked for."""
    assert _looks_like_internal_context_leak(
        "The rate is 0.1%. [doc_id=d0c00002 chunk=11 score=2.199 src=x.pdf]"
    ) is True


def test_a_normal_grounded_answer_is_not_flagged():
    """The C1 answer from the same battery, on the same build. A false
    positive here costs the user a real answer, so this is the guard rail."""
    good = (
        "Based on the retrieved excerpts for contract AB-2023-101, Delay "
        "Damages for the whole of the Works are 0.1% of the Contract Price "
        "per calendar day, capped at 10% of the Contract Price (Contract "
        "Data 8.8.1, src = AB-2023-101_Contract_Data.pdf)."
    )
    assert _looks_like_internal_context_leak(good) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The contract data lists a doc_id column and a score column.",
        "Attribution matters: cite the contract you used.",
        'Here is JSON you asked for: {"status": "success", "days": 30}',
    ],
)
def test_ordinary_text_is_not_flagged(text):
    assert _looks_like_internal_context_leak(text) is False


# -- the chokepoint -------------------------------------------------------


def test_sanitize_replaces_the_leak_with_the_fallback():
    """Both branches funnel through _sanitize_final_text, so this is what
    stops the leak being persisted, exported, and fed to sources."""
    assert _sanitize_final_text(LEAKED_ANSWER) == _TOOL_FORMAT_FALLBACK


def test_sanitize_leaves_a_real_answer_alone():
    good = "Delay Damages are 0.1% of the Contract Price per calendar day."
    assert _sanitize_final_text(good) == good


# -- the streaming path ---------------------------------------------------

_DEEPSEEK_CFG = {
    "provider": "deepseek",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "env_key": "DEEPSEEK_API_KEY",
    "default_model": "deepseek-chat",
}


@pytest.fixture(autouse=True)
def _disable_commissioning_remaining(monkeypatch):
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def deepseek_streaming(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _pa_agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _tool_then_final():
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        state["messages"] = [dict(m) for m in messages]
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant",
            "content": "Delay Damages are 0.1% of the Contract Price per day."}}}

    fake.state = state
    return fake


async def _tool_ok(self, tool_call, **kwargs):
    return {"name": "commissioning_checklist", "ok": True,
            "result": {"status": "success", "checklists_by_system": {"electrical": []}}}


def _mk_stream(deltas):
    async def fake_stream(self, messages, api_key, **kwargs):
        for d in deltas:
            yield d
    return fake_stream


def _run_turn(agent):
    # Deliberately the commissioning phrasing, matching the tool the fake
    # _call_llm returns: force_synthesis is set only after a DELIVERABLE tool
    # returns, and only then does the streamed branch run at all. A
    # domain-flavoured question here would leave the branch unentered and
    # every leak assertion below would pass vacuously.
    return _drain(agent.chat_stream(
        user_message="Generate a commissioning checklist for electrical.",
        history=[], project_id=None, conversation_id=None, user_id=None,
    ))


def _tokens(events):
    return "".join(e.get("content", "") for e in events if e.get("type") == "token")


def _sent_messages(call_llm):
    """The message list the last _call_llm saw, for asserting on the nudge."""
    return call_llm.state.get("messages") or []


def test_a_streamed_leak_never_reaches_the_client(deepseek_streaming):
    """The bytes must not go out, not merely be scrubbed from what is saved.

    Chunked mid-marker on purpose: the guard tests the ACCUMULATED text, so a
    marker split across two deltas must still be caught. Chunking it at the
    line boundary the flusher uses would let the check pass for the wrong
    reason.
    """
    deltas = [
        '{"status": "success", "wbs_id": "wbs-ccc4ee6c", "brief": "AUTHORI',
        'TATIVE REFERENCE CONTEXT \u2014 the material below was retrieved\n',
        "[doc_id=d0c00002 chunk=11 score=2.199 src=x.pdf]\n",
        "X:\\\\Example Drive\\\\Master Folder\\\\the project\\\\Contract Docs\n",
    ]
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    # Proof the streamed branch was entered AND the leak was caught there:
    # streaming bypasses _call_llm for synthesis, so a second invocation can
    # only be the forced no-tools retry this leak triggers. n == 1 would mean
    # the leak shipped; n == 2 means it was suppressed and re-asked.
    assert call_llm.state["n"] == 2, "leak should force a no-tools retry"
    text = _tokens(events)
    assert "AUTHORITATIVE REFERENCE CONTEXT" not in text
    assert "doc_id=" not in text
    assert "wbs_id" not in text
    # The customer's private path is the part that must never ship.
    assert "Example Drive" not in text and "Contract Docs" not in text
    assert events[-1]["type"] == "end"
    # The retry's answer is what the user gets -- a real one, not the
    # "retry or narrow the question" dead end an XML tool leak earns.
    assert "Delay Damages" in text
    assert text.strip() != _TOOL_FORMAT_FALLBACK
    # And the persisted/exported text matches what was streamed.
    assert "AUTHORITATIVE" not in (events[-1].get("content") or "")


def test_holding_the_leak_forces_a_retry_rather_than_ending_silent(deepseek_streaming):
    """Suppression alone would trade a leak for a blank turn. The user must
    still get an answer -- here, the non-streamed retry's."""
    deltas = ["AUTHORITATIVE REFERENCE CONTEXT \u2014 retrieved material\n"]
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    text = _tokens(events)
    assert "AUTHORITATIVE REFERENCE CONTEXT" not in text
    assert text.strip(), "a suppressed leak must not leave the turn empty"
    # The model is told WHY, in terms it can act on: the answer was in the
    # material it was handed, so "answer from it" -- not "stop promising",
    # which is the other hold's instruction and would not apply here.
    nudges = [
        m for m in _sent_messages(call_llm)
        if m.get("role") == "user"
        and "repeated the reference material" in str(m.get("content") or "")
    ]
    assert nudges, "the retry must carry the context-leak instruction"


def test_a_clean_stream_is_untouched(deepseek_streaming):
    deltas = [
        "Delay Damages for the whole of the Works are 0.1% of the ",
        "Contract Price per calendar day.\n",
        "The cap is 10% of the Contract Price (Contract Data 8.8.1).\n",
    ]
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    text = _tokens(events)
    assert "0.1% of the Contract Price per calendar day" in text
    assert "10% of the Contract Price" in text


def test_a_leak_forces_the_retry_on_its_own(deepseek_streaming, monkeypatch):
    """Not by borrowing the search-promise detector's verdict.

    Today a leak also sets promise_hold, but only by accident: the chokepoint
    has already swapped the leak for _TOOL_FORMAT_FALLBACK, whose "retry or
    narrow the question" wording happens to match _looks_like_search_preamble.
    Reword that fallback and the coupling silently breaks -- the leak would
    fall through to the plain else branch and the dead-end fallback would ship
    with no retry at all. Pinned here by taking the promise detector out of
    the picture entirely.
    """
    monkeypatch.setattr(
        "app.agents.runtime._final_text_needs_forced_retry",
        lambda *a, **k: False,
    )
    deltas = ["AUTHORITATIVE REFERENCE CONTEXT — retrieved material\n"]
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    assert call_llm.state["n"] == 2, "leak_hold alone must force the retry"
    text = _tokens(events)
    assert "AUTHORITATIVE REFERENCE CONTEXT" not in text
    assert "Delay Damages" in text
    assert text.strip() != _TOOL_FORMAT_FALLBACK


def test_a_retry_that_leaks_again_is_refused_too(deepseek_streaming):
    """One chance. A retry that hands back the context as well must not ship
    it just because it is the second attempt."""
    state = {"n": 0}

    async def leaky(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant",
            "content": (
                "[doc_id=d0c00002 chunk=11 score=2.199 src=x.pdf] the rate "
                "is stated in the Contract Data."
            )}}}

    leaky.state = state
    deltas = ["AUTHORITATIVE REFERENCE CONTEXT — retrieved material\n"]
    with patch.object(Agent, "_call_llm", leaky), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    text = _tokens(events)
    assert "doc_id=" not in text
    assert "AUTHORITATIVE" not in text
    assert _EMPTY_RESPONSE_FALLBACK in text
    assert "AUTHORITATIVE" not in (events[-1].get("content") or "")
    assert "doc_id=" not in (events[-1].get("content") or "")
