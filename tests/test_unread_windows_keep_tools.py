"""Half a document is not a deliverable.

Live on 1d3c98b -- the build that introduced char_offset. Asked for the m3
demolition quantity:

    "Limitation: The PDF is 16 pages long and the extraction result was
     truncated; I can only see the first portion (roughly through page 5).
     Because the later pages are not visible to me, I cannot confirm whether
     an m3 item appears deeper in the bill ... If you know the page number
     or the CESMM code you are looking for, I can try to narrow the search"

It never called boq_processor again. It could not: boq_processor is a
deliverable tool, so force_synthesis fired on the first result and the next
call went out with_tools=False.

So the envelope told the model to read the next window and the platform took
its tools away. That is the same failure the sympy_reasoning carve-out in
_should_force_synthesis already exists for -- an instruction the system then
prevents -- with a different tool.
"""
from __future__ import annotations

import json

import pytest

from app.agents.runtime import (
    _TOOL_RESULT_MAX_CHARS,
    _has_unread_windows,
    _should_force_synthesis,
    _tool_result_content,
)


def _content(payload, offset=0):
    return _tool_result_content(payload, offset=offset, max_chars=_TOOL_RESULT_MAX_CHARS)


def _big(rows: int = 400) -> dict:
    return {"items": [{"code": f"D{i}", "desc": f"Breakout item {i}",
                       "unit": "m", "qty": i} for i in range(rows)]}


def test_a_result_with_windows_left_is_not_finished():
    content = _content(_big())
    assert json.loads(content)["chars_remaining"] > 0
    assert _has_unread_windows(content) is True


def test_the_last_window_is_finished():
    """The whole point of the flag: it has to turn OFF, or the turn never
    settles into synthesis and burns iterations re-reading the tail."""
    payload = _big()
    offset, seen = 0, 0
    while True:
        content = _content(payload, offset=offset)
        parsed = json.loads(content)
        seen += 1
        if not parsed["next_char_offset"]:
            assert _has_unread_windows(content) is False
            break
        assert _has_unread_windows(content) is True
        offset = parsed["next_char_offset"]
        assert seen < 20


def test_a_small_result_is_finished_immediately():
    """No envelope at all, so nothing to keep reading."""
    content = _content({"items": [{"code": "D110"}]})
    assert "chars_remaining" not in content
    assert _has_unread_windows(content) is False


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        "[1, 2, 3]",
        '{"chars_remaining": "lots"}',
        '{"chars_remaining": null}',
        '{"status": "success"}',
    ],
)
def test_malformed_content_never_blocks_synthesis(content):
    """A parse failure must not strand the turn in tool mode forever -- that
    trades a truncated answer for no answer at all."""
    assert _has_unread_windows(content) is False


def test_the_deliverable_check_itself_is_unchanged():
    """This PR narrows WHEN force_synthesis fires, not WHAT counts as a
    deliverable. _should_force_synthesis keeps its own answers so the
    sympy_reasoning and non-deliverable carve-outs are not disturbed."""
    assert _should_force_synthesis(
        {"name": "boq_processor", "ok": True, "result": {"status": "success"}}
    ) is True
    assert _should_force_synthesis(
        {"name": "boq_processor", "ok": False, "result": {}}
    ) is False
    assert _should_force_synthesis("not a dict") is False


def test_the_two_conditions_are_independent():
    """A finished small result from a deliverable tool still locks synthesis;
    an unfinished one from the same tool does not. Both halves matter, so
    neither can be dropped without a test failing."""
    result = {"name": "boq_processor", "ok": True, "result": _big()}
    truncated = _content(_big())
    complete = _content({"items": [{"code": "D110"}]})

    assert _should_force_synthesis(result) and _has_unread_windows(truncated)
    assert _should_force_synthesis(result) and not _has_unread_windows(complete)


def test_the_boundary_is_the_serialized_cap_not_the_row_count():
    """Just under the cap finishes; just over does not. Pins that the flag
    tracks the real truncation rather than a guess about payload size."""
    small = {"pad": "x" * (_TOOL_RESULT_MAX_CHARS // 2)}
    big = {"pad": "x" * (_TOOL_RESULT_MAX_CHARS * 3)}
    assert _has_unread_windows(_content(small)) is False
    assert _has_unread_windows(_content(big)) is True


# -- the wiring, driven through a real turn --------------------------------
#
# Everything above passes with the guard removed from both call sites: the
# helper works, and nothing checks that anyone calls it. That gap IS the
# defect -- the first version of char_offset shipped with the instruction
# and without the tool to follow it.

import asyncio
from unittest.mock import patch

from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_04


def _recording_llm(second_content: str = "The m3 item is 945 under D550.9."):
    """Records with_tools for every call so the turn's tool state is visible."""
    state = {"calls": []}

    async def fake(_self, messages, api_key, **kwargs):
        state["calls"].append(kwargs.get("with_tools", True))
        if len(state["calls"]) == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "boq_processor",
                    "arguments": '{"file_path":"BOQ.pdf"}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": second_content}}}

    fake.state = state
    return fake


def _tool_returning(payload: dict):
    async def fake(self, tool_call, **kwargs):
        return {"name": "boq_processor", "ok": True, "result": payload}
    return fake


def _drain(agen):
    async def go():
        return [ev async for ev in agen]
    return asyncio.run(go())


def _run(agent):
    return _drain(agent.chat_stream(
        user_message="How many cubic metres of demolition are in the BOQ?",
        history=[], project_id=None, conversation_id=None, user_id=None,
    ))


def _agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["boq_processor"])


_GROQ_CFG = {
    "provider": "groq",
    "url": "https://api.groq.com/openai/v1/chat/completions",
    "env_key": "GROQ_API_KEY",
    "default_model": "meta-llama/llama-4-scout-17b-16e-instruct",
}


@pytest.fixture(autouse=True)
def _plain_turn(monkeypatch):
    """A non-streaming turn with a key present: this is about which call
    carries tools, not about how the answer is delivered."""
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_GROQ_CFG))
    monkeypatch.setenv("GROQ_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")
    monkeypatch.setenv("SYNTHESIS_STREAMING", "0")


def test_a_truncated_result_leaves_the_tools_armed():
    """Mutation killed: dropping `not _has_unread_windows(...)` from either
    call site -- which every helper-only test above survives."""
    llm = _recording_llm()
    with patch.object(Agent, "_call_llm", llm), \
         patch.object(Agent, "_run_tool_call", _tool_returning(_big())):
        _run(_agent())

    assert len(llm.state["calls"]) >= 2, "the turn must continue after the tool"
    assert llm.state["calls"][1] is not False, (
        "tools were disarmed while the model still had windows to read"
    )


def test_a_complete_result_still_locks_synthesis():
    """The guard must narrow force_synthesis, not disable it: a finished
    deliverable should still produce a clean tool-free answer call."""
    llm = _recording_llm()
    small = {"items": [{"code": "D110", "unit": "ha", "qty": 158}]}
    with patch.object(Agent, "_call_llm", llm), \
         patch.object(Agent, "_run_tool_call", _tool_returning(small)):
        _run(_agent())

    assert len(llm.state["calls"]) >= 2
    assert llm.state["calls"][1] is False, (
        "a complete deliverable should disarm tools for the answer call"
    )


def _run_chat(agent):
    """The non-streaming entry point. Both call sites make this decision and
    both have to make it the same way -- a guard on one is a fix on one code
    path, which is how the streamed branch ended up not knowing #454's rule."""
    return asyncio.run(agent.chat(
        user_message="How many cubic metres of demolition are in the BOQ?",
        history=[], project_id=None, conversation_id=None, user_id=None,
    ))


def test_chat_also_leaves_the_tools_armed_mid_document():
    """Mutation killed: dropping the guard from Agent.chat, which the
    chat_stream test above survives because it never runs this path."""
    llm = _recording_llm()
    with patch.object(Agent, "_call_llm", llm), \
         patch.object(Agent, "_run_tool_call", _tool_returning(_big())):
        _run_chat(_agent())

    assert len(llm.state["calls"]) >= 2
    assert llm.state["calls"][1] is not False


def test_chat_still_locks_synthesis_on_a_complete_result():
    llm = _recording_llm()
    small = {"items": [{"code": "D110", "unit": "ha", "qty": 158}]}
    with patch.object(Agent, "_call_llm", llm), \
         patch.object(Agent, "_run_tool_call", _tool_returning(small)):
        _run_chat(_agent())

    assert len(llm.state["calls"]) >= 2
    assert llm.state["calls"][1] is False


def test_a_second_call_with_an_offset_gets_the_later_window():
    """End to end, the thing the feature is for: read, see chars_remaining,
    call again with next_char_offset, get the part you had not seen.

    Mutation killed: hardcoding offset=0 at the call site, which every test
    that never sends a second tool call survives -- and which would leave the
    model looping on window one forever.
    """
    payload = _big()
    tool_messages: list = []
    state = {"n": 0}

    async def llm(_self, messages, api_key, **kwargs):
        state["n"] += 1
        tool_messages[:] = [
            m for m in messages
            if isinstance(m, dict) and m.get("role") == "tool"
        ]
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "boq_processor",
                    "arguments": '{"file_path":"BOQ.pdf"}'}}]}}}
        if state["n"] == 2:
            first = json.loads(tool_messages[0]["content"])
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c2", "type": "function", "function": {
                    "name": "boq_processor",
                    "arguments": json.dumps({
                        "file_path": "BOQ.pdf",
                        "char_offset": first["next_char_offset"],
                    })}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "Read both windows."}}}

    with patch.object(Agent, "_call_llm", llm), \
         patch.object(Agent, "_run_tool_call", _tool_returning(payload)):
        _run_chat(_agent())

    assert len(tool_messages) >= 2, "the second tool call must have happened"
    first = json.loads(tool_messages[0]["content"])
    second = json.loads(tool_messages[1]["content"])
    assert first["char_offset"] == 0
    assert second["char_offset"] == first["next_char_offset"] > 0
    assert second["preview"] != first["preview"]
    assert second["chars_remaining"] < first["chars_remaining"]
    # The windows join without a gap, which is the whole contract.
    full = json.dumps(payload, default=str)
    assert full.startswith(first["preview"] + second["preview"])
