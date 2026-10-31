"""S-9b item 6a: a no-newline paragraph must STREAM, not arrive in one burst.

The streamed-synthesis path used to flush a `token` event only at a newline
boundary (plus a 60-char floor), so a one/two-paragraph answer with few
newlines reached the client almost entirely at the end -- the ~11%
first-token-to-end G3 measured. The cadence now flushes a stable scrubbed
prefix held a safe margin behind the live edge, so a paragraph streams.

The margin is load-bearing for party-name withholding: scrubbing the WHOLE
prefix (never a slice) and never emitting within `safe_margin` of the edge is
what stops a withheld name leaking as a partial across two token events. These
tests pin both the cadence and that safety, and that `safe_margin=0` breaks the
safety (so the mutant dies).
"""
from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from app.agents.core.streaming import _safe_stream_margin
from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_04


# ── pure helper: margin derivation ──────────────────────────────────────────

@pytest.mark.parametrize("excerpts,grows", [
    ("This clause sets the completion period in calendar days.", False),
    ("Engineer: Acme International Construction and Engineering Company Limited", True),
])
def test_safe_stream_margin_grows_with_withheld_names(excerpts, grows):
    """The hold-back margin is the floor (64) when the excerpts name no party,
    and MUST grow past it when they name a long one -- a fixed-constant mutant
    (ignore the excerpts) dies here."""
    m = _safe_stream_margin(excerpts)
    assert m >= 64
    if grows:
        assert m > 64, "margin must grow to cover a long withheld party name"
    else:
        assert m == 64


# ── integration: the real streamed-synthesis path ──────────────────────────

_DEEPSEEK_CFG = {
    "provider": "deepseek",
    "url": "https://api.deepseek.com/v1/chat/completions",
    "env_key": "DEEPSEEK_API_KEY",
    "default_model": "deepseek-chat",
}


@pytest.fixture(autouse=True)
def _no_predispatch(monkeypatch):
    monkeypatch.setenv("AGENT_COMMISSIONING_PREDISPATCH", "0")


@pytest.fixture
def deepseek_streaming(monkeypatch):
    monkeypatch.setattr("app.agents.runtime._llm_config", lambda: dict(_DEEPSEEK_CFG))
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_04)
    monkeypatch.setenv("SYNTHESIS_STREAMING", "1")


def _pa_agent():
    return Agent(name="project-assistant", description="pa",
                 system_prompt="x", allowed_blocks=["construction"])


def _tool_then_final():
    state = {"n": 0}

    async def fake(_self, messages, api_key, **kwargs):
        state["n"] += 1
        if state["n"] == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {
                    "name": "commissioning_checklist",
                    "arguments": '{"systems":["electrical"]}'}}]}}}
        return {"status": "success", "choice": {"message": {
            "role": "assistant", "content": "NON-STREAMED fallback answer."}}}

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
    async def go():
        return [ev async for ev in agent.chat_stream(
            user_message="Generate a commissioning checklist for electrical.",
            history=[], project_id=None, conversation_id=None, user_id=None,
        )]
    return asyncio.run(go())


@pytest.mark.parametrize("deltas", [
    [f"point{i} " for i in range(60)],
    [f"seg{i}__" for i in range(70)],
])
def test_live_path_streams_a_no_newline_paragraph_progressively(deepseek_streaming, deltas):
    """The whole point of 6a: a long answer with NO newline, delivered as many
    small provider deltas, reaches the client as many token events spread
    through the turn -- not one burst at the end. Two delta shapes."""
    call_llm = _tool_then_final()
    with patch.object(Agent, "_call_llm", call_llm), \
         patch.object(Agent, "_run_tool_call", _tool_ok), \
         patch.object(Agent, "_stream_synthesis", _mk_stream(deltas)):
        events = _run_turn(_pa_agent())

    assert call_llm.state["n"] == 1, "synthesis should have streamed, not re-called _call_llm"
    token_events = [e for e in events if e.get("type") == "token"]
    assert len(token_events) >= 5, (
        f"a no-newline paragraph should stream in >=5 token events, got {len(token_events)}"
    )
    text = "".join(e.get("content", "") for e in token_events)
    assert deltas[0].strip() in text and deltas[-1].strip() in text
    assert "NON-STREAMED" not in text
    assert events[-1]["type"] == "end"


@pytest.mark.parametrize("msg", [
    "Export this conversation as a Word document.",
    "Save the answers 1-5 of this conversation as a docx.",
])
def test_answer_report_ask_takes_the_export_shortcut_not_the_llm(deepseek_streaming, msg):
    """_chat_stream_impl's FIRST `if` is the answer-report early-return. An
    export-shaped ask must return the compiled report WITHOUT calling the LLM;
    remove that guard (CI-M drop_rule) and the turn falls through to the model,
    so call_llm fires and the sentinel never appears -- this kills it."""
    call_llm = _tool_then_final()

    def fake_report(user_message, project_id, conversation_id, history, agent_name, owner_id=None):
        return "ANSWER-REPORT-SENTINEL", [{"name": "answers.docx"}]

    async def go():
        return [ev async for ev in _pa_agent().chat_stream(
            user_message=msg, history=[], project_id=None,
            conversation_id=None, user_id=None,
        )]

    with patch.object(Agent, "_call_llm", call_llm), \
         patch("app.agents.runtime._fulfill_answer_report", fake_report):
        events = asyncio.run(go())

    text = "".join(e.get("content", "") for e in events if e.get("type") == "token")
    assert "ANSWER-REPORT-SENTINEL" in text, "answer-report early-return did not fire"
    assert call_llm.state["n"] == 0, "an answer-report ask must not reach the LLM"
    assert events[-1]["type"] == "end"
