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

from app.agents.core.streaming import _stream_stable_prefix
from app.agents.runtime import Agent
from tests._fixture_values import STUB_TOKEN_04


# ── pure helper: cadence + margin safety ────────────────────────────────────

def _collect(full_text, scrub, margin, *, min_flush=24, delta=5):
    """Drive _stream_stable_prefix the way the loop does: grow the raw text a
    few chars at a time, flush each stable prefix, then flush the held-back
    tail (scrubbed independently, as the post-loop does). Returns the list of
    emitted segments."""
    raw = ""
    raw_emitted = 0
    emitted_len = 0
    out: list[str] = []
    i = 0
    while i < len(full_text):
        raw += full_text[i : i + delta]
        i += delta
        r = _stream_stable_prefix(raw, raw_emitted, emitted_len, margin, min_flush, scrub)
        if r is not None:
            seg, raw_emitted, emitted_len = r
            out.append(seg)
    tail = scrub(raw[raw_emitted:])
    if tail:
        out.append(tail)
    return out


def test_long_no_newline_answer_streams_in_many_segments():
    """A 400-char answer with NO newline flushes progressively, and the pieces
    reassemble to exactly the answer (identity scrub)."""
    full = "word " * 80  # 400 chars, no newline
    out = _collect(full, lambda t: t, margin=64)
    assert len(out) >= 5, f"expected progressive streaming, got {len(out)} segments"
    assert "".join(out) == full


def test_newline_answer_still_flushes_per_line():
    """Lines flush at their newline even when shorter than the margin -- the
    old behaviour is preserved for newline-delimited text."""
    full = "Line one.\nLine two.\nLine three.\n"
    out = _collect(full, lambda t: t, margin=64)
    assert len(out) >= 2
    assert "".join(out) == full


def test_stream_prefix_margin_prevents_name_straddle():
    """A withheld name split across the streaming edge is NEVER emitted as a
    partial: the margin keeps it unflushed until it is whole and scrubbed.

    The same feed with margin=0 DOES leak the partial -- that asymmetry is the
    mutant this test kills (drop/zero the margin and the name leaks)."""
    name = "Acme Engineering Limited"
    role = "the Engineer"
    full = "Under the contract the Employer appointed " + name + " to act as the Engineer for the works."

    def scrub(t):
        return t.replace(name, role)

    safe = "".join(_collect(full, scrub, margin=64))
    assert "Acme" not in safe, f"party name leaked under a safe margin: {safe!r}"
    assert role in safe

    leaky = "".join(_collect(full, scrub, margin=0))
    assert "Acme" in leaky, (
        "margin=0 should leak the partial name mid-stream; if it does not, this "
        "test no longer proves the margin is load-bearing"
    )


def test_tiny_sliver_held_until_worthwhile():
    """Below the min-flush size and with no newline, nothing is emitted yet --
    the client is not spammed with 1-2 char frames."""
    assert _stream_stable_prefix("short", 0, 0, 64, 24, lambda t: t) is None


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


def test_live_path_streams_a_no_newline_paragraph_progressively(deepseek_streaming):
    """The whole point of 6a: a long answer with NO newline, delivered as many
    small provider deltas, reaches the client as many token events spread
    through the turn -- not one burst at the end."""
    call_llm = _tool_then_final()
    # 60 small deltas, no newline anywhere: ~ a single long paragraph.
    deltas = [f"point{i} " for i in range(60)]
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
    assert "point0" in text and "point59" in text
    assert "NON-STREAMED" not in text
    assert events[-1]["type"] == "end"
