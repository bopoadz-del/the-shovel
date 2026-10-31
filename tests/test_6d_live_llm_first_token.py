"""6d: llm_first_token is measured on the LIVE /v1/chat/stream generation path.

The project-assistant agent's core chat_stream (app/agents/core/streaming.py)
had no mark_llm_dispatch -- the 6b stamp sits on the DRIVER agent -- so on the
live endpoint llm_first_token read ~1 ms while the real wait hid in
since_arrival. This drives the real core chat_stream with _call_llm mocked to
wait `gap` before a clean no-tool answer, calls first_token() on the first
token exactly as the endpoint does, and asserts llm_first_token tracks `gap`.
Dropping the mark_llm_dispatch stamp reverts it to the near-zero reading.
"""
from __future__ import annotations

import asyncio
import time

import pytest

import app.core.turn_timing as tt
from app.agents import get_agent, load_agents


def _clean_resp(_text: str) -> dict:
    return {
        "status": "ok",
        "choice": {"message": {"content": _text, "tool_calls": []}},
        "raw": {"model": "test-model"},
    }


async def _first_token_gap(agent, gap: float) -> float:
    """Run the real chat_stream; _call_llm waits `gap` then answers with no
    tools. first_token() is called on the first streamed token (as chat.py
    does). Returns the recorded llm_first_token."""
    async def _slow_call_llm(messages, api_key=None, **kw):
        await asyncio.sleep(gap)
        return _clean_resp("A provisional sum is a contingency allowance for work "
                           "not yet defined at tender; it is expended on the Engineer's "
                           "instruction and adjusted in the final account.")

    agent._call_llm = _slow_call_llm  # type: ignore[attr-defined]
    tt.mark_arrival()
    saw_token = False
    async for evt in agent.chat_stream(
        "What is a provisional sum?", history=[], project_id=None,
        conversation_id=None, user_id="6d-tester", attached_documents=[],
    ):
        if not saw_token and isinstance(evt, dict) and evt.get("type") == "token" and evt.get("content"):
            tt.first_token()          # the endpoint stamps the first token here
            saw_token = True
    assert saw_token, "no token streamed"
    return tt.snapshot()["llm_first_token"]


@pytest.mark.parametrize("gap", [0.20, 0.45])
def test_llm_first_token_tracks_the_dispatch_gap_on_the_core_path(gap, monkeypatch):
    # The real call is mocked; a dummy key only gets past the pre-dispatch guard.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-6d-key")
    load_agents()
    agent = get_agent("project-assistant")
    lft = asyncio.run(_first_token_gap(agent, gap))
    # The real dispatch->first-token wait is captured (not the ~0 of the
    # un-stamped path): at least most of the gap, and not wildly over it.
    assert lft >= gap * 0.7, (lft, gap)
    assert lft <= gap + 1.0, (lft, gap)
