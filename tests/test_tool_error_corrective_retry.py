"""A failed tool call triggers a corrective retry demand, never narration.

Phase-4 campaign find (F27), reproduced twice on bim-analyst: the first
tool call errored (construction without a valid action) and the model's
FINAL answer was "I will try bim_extract" -- an intention narrated, never
executed. Same family as the fabricated-dispatch findings: prose about
calls is not calls.

After an ERROR tool result the runtime now injects a corrective user-role
message demanding the corrected call (the mechanism proven by the
empty-router nudge), capped at two per turn so a hopeless tool cannot
loop forever.
"""
from __future__ import annotations

import json

import pytest

import app.agents.runtime as rt


def _tc(name, args=None):
    return {"id": "t1", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args or {})}}


def _wire(monkeypatch, tool_results, finals):
    """LLM: first N calls emit tool_calls, then finals. Tool: scripted."""
    calls = []
    t_iter = iter(tool_results)
    f_iter = iter(finals)

    async def fake_call_llm(self, messages, api_key=None, project_id=None,
                            with_tools=True, user_id=None, exclude_tools=None):
        calls.append([dict(m) for m in messages])
        if len(calls) <= len(tool_results):
            return {"status": "success",
                    "choice": {"message": {"role": "assistant", "content": "",
                                           "tool_calls": [_tc("construction")]}}}
        return {"status": "success",
                "choice": {"message": {"role": "assistant", "content": next(f_iter)}}}

    async def fake_run_tool_call(self, tc, **kw):
        return next(t_iter)

    monkeypatch.setattr(rt.Agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(rt.Agent, "_run_tool_call", fake_run_tool_call)
    monkeypatch.delenv("AGENT_FORCE_SYNTHESIS", raising=False)
    agent = rt.Agent(name="qa", description="", system_prompt="x",
                     allowed_blocks=["construction", "bim"])
    return agent, calls


ERR = {"name": "construction",
       "result": {"status": "error", "error": "Unknown action: None"}}
OK = {"name": "construction", "result": {"status": "success", "data": 1}}


@pytest.mark.asyncio
async def test_an_error_result_injects_the_corrective_demand(monkeypatch):
    """RED before the fix: the error tool result flowed back with no
    correction, and the model was free to narrate instead of retrying."""
    agent, calls = _wire(monkeypatch, [ERR], ["done"])
    res = await agent.chat("parse the model", api_key="k")
    assert res["status"] == "success"
    second = " ".join(str(m.get("content") or "") for m in calls[1])
    assert "FAILED" in second and "make it" in second.lower()


@pytest.mark.asyncio
async def test_a_successful_result_gets_no_error_nudge(monkeypatch):
    agent, calls = _wire(monkeypatch, [OK], ["done"])
    await agent.chat("parse the model", api_key="k")
    second = " ".join(str(m.get("content") or "") for m in calls[1])
    assert "FAILED" not in second


@pytest.mark.asyncio
async def test_the_nudge_is_capped_per_turn(monkeypatch):
    """Three consecutive errors: only two corrective demands, so a hopeless
    tool cannot ping-pong forever."""
    agent, calls = _wire(monkeypatch, [ERR, ERR, ERR], ["giving up"])
    await agent.chat("parse the model", api_key="k")
    final_msgs = calls[-1]
    nudges = sum(1 for m in final_msgs
                 if m.get("role") == "user" and "FAILED" in str(m.get("content")))
    assert nudges == 2
