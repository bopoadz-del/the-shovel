"""A routing verdict is not a deliverable -- dispatch must stay possible.

Live-fire campaign phase 3, F19, root cause found on the THIRD gate. Two
consecutive live probes (requests 143ac24a and 8c1c9589) produced a correct
slab volume with a fabricated "Routed to: formula_executor_v2" claim -- the
TIMING log showed no calculator call -- despite two contract revisions
demanding real dispatch. The machinery made compliance impossible:
AGENT_FORCE_SYNTHESIS (default ON) treats ANY successful non-search tool as
a deliverable, so after smart_orchestrator returned its verdict the next
model call ran with_tools=False. The agent was contractually required to
dispatch and structurally unable to; it resolved the conflict by inventing
the route and computing inline.

Fix is two-part, both fenced here:
  * smart_orchestrator joins search_project_documents in
    _NON_DELIVERABLE_TOOLS -- a verdict exists to be ACTED on, so its
    success must not disable tools for the next iteration;
  * on an EMPTY router match the runtime injects a corrective user-role
    message telling the model to dispatch or admit it has no tool -- an
    in-context instruction at the decision point, not a distant contract
    paragraph.
"""
from __future__ import annotations

import json

import pytest

import app.agents.runtime as rt
from tests._fixture_values import STUB_KEY_11


def _router_tool_call():
    return {"id": "t1", "type": "function",
            "function": {"name": "smart_orchestrator",
                         "arguments": json.dumps({"message": "slab volume"})}}


def _wire(monkeypatch, calls, matched_actions):
    async def fake_call_llm(self, messages, api_key=None, project_id=None,
                            with_tools=True, user_id=None, exclude_tools=None):
        calls.append({"with_tools": with_tools,
                      "messages": [dict(m) for m in messages]})
        if len(calls) == 1:
            return {"status": "success",
                    "choice": {"message": {"role": "assistant", "content": "",
                                           "tool_calls": [_router_tool_call()]}}}
        return {"status": "success",
                "choice": {"message": {"role": "assistant",
                                       "content": "24 cubic metres"}}}

    async def fake_run_tool_call(self, tc, **kw):
        return {"name": "smart_orchestrator",
                "result": {"status": "success",
                           "matched_actions": matched_actions,
                           "action_queue": ["intelligent_workflow"],
                           "routing_mode": "keyword"}}

    monkeypatch.setattr(rt.Agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(rt.Agent, "_run_tool_call", fake_run_tool_call)
    monkeypatch.delenv("AGENT_FORCE_SYNTHESIS", raising=False)
    return rt.Agent(name="smart-orchestrator", description="", system_prompt="route",
                    allowed_blocks=["smart_orchestrator", "formula_executor_v2"])


@pytest.mark.asyncio
async def test_the_iteration_after_a_router_verdict_still_offers_tools(monkeypatch):
    """RED before the fix: force-synthesis fired on the router's success and
    the second call ran with_tools=False -- dispatch was impossible."""
    calls = []
    agent = _wire(monkeypatch, calls, matched_actions=[])
    res = await agent.chat("concrete volume for a slab 12 x 8 x 0.25", api_key=STUB_KEY_11)
    assert res["status"] == "success"
    assert len(calls) >= 2
    assert calls[1]["with_tools"] is True, "router verdict disabled tools"


@pytest.mark.asyncio
async def test_an_empty_router_match_injects_the_dispatch_correction(monkeypatch):
    calls = []
    agent = _wire(monkeypatch, calls, matched_actions=[])
    await agent.chat("concrete volume for a slab 12 x 8 x 0.25", api_key=STUB_KEY_11)
    joined = " ".join(str(m.get("content") or "") for m in calls[1]["messages"])
    assert "router matched NO action" in joined
    assert "have not actually called" in joined


@pytest.mark.asyncio
async def test_a_mid_batch_nudge_does_not_split_tool_results(monkeypatch):
    """User nudges must wait until every tool_call in the assistant batch
    has a tool result. Inserting one after fetch_document:2 made Kimi 400
    Missing fetch_document:3,:4 on the live IPC_NEG turn."""
    calls = []

    async def fake_call_llm(self, messages, api_key=None, project_id=None,
                            with_tools=True, user_id=None, exclude_tools=None):
        calls.append({"messages": [dict(m) for m in messages]})
        if len(calls) == 1:
            return {"status": "success", "choice": {"message": {
                "role": "assistant", "content": "",
                "tool_calls": [
                    {"id": "fetch_document:1", "type": "function",
                     "function": {"name": "smart_orchestrator",
                                  "arguments": "{}"}},
                    {"id": "fetch_document:2", "type": "function",
                     "function": {"name": "smart_orchestrator",
                                  "arguments": "{}"}},
                    {"id": "fetch_document:3", "type": "function",
                     "function": {"name": "smart_orchestrator",
                                  "arguments": "{}"}},
                    {"id": "fetch_document:4", "type": "function",
                     "function": {"name": "smart_orchestrator",
                                  "arguments": "{}"}},
                ]}}}
        return {"status": "success",
                "choice": {"message": {"role": "assistant", "content": "ok"}}}

    async def fake_run_tool_call(self, tc, **kw):
        return {"name": "smart_orchestrator",
                "result": {"status": "success", "matched_actions": [],
                           "action_queue": ["intelligent_workflow"],
                           "routing_mode": "keyword"}}

    monkeypatch.setattr(rt.Agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(rt.Agent, "_run_tool_call", fake_run_tool_call)
    monkeypatch.delenv("AGENT_FORCE_SYNTHESIS", raising=False)
    agent = rt.Agent(name="smart-orchestrator", description="", system_prompt="route",
                    allowed_blocks=["smart_orchestrator"])
    await agent.chat("issue the payment certificate", api_key=STUB_KEY_11)
    roles = [(m.get("role"), m.get("tool_call_id")) for m in calls[1]["messages"]]
    # After the assistant tool-call message: four tools, THEN the nudge.
    assistant_idx = next(i for i, r in enumerate(roles) if r[0] == "assistant")
    following = roles[assistant_idx + 1:]
    assert [r[0] for r in following[:4]] == ["tool", "tool", "tool", "tool"]
    assert all(r[1] for r in following[:4])
    assert following[4][0] == "user"


@pytest.mark.asyncio
async def test_a_matched_router_verdict_injects_no_correction(monkeypatch):
    """The nudge is for the empty-match failure mode only."""
    calls = []
    agent = _wire(monkeypatch, calls,
                  matched_actions=[{"action": "drawing_qto_extract"}])
    await agent.chat("do QTO on this drawing", api_key=STUB_KEY_11)
    joined = " ".join(str(m.get("content") or "") for m in calls[1]["messages"])
    assert "router matched NO action" not in joined


@pytest.mark.asyncio
async def test_a_deliverable_tool_still_forces_synthesis(monkeypatch):
    """The Groq tool-loop protection force-synthesis exists for must keep
    working: a real deliverable (boq_processor) still disables tools next."""
    calls = []

    async def fake_call_llm(self, messages, api_key=None, project_id=None,
                            with_tools=True, user_id=None, exclude_tools=None):
        calls.append({"with_tools": with_tools})
        if len(calls) == 1:
            return {"status": "success",
                    "choice": {"message": {
                        "role": "assistant", "content": "",
                        "tool_calls": [{"id": "t1", "type": "function",
                                        "function": {"name": "boq_processor",
                                                     "arguments": "{}"}}]}}}
        return {"status": "success",
                "choice": {"message": {"role": "assistant", "content": "done"}}}

    async def fake_run_tool_call(self, tc, **kw):
        return {"name": "boq_processor",
                "result": {"status": "success", "line_items": []}}

    monkeypatch.setattr(rt.Agent, "_call_llm", fake_call_llm)
    monkeypatch.setattr(rt.Agent, "_run_tool_call", fake_run_tool_call)
    monkeypatch.delenv("AGENT_FORCE_SYNTHESIS", raising=False)
    agent = rt.Agent(name="qs", description="", system_prompt="qs",
                     allowed_blocks=["boq_processor"])
    await agent.chat("parse the boq", api_key=STUB_KEY_11)
    assert calls[1]["with_tools"] is False


# -- the predicate itself -------------------------------------------------

def test_empty_verdict_shapes():
    f = rt._empty_router_verdict
    assert f({"name": "smart_orchestrator",
              "result": {"matched_actions": []}}) is True
    assert f({"name": "smart_orchestrator",
              "result": {"matched_actions": [{"action": "x"}]}}) is False
    assert f({"name": "boq_processor", "result": {"matched_actions": []}}) is False
    assert f({"name": "smart_orchestrator", "result": "oops"}) is False
    assert f(None) is False


def test_router_and_search_are_non_deliverable():
    assert "smart_orchestrator" in rt._NON_DELIVERABLE_TOOLS
    assert "search_project_documents" in rt._NON_DELIVERABLE_TOOLS
