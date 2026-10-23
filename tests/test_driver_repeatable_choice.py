"""The same question chooses the same tool every time.

A driver step that offers tools is a decision call: it goes out unsampled
(temperature 0, and a fixed seed where the provider takes one), and the
request the driver builds for it is identical run to run. A call with no
tools only writes prose and keeps the agent's own temperature. Synthetic
tools, questions and model throughout."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random

import pytest

from app.agents.driver import llm, loop, tools
from tests._fixture_values import STUB_TOKEN_09


def _agent():
    from app.agents.runtime import Agent

    return Agent(name="driver-probe", description="probe", system_prompt="", allowed_blocks=[])


def _schema(name, description):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {}}}}


_SYNTHETIC_TOOLS = [
    _schema("widget_cost_estimate", "Works out the cost of a widget order from its quantity and unit rate."),
    _schema("widget_order_list", "Lists the widget orders held in the project's documents."),
    _schema("widget_report_draft", "Drafts a widget progress report as a document."),
]

_QUESTIONS = [
    "What would 40 widgets at 12 each cost?",
    "Which widget orders do we hold?",
    "Draft the widget progress report for this week.",
]


@pytest.mark.parametrize("provider,seeded", [("deepseek", False), ("openrouter", True)])
def test_decision_calls_send_deterministic_settings(monkeypatch, provider, seeded):
    from app.agents.runtime import _llm_config

    monkeypatch.setenv("LLM_PROVIDER", provider)
    agent = _agent()
    agent.temperature = 0.7
    msgs = [{"role": "user", "content": "hi"}]
    for stream in (False, True):
        decision = llm._payload(_llm_config(), agent, msgs, _SYNTHETIC_TOOLS, "", stream=stream)
        assert decision["temperature"] == 0.0
        assert decision.get("seed") == (llm.DECISION_SEED if seeded else None)
        assert decision["tool_choice"] == "auto"
        prose = llm._payload(_llm_config(), agent, msgs, None, "", stream=stream)
        assert prose["temperature"] == 0.7 and "seed" not in prose and "tools" not in prose


def test_the_non_streamed_call_sends_the_decision_settings(monkeypatch):
    sent = {}

    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, headers=None):
            sent.update(json)
            return FakeResp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_09)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    asyncio.run(llm.call(_agent(), [{"role": "user", "content": "hi"}], _SYNTHETIC_TOOLS))
    assert sent["temperature"] == 0.0


def _model(requests):
    """A synthetic model: a sampled call (temperature above 0) may choose any
    tool; an unsampled one answers as a fixed function of the request."""
    def choose(payload):
        names = [t["function"]["name"] for t in payload["tools"]]
        if payload["temperature"] > 0:
            return random.choice(names)
        digest = hashlib.sha256(json.dumps(
            {"messages": payload["messages"], "tools": payload["tools"], "seed": payload.get("seed")},
            sort_keys=True, default=str).encode()).digest()
        ask = payload["messages"][-1]["content"].lower()
        # The tool whose name shares most words with the ask; the request
        # digest breaks a tie, so only a changed request can change the pick.
        return max(names, key=lambda n: (sum(w in ask for w in n.split("_")),
                                         digest[names.index(n) % len(digest)]))

    async def fake_stream(agent, messages, offered, model="", timeout=120.0):
        from app.agents.runtime import _llm_config

        payload = llm._payload(_llm_config(), agent, messages, offered, model, stream=True)
        if not offered:
            yield "done", {"status": "success", "message": {"content": "Done."}}
            return
        requests.append(json.dumps(payload, sort_keys=True, default=str))
        if any(m.get("role") == "tool" for m in messages):
            yield "content", "Done."
            yield "done", {"status": "success", "message": {"content": "Done."}}
            return
        call = {"id": "c1", "type": "function",
                "function": {"name": choose(payload), "arguments": "{}"}}
        yield "done", {"status": "success", "message": {"content": "", "tool_calls": [call]}}
    return fake_stream


@pytest.mark.parametrize("question", _QUESTIONS)
def test_the_same_question_chooses_the_same_tool_five_times(monkeypatch, question):
    import app.agents.runtime as rt
    import app.core.rag.inject as inject

    def no_retrieval(*a, **k):
        raise AssertionError("driver mode must not retrieve automatically")

    monkeypatch.setattr(inject, "rag_inject", no_retrieval)
    monkeypatch.setattr(rt, "rag_inject", no_retrieval)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr(tools, "offered", lambda *a, **k: list(_SYNTHETIC_TOOLS))

    chosen, first_requests = [], []
    for _ in range(5):
        requests = []
        monkeypatch.setattr(llm, "stream", _model(requests))
        agent = _agent()
        agent.temperature = 0.7  # the agent's own setting must not reach a decision

        async def fake_tool(call, api_key=None, project_id=None, conversation_id=None, **kw):
            return {"ok": True, "result": {"name": call["function"]["name"]}}

        monkeypatch.setattr(agent, "_run_tool_call", fake_tool)

        async def go():
            return [e async for e in loop.stream(agent, user_message=question)]

        events = asyncio.run(go())
        chosen.append(tuple(e["tool"] for e in events if e["type"] == "tool_call"))
        first_requests.append(requests[0])

    assert len(set(chosen)) == 1 and chosen[0], chosen
    assert len(set(first_requests)) == 1  # the driver builds the same request each run
