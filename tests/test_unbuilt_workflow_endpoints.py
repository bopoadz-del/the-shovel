"""X4 at the two chat endpoints: a workflow that built nothing gives the turn
back to the agent, and a built one keeps it.

* ``/v1/chat/stream`` (``app.routers.chat``): the redirect cancels the
  speculative call; ``_assistant_after_unbuilt_workflow`` is the assistant
  stream it hands an unbuilt turn to.
* ``/v1/agents/{name}/chat/stream`` (``app.routers.agents``): ``event_stream``
  routes a predefined action to ``_stream_from_predefined`` with ``_agent_sse``
  (the endpoint's own agent stream) as its fallback.

The build step is also driven directly (``PlanExecutor._step_build_wbs``).
Every input is invented; two per decision.
"""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import app.routers.agents as ag
import app.routers.chat as chat_mod
from app.containers.construction import ConstructionContainer
from app.core import plan_executor, turn_timing
from app.main import app
from app.schemas.execution_plan import PlanStep
from app.schemas.project_session import ProjectSession
from tests import _wbs_fakes
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


def _refusing_engine(monkeypatch, words="no schedulable scope"):
    async def fake_generate_wbs(self, _ctx, _params):  # noqa: ANN001 - test double
        return {"status": "error", "action": "generate_wbs", "error": words}
    monkeypatch.setattr(ConstructionContainer, "generate_wbs", fake_generate_wbs)


def _agent_answers(monkeypatch, text):
    async def fake_chat_stream(self, message, **kwargs):
        yield {"type": "start", "agent": self.name}
        yield {"type": "token", "content": text}
        yield {"type": "end", "iterations": 1}
    monkeypatch.setattr("app.agents.runtime.Agent.chat_stream", fake_chat_stream)


def _count_first_tokens(monkeypatch):
    seen = []
    monkeypatch.setattr(turn_timing, "first_token", lambda: seen.append(1))
    return seen


def _tokens(body):
    out = []
    for line in body.splitlines():
        if line.startswith("data: "):
            try:
                e = json.loads(line[6:])
            except ValueError:
                continue
            if e.get("type") == "token":
                out.append(e.get("content", ""))
    return "".join(out)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── the build step, driven directly ───────────────────────────────────────────

@pytest.mark.parametrize("words", [
    "no schedulable scope: list the activities",
    "brief names no work to break down",
])
def test_step_build_wbs_raises_the_engine_refusal(monkeypatch, words):
    _refusing_engine(monkeypatch, words)
    session = ProjectSession.new("wf-ep", user_id="system")
    step = PlanStep(type="build_wbs", args={"brief": "x"})
    with pytest.raises(plan_executor.PlanExecutionError) as err:
        asyncio.run(plan_executor.PlanExecutor()._step_build_wbs(step, session))
    assert str(err.value) == words
    assert session.data.get("activities") is None


@pytest.mark.parametrize("brief", [
    "Set out grid, excavate pits, cast footings.",
    "Erect scaffold, render facade, strike scaffold.",
])
def test_step_build_wbs_stages_a_built_breakdown(monkeypatch, brief):
    _wbs_fakes.install(monkeypatch)
    session = ProjectSession.new("wf-ep", user_id="system")
    step = PlanStep(type="build_wbs", args={"brief": brief})
    value = asyncio.run(plan_executor.PlanExecutor()._step_build_wbs(step, session))
    assert value["actual_count"] == len(session.data["activities"])
    assert len(session.data["activities"]) > 0


# ── /v1/chat/stream: _assistant_after_unbuilt_workflow ────────────────────────

def _route_chat_to_schedule(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")

    async def fake_understand(prompt, has_documents=False):
        return {"workflow": "schedule", "action": "generate_wbs", "mode": "answer",
                "deliverable": False, "params": {}}
    monkeypatch.setattr(chat_mod, "understand_intent", fake_understand)

    async def fake_classify(prompt):
        return (None, 0.0)
    monkeypatch.setattr(chat_mod, "_classify_intent", fake_classify)


@pytest.mark.parametrize("prompt,agent_text", [
    ("Wall 14 m x 3 m, a mason lays 1.5 m2 per hour: hours, then a programme?",
     "Area 42 m2, 28 hours."),
    ("Ditch 40 x 0.8 x 1.2 m at 16 m3 a day, then pipe laying: programme?",
     "Volume 38.4 m3, 2.4 days."),
])
def test_chat_stream_unbuilt_workflow_gives_the_turn_to_the_assistant(client, monkeypatch, prompt, agent_text):
    _route_chat_to_schedule(monkeypatch)
    _refusing_engine(monkeypatch)
    _agent_answers(monkeypatch, agent_text)
    seen = _count_first_tokens(monkeypatch)
    r = client.post("/v1/chat/stream", json={"prompt": prompt}, headers=H)
    assert r.status_code == 200
    assert agent_text in _tokens(r.text)
    assert "could not build the schedule" not in r.text
    assert len(seen) >= 1


@pytest.mark.parametrize("prompt", [
    "Set out grid, excavate pits, cast footings: schedule it.",
    "Erect scaffold, render facade, strike scaffold: schedule it.",
])
def test_chat_stream_built_workflow_keeps_the_turn(client, monkeypatch, prompt):
    _route_chat_to_schedule(monkeypatch)
    _wbs_fakes.install(monkeypatch)
    _agent_answers(monkeypatch, "AGENT-SHOULD-NOT-ANSWER")
    r = client.post("/v1/chat/stream", json={"prompt": prompt}, headers=H)
    assert r.status_code == 200
    assert "AGENT-SHOULD-NOT-ANSWER" not in r.text
    assert "working days" in _tokens(r.text)


# ── /v1/agents/{name}/chat/stream: event_stream + _agent_sse ──────────────────

def _route_agents_to_schedule(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")

    async def fake_select(message, agent):
        return agent, {"requested": agent.name, "final": agent.name,
                       "action": "generate_wbs", "confidence": 0.9}
    monkeypatch.setattr(ag, "select_agent_for_message", fake_select)


@pytest.mark.parametrize("message,agent_text", [
    ("Wall 14 m x 3 m, a mason lays 1.5 m2 per hour: hours, then a programme?",
     "Area 42 m2, 28 hours."),
    ("Ditch 40 x 0.8 x 1.2 m at 16 m3 a day, then pipe laying: programme?",
     "Volume 38.4 m3, 2.4 days."),
])
def test_agents_event_stream_unbuilt_workflow_falls_back_to_agent_sse(client, monkeypatch, message, agent_text):
    _route_agents_to_schedule(monkeypatch)
    _refusing_engine(monkeypatch)
    _agent_answers(monkeypatch, agent_text)
    seen = _count_first_tokens(monkeypatch)
    r = client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": message}, headers=H)
    assert r.status_code == 200
    assert agent_text in _tokens(r.text)
    assert "could not build the schedule" not in r.text
    assert len(seen) >= 1


@pytest.mark.parametrize("message", [
    "Set out grid, excavate pits, cast footings: schedule it.",
    "Erect scaffold, render facade, strike scaffold: schedule it.",
])
def test_agents_event_stream_built_workflow_stays_predefined(client, monkeypatch, message):
    _route_agents_to_schedule(monkeypatch)
    _wbs_fakes.install(monkeypatch)
    _agent_answers(monkeypatch, "AGENT-SHOULD-NOT-ANSWER")
    r = client.post("/v1/agents/project-assistant/chat/stream",
                    json={"message": message}, headers=H)
    assert r.status_code == 200
    assert "AGENT-SHOULD-NOT-ANSWER" not in r.text
    assert '"mode": "predefined"' in r.text
