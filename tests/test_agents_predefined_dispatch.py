"""The pilot uses /v1/agents/{name}/chat/stream. When the reasoner picks a real
container action AND ORCHESTRATOR_PREDEFINED is on, the endpoint must route it
through the predefined generic-container dispatch instead of the agent tool loop
(which only hand-wires a couple actions and otherwise gives a short answer).
Regression guard for the endpoint-wiring gap.
"""
from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06


def _token(c: TestClient) -> str:
    email = f"agpd-{uuid.uuid4().hex[:8]}@x.com"
    c.post("/v1/users/register", json={"email": email, "password": PW_06})
    return c.post("/v1/users/login",
                  json={"email": email, "password": PW_06}).json()["token"]


def test_agents_endpoint_routes_container_action_to_predefined(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")

    import app.routers.agents as ag
    import app.routers.chat as chat_mod

    async def fake_select(message, agent):
        return agent, {"requested": agent.name, "final": agent.name,
                       "action": "risk_register_auto_populate", "confidence": 0.6}
    monkeypatch.setattr(ag, "select_agent_for_message", fake_select)
    monkeypatch.setattr(chat_mod, "_predefined_workflows",
                        lambda: {"risk_register_auto_populate"})

    async def fake_predefined(**kwargs):
        assert kwargs["action"] == "risk_register_auto_populate"
        yield 'data: {"type": "start", "mode": "predefined"}\n\n'
        yield 'data: {"type": "token", "content": "RISK-REGISTER-OK "}\n\n'
        yield 'data: {"type": "end", "mode": "predefined"}\n\n'
    monkeypatch.setattr(chat_mod, "_stream_from_predefined", fake_predefined)

    with TestClient(app) as c:
        tok = _token(c)
        r = c.post("/v1/agents/project-assistant/chat/stream",
                   headers={"Authorization": f"Bearer {tok}"},
                   json={"message": "populate a risk register", "project_id": None})
        body = r.text

    assert r.status_code == 200
    assert "RISK-REGISTER-OK" in body            # predefined stream reached the client
    assert '"mode": "predefined"' in body


def test_agents_endpoint_keeps_agent_path_for_non_container_action(monkeypatch):
    """A plain Q&A action must NOT divert to predefined — it stays on the agent
    path (predefined stream must not be invoked)."""
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")

    import app.routers.agents as ag
    import app.routers.chat as chat_mod

    async def fake_select(message, agent):
        return agent, {"requested": agent.name, "final": agent.name,
                       "action": "chat", "confidence": 0.3}
    monkeypatch.setattr(ag, "select_agent_for_message", fake_select)
    monkeypatch.setattr(chat_mod, "_predefined_workflows",
                        lambda: {"risk_register_auto_populate"})  # 'chat' not in it

    called = {"predefined": False}

    async def fake_predefined(**kwargs):
        called["predefined"] = True
        yield 'data: {}\n\n'
    monkeypatch.setattr(chat_mod, "_stream_from_predefined", fake_predefined)

    with TestClient(app) as c:
        tok = _token(c)
        c.post("/v1/agents/project-assistant/chat/stream",
               headers={"Authorization": f"Bearer {tok}"},
               json={"message": "what is a WBS?", "project_id": None})

    assert called["predefined"] is False
