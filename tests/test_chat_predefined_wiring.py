"""Phase 1 wiring: /v1/chat/stream routes a known workflow (schedule) to the
orchestrator's predefined reasoning when ORCHESTRATOR_PREDEFINED is on. Intent
decides the form — 'produce' carries a download offer, a question does not."""
from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
import app.routers.chat as chat_mod
from tests._dev_key import DEV_KEY
from tests import _wbs_fakes

H = {"Authorization": "Bearer " + DEV_KEY}


def _events_from_generator(action, message, project_id, monkeypatch, doc_ids=None):
    monkeypatch.setattr("app.core.projects.get_project",
                        lambda pid, user_id=None, **kw: {"id": pid, "name": "DC"})

    async def run():
        evs = []
        gen = chat_mod._stream_from_predefined(
            action=action, user_message=message, project_id=project_id,
            user_id="u1", session_id="sess", document_ids=doc_ids or [])
        async for chunk in gen:
            for line in chunk.splitlines():
                if line.startswith("data: "):
                    evs.append(json.loads(line[6:]))
        return evs

    return asyncio.run(run())


def test_predefined_stream_produce_has_offer(monkeypatch):
    _wbs_fakes.install(monkeypatch)
    evs = _events_from_generator("generate_wbs", "produce the schedule", "proj1", monkeypatch)
    start = next(e for e in evs if e["type"] == "start")
    end = next(e for e in evs if e["type"] == "end")
    assert start["mode"] == "predefined"
    assert end["workflow"] == "generate_wbs"
    assert end["exports"] and end["exports"][0]["endpoint"].endswith("/export/schedule-from-brief")
    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert "working days" in text and "download" in text.lower()


def test_predefined_stream_question_has_no_offer(monkeypatch):
    evs = _events_from_generator("generate_wbs", "how long is procurement on the schedule?", "proj1", monkeypatch)
    end = next(e for e in evs if e["type"] == "end")
    assert end["exports"] == []
    text = "".join(e.get("content", "") for e in evs if e["type"] == "token")
    assert "download" not in text.lower()


def test_gating_condition(monkeypatch):
    monkeypatch.delenv("ORCHESTRATOR_PREDEFINED", raising=False)
    assert not chat_mod._predefined_enabled()
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
    assert chat_mod._predefined_enabled()
    assert "generate_wbs" in chat_mod._predefined_workflows()


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_endpoint_routes_to_predefined_via_dynamic_understand(client, monkeypatch):
    """With the flag on, dynamic UNDERSTAND (mocked) recognizes the workflow and
    the endpoint streams the predefined path (deterministic, no LLM in exec)."""
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")

    async def fake_understand(prompt, has_documents=False):
        return {"workflow": "schedule", "action": "generate_wbs", "mode": "produce",
                "deliverable": True, "params": {"target_count": 40, "project_type": "data_center"}}
    monkeypatch.setattr(chat_mod, "understand_intent", fake_understand)

    async def fake_classify(prompt):   # keep the pre-branch classify fast/offline
        return (None, 0.0)
    monkeypatch.setattr(chat_mod, "_classify_intent", fake_classify)

    pid = client.post("/v1/projects", json={"name": "PredWire", "client": "A"}, headers=H).json()["id"]
    r = client.post("/v1/chat/stream",
                    json={"prompt": "produce the schedule", "project_id": pid},
                    headers=H)
    assert r.status_code == 200
    evs = [json.loads(ln[6:]) for ln in r.text.splitlines() if ln.startswith("data: ")]
    # Orchestrator emits a generic `start` first, then the predefined events.
    assert any(e["type"] == "start" for e in evs)
    end = next(e for e in evs if e["type"] == "end")
    assert end["workflow"] == "generate_wbs"
    assert end["exports"], "produce mode should carry a download offer"


def test_a_driver_mode_turn_is_never_taken_by_the_predefined_shortcut(client, monkeypatch):
    """Lanes 1 and 3 CORE: a cash-flow / look-ahead ask with driver mode on was
    answered in 'predefined' mode. In driver mode the model picks workflows."""
    monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
    monkeypatch.setenv("DRIVER_MODE", "request")

    async def fake_understand(prompt, has_documents=False):
        return {"workflow": "schedule", "action": "generate_wbs", "mode": "produce",
                "deliverable": True, "params": {}}
    monkeypatch.setattr(chat_mod, "understand_intent", fake_understand)

    async def fake_classify(prompt):
        return (None, 0.0)
    monkeypatch.setattr(chat_mod, "_classify_intent", fake_classify)

    from app.agents import driver
    assert not chat_mod._predefined_enabled() or not driver.enabled()
    token = driver._REQUESTED.set(True)
    try:
        assert driver.enabled()
        assert chat_mod._predefined_enabled() is False
    finally:
        driver._REQUESTED.reset(token)
    assert chat_mod._predefined_enabled() is True
