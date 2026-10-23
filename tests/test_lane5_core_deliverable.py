"""CORE deliverable dispatch: the workflow the ask names is the one that runs."""
from __future__ import annotations

import json

import pytest

from app.core.predefined_reasoning import DELIVER_REGISTRY, WORKFLOW_REGISTRY
from app.schemas.execution_plan import ExecutionPlan
from tests.lane5_core_support import call
from tests._dev_key import DEV_KEY


@pytest.fixture(autouse=True)
def _driver_stream(monkeypatch):
    from tests.lane5_core_support import install_driver_stream

    install_driver_stream(monkeypatch)


def _plan(_context):
    return ExecutionPlan(steps=[])


def test_naming_one_workflow_does_not_run_another(monkeypatch):
    def deliver_panel(_session, _deliverable, _message):
        return "The panel ledger total is 17."

    def deliver_north(_session, _deliverable, _message):
        return "The north ledger total is 19."

    keys = ("lane5_panel_ledger", "lane5_north_ledger")
    WORKFLOW_REGISTRY["lane5_panel_ledger"] = _plan
    WORKFLOW_REGISTRY["lane5_north_ledger"] = _plan
    DELIVER_REGISTRY["lane5_panel_ledger"] = deliver_panel
    DELIVER_REGISTRY["lane5_north_ledger"] = deliver_north

    async def intent(_prompt, has_documents=False):
        return {
            "action": "lane5_north_ledger",
            "workflow": "lane5_north_ledger",
            "mode": "deliverable",
            "params": {},
        }

    try:
        from fastapi.testclient import TestClient

        from app.agents.driver import llm
        from app.main import app
        from tests.lane5_core_support import scripted

        monkeypatch.setenv("ENV", "development")
        monkeypatch.setenv("DRIVER_MODE", "request")
        monkeypatch.setenv("ORCHESTRATOR_PREDEFINED", "1")
        monkeypatch.setattr("app.routers.chat.understand_intent", intent)
        seen: list = []
        monkeypatch.setattr(llm, "call", scripted([
            {"content": "", "tool_calls": [call("run_workflow", {
                "workflow": "lane5_panel_ledger", "deliverable": False,
            }, "c1")]},
            {"content": "The panel ledger total is 17."},
        ], seen))
        with TestClient(app) as client:
            raw = client.post(
                "/v1/chat/stream",
                json={"message": "Prepare the panel ledger.", "project_id": None, "history": []},
                headers={"Authorization": "Bearer " + DEV_KEY, "X-Fork-Driver": "on"},
            ).text
    finally:
        for key in keys:
            WORKFLOW_REGISTRY.pop(key, None)
            DELIVER_REGISTRY.pop(key, None)
    events = [
        json.loads(line[len("data: "):])
        for line in raw.split("\n") if line.startswith("data: ")
    ]
    text = "".join(event.get("content", "") for event in events if event.get("type") == "token")
    assert "panel ledger total is 17" in text
    assert "north ledger total is 19" not in text
