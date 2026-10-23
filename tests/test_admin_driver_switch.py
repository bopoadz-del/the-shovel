"""The admin-only driver-mode switch: an admin turns driver mode on for one
user; with DRIVER_MODE=request that user's turns take the driver, nobody
else's do. Synthetic users and a scripted model."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.core import jwt_auth
from app.core import users as users_store


def _headers(user_id, email, role):
    users_store.ensure_user_exists(user_id, role=role, email=email)
    return {"Authorization": f"Bearer {jwt_auth.create_token(user_id)}"}


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_only_an_admin_switches_driver_mode(client):
    admin = _headers("dm-admin", "dm-admin@local", "admin")
    user = _headers("dm-user", "dm-user@local", "user")
    r = client.post("/v1/admin/driver-mode", json={"email": "dm-user@local", "on": True}, headers=user)
    assert r.status_code in (401, 403)
    r = client.post("/v1/admin/driver-mode", json={"email": "dm-user@local", "on": True}, headers=admin)
    assert r.status_code == 200 and r.json()["driver_mode"] is True
    listed = client.get("/v1/admin/driver-mode", headers=admin).json()
    assert "dm-user@local" in [u["email"] for u in listed["users"]]
    client.post("/v1/admin/driver-mode", json={"email": "dm-user@local", "on": False}, headers=admin)
    assert users_store.driver_mode_for("dm-user") is False
    assert client.post("/v1/admin/driver-mode", json={"email": "nobody@local", "on": True},
                       headers=admin).status_code == 404


def test_a_switched_on_user_takes_the_driver_and_others_do_not(client, monkeypatch):
    from app.agents.driver import llm, loop

    monkeypatch.setenv("DRIVER_MODE", "request")
    on = _headers("dm-on", "dm-on@local", "user")
    off = _headers("dm-off", "dm-off@local", "user")
    users_store.set_driver_mode("dm-on@local", True)
    calls = []

    async def fake_call(agent, messages, offered, model="", timeout=120.0):
        calls.append(1)
        return {"status": "success", "message": {"content": "Driver here."}}

    monkeypatch.setattr(llm, "call", fake_call)
    body = {"message": "hello", "history": []}
    text = client.post("/v1/chat/stream", json=body, headers=on).text
    events = [json.loads(line[6:]) for line in text.split("\n") if line.startswith("data: ")]
    assert any(e.get("type") == "start" and e.get("mode") == "driver" for e in events)

    async def no_driver(*a, **k):
        raise AssertionError("driver ran for a user who is not switched on")
        yield  # pragma: no cover

    monkeypatch.setattr(loop, "stream", no_driver)
    text = client.post("/v1/chat/stream", json=body, headers=off).text
    assert '"mode": "driver"' not in text
    users_store.set_driver_mode("dm-on@local", False)


def test_the_switch_is_ignored_unless_the_platform_runs_request_mode(client, monkeypatch):
    from app.agents import driver

    monkeypatch.delenv("DRIVER_MODE", raising=False)
    driver.mark_request({}, user_switched_on=True)
    assert driver.enabled() is False
