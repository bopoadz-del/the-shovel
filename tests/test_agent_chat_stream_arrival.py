"""agent_chat_stream at a turn's arrival, before any agent runs.

- With DRIVER_MODE=request, a turn takes driver mode when an admin switched
  it on for the calling user, even without the request header.
- A name that is no agent is a 404, never a crash further down.

Agent names and user ids are invented for this file.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.agents import driver
from app.core import users
from app.routers.agents import agent_chat_stream


async def _arrive(name: str, user_id: str) -> tuple[int, bool]:
    request = SimpleNamespace(headers={}, query_params={})
    try:
        await agent_chat_stream(name, request, auth={"user_id": user_id, "role": "user"})
    except HTTPException as exc:
        return exc.status_code, driver.enabled()
    return 200, driver.enabled()


@pytest.mark.parametrize("name, user_id, switched_on", [
    ("no-such-agent-quarry", "u-arrival-1", True),
    ("no-such-agent-lantern", "u-arrival-2", False),
])
def test_user_switch_decides_driver_mode_on_request(monkeypatch, name, user_id, switched_on):
    monkeypatch.setenv("DRIVER_MODE", "request")
    monkeypatch.setattr(users, "driver_mode_for", lambda uid: switched_on if uid == user_id else not switched_on)
    status, enabled = asyncio.run(_arrive(name, user_id))
    assert status == 404
    assert enabled is switched_on
