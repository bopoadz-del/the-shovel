"""SESSION ANSWER ROUTING: an answer is stored and shown only in its own session.

A slow ask starts in session X. While it is still running, a fresh session Y
is opened and asked something else; Y finishes first, then X finishes. The
answer to X must be in X's stream and X's stored history only, never in Y's,
and Y's answer never in X's. That holds on the live stream, after a reload
(the history the workspace fetches when a session is reopened), and after a
server restart (a new app process reading the same database).

The last test is the control: it plants X's answer in Y by hand and shows
the same reading reports it, so a pass above is not a reading that cannot
see a leak.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.main import app
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]

# Each session's ask and the answer the stand-in model gives to it. The
# answers share no words that would let one be mistaken for the other.
ASK_X = "How many inspection chambers are on the northern run?"
ANSWER_X = "The northern run has forty two inspection chambers."
ASK_Y = "What is the total length of the southern culvert?"
ANSWER_Y = "The southern culvert measures three hundred metres overall."

_HOLD_SECONDS = 30


def _register(client) -> dict:
    email = f"route-{_RUN}-{uuid.uuid4().hex[:6]}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers) -> str:
    res = client.post("/v1/projects", json={"name": f"Route-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _stream(client, headers, project_id: str, conversation_id: str, message: str) -> str:
    res = client.post(
        "/v1/chat/stream",
        headers=headers,
        json={
            "message": message,
            "project_id": project_id,
            "conversation_id": conversation_id,
            "session_id": conversation_id,
        },
    )
    assert res.status_code == 200, res.text
    return res.text


def _stored(client, headers, conversation_id: str) -> list[dict]:
    res = client.get(
        f"/v1/agents/conversations/{conversation_id}/messages", headers=headers
    )
    assert res.status_code == 200, res.text
    return res.json()["messages"]


def _contents(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def _assert_only_own(text: str, own_ask: str, own_answer: str, other_ask: str,
                     other_answer: str, where: str) -> None:
    assert own_answer in text, f"{where}: its own answer is missing"
    assert other_answer not in text, f"{where}: shows the other session's answer"
    assert other_ask not in text, f"{where}: shows the other session's question"
    if own_ask is not None:
        assert own_ask in text, f"{where}: its own question is missing"


class _Order:
    """X's model call waits until Y's whole turn has come back."""

    def __init__(self) -> None:
        self.y_done = threading.Event()
        self.x_started = threading.Event()
        self.x_finished_at: float | None = None
        self.y_finished_at: float | None = None


def _routing_llm(order: _Order):
    async def _inner(self, messages, api_key, project_id=None, **kwargs):
        seen = "\n".join(str(m.get("content") or "") for m in messages or [])
        if ASK_X in seen:
            order.x_started.set()
            deadline = time.monotonic() + _HOLD_SECONDS
            while not order.y_done.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            text = ANSWER_X
        elif ASK_Y in seen:
            text = ANSWER_Y
        else:
            text = "No figure for this question."
        return {
            "status": "success",
            "choice": {"message": {"content": text, "tool_calls": []}},
            "raw": {},
        }
    return _inner


@pytest.fixture
def _llm(monkeypatch):
    order = _Order()
    monkeypatch.setattr(Agent, "_call_llm", _routing_llm(order))
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    return order


def _restart(monkeypatch) -> None:
    """Drop every in-process handle on the database, as a new process would."""
    from app.core import agent_memory, db

    try:
        db.get_engine().dispose()
    finally:
        db._session_factory_for_engine.cache_clear()
        db._engine_for_url.cache_clear()
    monkeypatch.setattr(agent_memory, "_initialized", False)
    monkeypatch.setattr(agent_memory, "_initialized_for_url", None)


def test_a_late_answer_stays_in_its_own_session(_llm, monkeypatch):
    order = _llm
    with TestClient(app) as client:
        headers = _register(client)
        pid = _project(client, headers)
        sx = f"ws-{pid}-{int(time.time() * 1000)}"
        sy = f"ws-{pid}-{int(time.time() * 1000) + 1}"

        out: dict[str, str] = {}
        errors: list[BaseException] = []

        def _ask_x() -> None:
            try:
                out["x"] = _stream(client, headers, pid, sx, ASK_X)
                order.x_finished_at = time.monotonic()
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append(exc)

        worker = threading.Thread(target=_ask_x, daemon=True)
        worker.start()
        assert order.x_started.wait(_HOLD_SECONDS), "X's turn never reached the model"

        # Y is opened and asked while X is still waiting on its model call.
        out["y"] = _stream(client, headers, pid, sy, ASK_Y)
        order.y_finished_at = time.monotonic()
        order.y_done.set()
        worker.join(_HOLD_SECONDS + 30)
        assert not worker.is_alive(), "X's turn did not finish"
        assert not errors, errors

        # The ordering this test is about actually happened: X ended after Y.
        assert order.x_finished_at is not None and order.y_finished_at is not None
        assert order.x_finished_at > order.y_finished_at

        # Live stream: each response carries only its own session's answer.
        _assert_only_own(out["x"], None, ANSWER_X, ASK_Y, ANSWER_Y, "X stream")
        _assert_only_own(out["y"], None, ANSWER_Y, ASK_X, ANSWER_X, "Y stream")

        # Reload: the history the workspace fetches when a session is reopened.
        _assert_only_own(_contents(_stored(client, headers, sx)),
                         ASK_X, ANSWER_X, ASK_Y, ANSWER_Y, "X after reload")
        _assert_only_own(_contents(_stored(client, headers, sy)),
                         ASK_Y, ANSWER_Y, ASK_X, ANSWER_X, "Y after reload")

    # Restart: a new app process on the same database.
    _restart(monkeypatch)
    with TestClient(app) as client:
        _assert_only_own(_contents(_stored(client, headers, sx)),
                         ASK_X, ANSWER_X, ASK_Y, ANSWER_Y, "X after restart")
        _assert_only_own(_contents(_stored(client, headers, sy)),
                         ASK_Y, ANSWER_Y, ASK_X, ANSWER_X, "Y after restart")
        listed = client.get(f"/v1/projects/{pid}/conversations", headers=headers)
        assert listed.status_code == 200, listed.text
        assert {sx, sy} <= {row["id"] for row in listed.json()["conversations"]}


def test_the_reading_reports_a_planted_leak():
    """Control: the same reading fails when X's answer is stored in Y."""
    from app.core import agent_memory

    sy = f"ws-planted-{uuid.uuid4().hex[:8]}"
    agent_memory.get_or_create_conversation(sy, "project-assistant")
    agent_memory.append_message(sy, "user", ASK_Y)
    agent_memory.append_message(sy, "assistant", ANSWER_Y)
    agent_memory.append_message(sy, "assistant", ANSWER_X)
    text = _contents(agent_memory.get_messages(sy))
    with pytest.raises(AssertionError, match="shows the other session's answer"):
        _assert_only_own(text, ASK_Y, ANSWER_Y, ASK_X, ANSWER_X, "Y")
