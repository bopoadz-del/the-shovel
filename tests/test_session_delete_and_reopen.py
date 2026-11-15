"""W1: a chat session is kept, reopened whole, and can be deleted by its owner.

A reopened session returns every stored turn (not only the recent window the
model sees as context). Deleting a session removes its row and messages; it
answers 404 to anyone else, and to a session on another project, exactly like
a missing one, and leaves the caller's other sessions in place.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import agent_memory as am
from app.main import app
from app.routers.projects import delete_project_conversation
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"w1-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    login = client.post("/v1/users/login", json={"email": email, "password": PW_06})
    return {"Authorization": f"Bearer {login.json()['token']}"}


def _owner_id(client, headers) -> str:
    return client.get("/v1/users/me", headers=headers).json()["user_id"]


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _session(pid: str, owner: str, turns: list[tuple[str, str]]) -> str:
    cid = f"ws-{pid}-{uuid.uuid4().hex[:10]}"
    am.get_or_create_conversation(cid, "project-assistant", project_id=pid, owner_id=owner)
    for role, text in turns:
        am.append_message(cid, role, text)
    return cid


def _endpoint(method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", ()):
            return route.endpoint
    return None


def _listed(client, headers, pid: str) -> list[str]:
    res = client.get(f"/v1/projects/{pid}/conversations", headers=headers)
    assert res.status_code == 200, res.text
    return [row["id"] for row in res.json()["conversations"]]


@pytest.mark.parametrize("turn_count", [46, 61])
def test_a_reopened_session_returns_every_stored_turn(client, turn_count):
    headers = _register(client, f"long{turn_count}")
    pid = _project(client, headers, f"Long-{turn_count}-{_RUN}")
    turns = [
        ("user" if i % 2 == 0 else "assistant", f"turn number {i} of {turn_count}")
        for i in range(turn_count)
    ]
    cid = _session(pid, _owner_id(client, headers), turns)

    res = client.get(f"/v1/agents/conversations/{cid}/messages", headers=headers)
    assert res.status_code == 200, res.text
    contents = [m["content"] for m in res.json()["messages"]]
    assert len(contents) == turn_count
    assert contents[0] == f"turn number 0 of {turn_count}"
    assert contents[-1] == f"turn number {turn_count - 1} of {turn_count}"


@pytest.mark.parametrize(
    "first,second",
    [("piling rig access route", "crane base bearing check"),
     ("rebar lap length query", "formwork strike times")],
)
def test_owner_deletes_one_session_and_keeps_the_other(client, first, second):
    headers = _register(client, f"del-{first[:5]}")
    pid = _project(client, headers, f"Delete-{first[:5]}-{_RUN}")
    owner = _owner_id(client, headers)
    # The sidebar's delete action is served by the project session route.
    assert _endpoint(
        "DELETE", "/v1/projects/{project_id}/conversations/{conversation_id}"
    ) is delete_project_conversation
    gone = _session(pid, owner, [("user", first), ("assistant", "noted")])
    kept = _session(pid, owner, [("user", second), ("assistant", "noted")])

    res = client.delete(f"/v1/projects/{pid}/conversations/{gone}", headers=headers)
    assert res.status_code == 200, res.text
    assert res.json() == {"status": "deleted", "conversation_id": gone}

    assert am.get_conversation(gone) is None
    assert am.get_messages(gone, limit=None) == []
    assert _listed(client, headers, pid) == [kept]
    assert [m["content"] for m in am.get_messages(kept, limit=None)] == [second, "noted"]

    again = client.delete(f"/v1/projects/{pid}/conversations/{gone}", headers=headers)
    assert again.status_code == 404


@pytest.mark.parametrize("label", ["stranger", "other-project"])
def test_delete_answers_404_outside_the_callers_own_project_session(client, label):
    owner = _register(client, f"own-{label}")
    pid = _project(client, owner, f"Own-{label}-{_RUN}")
    cid = _session(pid, _owner_id(client, owner), [("user", f"kept for {label}")])

    if label == "stranger":
        caller = _register(client, f"str-{label}")
        target_pid = pid
    else:
        caller = owner
        target_pid = _project(client, owner, f"Elsewhere-{label}-{_RUN}")

    res = client.delete(f"/v1/projects/{target_pid}/conversations/{cid}", headers=caller)
    assert res.status_code == 404, res.text
    assert am.get_conversation(cid) is not None
    assert [m["content"] for m in am.get_messages(cid, limit=None)] == [f"kept for {label}"]


@pytest.mark.parametrize("text", ["sheet pile toe level", "blinding concrete grade"])
def test_a_session_removed_by_a_concurrent_request_is_not_reported_deleted(
    client, monkeypatch, text
):
    """Another request (a second tab) deletes the row after this request's
    access check passed: this request answers 404, not "deleted"."""
    headers = _register(client, f"race-{text[:5]}")
    pid = _project(client, headers, f"Race-{text[:5]}-{_RUN}")
    cid = _session(pid, _owner_id(client, headers), [("user", text)])

    real_delete = am.delete_conversation

    def _other_tab_won(conversation_id: str) -> bool:
        real_delete(conversation_id)  # the other request's delete lands first
        return real_delete(conversation_id)

    monkeypatch.setattr(am, "delete_conversation", _other_tab_won)
    res = client.delete(f"/v1/projects/{pid}/conversations/{cid}", headers=headers)
    assert res.status_code == 404, res.text
    assert am.get_conversation(cid) is None
