"""Chat sessions are per project and per owner.

Opening a project must not resume another thread: a new conversation id
returns none of an older thread's messages. The sidebar lists only the
caller's sessions on that project, newest first, including a session whose
row was stored before a project id was known. Another user who can open
the same project does not see them. The owner can rename a session; the
name is stored, length-limited, and cannot contain HTML.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.agents.runtime import Agent
from app.main import app
from tests._fixture_values import PW_06, STUB_TOKEN_02

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"sess-{suffix}-{_RUN}@x.com"
    reg = client.post(
        "/v1/users/register", json={"email": email, "password": PW_06}
    )
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _share(project_id: str) -> None:
    from app.core.db import SessionLocal
    from app.core.models import Project

    with SessionLocal() as db:
        row = db.get(Project, project_id)
        row.origin = "admin_drive_approved"
        row.is_approved = True
        db.commit()


def _fake_llm(text: str = "session answer"):
    async def _inner(self, messages, api_key, project_id=None, **kwargs):
        return {
            "status": "success",
            "choice": {"message": {"content": text, "tool_calls": []}},
            "raw": {},
        }
    return _inner


def _chat(client, headers, project_id: str, conversation_id: str, message: str):
    # The workspace composer posts /v1/chat/stream. A caller who can open
    # the project must be able to take a turn; the JSON agent route is not
    # what the UI uses.
    res = client.post(
        "/v1/chat/stream",
        headers=headers,
        json={
            "message": message,
            "project_id": project_id,
            "conversation_id": conversation_id,
        },
    )
    assert res.status_code == 200, res.text
    assert "Conversation not found" not in res.text
    return res


def _ids(res) -> list[str]:
    assert res.status_code == 200, res.text
    return [row["id"] for row in res.json()["conversations"]]


@pytest.fixture
def _llm(monkeypatch):
    monkeypatch.setattr(Agent, "_call_llm", _fake_llm("session answer"))
    monkeypatch.setattr(
        "app.agents.runtime.project_is_rag_ready", lambda _pid: True
    )
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


def test_a_new_conversation_does_not_contain_an_older_thread(client, _llm):
    headers = _register(client, "fresh")
    pid = _project(client, headers, f"Fresh-{_RUN}")
    older = f"ws-{pid}-111"
    newer = f"ws-{pid}-222"
    _chat(client, headers, pid, older, "older question")
    hist = client.get(
        f"/v1/agents/conversations/{newer}/messages", headers=headers
    )
    assert hist.status_code == 200, hist.text
    assert hist.json()["messages"] == []
    _chat(client, headers, pid, newer, "newer question")
    again = client.get(
        f"/v1/agents/conversations/{newer}/messages", headers=headers
    )
    contents = [m["content"] for m in again.json()["messages"]]
    assert "newer question" in contents
    assert "older question" not in contents


def test_shared_project_lists_only_the_callers_sessions_newest_first(client, _llm):
    owner = _register(client, "own")
    member = _register(client, "mem")
    pid = _project(client, owner, f"Shared-{_RUN}")
    _share(pid)
    owner_old = f"ws-{pid}-100"
    owner_new = f"ws-{pid}-300"
    member_cid = f"ws-{pid}-200"
    _chat(client, owner, pid, owner_old, "owner older")
    _chat(client, member, pid, member_cid, "member note")
    _chat(client, owner, pid, owner_new, "owner newer")

    member_rows = client.get(
        f"/v1/projects/{pid}/conversations", headers=member
    )
    assert _ids(member_rows) == [member_cid]
    assert member_rows.json()["conversations"][0]["title"] == "member note"

    owner_rows = client.get(f"/v1/projects/{pid}/conversations", headers=owner)
    assert _ids(owner_rows) == [owner_new, owner_old]
    assert member_cid not in _ids(owner_rows)

    stranger = _register(client, "str")
    hidden = client.get(f"/v1/projects/{pid}/conversations", headers=stranger)
    # A caller who can open a shared project lists only their own sessions.
    assert hidden.status_code == 200, hidden.text
    assert _ids(hidden) == []


def test_unbound_session_is_listed_for_its_project_owner_only(client):
    """A session id that names the project is listed even when the row was
    stored before a project id was written. It is not listed on any other
    project, and not to someone who cannot open this one."""
    from app.core import agent_memory as am

    owner = _register(client, "unb")
    pid = _project(client, owner, f"Unbound-{_RUN}")
    other = _project(client, owner, f"Other-{_RUN}")
    cid = f"ws-{pid}-{uuid.uuid4().hex[:6]}"
    am.get_or_create_conversation(cid, "project-assistant", project_id=None)
    am.append_message(cid, "user", "kept without a project id")

    listed = client.get(f"/v1/projects/{pid}/conversations", headers=owner)
    assert cid in _ids(listed)
    elsewhere = client.get(f"/v1/projects/{other}/conversations", headers=owner)
    assert cid not in _ids(elsewhere)

    stranger = _register(client, "unb-str")
    denied = client.get(f"/v1/projects/{pid}/conversations", headers=stranger)
    assert denied.status_code == 404


def test_owner_renames_a_session_and_others_cannot(client, _llm):
    owner = _register(client, "ren")
    pid = _project(client, owner, f"Rename-{_RUN}")
    _share(pid)
    member = _register(client, "ren-mem")
    cid = f"ws-{pid}-400"
    _chat(client, owner, pid, cid, "original title text")

    renamed = client.patch(
        f"/v1/projects/{pid}/conversations/{cid}",
        headers=owner,
        json={"title": "  Drainage review  "},
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == "Drainage review"
    listed = client.get(f"/v1/projects/{pid}/conversations", headers=owner)
    match = [c for c in listed.json()["conversations"] if c["id"] == cid]
    assert match and match[0]["title"] == "Drainage review"

    html = client.patch(
        f"/v1/projects/{pid}/conversations/{cid}",
        headers=owner,
        json={"title": "<b>Drainage</b>"},
    )
    assert html.status_code == 400, html.text

    too_long = client.patch(
        f"/v1/projects/{pid}/conversations/{cid}",
        headers=owner,
        json={"title": "x" * 81},
    )
    assert too_long.status_code == 400, too_long.text

    empty = client.patch(
        f"/v1/projects/{pid}/conversations/{cid}",
        headers=owner,
        json={"title": "   "},
    )
    assert empty.status_code == 400, empty.text

    stolen = client.patch(
        f"/v1/projects/{pid}/conversations/{cid}",
        headers=member,
        json={"title": "not yours"},
    )
    assert stolen.status_code == 404, stolen.text
    still = client.get(f"/v1/projects/{pid}/conversations", headers=owner)
    match = [c for c in still.json()["conversations"] if c["id"] == cid]
    assert match[0]["title"] == "Drainage review"
