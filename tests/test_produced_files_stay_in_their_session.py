"""Exam (advisor 10:30Z, 16 Nov): a file the product produced in chat
session X is listed in X only -- never in a NEW session Y of the same user,
never for another user.

TESTER (04:20Z) opened a new session and its "This session" group listed two
files "made by the product in this chat session" from an earlier session.
This exam builds that sequence end to end through the routes the browser
uses: answer and export in X, then open Y (a client-made session id, as the
"New chat" button makes them), attach a file in Y, and read Y's list.

Every project, user, session and answer here is created by the test.
"""
from __future__ import annotations

import io
import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, suffix: str) -> dict:
    email = f"w2scope-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _answer(cid: str, pid: str, ask: str, text: str) -> None:
    from app.core import agent_memory

    agent_memory.get_or_create_conversation(cid, "project_assistant", pid)
    agent_memory.append_message(cid, "user", ask)
    agent_memory.append_message(cid, "assistant", text)


def _export(client, headers, pid, cid, fmt):
    path = app.url_path_for("export_conversation_message", project_id=pid, conversation_id=cid)
    res = client.post(f"{path}?format={fmt}", headers=headers)
    assert res.status_code == 200, res.text
    return res


def _listed(client, headers, pid, sid) -> list[dict]:
    res = client.get(f"/v1/projects/{pid}/conversations/{sid}/documents", headers=headers)
    assert res.status_code in (200, 404), res.text
    return res.json().get("documents", []) if res.status_code == 200 else []


def _attach(client, headers, pid, sid, name: str) -> str:
    res = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": (name, io.BytesIO(b"Invented attached note for the exam.\n"), "text/plain")},
        data={"session_id": sid},
        headers=headers,
    )
    assert res.status_code in (200, 201), res.text
    body = res.json()
    return (body.get("document") or body).get("id")


@pytest.mark.parametrize("second_session_suffix", ["2000", "mwiqvj9k28n987unh726"])
def test_a_produced_file_is_listed_only_in_the_session_that_made_it(client, second_session_suffix):
    owner = _register(client, f"own{second_session_suffix[:4]}")
    stranger = _register(client, f"oth{second_session_suffix[:4]}")
    pid = _project(client, owner, f"W2scope-{second_session_suffix[:4]}-{_RUN}")
    x = f"ws-{pid}-1000"
    _answer(x, pid, "Summarise the pour sequence.", "Pour bay 1 first, then bay 3, then bay 2.")
    _export(client, owner, pid, x, "docx")
    _export(client, owner, pid, x, "xlsx")
    made_in_x = {d["id"] for d in _listed(client, owner, pid, x) if d["origin"] == "produced"}
    assert len(made_in_x) == 2, made_in_x

    # A NEW session of the same user: before its first message, after an
    # attach, and after an answer.
    y = f"ws-{pid}-{second_session_suffix}"
    assert _listed(client, owner, pid, y) == []
    attached = _attach(client, owner, pid, y, "invented_exam_note.txt")
    _answer(y, pid, "Read the attached note.", "The note is an invented exam note.")
    in_y = _listed(client, owner, pid, y)
    assert {d["id"] for d in in_y} == {attached}, in_y
    assert not made_in_x & {d["id"] for d in in_y}

    # X still lists its own files; another user sees none of them.
    assert made_in_x <= {d["id"] for d in _listed(client, owner, pid, x)}
    for sid in (x, y):
        assert not made_in_x & {d["id"] for d in _listed(client, stranger, pid, sid)}
