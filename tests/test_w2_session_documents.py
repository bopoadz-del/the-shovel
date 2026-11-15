"""W2 (exam 11.3, backend half): each chat session has its own documents.

* A file uploaded inside a session records that session. It is listed under
  that session and under no other.
* A session id that names a different project is refused.
* A session upload is stored and pinned. It is never handed to the project
  indexer, even when the caller may add to the project's search.
* A question asked in the session is told about the session's files without a
  marker or a pin. A question in another session of the same project is not.
* One user's session files never reach another user's session.

Every project, user, session and document here is created by the test.
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
    email = f"w2-{suffix}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _project(client, headers, name: str) -> str:
    res = client.post("/v1/projects", json={"name": name}, headers=headers)
    assert res.status_code in (200, 201), res.text
    return res.json()["id"]


def _upload(client, headers, pid, name, body, session_id=None):
    data = {"session_id": session_id} if session_id else None
    return client.post(
        f"/v1/projects/{pid}/documents", headers=headers, data=data,
        files={"file": (name, body, "text/plain")},
    )


def _session_docs(client, headers, pid, sid):
    res = client.get(
        f"/v1/projects/{pid}/conversations/{sid}/documents", headers=headers,
    )
    assert res.status_code == 200, res.text
    return res.json()["documents"]


# ── a session upload is listed under its session only ────────────────────────

@pytest.mark.parametrize("name,body", [
    ("kerb_setting_out.txt", b"Kerb line offset 450 mm from channel."),
    ("formwork_strike_times.txt", b"Soffit forms struck after 7 days."),
])
def test_a_session_upload_is_listed_under_that_session_only(client, name, body):
    headers = _register(client, f"list{name[:4]}")
    pid = _project(client, headers, f"W2-list-{name[:4]}-{_RUN}")
    mine, other = f"ws-{pid}-1000", f"ws-{pid}-2000"
    res = _upload(client, headers, pid, name, body, session_id=mine)
    assert res.status_code == 201, res.text
    doc = res.json()["document"]

    listed = _session_docs(client, headers, pid, mine)
    assert [d["id"] for d in listed] == [doc["id"]]
    assert listed[0]["origin"] == "uploaded"
    assert listed[0]["original_name"] == name
    assert _session_docs(client, headers, pid, other) == []


@pytest.mark.parametrize("with_session", [False, True])
def test_a_plain_upload_belongs_to_no_session(client, with_session):
    headers = _register(client, f"plain{int(with_session)}")
    pid = _project(client, headers, f"W2-plain-{int(with_session)}-{_RUN}")
    sid = f"ws-{pid}-3000"
    res = _upload(client, headers, pid, "site_diary_week_9.txt", b"Rain on day 2.",
                  session_id=sid if with_session else None)
    assert res.status_code == 201, res.text
    listed = _session_docs(client, headers, pid, sid)
    assert len(listed) == (1 if with_session else 0)


@pytest.mark.parametrize("bad", ["foreign", "malformed"])
def test_a_session_id_from_another_project_is_refused(client, bad):
    headers = _register(client, f"bad{bad[:3]}")
    pid = _project(client, headers, f"W2-bad-{bad}-{_RUN}")
    other_pid = _project(client, headers, f"W2-other-{bad}-{_RUN}")
    sid = f"ws-{other_pid}-4000" if bad == "foreign" else "not-a-session"
    res = _upload(client, headers, pid, "pour_record.txt", b"Pour 3 complete.", session_id=sid)
    assert res.status_code == 400, res.text
    good = _upload(client, headers, pid, "pour_record.txt", b"Pour 3 complete.",
                   session_id=f"ws-{pid}-4000")
    assert good.status_code == 201, good.text


def test_listing_a_session_of_another_project_is_refused(client):
    headers = _register(client, "listbad")
    pid = _project(client, headers, f"W2-listbad-{_RUN}")
    other_pid = _project(client, headers, f"W2-listbad-o-{_RUN}")
    res = client.get(
        f"/v1/projects/{pid}/conversations/ws-{other_pid}-1/documents", headers=headers,
    )
    assert res.status_code == 404
    assert _session_docs(client, headers, pid, f"ws-{pid}-1") == []


def test_another_users_session_files_are_not_listed(client):
    owner = _register(client, "own")
    pid = _project(client, owner, f"W2-own-{_RUN}")
    sid = f"ws-{pid}-5000"
    assert _upload(client, owner, pid, "crane_lift_plan.txt", b"Lift 12 t at 30 m.",
                   session_id=sid).status_code == 201
    stranger = _register(client, "stranger")
    res = client.get(f"/v1/projects/{pid}/conversations/{sid}/documents", headers=stranger)
    # The stranger cannot open the project at all, so the session is not theirs.
    assert res.status_code == 404
    assert len(_session_docs(client, owner, pid, sid)) == 1


# ── a session upload is stored and pinned, never indexed ─────────────────────

@pytest.mark.parametrize("with_session,indexed", [(False, True), (True, False)])
def test_a_session_upload_is_not_handed_to_the_indexer(client, monkeypatch,
                                                       with_session, indexed):
    from app.core import doc_index, privileges

    calls: list[tuple] = []
    monkeypatch.setattr(privileges, "caller_may_add_to_project_rag", lambda _role: True)
    monkeypatch.setattr(doc_index, "maybe_eager_index", lambda *a, **k: calls.append(a))
    headers = _register(client, f"idx{int(with_session)}")
    pid = _project(client, headers, f"W2-idx-{int(with_session)}-{_RUN}")
    res = _upload(client, headers, pid, "rebar_schedule_b2.txt", b"B2 bars T16 at 200.",
                  session_id=f"ws-{pid}-6000" if with_session else None)
    assert res.status_code == 201, res.text
    assert bool(calls) is indexed
    status = ((res.json()["document"].get("metadata") or {}).get("indexing") or {})
    if with_session:
        assert status.get("status") == "not_indexed"
        assert "session" in (status.get("detail") or "").lower()


# ── a question in the session is told about the session's files ──────────────

@pytest.fixture
def seen_messages(monkeypatch):
    seen: list[list[dict]] = []

    async def _fake_llm(self, messages, api_key, project_id=None, **kwargs):
        seen.append([dict(m) for m in messages])
        return {"status": "success",
                "choice": {"message": {"content": "Read it.", "tool_calls": []}},
                "raw": {}}

    monkeypatch.setattr(Agent, "_call_llm", _fake_llm)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)
    return seen


def _attached_note(seen: list[list[dict]]) -> str:
    return "\n".join(
        str(m.get("content") or "") for turn in seen for m in turn
        if "ATTACHED DOCUMENT(S) THIS TURN" in str(m.get("content") or "")
    )


@pytest.mark.parametrize("ask_in,expect_pinned", [("same", True), ("other", False)])
def test_the_sessions_files_are_pinned_to_its_questions(client, seen_messages,
                                                        ask_in, expect_pinned):
    headers = _register(client, f"pin{ask_in}")
    pid = _project(client, headers, f"W2-pin-{ask_in}-{_RUN}")
    sid = f"ws-{pid}-7000"
    doc = _upload(client, headers, pid, "dewatering_log_march.txt",
                  b"Well 4 drawdown 2.1 m.", session_id=sid).json()["document"]
    res = client.post("/v1/chat/stream", headers=headers, json={
        "message": "What does the log say about drawdown?",
        "project_id": pid,
        "conversation_id": sid if ask_in == "same" else f"ws-{pid}-8000",
    })
    assert res.status_code == 200, res.text
    note = _attached_note(seen_messages)
    assert (doc["id"] in note) is expect_pinned
