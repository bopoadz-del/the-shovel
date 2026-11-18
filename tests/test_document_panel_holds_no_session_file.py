"""The project's document panel holds no chat session's file (TESTER 10:40Z).

The panel is filled from two server routes and one client step:

* ``GET /v1/projects/{id}`` (the first page, on opening the project) and
  ``GET /v1/projects/{id}/documents`` (further pages): a session file is in
  neither, from any chat, because neither route binds a session scope.
* the upload response (``POST /v1/projects/{id}/documents``): the client
  added this row to the project's list by itself, and that list outlives
  the chat. The row was a session file, so the panel showed it in every
  later chat. The client now keeps such a row out of the project's list
  (``belongsToProjectList`` in ``frontend/src/chat/sessionDocuments.js``)
  by reading ``metadata.session_id`` on the row; this test pins that field
  on the response, so the rule cannot lose the field it reads.
* ``GET .../conversations/{cid}/documents`` is the session's own strip: the
  file is there, in its own chat only.

The session-scope enumeration test (``test_session_scope_enumeration.py``)
walks every server function that READS documents; the leak was a row the
client wrote into its own list from a write's response, which no reader
walk can see. Every project, user, session and file here is made by the test.
"""
from __future__ import annotations

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


def _headers(client, suffix: str) -> dict:
    email = f"panel-{suffix}-{_RUN}@x.com"
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


def _upload(client, headers, pid, name, body, session_id=None) -> dict:
    data = {"session_id": session_id} if session_id else None
    res = client.post(
        f"/v1/projects/{pid}/documents", headers=headers, data=data,
        files={"file": (name, body, "text/plain")},
    )
    assert res.status_code == 201, res.text
    return res.json()["document"]


def _panel_ids(client, headers, pid) -> tuple[set, set]:
    """Ids the panel can show: the project detail's first page and the page route."""
    detail = client.get(f"/v1/projects/{pid}", headers=headers)
    assert detail.status_code == 200, detail.text
    page = client.get(f"/v1/projects/{pid}/documents", headers=headers)
    assert page.status_code == 200, page.text
    return ({d["id"] for d in detail.json()["documents"]},
            {d["id"] for d in page.json()["documents"]})


def _strip_ids(client, headers, pid, sid) -> set:
    res = client.get(f"/v1/projects/{pid}/conversations/{sid}/documents", headers=headers)
    assert res.status_code == 200, res.text
    return {d["id"] for d in res.json()["documents"]}


@pytest.mark.parametrize("name,body", [
    ("bearing_pad_schedule.txt", b"Elastomeric pads 300 x 400 x 41 at piers P2 to P5."),
    ("curing_compound_note.txt", b"Apply curing compound within 30 minutes of finishing."),
])
def test_a_session_file_is_in_its_own_strip_and_in_no_project_panel(client, name, body):
    headers = _headers(client, name[:6])
    pid = _project(client, headers, f"Panel-{name[:6]}-{_RUN}")
    own, other = f"ws-{pid}-1000", f"ws-{pid}-2000"
    project_doc = _upload(client, headers, pid, "site_rules.txt", b"Hard hats at all times.")
    session_doc = _upload(client, headers, pid, name, body, session_id=own)

    # The panel's routes take no chat id: what they list is what every chat,
    # the uploading one included, shows as the project's documents.
    detail_ids, page_ids = _panel_ids(client, headers, pid)
    assert session_doc["id"] not in detail_ids
    assert session_doc["id"] not in page_ids
    assert project_doc["id"] in detail_ids and project_doc["id"] in page_ids

    # The session's own strip lists it; another chat's strip does not.
    assert session_doc["id"] in _strip_ids(client, headers, pid, own)
    assert session_doc["id"] not in _strip_ids(client, headers, pid, other)


@pytest.mark.parametrize("session", [True, False])
def test_the_upload_response_names_the_session_the_client_rule_reads(client, session):
    """``belongsToProjectList`` reads ``document.metadata.session_id`` off the
    upload response. A session upload must carry it; a project upload must not
    (or the client would drop a real project document from the panel)."""
    headers = _headers(client, f"resp{int(session)}")
    pid = _project(client, headers, f"Panel-resp-{int(session)}-{_RUN}")
    sid = f"ws-{pid}-3000"
    doc = _upload(client, headers, pid, "pour_card_7.txt", b"Pour 7 slump 90 mm.",
                  session_id=sid if session else None)
    meta = doc.get("metadata") or {}
    if session:
        assert meta.get("session_id") == sid
    else:
        assert not meta.get("session_id")
        detail_ids, page_ids = _panel_ids(client, headers, pid)
        assert doc["id"] in detail_ids and doc["id"] in page_ids
