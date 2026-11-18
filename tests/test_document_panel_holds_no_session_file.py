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


# ── a second user on the same project (grader 12:50Z, bar 4) ────────────────

def _share(project_id: str) -> None:
    """Make the project one a second user can open (an approved platform
    project), the way tests/test_chat_session_scope.py shares one."""
    from app.core.db import SessionLocal
    from app.core.models import Project

    with SessionLocal() as db:
        row = db.get(Project, project_id)
        row.origin = "admin_drive_approved"
        row.is_approved = True
        db.commit()


@pytest.mark.parametrize("name,body", [
    ("grout_mix_record.txt", b"Grout 1:3 by volume, 28-day 40 MPa."),
    ("anchor_bolt_list.txt", b"M24 anchors at base plates B1 to B9."),
])
def test_a_second_user_sees_the_session_file_nowhere(client, name, body):
    first = _headers(client, f"u1{name[:5]}")
    second = _headers(client, f"u2{name[:5]}")
    pid = _project(client, first, f"Panel-2u-{name[:5]}-{_RUN}")
    _share(pid)
    own = f"ws-{pid}-7000"
    session_doc = _upload(client, first, pid, name, body, session_id=own)
    project_doc = _upload(client, first, pid, "gate_hours.txt", b"Gate open 06:00 to 18:00.")

    detail_ids, page_ids = _panel_ids(client, second, pid)
    assert session_doc["id"] not in detail_ids and session_doc["id"] not in page_ids
    assert project_doc["id"] in detail_ids and project_doc["id"] in page_ids
    # Neither the second user's own chat strip nor the first user's chat id
    # lists it for the second user.
    assert session_doc["id"] not in _strip_ids(client, second, pid, f"ws-{pid}-8000")
    strip = client.get(f"/v1/projects/{pid}/conversations/{own}/documents", headers=second)
    assert strip.status_code == 404 or session_doc["id"] not in {d["id"] for d in strip.json()["documents"]}


# ── by id: delete, fetch, ask (advisor ruling 12:39Z) ───────────────────────

def _doc_url(pid, did, tail=""):
    return f"/v1/projects/{pid}/documents/{did}{tail}"


@pytest.mark.parametrize("name,body", [
    ("formwork_strike_log.txt", b"Soffit forms struck after 10 days at pour 4."),
    ("pile_test_note.txt", b"Static load test on pile P-17 to 2.0 x working load."),
])
def test_a_session_file_is_deleted_only_from_its_own_chat_by_its_maker(client, name, body):
    first = _headers(client, f"d1{name[:5]}")
    second = _headers(client, f"d2{name[:5]}")
    pid = _project(client, first, f"Panel-del-{name[:5]}-{_RUN}")
    _share(pid)
    own, other = f"ws-{pid}-1100", f"ws-{pid}-2200"
    did = _upload(client, first, pid, name, body, session_id=own)["id"]

    assert client.delete(_doc_url(pid, did), headers=first).status_code == 404
    assert client.delete(_doc_url(pid, did), headers=first, params={"session_id": other}).status_code == 404
    assert client.delete(_doc_url(pid, did), headers=second, params={"session_id": own}).status_code == 404
    assert did in _strip_ids(client, first, pid, own)  # still there

    res = client.delete(_doc_url(pid, did), headers=first, params={"session_id": own})
    assert res.status_code == 200, res.text
    assert did not in _strip_ids(client, first, pid, own)


@pytest.mark.parametrize("name,body", [
    ("cube_results_w12.txt", b"Cubes 7-day 31 MPa, 28-day 44 MPa."),
    ("crane_lift_plan.txt", b"Lift 12 t at 18 m radius, 50 t crawler."),
])
def test_the_maker_on_a_shared_project_deletes_their_session_file_from_its_chat(client, name, body):
    """The user who attached it deletes it, though they do not own the project."""
    owner = _headers(client, f"o{name[:5]}")
    maker = _headers(client, f"m{name[:5]}")
    pid = _project(client, owner, f"Panel-mk-{name[:5]}-{_RUN}")
    _share(pid)
    own = f"ws-{pid}-3300"
    did = _upload(client, maker, pid, name, body, session_id=own)["id"]
    assert client.delete(_doc_url(pid, did), headers=owner, params={"session_id": own}).status_code == 404
    res = client.delete(_doc_url(pid, did), headers=maker, params={"session_id": own})
    assert res.status_code == 200, res.text


@pytest.mark.parametrize("name", ["site_rules_a.txt", "site_rules_b.txt"])
def test_a_project_document_is_deleted_by_its_owner_as_before(client, name):
    owner = _headers(client, f"po{name[-5]}")
    other = _headers(client, f"px{name[-5]}")
    pid = _project(client, owner, f"Panel-pd-{name[-5]}-{_RUN}")
    _share(pid)
    did = _upload(client, owner, pid, name, b"Hard hats at all times.")["id"]
    assert client.delete(_doc_url(pid, did), headers=other).status_code == 404  # owner-only, as before
    res = client.delete(_doc_url(pid, did), headers=owner)
    assert res.status_code == 200, res.text
    _detail, page_ids = _panel_ids(client, owner, pid)
    assert did not in page_ids


@pytest.mark.parametrize("method,tail,handler", [
    ("DELETE", "", "delete_document"),
    ("GET", "/preview", "preview_document"),
    ("GET", "/preview/raw", "preview_document_raw"),
    ("GET", "/download", "download_document"),
])
def test_the_by_id_routes_are_the_handlers_these_tests_call(method, tail, handler):
    """The HTTP routes exercised below are app.routers.projects' handlers."""
    from app.routers import projects as projects_router

    path = "/v1/projects/{project_id}/documents/{document_id}" + tail
    found = [r.endpoint for r in app.routes
             if getattr(r, "path", None) == path and method in (getattr(r, "methods", None) or ())]
    assert found and found[0] is getattr(projects_router, handler)


@pytest.mark.parametrize("name,body", [
    ("joint_layout.txt", b"Joints at 6 m centres."),
    ("drain_falls.txt", b"Falls of 1 in 80 to gullies."),
])
def test_raw_preview_of_a_session_file_is_reached_only_from_its_own_chat(client, name, body):
    """``/preview/raw`` serves PDFs only: from the file's own chat a text file
    is reached and refused as not a PDF (400); from anywhere else it is not
    found (404)."""
    headers = _headers(client, f"raw{name[:4]}")
    pid = _project(client, headers, f"Panel-raw-{name[:4]}-{_RUN}")
    own, other = f"ws-{pid}-8800", f"ws-{pid}-9900"
    did = _upload(client, headers, pid, name, body, session_id=own)["id"]
    url = _doc_url(pid, did, "/preview/raw")
    assert client.get(url, headers=headers).status_code == 404
    assert client.get(url, headers=headers, params={"session_id": other}).status_code == 404
    assert client.get(url, headers=headers, params={"session_id": own}).status_code == 400


@pytest.mark.parametrize("tail", ["/preview", "/download"])
def test_a_session_file_is_fetched_by_id_only_from_its_own_chat(client, tail):
    first = _headers(client, f"f1{tail[1:4]}")
    second = _headers(client, f"f2{tail[1:4]}")
    pid = _project(client, first, f"Panel-fetch-{tail[1:4]}-{_RUN}")
    _share(pid)
    own, other = f"ws-{pid}-4400", f"ws-{pid}-5500"
    did = _upload(client, first, pid, "rebar_schedule_s3.txt", b"T16 at 150 c/c both ways.",
                  session_id=own)["id"]
    project_did = _upload(client, first, pid, "access_plan.txt", b"Access from the north gate.")["id"]
    url = _doc_url(pid, did, tail)

    assert client.get(url, headers=first).status_code == 404
    assert client.get(url, headers=first, params={"session_id": other}).status_code == 404
    assert client.get(url, headers=second, params={"session_id": own}).status_code == 404
    res = client.get(url, headers=first, params={"session_id": own})
    assert res.status_code == 200, res.text
    # A project document needs no chat, for anyone who can open the project.
    assert client.get(_doc_url(pid, project_did, tail), headers=second).status_code == 200


@pytest.fixture
def _llm(monkeypatch):
    from app.agents.runtime import Agent
    from tests._fixture_values import STUB_TOKEN_02

    async def _reply(self, messages, api_key, project_id=None, **kwargs):
        return {"status": "success", "choice": {"message": {"content": "noted", "tool_calls": []}}, "raw": {}}

    monkeypatch.setattr(Agent, "_call_llm", _reply)
    monkeypatch.setattr("app.agents.runtime.project_is_rag_ready", lambda _pid: True)
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", STUB_TOKEN_02)


@pytest.mark.parametrize("name,body", [
    ("waterstop_detail.txt", b"PVC waterstop 240 mm at all construction joints."),
    ("sleeve_schedule.txt", b"Sleeves 150 mm at grid C/4 and C/7."),
])
def test_asking_about_a_session_file_by_id_works_only_from_its_own_chat(client, _llm, name, body):
    headers = _headers(client, f"a{name[:5]}")
    pid = _project(client, headers, f"Panel-ask-{name[:5]}-{_RUN}")
    own, other = f"ws-{pid}-6600", f"ws-{pid}-7700"
    did = _upload(client, headers, pid, name, body, session_id=own)["id"]

    def ask(conversation_id):
        return client.post("/v1/chat/stream", headers=headers, json={
            "message": "What does this document say?", "project_id": pid,
            "conversation_id": conversation_id, "document_ids": [did]})

    assert ask(other).status_code == 404
    res = ask(own)
    assert res.status_code == 200, res.text


# ── a session file with no stored original, from its own chat ───────────────

def _session_file_without_original(client, headers, pid, sid, name, tmp_path) -> str:
    """A session file whose original was never stored (its maker read off a
    real upload's metadata), as the preview/download fallbacks see one."""
    from app.core import projects as store

    maker = _upload(client, headers, pid, "probe.txt", b"probe", session_id=sid)["metadata"]["uploader_id"]
    return store.add_document(
        project_id=pid, original_name=name, file_path=str(tmp_path / "never-written.bin"), size=0,
        metadata={"session_id": sid, "uploader_id": maker, "origin": "uploaded"})["id"]


@pytest.mark.parametrize("tail,indexed,expected", [
    ("/preview", "Laps 40 diameters.", 200),
    ("/download", "Laps 40 diameters.", 200),
    ("/preview", "", 404),
    ("/download", "", 404),
    ("/preview/raw", "", 404),
])
def test_a_session_file_without_its_original_falls_back_only_in_its_own_chat(
        client, monkeypatch, tmp_path, tail, indexed, expected):
    from app.routers import projects as projects_router

    headers = _headers(client, f"nf{tail[1:4]}{len(indexed)}")
    pid = _project(client, headers, f"Panel-nofile-{tail[1:4]}-{len(indexed)}-{_RUN}")
    own = f"ws-{pid}-1212"
    did = _session_file_without_original(client, headers, pid, own, "lap_schedule.pdf", tmp_path)
    monkeypatch.setattr(projects_router, "_indexed_text", lambda _owner, _did: indexed)

    url = _doc_url(pid, did, tail)
    assert client.get(url, headers=headers).status_code == 404  # no chat: not found
    res = client.get(url, headers=headers, params={"session_id": own})
    assert res.status_code == expected, res.text
    if expected == 200:
        assert indexed in res.text


# ── by id: the metadata update and the replace-bytes routes (#451 follow-up) ──

_DRIVE_ID = "invented_drive_id_0123456789"


@pytest.mark.parametrize("method,tail,handler", [
    ("PATCH", "", "update_document"),
    ("POST", "/bytes", "restore_document_bytes"),
])
def test_the_write_by_id_routes_are_the_handlers_these_tests_call(method, tail, handler):
    test_the_by_id_routes_are_the_handlers_these_tests_call(method, tail, handler)


def _replace(client, headers, pid, did, name, body, session_id=None):
    return client.post(_doc_url(pid, did, "/bytes"), headers=headers,
                       params={"session_id": session_id} if session_id else None,
                       files={"file": (name, body, "text/plain")})


def _patch(client, headers, pid, did, session_id=None):
    return client.patch(_doc_url(pid, did), headers=headers,
                        params={"session_id": session_id} if session_id else None,
                        json={"drive_file_id": _DRIVE_ID})


@pytest.mark.parametrize("name,body", [
    ("rebar_schedule_l3.txt", b"T16 at 150 centres both ways."),
    ("waterproofing_note.txt", b"Two coats of bitumen membrane to the retaining wall."),
])
def test_a_session_file_is_replaced_only_from_its_own_chat_by_its_maker(client, name, body):
    owner = _headers(client, f"rb1{name[:5]}")
    maker = _headers(client, f"rb2{name[:5]}")
    pid = _project(client, owner, f"Panel-rb-{name[:5]}-{_RUN}")
    _share(pid)
    own, other = f"ws-{pid}-4100", f"ws-{pid}-4200"
    did = _upload(client, maker, pid, name, body, session_id=own)["id"]
    new = body + b" Revised."

    assert _replace(client, maker, pid, did, name, new).status_code == 404
    assert _replace(client, maker, pid, did, name, new, session_id=other).status_code == 404
    assert _replace(client, owner, pid, did, name, new, session_id=own).status_code == 404
    res = _replace(client, maker, pid, did, name, new, session_id=own)
    assert res.status_code == 200, res.text
    assert res.json()["document"]["id"] == did
    got = client.get(_doc_url(pid, did, "/download"), headers=maker, params={"session_id": own})
    assert got.status_code == 200 and got.content == new


@pytest.mark.parametrize("name", ["permit_hours_a.txt", "permit_hours_b.txt"])
def test_a_project_document_is_replaced_with_no_chat_as_before(client, name):
    owner = _headers(client, f"rp{name[-5]}")
    pid = _project(client, owner, f"Panel-rp-{name[-5]}-{_RUN}")
    did = _upload(client, owner, pid, name, b"Permits 07:00 to 17:00.")["id"]
    res = _replace(client, owner, pid, did, name, b"Permits 06:00 to 18:00.")
    assert res.status_code == 200, res.text
    assert res.json()["document"]["id"] == did


@pytest.mark.parametrize("name,body", [
    ("survey_benchmarks.txt", b"TBM 1 at +12.450 m."),
    ("dewatering_log.txt", b"Wellpoints drawn down 2.4 m by day 3."),
])
def test_a_session_file_is_updated_only_from_its_own_chat_by_its_maker(client, name, body):
    owner = _headers(client, f"pt1{name[:5]}")
    maker = _headers(client, f"pt2{name[:5]}")
    pid = _project(client, owner, f"Panel-pt-{name[:5]}-{_RUN}")
    _share(pid)
    own, other = f"ws-{pid}-5100", f"ws-{pid}-5200"
    did = _upload(client, maker, pid, name, body, session_id=own)["id"]

    assert _patch(client, maker, pid, did).status_code == 404
    assert _patch(client, maker, pid, did, session_id=other).status_code == 404
    # The project's owner is not the file's maker: not found, from its chat too.
    assert _patch(client, owner, pid, did, session_id=own).status_code == 404
    res = _patch(client, maker, pid, did, session_id=own)
    assert res.status_code == 200, res.text
    assert (res.json().get("metadata") or {}).get("drive_file_id") == _DRIVE_ID


@pytest.mark.parametrize("name", ["hoarding_line_a.txt", "hoarding_line_b.txt"])
def test_a_project_document_is_updated_by_its_owner_as_before(client, name):
    owner = _headers(client, f"uo{name[-5]}")
    other = _headers(client, f"ux{name[-5]}")
    pid = _project(client, owner, f"Panel-uo-{name[-5]}-{_RUN}")
    _share(pid)
    did = _upload(client, owner, pid, name, b"Hoarding 2.4 m high on the north line.")["id"]
    assert _patch(client, other, pid, did).status_code == 403  # owner or admin only, as before
    res = _patch(client, owner, pid, did)
    assert res.status_code == 200, res.text
    assert (res.json().get("metadata") or {}).get("drive_file_id") == _DRIVE_ID
