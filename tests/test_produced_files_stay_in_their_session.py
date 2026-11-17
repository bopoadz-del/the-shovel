"""Exam (advisor 10:30Z, 16 Nov): a file the product produced in chat
session X is listed in X only -- never in a NEW session Y of the same user,
never for another user.

The first test reads the "This session" route. The second reads the
project's own document list (the project detail and its paginated pages),
which every session of every user with access shows: a session's files,
uploaded or produced, are not listed there, so they cannot appear in another
session, while the project's own documents still are.

TESTER (04:20Z) opened a new session and its "This session" group listed two
files "made by the product in this chat session" from an earlier session.
This exam builds that sequence end to end through the routes the browser
uses: answer and export in X, then open Y (a client-made session id, as the
"New chat" button makes them), attach a file in Y, and read Y's list.

Every project, user, session and answer here is created by the test.
"""
from __future__ import annotations

import io
import json
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


def _store_metadata_as_json_string(doc_id: str, meta: dict) -> None:
    """Write ``meta`` the way some live rows hold it: one JSON string."""
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, doc_id)
        row.metadata_ = json.dumps(meta)
        session.commit()


def _clear_metadata(doc_id: str) -> None:
    """Leave a row's metadata SQL NULL, as a bulk insert does."""
    from sqlalchemy import null, update

    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        session.execute(update(Document).where(Document.id == doc_id).values(metadata_=null()))
        session.commit()


def _project_list(client, headers, pid) -> tuple[list[str], dict]:
    detail = client.get(f"/v1/projects/{pid}", headers=headers)
    assert detail.status_code == 200, detail.text
    return [d["id"] for d in detail.json()["documents"]], detail.json()


@pytest.mark.parametrize("second_session_suffix", ["2000", "mwiqvj9k28n987unh726"])
def test_the_projects_document_list_holds_no_sessions_files(client, second_session_suffix):
    from app.core import projects as store

    owner = _register(client, f"plist-own{second_session_suffix[:4]}")
    stranger = _register(client, f"plist-oth{second_session_suffix[:4]}")
    owner_id = client.get("/v1/users/me", headers=owner).json()
    owner_id = owner_id.get("user_id") or owner_id.get("id")
    pid = _project(client, owner, f"W2scope-plist-{second_session_suffix[:4]}-{_RUN}")
    store.approve_project(pid)  # shared: every user opens it, as on Master Corpus

    # The project's own documents: an upload with no session, a legacy row
    # whose metadata is one JSON string, a row that names the key with no
    # session in it, a row stored with no metadata and a bulk-inserted row
    # whose metadata is SQL NULL.
    plain = client.post(
        f"/v1/projects/{pid}/documents",
        files={"file": ("invented_site_rules.txt", io.BytesIO(b"Invented site rules.\n"), "text/plain")},
        headers=owner,
    )
    assert plain.status_code in (200, 201), plain.text
    plain_id = (plain.json().get("document") or plain.json())["id"]
    legacy_project = store.add_document(pid, "invented_legacy_register.md")["id"]
    _store_metadata_as_json_string(legacy_project, {"provenance": "drive", "uploader_id": owner_id})
    keyed_no_session = store.add_document(
        pid, "invented_keyed_note.md", metadata={"session_id": None, "provenance": "drive"})["id"]
    no_metadata = store.add_document(pid, "invented_drawing_index.pdf")["id"]
    bulk_null = store.add_document(pid, "invented_bulk_schedule.pdf")["id"]
    _clear_metadata(bulk_null)
    projects_own = {plain_id, legacy_project, keyed_no_session, no_metadata, bulk_null}

    # Sessions' files, all newer than the project's: two produced in X, one
    # attached in Y, and a legacy session row held as one JSON string.
    x, y = f"ws-{pid}-1000", f"ws-{pid}-{second_session_suffix}"
    _answer(x, pid, "Summarise the pour sequence.", "Pour bay 1 first, then bay 3, then bay 2.")
    _export(client, owner, pid, x, "docx")
    _export(client, owner, pid, x, "xlsx")
    made_in_x = {d["id"] for d in _listed(client, owner, pid, x) if d["origin"] == "produced"}
    assert len(made_in_x) == 2, made_in_x
    attached_in_y = _attach(client, owner, pid, y, "invented_exam_note.txt")
    legacy_session = store.add_document(pid, "invented_legacy_session_note.md")["id"]
    _store_metadata_as_json_string(
        legacy_session, {"session_id": x, "uploader_id": owner_id, "origin": "uploaded"})
    sessions_files = made_in_x | {attached_in_y, legacy_session}

    for viewer in (owner, stranger):
        ids, detail = _project_list(client, viewer, pid)
        assert set(ids) == projects_own, ids
        assert detail["document_count"] == len(projects_own)
        assert detail["held_document_count"] == len(projects_own)

        # Every page is the project's own documents: the session rows are
        # left out by the query, so a page is never short or empty.
        paged = []
        for offset in range(len(projects_own) + 1):
            page = client.get(
                f"/v1/projects/{pid}/documents?limit=1&offset={offset}", headers=viewer)
            assert page.status_code == 200, page.text
            assert page.json()["total"] == len(projects_own)
            paged += [d["id"] for d in page.json()["documents"]]
        assert paged == ids, (paged, ids)
        assert not sessions_files & set(paged)

    # The session files are still there for the user who made them, each in
    # its own session.
    assert made_in_x | {legacy_session} == {d["id"] for d in _listed(client, owner, pid, x)}
    assert {attached_in_y} == {d["id"] for d in _listed(client, owner, pid, y)}
