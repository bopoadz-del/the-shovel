"""G-3 (grader, #342): listing a chat session's documents reads the
session's own rows from the database, not every document of the project.

``list_session_documents`` runs on every chat turn and every export. It used
to select every document row of the project and filter session and uploader
in Python, so the read grew with the project. Here a project holds a growing
number of other documents (other sessions, another uploader in the same
session, project uploads, and a legacy row whose metadata is stored as an
encoded JSON string); the statement the listing runs is captured from the
real engine and re-run, and the rows it returns must be the session's own,
whatever the project's size. The listing's answer must not change.

Every project, user, session and document here is created by the test.
"""
from __future__ import annotations

import json
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event

from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _store_metadata_as_json_string(doc_id: str, meta: dict) -> None:
    """Write ``meta`` the way some live rows hold it: one JSON string."""
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as session:
        row = session.get(Document, doc_id)
        row.metadata_ = json.dumps(meta)
        session.commit()


@pytest.mark.parametrize("others", [6, 120])
def test_session_listing_reads_only_the_sessions_rows(client, others):
    from app.core import projects as store
    from app.core.db import get_engine

    engine = get_engine()

    email = f"g3-{others}-{_RUN}@x.com"
    reg = client.post("/v1/users/register", json={"email": email, "password": PW_06})
    assert reg.status_code in (200, 201, 409), reg.text
    token = client.post("/v1/users/login", json={"email": email, "password": PW_06}).json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    me = client.get("/v1/users/me", headers=headers).json()
    owner, stranger = me.get("user_id") or me.get("id"), f"u-str-{_RUN}"
    res = client.post("/v1/projects", json={"name": f"G3-bounded-{others}-{_RUN}"}, headers=headers)
    assert res.status_code in (200, 201), res.text
    pid = res.json()["id"]
    sid, other_sid = f"ws-{pid}-1700000000001", f"ws-{pid}-1700000000002"

    def add(name: str, meta: dict | None) -> str:
        return store.add_document(pid, name, metadata=meta)["id"]

    mine = {
        add("pour-plan.md", {"session_id": sid, "uploader_id": owner, "origin": "uploaded"}),
        add("answer.xlsx", {"session_id": sid, "uploader_id": owner, "origin": "produced"}),
    }
    legacy_meta = {"session_id": sid, "uploader_id": owner, "origin": "uploaded"}
    legacy = add("legacy-diary.md", legacy_meta)
    _store_metadata_as_json_string(legacy, legacy_meta)
    mine.add(legacy)

    for i in range(others):
        kind = i % 4
        if kind == 0:
            add(f"other-session-{i}.md", {"session_id": other_sid, "uploader_id": owner})
        elif kind == 1:
            add(f"same-session-other-user-{i}.md", {"session_id": sid, "uploader_id": stranger})
        elif kind == 2:
            add(f"project-upload-{i}.pdf", None)
        else:
            doc = add(f"legacy-other-{i}.md", {"session_id": other_sid, "uploader_id": owner})
            _store_metadata_as_json_string(doc, {"session_id": other_sid, "uploader_id": owner})

    captured: list[tuple[str, object]] = []

    def capture(conn, cursor, statement, parameters, context, executemany):
        if "FROM documents" in statement and "metadata" in statement and "ORDER BY" in statement:
            captured.append((statement, parameters))

    event.listen(engine, "after_cursor_execute", capture)
    try:
        listed = store.list_session_documents(pid, sid, owner)
    finally:
        event.remove(engine, "after_cursor_execute", capture)

    # The answer is the session's own files, the legacy row included.
    assert {d["id"] for d in listed} == mine
    assert {d["origin"] for d in listed} == {"uploaded", "produced"}

    # The read is bounded by the session, not by the project.
    assert len(captured) == 1, [s for s, _ in captured]
    statement, parameters = captured[0]
    with engine.connect() as conn:
        rows_read = conn.exec_driver_sql(statement, parameters).fetchall()
    assert len(rows_read) == len(mine), (len(rows_read), others)
