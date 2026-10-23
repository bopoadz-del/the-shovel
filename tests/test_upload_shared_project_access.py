"""Upload access follows OPEN access (2026-07-25 triage follow-up).

The pilot works on shared projects: anyone who can open a project can add
documents to it. The old owner-only gate 404'd the upload button on every
project the caller didn't create — reported as "upload is broken".
Destructive mutations stay owner-only.
"""
import io
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from app.core import users as users_store
from tests._fixture_values import PW_10


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, email):
    r = client.post("/v1/users/register", json={"email": email, "password": PW_10})
    assert r.status_code == 201, r.text
    r = client.post("/v1/users/login", json={"email": email, "password": PW_10})
    return r.json()["token"], r.json()["user"]["id"]


def _upload(client, token, pid, name="note.txt"):
    return client.post(
        f"/v1/projects/{pid}/documents",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (name, io.BytesIO(b"hello shared upload"), "text/plain")},
    )


def test_admin_can_upload_to_another_users_project(client):
    tok_a, uid_a = _register(client, "owner-up@example.com")
    tok_b, uid_b = _register(client, "admin-up@example.com")
    from app.core.db import SessionLocal
    from app.core.models import User
    with SessionLocal() as db:
        u = db.get(User, uid_b)
        u.role = "admin"
        db.commit()

    r = client.post("/v1/projects", json={"name": "Owner A project up"},
                    headers={"Authorization": f"Bearer {tok_a}"})
    pid = r.json()["id"]

    r = _upload(client, tok_b, pid)
    assert r.status_code == 201, r.text
    assert r.json()["document"]["original_name"] == "note.txt"


def test_user_can_upload_to_admin_approved_shared_project(client):
    tok_a, uid_a = _register(client, "owner-c@example.com")
    tok_b, _ = _register(client, "tester-c@example.com")

    r = client.post("/v1/projects", json={"name": "Shared pilot project up"},
                    headers={"Authorization": f"Bearer {tok_a}"})
    pid = r.json()["id"]
    # mark as admin-approved shared platform project
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as db:
        p = db.get(Project, pid)
        p.origin = "admin_drive_approved"
        p.is_approved = True
        db.commit()

    r = _upload(client, tok_b, pid, "tester-doc.txt")
    assert r.status_code == 201, r.text


def test_regular_user_still_404_on_private_project(client):
    tok_a, _ = _register(client, "owner-d@example.com")
    tok_b, _ = _register(client, "stranger-d@example.com")

    r = client.post("/v1/projects", json={"name": "Private project up"},
                    headers={"Authorization": f"Bearer {tok_a}"})
    pid = r.json()["id"]

    r = _upload(client, tok_b, pid)
    assert r.status_code == 404
