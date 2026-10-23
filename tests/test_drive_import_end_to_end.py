"""Google Drive, end to end, against a stubbed Google.

Disconnect used to delete only our copy of the token. The grant stayed live at
Google: the app kept showing as connected in the user's Google account, and
the refresh token kept working for anyone holding a copy of it.
"""
import asyncio
import time

import httpx
import pytest

from app.core import drive_auth
from tests._fixture_values import ENV_SECRET_05, PW_06


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


def _stub_revoke(monkeypatch, outcome=True):
    seen = []

    async def fake(token):
        seen.append(token)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(drive_auth, "_revoke_request", fake)
    return seen


def test_disconnect_revokes_the_whole_grant_at_google(monkeypatch):
    seen = _stub_revoke(monkeypatch)
    drive_auth.save_token("u1", {"access_token": "AT", "refresh_token": "RT",
                                 "expiry": time.time() + 999})
    out = asyncio.run(drive_auth.revoke_and_clear("u1"))
    # The refresh token, not the access token: revoking it ends the grant.
    assert seen == ["RT"]
    assert out == {"was_connected": True, "revoked": True}
    assert drive_auth.load_token("u1") is None


@pytest.mark.parametrize("outcome", [False, httpx.ConnectError("down")])
def test_a_failed_revoke_still_disconnects_and_says_so(monkeypatch, outcome):
    _stub_revoke(monkeypatch, outcome)
    drive_auth.save_token("u1", {"access_token": "AT", "refresh_token": "RT",
                                 "expiry": time.time() + 999})
    out = asyncio.run(drive_auth.revoke_and_clear("u1"))
    assert out == {"was_connected": True, "revoked": False}
    assert drive_auth.load_token("u1") is None


def test_disconnecting_when_not_connected_calls_nobody(monkeypatch):
    seen = _stub_revoke(monkeypatch)
    out = asyncio.run(drive_auth.revoke_and_clear("u1"))
    assert seen == []
    assert out == {"was_connected": False, "revoked": False}


def test_one_users_disconnect_leaves_another_users_grant_alone(monkeypatch):
    seen = _stub_revoke(monkeypatch)
    drive_auth.save_token("u1", {"access_token": "A1", "refresh_token": "R1",
                                 "expiry": time.time() + 999})
    drive_auth.save_token("u2", {"access_token": "A2", "refresh_token": "R2",
                                 "expiry": time.time() + 999})
    asyncio.run(drive_auth.revoke_and_clear("u1"))
    assert seen == ["R1"]
    assert drive_auth.load_token("u2")["refresh_token"] == "R2"


# ── the folder walker, against an in-memory Drive ───────────────────────────

from tests.test_drive_index_folder import (  # noqa: E402
    _FakeDriveClient,
    isolated_data_dir,  # noqa: F401 - fixture
)

PDF = "application/pdf"


async def _walk(monkeypatch, tree, files, project_id):
    from app.routers import drive as drive_router

    monkeypatch.setattr(drive_router.httpx, "AsyncClient",
                        lambda *a, **kw: _FakeDriveClient(tree, files))
    monkeypatch.setattr(drive_router.doc_index, "maybe_eager_index", lambda *a, **kw: None)
    return await drive_router._walk_drive_folder_into_project(
        project_id=project_id, user_id="system", access_token="t", folder_id="root")


def _visible(project_id):
    from app.core import projects

    return [d for d in projects.list_documents(project_id) if d.get("retrieval_visible", True)]


@pytest.mark.asyncio
async def test_a_file_changed_in_drive_replaces_what_was_imported(isolated_data_dir, monkeypatch):  # noqa: F811
    """Same Drive file id, new bytes: the old version must stop answering.
    It was imported as a SECOND document and the stale one stayed searchable."""
    from app.core import projects

    pid = projects.create_project(name="Drive change", user_id="system")["id"]
    tree = {"root": [{"id": "f1", "name": "spec.pdf", "mimeType": PDF}]}
    await _walk(monkeypatch, tree, {"f1": (PDF, b"curing is 7 days")}, pid)
    second = await _walk(monkeypatch, tree, {"f1": (PDF, b"curing is 10 days")}, pid)

    assert second["imported_count"] == 1
    visible = _visible(pid)
    assert len(visible) == 1, [d["original_name"] for d in visible]
    assert visible[0]["id"] == second["imported"][0]["doc_id"]
    hidden = [d for d in projects.list_documents(pid) if not d.get("retrieval_visible", True)]
    assert len(hidden) == 1 and hidden[0]["superseded_by"] == visible[0]["id"]


@pytest.mark.asyncio
async def test_an_unchanged_file_is_not_imported_twice(isolated_data_dir, monkeypatch):  # noqa: F811
    from app.core import projects

    pid = projects.create_project(name="Drive same", user_id="system")["id"]
    tree = {"root": [{"id": "f1", "name": "spec.pdf", "mimeType": PDF}]}
    files = {"f1": (PDF, b"curing is 7 days")}
    await _walk(monkeypatch, tree, files, pid)
    again = await _walk(monkeypatch, tree, files, pid)
    assert again["imported_count"] == 0
    assert len(projects.list_documents(pid)) == 1


@pytest.mark.asyncio
async def test_an_empty_file_is_reported_not_indexed(isolated_data_dir, monkeypatch):  # noqa: F811
    from app.core import projects
    from app.routers import drive as drive_router

    class _Empty(_FakeDriveClient):
        async def get(self, url, **kw):
            r = await super().get(url, **kw)
            if (kw.get("params") or {}).get("alt") == "media" and "/files/e1" in url:
                r.content = b""
            return r

    pid = projects.create_project(name="Drive empty", user_id="system")["id"]
    tree = {"root": [{"id": "e1", "name": "empty.pdf", "mimeType": PDF},
                     {"id": "f1", "name": "spec.pdf", "mimeType": PDF},
                     {"id": "d1", "name": "drawing.dwg", "mimeType": "application/acad"}]}
    files = {"e1": (PDF, b""), "f1": (PDF, b"ok"), "d1": ("application/acad", b"x")}
    monkeypatch.setattr(drive_router.httpx, "AsyncClient", lambda *a, **kw: _Empty(tree, files))
    monkeypatch.setattr(drive_router.doc_index, "maybe_eager_index", lambda *a, **kw: None)
    out = await drive_router._walk_drive_folder_into_project(
        project_id=pid, user_id="system", access_token="t", folder_id="root",
        include_extensions=[".pdf"])

    assert [i["name"] for i in out["imported"]] == ["spec.pdf"]      # the job went on
    reasons = {s["name"]: s["reason"] for s in out["skipped"]}
    assert "empty" in reasons["empty.pdf"]
    assert "allowlist" in reasons["drawing.dwg"]


@pytest.mark.asyncio
async def test_a_file_deleted_in_drive_stays_in_the_project(isolated_data_dir, monkeypatch):  # noqa: F811
    """Decided from the code, and deliberately: an imported document is the
    project's record. Re-walking a folder never deletes; the user removes a
    document in the project."""
    from app.core import projects

    pid = projects.create_project(name="Drive delete", user_id="system")["id"]
    await _walk(monkeypatch, {"root": [{"id": "f1", "name": "spec.pdf", "mimeType": PDF}]},
                {"f1": (PDF, b"kept")}, pid)
    await _walk(monkeypatch, {"root": []}, {}, pid)
    assert [d["original_name"] for d in _visible(pid)] == ["spec.pdf"]


# ── one user's connection is never another user's ───────────────────────────

def _user(c):
    import uuid

    email = f"drv-{uuid.uuid4().hex[:8]}@x.com"
    c.post("/v1/users/register", json={"email": email, "password": PW_06})
    body = c.post("/v1/users/login", json={"email": email, "password": PW_06}).json()
    me = c.get("/v1/users/me", headers={"Authorization": f"Bearer {body['token']}"}).json()
    return {"Authorization": f"Bearer {body['token']}"}, (me.get("user_id") or me.get("id"))


def test_user_y_cannot_list_or_import_through_user_xs_connection(monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "cid")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", ENV_SECRET_05)
    with TestClient(app) as c:
        x, x_id = _user(c)
        y, _ = _user(c)
        drive_auth.save_token(x_id, {"access_token": "AX", "refresh_token": "RX",
                                     "expiry": time.time() + 999, "email": "x@example.com"})
        assert c.get("/v1/drive/status", headers=x).json()["connected"] is True
        assert c.get("/v1/drive/status", headers=y).json()["connected"] is False
        assert c.get("/v1/drive/files", headers=y).status_code == 409

        x_project = c.post("/v1/projects", headers=x, json={"name": "X only"}).json()["id"]
        # Y, not connected and not the owner, cannot import into X's project.
        r = c.post(f"/v1/projects/{x_project}/drive/index-folder", headers=y,
                   json={"folder_id": "root"})
        assert r.status_code == 404, r.text
        # Nor read X's import job.
        from app.routers import drive as drive_router

        drive_router._DRIVE_FOLDER_JOBS["job-x"] = {"job_id": "job-x", "status": "done",
                                                    "project_id": x_project}
        try:
            assert c.get(f"/v1/projects/{x_project}/drive/index-folder/job/job-x",
                         headers=x).status_code == 200
            assert c.get(f"/v1/projects/{x_project}/drive/index-folder/job/job-x",
                         headers=y).status_code == 404
        finally:
            drive_router._DRIVE_FOLDER_JOBS.pop("job-x", None)
