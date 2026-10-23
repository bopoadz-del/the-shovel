"""Tests for the Project entity, readiness gate, and execution-intent model.

Roadmap V2 · Part 0 (0.1 / 0.2 / 0.3).
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import requires_construction_kit
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client, name="Test Project"):
    r = client.post("/v1/projects", json={"name": name, "client": "ACME"}, headers=H)
    assert r.status_code == 201, r.text
    return r.json()


def _attach(client, pid, filename, role=None):
    files = {"file": (filename, b"%PDF-1.4 test content", "application/pdf")}
    data = {"role": role} if role else {}
    return client.post(
        f"/v1/projects/{pid}/documents", files=files, data=data, headers=H
    )


# ── 0.1 Project entity ──────────────────────────────────────────────────────

def test_create_and_get_project(client):
    proj = _new_project(client, "the client Phase 1")
    assert proj["name"] == "the client Phase 1"
    assert proj["client"] == "ACME"
    assert proj["status"] == "active"

    got = client.get(f"/v1/projects/{proj['id']}", headers=H)
    assert got.status_code == 200
    assert got.json()["id"] == proj["id"]
    assert got.json()["documents"] == []


# ── F4: project location (daily_site_report weather source) ─────────────────

def test_create_project_with_location(client):
    r = client.post("/v1/projects",
                    json={"name": "Riyadh Tower", "location": "Riyadh"}, headers=H)
    assert r.status_code == 201, r.text
    proj = r.json()
    assert proj["location"] == "Riyadh"


def test_create_project_without_location_is_none(client):
    proj = _new_project(client, "No Location Project")
    assert proj["location"] is None


def test_patch_project_sets_and_clears_location(client):
    proj = _new_project(client, "Location Update Project")
    assert proj["location"] is None
    # set
    r = client.patch(f"/v1/projects/{proj['id']}",
                     json={"location": "Jeddah"}, headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["location"] == "Jeddah"
    # confirm it persisted on GET
    got = client.get(f"/v1/projects/{proj['id']}", headers=H)
    assert got.json()["location"] == "Jeddah"
    # clear (empty string -> None)
    r2 = client.patch(f"/v1/projects/{proj['id']}",
                      json={"location": ""}, headers=H)
    assert r2.status_code == 200
    assert r2.json()["location"] is None


def test_patch_missing_project_404(client):
    assert client.patch("/v1/projects/nope1234",
                        json={"location": "X"}, headers=H).status_code == 404


def test_list_projects_includes_created(client):
    proj = _new_project(client, "Listed Project")
    r = client.get("/v1/projects", headers=H)
    assert r.status_code == 200
    ids = [p["id"] for p in r.json()["projects"]]
    assert proj["id"] in ids


def test_get_missing_project_404(client):
    assert client.get("/v1/projects/nope1234", headers=H).status_code == 404


def test_project_requires_auth(client):
    assert client.get("/v1/projects").status_code in (401, 403)


def test_delete_project(client):
    proj = _new_project(client, "Doomed Project")
    assert client.delete(f"/v1/projects/{proj['id']}", headers=H).status_code == 200
    assert client.get(f"/v1/projects/{proj['id']}", headers=H).status_code == 404


def test_delete_soft_archives_and_preserves_the_row(client):
    """Delete must HIDE the project (UI + retrieval) but NEVER hard-delete the
    row. `chunks.project_id` is ON DELETE CASCADE, so keeping the row is what
    keeps the RAG — the operator principle 'delete the UI, never the RAG'."""
    proj = _new_project(client, "Archive Me")
    pid = proj["id"]
    assert client.delete(f"/v1/projects/{pid}", headers=H).status_code == 200
    # hidden from detail + listing
    assert client.get(f"/v1/projects/{pid}", headers=H).status_code == 404
    listed = [p["id"] for p in client.get("/v1/projects", headers=H).json()["projects"]]
    assert pid not in listed
    # but the row survives as 'archived' → its chunks are NOT cascade-deleted
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as s:
        row = s.get(Project, pid)
        assert row is not None and row.status == "archived"


def test_approve_project_makes_it_admin_visible(client):
    """approve_project flips an EXISTING user-created project to admin-approved
    so it appears in the sidebar (which filters origin='admin_drive_approved').
    Mirrors the visibility a Drive-approved project gets, without re-importing."""
    from app.core import projects as store
    proj = _new_project(client, "the client project Bills of Quantities")
    pid = proj["id"]
    assert store.approve_project(pid) is True
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as s:
        row = s.get(Project, pid)
        assert row.is_approved is True
        assert row.origin == "admin_drive_approved"
    # idempotent + safe on a missing id
    assert store.approve_project(pid) is True
    assert store.approve_project("does-not-exist") is False


def test_purge_archived_project_only_touches_archived(client):
    """purge_archived_project PERMANENTLY removes an ARCHIVED project's row +
    chunks, but REFUSES active projects and protected master/backing ids — so
    the never-delete-RAG rule still holds for everything live."""
    from app.core import projects as store
    proj = _new_project(client, "Junk To Purge")
    pid = proj["id"]
    # active project: refused
    assert store.purge_archived_project(pid) == "not_archived"
    # protected master ids: refused even if somehow archived
    assert store.purge_archived_project("master_corpus") == "protected"
    assert store.purge_archived_project("projects_folder") == "protected"
    # archive it, THEN purge succeeds and the row is gone
    assert store.archive_project(pid) is True
    assert store.purge_archived_project(pid) == "purged"
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as s:
        assert s.get(Project, pid) is None  # row permanently removed
    # missing id: refused cleanly
    assert store.purge_archived_project("nope-xyz") == "not_found"


# ── 0.3 Execution-intent model — attaching a document runs nothing ──────────

def test_document_upload_rejects_oversize(client, monkeypatch):
    """A too-large upload must be rejected with 413 BEFORE being read into
    memory — otherwise one big BIM file OOMs the shared instance.

    Drives the real env var rather than monkeypatching a module constant: the
    limit is now read per-request from app.core.upload_limits, so patching an
    import-time attribute would test a binding the product no longer has.
    """
    monkeypatch.setenv("MAX_DOC_UPLOAD_SIZE", "64")
    proj = _new_project(client, "Size Guard")
    files = {"file": ("big.pdf", b"%PDF-1.4 " + b"x" * 500, "application/pdf")}
    r = client.post(
        f"/v1/projects/{proj['id']}/documents", files=files, headers=H
    )
    assert r.status_code == 413, r.text


def test_attaching_document_stores_only(client):
    proj = _new_project(client)
    r = _attach(client, proj["id"], "site_drawing.pdf")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "stored"
    # the response explicitly states no analysis was run
    assert "no analysis" in body["message"].lower()
    assert body["document"]["doc_type"] == "drawing"


def test_document_role_auto_classified(client):
    proj = _new_project(client)
    _attach(client, proj["id"], "baseline_programme.pdf")
    _attach(client, proj["id"], "daily_report_01.pdf")
    _attach(client, proj["id"], "weekly_report_01.pdf")
    docs = client.get(f"/v1/projects/{proj['id']}", headers=H).json()["documents"]
    roles = sorted(d["doc_role"] for d in docs)
    assert roles == ["baseline_schedule", "daily_report", "weekly_report"]


def test_explicit_role_override(client):
    proj = _new_project(client)
    r = _attach(client, proj["id"], "random.pdf", role="weekly_report")
    assert r.json()["document"]["doc_role"] == "weekly_report"


# ── 0.2 Readiness gate ──────────────────────────────────────────────────────

def test_new_project_is_not_ready(client):
    proj = _new_project(client)
    readiness = proj["readiness"]
    assert readiness["ready"] is False
    assert set(readiness["missing"]) == {
        "baseline_schedule", "daily_reports", "weekly_reports", "aconex",
    }


def test_readiness_progresses_as_inputs_arrive(client):
    proj = _new_project(client)
    pid = proj["id"]

    _attach(client, pid, "baseline_schedule.pdf")
    _attach(client, pid, "daily_report.pdf")
    _attach(client, pid, "weekly_report.pdf")
    readiness = client.get(f"/v1/projects/{pid}", headers=H).json()["readiness"]
    # docs present but Aconex still missing → still not ready
    assert readiness["ready"] is False
    assert readiness["missing"] == ["aconex"]

    client.post(f"/v1/projects/{pid}/connectors/aconex",
                json={"connected": True}, headers=H)
    readiness = client.get(f"/v1/projects/{pid}", headers=H).json()["readiness"]
    assert readiness["ready"] is True
    assert readiness["missing"] == []


def test_progress_blocked_when_not_ready(client):
    proj = _new_project(client)
    r = client.post(f"/v1/projects/{proj['id']}/progress",
                     json={"planned_percent": 40, "actual_percent": 35}, headers=H)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "not_ready"
    assert "baseline_schedule" in body["missing"]


@requires_construction_kit
def test_progress_runs_when_ready(client):
    proj = _new_project(client)
    pid = proj["id"]
    _attach(client, pid, "baseline_schedule.pdf")
    _attach(client, pid, "daily_report.pdf")
    _attach(client, pid, "weekly_report.pdf")
    client.post(f"/v1/projects/{pid}/connectors/aconex",
                json={"connected": True}, headers=H)

    r = client.post(f"/v1/projects/{pid}/progress", json={
        "planned_percent": 45, "actual_percent": 38, "contract_value": 12_000_000,
    }, headers=H)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "success"
    overall = body["tracker"]["overall_progress"]
    assert overall["planned_percent"] == 45
    assert overall["actual_percent"] == 38
    assert overall["variance_percent"] == -7
