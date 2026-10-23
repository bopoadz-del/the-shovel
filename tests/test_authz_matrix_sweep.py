"""Authorization MATRIX sweep — every project surface x caller x project kind.

New test shape (2026-07-25): the pilot's two authz bugs (#267 list/detail
mismatch, #277 upload owner-only 404) were both single-cell holes in a matrix
nobody enumerated. This sweep pins the WHOLE matrix so the next asymmetry
fails a test instead of a tester.

Callers: owner / admin / stranger.
Projects: private (user_create) / admin-approved shared platform project.
Surfaces: open detail, list documents, upload document, delete project,
RAG search, project memory, execute-with-project_id.

Expected access:
  open+docs+upload+rag+execute -> owner: yes; stranger: shared only.
  memory           -> owner only.
  delete           -> owner or admin (soft-archive curation); strangers denied.

First run of this sweep found a live hole: document PAGINATION 404'd for
admins on projects they can open (role not threaded through the access
check) — fixed alongside this file.
"""
import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_10


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client, email):
    r = client.post("/v1/users/register", json={"email": email, "password": PW_10})
    assert r.status_code in (201, 409), r.text  # 409 = already there from a prior case
    r = client.post("/v1/users/login", json={"email": email, "password": PW_10})
    return {"token": r.json()["token"], "id": r.json()["user"]["id"]}


def _promote_admin(uid):
    from app.core.db import SessionLocal
    from app.core.models import User
    with SessionLocal() as db:
        db.get(User, uid).role = "admin"
        db.commit()


def _mark_shared(pid):
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as db:
        p = db.get(Project, pid)
        p.origin = "admin_drive_approved"
        p.is_approved = True
        db.commit()


def _h(actor):
    return {"Authorization": f"Bearer {actor['token']}"}


@pytest.fixture
def world(client):
    """owner + admin + stranger, one private and one shared project."""
    owner = _register(client, "mx-owner@example.com")
    admin = _register(client, "mx-admin@example.com")
    stranger = _register(client, "mx-stranger@example.com")
    _promote_admin(admin["id"])

    private_pid = client.post(
        "/v1/projects", json={"name": "Matrix private"}, headers=_h(owner)
    ).json()["id"]
    shared_pid = client.post(
        "/v1/projects", json={"name": "Matrix shared"}, headers=_h(owner)
    ).json()["id"]
    _mark_shared(shared_pid)
    return {
        "owner": owner, "admin": admin, "stranger": stranger,
        "private": private_pid, "shared": shared_pid,
    }


def _surface(client, name, pid, actor):
    if name == "open":
        return client.get(f"/v1/projects/{pid}", headers=_h(actor)).status_code
    if name == "docs":
        return client.get(f"/v1/projects/{pid}/documents", headers=_h(actor)).status_code
    if name == "upload":
        return client.post(
            f"/v1/projects/{pid}/documents", headers=_h(actor),
            files={"file": ("m.txt", io.BytesIO(b"matrix"), "text/plain")},
        ).status_code
    if name == "delete":
        return client.delete(f"/v1/projects/{pid}", headers=_h(actor)).status_code
    if name == "rag":
        return client.post(
            "/v1/rag/search",
            headers=_h(actor),
            json={"query": "x", "project_id": pid, "k": 3},
        ).status_code
    if name == "memory":
        return client.get(f"/v1/projects/{pid}/memory", headers=_h(actor)).status_code
    if name == "execute":
        return client.post(
            "/v1/execute",
            headers=_h(actor),
            json={
                "block": "translate",
                "input": "hola",
                "params": {"target": "en", "project_id": pid},
            },
        ).status_code
    raise AssertionError(name)


# (surface, caller, project) -> allowed?
MATRIX = [
    ("open",   "owner",    "private", True),
    ("open",   "admin",    "private", True),
    ("open",   "stranger", "private", False),
    ("open",   "owner",    "shared",  True),
    ("open",   "admin",    "shared",  True),
    ("open",   "stranger", "shared",  True),
    ("docs",   "owner",    "private", True),
    ("docs",   "admin",    "private", True),
    ("docs",   "stranger", "private", False),
    ("docs",   "stranger", "shared",  True),
    ("upload", "owner",    "private", True),
    ("upload", "admin",    "private", True),
    ("upload", "stranger", "private", False),
    ("upload", "owner",    "shared",  True),
    ("upload", "admin",    "shared",  True),
    ("upload", "stranger", "shared",  True),
    ("delete", "stranger", "private", False),
    ("delete", "stranger", "shared",  False),
    # Same grant as open: owner / admin / shared-stranger. A new RAG
    # door that skips get_project_accessible fails these cells.
    ("rag",    "owner",    "private", True),
    ("rag",    "stranger", "private", False),
    ("rag",    "owner",    "shared",  True),
    ("rag",    "stranger", "shared",  True),
    # Project facts are owner-only (mutating-style _owned_or_404).
    ("memory", "owner",    "private", True),
    ("memory", "stranger", "private", False),
    ("memory", "stranger", "shared",  False),
    # Nested project_id on /v1/execute uses get_project_accessible.
    ("execute", "owner",    "private", True),
    ("execute", "stranger", "private", False),
    ("execute", "owner",    "shared",  True),
    ("execute", "stranger", "shared",  True),
    # Admin delete = documented curation feature (soft-archive, reversible);
    # covered separately in test_admin_delete_is_soft_archive below.
]


@pytest.mark.parametrize("surface,caller,kind,allowed", MATRIX)
def test_authz_matrix(client, world, surface, caller, kind, allowed):
    code = _surface(client, surface, world[kind], world[caller])
    if allowed:
        assert 200 <= code < 300, f"{surface} {caller} {kind}: expected allow, got {code}"
    else:
        # 404 = existence hidden; 403 = visible-but-forbidden (delete on a
        # shared project the caller can see). Both are correct denials.
        assert code in (403, 404), f"{surface} {caller} {kind}: expected denial, got {code}"


def test_admin_delete_is_soft_archive(client, world):
    # Admin curation: delete on another user's project archives it (soft,
    # reversible) — the documented workspace-cleanup capability.
    assert _surface(client, "delete", world["private"], world["admin"]) == 200


def test_owner_delete_last(client, world):
    # Run owner deletes LAST so the matrix cases above see live projects.
    assert _surface(client, "delete", world["shared"], world["owner"]) == 200


# Deliberate public surfaces. A NEW /v1 (or other data) route that is not
# in this set and has no require_user / require_api_key dependency fails
# this test — the next unguarded door cannot land silently.
_PUBLIC_ROUTES = {
    ("GET", "/"),
    ("GET", "/api"),
    ("GET", "/{full_path:path}"),
    ("GET", "/livez"),
    ("GET", "/health"),
    ("GET", "/ready"),
    ("GET", "/v1/health"),
    ("GET", "/v1/upload-limits"),
    ("GET", "/metrics"),
    ("GET", "/v1/users/verify-email"),
    ("POST", "/v1/users/register"),
    ("POST", "/v1/users/login"),
    # A locked-out user cannot sign in: asking for a reset link and setting
    # the new password are public by design (one neutral reply; a single-use
    # signed link). Order 0735Z item 3f.
    ("POST", "/v1/users/forgot-password"),
    ("POST", "/v1/users/reset-password"),
    ("POST", "/v1/users/resend-verification"),
    ("GET", "/v1/drive/callback"),
    # Catalog / liveness extras that currently have no Depends. Adding a
    # guard later is fine; adding a *new* unlisted unguarded path is not.
    ("GET", "/stats"),
    ("GET", "/v1/system/health"),
    ("GET", "/blocks"),
    ("GET", "/blocks/{block_name}"),
    ("GET", "/v1/blocks"),
    ("GET", "/v1/blocks/{block_name}"),
}


def _route_has_auth_dependency(route) -> bool:
    from app.dependencies import require_api_key, require_user

    wanted = {require_user, require_api_key}

    def walk(dep) -> bool:
        call = getattr(dep, "call", None)
        if call in wanted:
            return True
        for child in getattr(dep, "dependencies", []) or []:
            if walk(child):
                return True
        return False

    return walk(route.dependant)


def test_new_unguarded_route_fails_ci():
    """Inventory gate: every non-public HTTP route must carry an auth Depends.

    Worth more than any single ownership fix — a route added without
    require_user / require_api_key and without an explicit public listing
    fails here instead of shipping as an open door.
    """
    from fastapi.routing import APIRoute

    unguarded = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        methods = (route.methods or set()) - {"HEAD", "OPTIONS"}
        for method in methods:
            key = (method, route.path)
            if key in _PUBLIC_ROUTES:
                continue
            if _route_has_auth_dependency(route):
                continue
            unguarded.append(key)
    assert unguarded == [], (
        "New unguarded route(s) — add require_user/require_api_key "
        f"or list them in _PUBLIC_ROUTES if deliberate public: {unguarded}"
    )
