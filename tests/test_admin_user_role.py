"""Admin role management (ORDERS_02NOV_1545Z): POST /v1/admin/users/{id}/role
sets a user's role under the users table's own vocabulary, behind the existing
admin gate, idempotent, audit-logged. privileges.py is untouched -- this is the
MECHANISM that lets the battery account become an indexing (admin) uploader
instead of a hand edit of the live database. Synthetic users only."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core import jwt_auth
from app.core import users as users_store


def _headers(user_id, email, role):
    users_store.ensure_user_exists(user_id, role=role, email=email)
    return {"Authorization": f"Bearer {jwt_auth.create_token(user_id)}"}


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.mark.parametrize("start_role,new_role,expect_changed", [
    ("user", "admin", True),    # the battery-account promotion path
    ("admin", "user", True),    # and the demotion path, same mechanism
    ("admin", "admin", False),  # idempotent: same role -> 200 no-op
])
def test_admin_sets_a_users_role_idempotently(client, start_role, new_role, expect_changed):
    admin = _headers("role-admin", "role-admin@local", "admin")
    uid = f"role-subject-{start_role}-{new_role}"
    users_store.ensure_user_exists(uid, role=start_role, email=f"{uid}@local")

    r = client.post(f"/v1/admin/users/{uid}/role", json={"role": new_role}, headers=admin)
    assert r.status_code == 200
    assert r.json() == {"user_id": uid, "role": new_role, "changed": expect_changed}
    assert users_store.get_user_by_id(uid)["role"] == new_role
    # a second identical call is always the no-op
    again = client.post(f"/v1/admin/users/{uid}/role", json={"role": new_role}, headers=admin)
    assert again.status_code == 200 and again.json()["changed"] is False


@pytest.mark.parametrize("caller_role", ["user", "admin"])
def test_only_an_admin_reaches_the_role_endpoint(client, caller_role):
    caller = _headers(f"role-caller-{caller_role}", f"role-caller-{caller_role}@local", caller_role)
    users_store.ensure_user_exists("role-target", role="user", email="role-target@local")
    r = client.post("/v1/admin/users/role-target/role", json={"role": "admin"}, headers=caller)
    assert r.status_code == (200 if caller_role == "admin" else 403)


@pytest.mark.parametrize("bad_role", ["superuser", ""])
def test_a_role_outside_the_check_constraint_is_rejected(client, bad_role):
    admin = _headers("role-admin2", "role-admin2@local", "admin")
    users_store.ensure_user_exists("role-bad", role="user", email="role-bad@local")
    r = client.post("/v1/admin/users/role-bad/role", json={"role": bad_role}, headers=admin)
    assert r.status_code == 422
    assert users_store.get_user_by_id("role-bad")["role"] == "user"  # nothing written


@pytest.mark.parametrize("unknown_id", ["no-such-user", "ghost-42"])
def test_an_unknown_user_is_404(client, unknown_id):
    admin = _headers("role-admin3", "role-admin3@local", "admin")
    r = client.post(f"/v1/admin/users/{unknown_id}/role", json={"role": "admin"}, headers=admin)
    assert r.status_code == 404
