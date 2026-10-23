"""Workflow tenant isolation — a saved workflow is private to its owner.

Regression for the IDOR where any authenticated user could list, read,
delete, or run another user's saved workflows (the store had no owner column).
"""

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]
_STEPS = [{"block": "formula_executor", "params": {}, "label": "Compute"}]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _register_and_login(client, suffix: str) -> dict:
    email = f"wf-{suffix}-{_RUN}@x.com"
    client.post("/v1/users/register", json={"email": email, "password": PW_06})
    token = client.post(
        "/v1/users/login", json={"email": email, "password": PW_06}
    ).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_workflow_is_private_to_its_owner(client):
    owner = _register_and_login(client, "owner")
    attacker = _register_and_login(client, "attacker")

    # Owner saves a workflow.
    resp = client.post(
        "/v1/workflows", headers=owner,
        json={"name": "Owner's Workflow", "steps": _STEPS},
    )
    assert resp.status_code == 201, resp.text
    wid = resp.json()["id"]

    # Attacker must not be able to read, delete, or run it.
    assert client.get(f"/v1/workflows/{wid}", headers=attacker).status_code == 404
    assert client.delete(f"/v1/workflows/{wid}", headers=attacker).status_code == 404
    run = client.post(
        f"/v1/workflows/{wid}/run", headers=attacker, json={"initial_input": "x"}
    )
    assert run.status_code == 404, run.text

    # Attacker's listing must not include it.
    attacker_list = client.get("/v1/workflows", headers=attacker).json()["workflows"]
    assert wid not in [w["id"] for w in attacker_list]

    # The attacker's failed delete must not have removed it — owner still sees it.
    assert client.get(f"/v1/workflows/{wid}", headers=owner).status_code == 200
    owner_list = client.get("/v1/workflows", headers=owner).json()["workflows"]
    assert wid in [w["id"] for w in owner_list]

    # The owner can delete their own workflow.
    assert client.delete(f"/v1/workflows/{wid}", headers=owner).status_code == 200
