"""Audit entries carry the acting user_id — Stream A."""
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _token(client, email):
    client.post("/v1/users/register",
                json={"email": email, "password": PW_06})
    return client.post("/v1/users/login",
                       json={"email": email, "password": PW_06}).json()["token"]


def test_project_audit_records_user_id(client):
    alice = {"Authorization": f"Bearer {_token(client, f'au-{_RUN}@x.com')}"}
    pid = client.post("/v1/projects", json={"name": "Audited"},
                      headers=alice).json()["id"]
    entries = client.get(f"/v1/projects/{pid}/audit", headers=alice).json()["entries"]
    created = [e for e in entries if e["event"] == "project.created"]
    assert created and created[0]["user_id"] != "system"


def test_legacy_audit_records_system_user(client):
    legacy = {"Authorization": "Bearer " + DEV_KEY}
    pid = client.post("/v1/projects", json={"name": "Legacy Audited"},
                      headers=legacy).json()["id"]
    entries = client.get(f"/v1/projects/{pid}/audit", headers=legacy).json()["entries"]
    created = [e for e in entries if e["event"] == "project.created"]
    assert created and created[0]["user_id"] == "system"
