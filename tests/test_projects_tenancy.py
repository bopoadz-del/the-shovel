"""Multi-tenant isolation at the projects API layer — Stream A."""
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_06

LEGACY = {"Authorization": "Bearer " + DEV_KEY}

# Run-unique suffix so each test run creates fresh users with no prior history.
_RUN = uuid.uuid4().hex[:8]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _user_token(client, email):
    client.post("/v1/users/register",
                json={"email": email, "password": PW_06})
    return client.post("/v1/users/login",
                       json={"email": email, "password": PW_06}).json()["token"]


def test_user_only_sees_own_projects(client):
    """Private user_create rows stay tenant-scoped; shared GK may appear.

    After the system_seed / RAG_GENERAL_KNOWLEDGE_PROJECTS ACL grant,
    GET /v1/projects for a role=user includes approved GK (e.g. name
    'General Knowledge', id curated_kb / training_material). That is
    intended. This test still proves Alice cannot see Bob's project
    and Bob cannot see Alice's.
    """
    alice = {"Authorization": f"Bearer {_user_token(client, f'ten-alice-{_RUN}@x.com')}"}
    bob = {"Authorization": f"Bearer {_user_token(client, f'ten-bob-{_RUN}@x.com')}"}
    a_pid = client.post("/v1/projects", json={"name": "Alice P"},
                        headers=alice).json()["id"]
    b_pid = client.post("/v1/projects", json={"name": "Bob P"}, headers=bob).json()["id"]
    alice_list = client.get("/v1/projects", headers=alice).json()["projects"]
    bob_list = client.get("/v1/projects", headers=bob).json()["projects"]
    alice_names = [p["name"] for p in alice_list]
    bob_names = [p["name"] for p in bob_list]
    assert "Alice P" in alice_names
    assert "Bob P" not in alice_names
    assert "Bob P" in bob_names
    assert "Alice P" not in bob_names
    assert all(p["id"] != a_pid for p in bob_list)
    assert all(p["id"] != b_pid for p in alice_list)
    # Shared GK (General Knowledge / curated_kb / system_seed) may also
    # appear; that is the product grant. Do not exact-match the name list.


def test_cross_tenant_get_returns_404(client):
    alice = {"Authorization": f"Bearer {_user_token(client, f'x-alice-{_RUN}@x.com')}"}
    bob = {"Authorization": f"Bearer {_user_token(client, f'x-bob-{_RUN}@x.com')}"}
    pid = client.post("/v1/projects", json={"name": "Alice Secret"},
                      headers=alice).json()["id"]
    assert client.get(f"/v1/projects/{pid}", headers=bob).status_code == 404
    assert client.delete(f"/v1/projects/{pid}", headers=bob).status_code == 404


def test_cross_tenant_document_and_memory_404(client):
    alice = {"Authorization": f"Bearer {_user_token(client, f'd-alice-{_RUN}@x.com')}"}
    bob = {"Authorization": f"Bearer {_user_token(client, f'd-bob-{_RUN}@x.com')}"}
    pid = client.post("/v1/projects", json={"name": "Alice Docs"},
                      headers=alice).json()["id"]
    files = {"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    assert client.post(f"/v1/projects/{pid}/documents", files=files,
                       headers=bob).status_code == 404
    assert client.get(f"/v1/projects/{pid}/memory",
                      headers=bob).status_code == 404


def test_legacy_key_sees_system_projects(client):
    pid = client.post("/v1/projects", json={"name": "Legacy P"},
                      headers=LEGACY).json()["id"]
    listed = client.get("/v1/projects", headers=LEGACY).json()["projects"]
    assert pid in [p["id"] for p in listed]
