"""Session ownership for /v1/project/ask — Stream A."""
import json
import uuid
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.routers import project as project_router
from app.core.session_store import InMemorySessionStore
from tests._dev_key import DEV_KEY
from tests._fixture_values import PW_06

_RUN = uuid.uuid4().hex[:8]


class _MockReasoner:
    """Lightweight stand-in for ProjectReasonerBlock — no LLM calls needed."""

    async def process(self, inputs):
        return {
            "status": "success",
            "answer": "mock answer",
            "understanding": "",
            "plan": None,
            "execution": None,
        }


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tenancy_data")
    import os
    os.environ.setdefault("DATA_DIR", str(tmp))

    store = InMemorySessionStore()

    with TestClient(app) as c:
        # Inject shared fresh store and mock reasoner for the whole module.
        project_router._store = store
        project_router._reasoner_factory = lambda: _MockReasoner()
        yield c


def _token(client, email):
    client.post("/v1/users/register",
                json={"email": email, "password": PW_06})
    return client.post("/v1/users/login",
                       json={"email": email, "password": PW_06}).json()["token"]


def test_session_is_owned_by_creator(client):
    alice = {"Authorization": f"Bearer {_token(client, f'sa-{_RUN}@x.com')}"}
    bob = {"Authorization": f"Bearer {_token(client, f'sb-{_RUN}@x.com')}"}
    sid = f"shared-{_RUN}-001"
    r = client.post("/v1/project/ask",
                    json={"session_id": sid, "request": "hello"}, headers=alice)
    assert r.status_code == 200, r.text
    r2 = client.post("/v1/project/ask",
                     json={"session_id": sid, "request": "intrude"}, headers=bob)
    assert r2.status_code == 404


def test_owner_can_continue_own_session(client):
    alice = {"Authorization": f"Bearer {_token(client, f'sa2-{_RUN}@x.com')}"}
    sid = f"own-{_RUN}-002"
    assert client.post("/v1/project/ask",
                       json={"session_id": sid, "request": "turn 1"},
                       headers=alice).status_code == 200
    assert client.post("/v1/project/ask",
                       json={"session_id": sid, "request": "turn 2"},
                       headers=alice).status_code == 200


def test_legacy_key_session_works(client):
    legacy = {"Authorization": "Bearer " + DEV_KEY}
    assert client.post("/v1/project/ask",
                       json={"session_id": f"legacy-{_RUN}-003", "request": "hi"},
                       headers=legacy).status_code == 200


def test_drive_import_cross_tenant_404(client):
    alice = {"Authorization": f"Bearer {_token(client, f'da-{_RUN}@x.com')}"}
    bob = {"Authorization": f"Bearer {_token(client, f'db-{_RUN}@x.com')}"}
    pid = client.post("/v1/projects", json={"name": "Alice DriveProj"},
                      headers=alice).json()["id"]
    r = client.post(f"/v1/projects/{pid}/drive/import",
                    json={"file_id": "x", "name": "test.pdf"}, headers=bob)
    assert r.status_code == 404
