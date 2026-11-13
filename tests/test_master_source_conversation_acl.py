"""The physical master-corpus source id is owner-only in the conversation ACL,
whatever origin its project row carries (P-S1).

The shared-platform grant opens ``system_seed`` rows to every signed-in user.
A master-source row stamped ``system_seed`` (a boot seed, a restore, a test
wipe) must not open the whole corpus's conversations; a real general-knowledge
``system_seed`` project must stay open. Every project here is synthetic and
the source id is this test's own (monkeypatched), never a live id.
"""
import uuid

import pytest
from fastapi.testclient import TestClient

from app.core import agent_memory as am
from app.core import projects as store
from app.main import app
from tests._fixture_values import PW_10


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _login(client, email):
    r = client.post("/v1/users/register", json={"email": email, "password": PW_10})
    assert r.status_code in (201, 409), r.text
    r = client.post("/v1/users/login", json={"email": email, "password": PW_10})
    body = r.json()
    return {"Authorization": f"Bearer {body['token']}"}, body["user"]["id"]


@pytest.fixture
def world(client, monkeypatch):
    owner_h, owner_id = _login(client, "ps1-owner@example.com")
    stranger_h, _ = _login(client, "ps1-stranger@example.com")
    tag = uuid.uuid4().hex[:8]
    ids = {"general": f"ps1_gk_{tag}", "source": f"ps1_src_{tag}"}
    monkeypatch.setattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", ids["source"])
    for pid in ids.values():
        # Same owner, same origin, same approval: the ONLY difference is that
        # one id is the master-corpus source.
        store.create_project(
            name=f"PS1 {pid}", user_id=owner_id, is_approved=True,
            project_id=pid, origin="system_seed",
        )
    yield {"owner": owner_h, "stranger": stranger_h, "ids": ids}
    for pid in ids.values():
        store.delete_project(pid)


@pytest.mark.parametrize("which, stranger_status", [("general", 200), ("source", 404)])
def test_ws_conversation_master_source_is_owner_only_even_as_system_seed(
    client, world, which, stranger_status,
):
    cid = f"ws-{world['ids'][which]}-1740000000000"
    url = f"/v1/agents/conversations/{cid}/messages"
    stranger = client.get(url, headers=world["stranger"])
    assert stranger.status_code == stranger_status, stranger.text
    owner = client.get(url, headers=world["owner"])
    assert owner.status_code == 200, owner.text


@pytest.mark.parametrize("which, stranger_status", [("general", 200), ("source", 404)])
def test_stored_conversation_bound_to_master_source_is_owner_only(
    client, world, which, stranger_status,
):
    cid = f"ps1-{which}-{uuid.uuid4().hex[:8]}"
    am.get_or_create_conversation(cid, "project-assistant", project_id=world["ids"][which])
    am.append_message(cid, "user", f"note on {which}")
    url = f"/v1/agents/conversations/{cid}/messages"
    stranger = client.get(url, headers=world["stranger"])
    assert stranger.status_code == stranger_status, stranger.text
    owner = client.get(url, headers=world["owner"])
    assert owner.status_code == 200, owner.text
    assert f"note on {which}" in [m["content"] for m in owner.json()["messages"]]


@pytest.mark.parametrize(
    "project_id, expected",
    [("ps1_any_general", True), ("ps1_the_source", False)],
)
def test_shared_grant_applies_to_every_id_but_the_master_source(monkeypatch, project_id, expected):
    from app.routers.agents import _shared_grant_applies

    monkeypatch.setattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", "ps1_the_source")
    assert _shared_grant_applies(project_id) is expected
