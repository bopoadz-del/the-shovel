"""Tests for project memory — durable facts across documents. Roadmap V2 · Epic 3."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core import projects as store
from tests._dev_key import DEV_KEY
from app.core.project_memory import (
    build_memory_context,
    extract_facts,
    remember_from_result,
)

H = {"Authorization": "Bearer " + DEV_KEY}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _new_project(client):
    return client.post("/v1/projects", json={"name": "Memory Test"}, headers=H).json()


# ── fact extraction ─────────────────────────────────────────────────────────

def test_extract_facts_pulls_durable_fields():
    result = {
        "doc_type": "contract",
        "contract_value": 12000000,
        "completion_date": "2027-03-01",
        "details": {"ld_rate": "0.1% per day", "employer": "the client project Co"},
        "noise_field": "ignore me",
    }
    facts = {f["key"]: f["value"] for f in extract_facts(result)}
    assert facts["contract_value"] == "12000000"
    assert facts["completion_date"] == "2027-03-01"
    assert facts["ld_rate"] == "0.1% per day"
    assert facts["employer"] == "the client project Co"
    assert "noise_field" not in facts


def test_extract_facts_ignores_junk_values():
    facts = extract_facts({"contract_value": 0, "revision": "none", "scale": ""})
    assert facts == []


# ── store CRUD ──────────────────────────────────────────────────────────────

def test_set_list_and_get_fact():
    proj = store.create_project("Direct Store Test")
    store.set_fact(proj["id"], "ld_rate", "0.1%/day")
    assert store.get_fact(proj["id"], "ld_rate")["value"] == "0.1%/day"
    assert len(store.list_facts(proj["id"])) == 1


def test_set_fact_upserts_on_same_key():
    proj = store.create_project("Upsert Test")
    store.set_fact(proj["id"], "contract_value", "10000000")
    store.set_fact(proj["id"], "contract_value", "11500000")
    facts = store.list_facts(proj["id"])
    assert len(facts) == 1                       # not duplicated
    assert facts[0]["value"] == "11500000"       # updated


def test_search_facts_by_keyword():
    proj = store.create_project("Search Test")
    store.set_fact(proj["id"], "completion_date", "2027-03-01")
    store.set_fact(proj["id"], "contract_value", "12000000")
    hits = store.search_facts(proj["id"], "completion")
    assert len(hits) == 1 and hits[0]["key"] == "completion_date"


# ── accumulation + retrieval ────────────────────────────────────────────────

def test_remember_then_recall_across_documents():
    proj = store.create_project("Accumulation Test")
    pid = proj["id"]
    # document 1 — a contract
    remember_from_result(pid, {"contract_value": 12000000, "ld_rate": "0.1%/day"},
                         source_document="contract.pdf")
    # document 2 — a drawing
    remember_from_result(pid, {"drawing_number": "A-101", "revision": "C"},
                         source_document="plan.pdf")
    # later: ask about the LD rate — answered from memory, no contract re-attached
    ctx = build_memory_context(pid, "ld rate")
    assert "0.1%/day" in ctx
    # all four facts accumulated across both documents
    assert len(store.list_facts(pid)) == 4


def test_build_memory_context_empty_when_no_facts():
    proj = store.create_project("Empty Memory")
    assert build_memory_context(proj["id"]) == ""


# ── API ─────────────────────────────────────────────────────────────────────

def test_memory_api_roundtrip(client):
    proj = _new_project(client)
    pid = proj["id"]

    r = client.post(f"/v1/projects/{pid}/memory",
                     json={"key": "retention_percent", "value": "5"}, headers=H)
    assert r.status_code == 201

    r = client.get(f"/v1/projects/{pid}/memory", headers=H)
    assert r.status_code == 200
    assert r.json()["count"] == 1

    r = client.get(f"/v1/projects/{pid}/memory?q=retention", headers=H)
    assert r.json()["count"] == 1

    r = client.delete(f"/v1/projects/{pid}/memory/retention_percent", headers=H)
    assert r.status_code == 200
    assert client.get(f"/v1/projects/{pid}/memory", headers=H).json()["count"] == 0


def test_memory_api_404_for_missing_project(client):
    assert client.get("/v1/projects/nope9999/memory", headers=H).status_code == 404


def test_chat_injects_project_memory():
    """Roadmap V2 · Epic 4 Slice B — chat scoped to a project sees its memory."""
    from app.routers.chat import _with_project_memory
    store.init_db()
    proj = store.create_project("Chat Memory Test")  # owner defaults to "system"
    store.set_fact(proj["id"], "ld_rate", "0.1%/day")
    # The project owner sees its memory injected into the prompt.
    out = _with_project_memory("what is the LD rate?", proj["id"], "system")
    assert "0.1%/day" in out
    assert "what is the LD rate?" in out
    # no project scope → prompt unchanged
    assert _with_project_memory("hi", None, "system") == "hi"
