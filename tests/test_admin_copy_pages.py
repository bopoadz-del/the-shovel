"""Plan/execute copy of selected pages into a new project (additive only)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import projects as projects_mod
from app.dependencies import require_api_key
from app.main import app

URL = "/v1/admin/projects/{}/copy-pages"


@pytest.fixture(autouse=True)
def _admin_override():
    app.dependency_overrides[require_api_key] = lambda: {
        "user_id": "system", "role": "admin",
    }
    yield
    app.dependency_overrides.clear()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("DATA_ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.setattr(projects_mod, "_initialized", False)
    projects_mod.init_db()
    from app.core.rag import vector_store as _vs
    _vs.reset_store_cache()
    store = _vs.get_store()
    src = projects_mod.create_project("source corpus")["id"]

    def add(name, pages, layer=None):
        doc = projects_mod.add_document(src, name, size=10)
        texts = [f"{name} chunk {i} page {p}" for i, p in enumerate(pages)]
        emb = np.random.default_rng(len(texts)).random((len(texts), store.dim)).astype(np.float32)
        store.upsert_chunks(src, doc["id"], texts, emb, pages=pages,
                            knowledge_layer=layer, ocr_flags=[p == 3 for p in pages])
        return doc["id"]

    pdf = add("spec.pdf", [1, 1, 2, 3, 3, 3, 5])
    sheet = add("bill.xlsx", [None, None, None])
    return {"store": store, "src": src, "pdf": pdf, "sheet": sheet}


def _source_fingerprint(store, src, doc_ids):
    rows = []
    for d in doc_ids:
        for r in store.doc_chunk_rows(src, d, with_vectors=True):
            rows.append((r["chunk_id"], r["page"], r["text"], r["embedding"].tobytes()))
    return hashlib.sha256(repr(rows).encode()).hexdigest(), len(rows)


def _plan(client, target, body):
    r = client.post(URL.format(target), json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_plan_counts_selected_pages_and_writes_nothing(env):
    body = {"source_project_id": env["src"],
            "selections": [{"doc_id": env["pdf"], "pages": [1, 3]}]}
    with TestClient(app) as client:
        out = _plan(client, "core-a", body)
        other = _plan(client, "core-a", {**body, "selections": [{"doc_id": env["pdf"], "pages": [2, 4]}]})
    plan = out["plan"]
    assert out["executed"] is False
    assert plan["eligible"] is True
    assert plan["chunks_taken"] == 5 and plan["pages_found"] == 2
    assert plan["selections"][0]["ocr_chunks_taken"] == 3
    # A different selection measures differently and hashes differently.
    assert other["plan"]["chunks_taken"] == 1
    assert other["plan"]["selections"][0]["pages_missing"] == [4]
    assert other["plan"]["plan_hash"] != plan["plan_hash"]
    assert projects_mod.get_project("core-a") is None


def test_execute_copies_only_selected_pages_and_keeps_page(env):
    store = env["store"]
    before = _source_fingerprint(store, env["src"], [env["pdf"], env["sheet"]])
    sels = [{"doc_id": env["pdf"], "pages": [3, 5]}, {"doc_id": env["sheet"], "pages": "all"}]
    body = {"source_project_id": env["src"], "selections": sels,
            "profile": {"answer_path": "harnessed"}}
    with TestClient(app) as client:
        plan = _plan(client, "core-b", body)["plan"]
        no_profile = _plan(client, "core-b", {**body, "profile": None})["plan"]
        out = _plan(client, "core-b", {**body, "execute": True, "reason": "core set",
                                        "expected_plan_hash": plan["plan_hash"]})
    # The profile is part of what the hash approves.
    assert no_profile["plan_hash"] != plan["plan_hash"]
    assert out["executed"] is True, out
    assert out["chunks_copied"] == 4 + 3
    by_src = {d["source_doc_id"]: d["doc_id"] for d in out["documents"]}
    pdf_rows = store.doc_chunk_rows("core-b", by_src[env["pdf"]])
    assert [r["page"] for r in pdf_rows] == [3, 3, 3, 5]
    assert [r["ocr"] for r in pdf_rows] == [True, True, True, False]
    assert [r["page"] for r in store.doc_chunk_rows("core-b", by_src[env["sheet"]])] == [None] * 3
    new_doc = projects_mod.get_document(by_src[env["pdf"]])
    assert new_doc["project_id"] == "core-b" and new_doc["chunk_count"] == 4
    assert new_doc["metadata"]["copied_from"]["doc_id"] == env["pdf"]
    assert projects_mod.get_project("core-b")["profile"] == {"answer_path": "harnessed"}
    # The source is only read.
    assert _source_fingerprint(store, env["src"], [env["pdf"], env["sheet"]]) == before
    assert projects_mod.get_document(env["pdf"])["project_id"] == env["src"]


def test_execute_refuses_wrong_hash_and_missing_reason(env):
    body = {"source_project_id": env["src"], "selections": [{"doc_id": env["pdf"], "pages": [1]}]}
    with TestClient(app) as client:
        plan = _plan(client, "core-c", body)["plan"]
        stale = _plan(client, "core-c", {**body, "execute": True, "reason": "x",
                                          "expected_plan_hash": "0" * 64})
        no_reason = _plan(client, "core-c", {**body, "execute": True, "reason": "  ",
                                              "expected_plan_hash": plan["plan_hash"]})
    assert stale["executed"] is False and "plan_hash" in stale["refused"]
    assert no_reason["executed"] is False and "reason" in no_reason["refused"]
    assert projects_mod.get_project("core-c") is None


def test_existing_target_and_empty_page_hit_are_not_eligible(env):
    projects_mod.create_project("already here", project_id="core-d")
    body = {"source_project_id": env["src"], "selections": [{"doc_id": env["pdf"], "pages": [2]}]}
    no_hit = {"source_project_id": env["src"], "selections": [{"doc_id": env["pdf"], "pages": [4, 6]}]}
    with TestClient(app) as client:
        taken = _plan(client, "core-d", body)["plan"]
        out = _plan(client, "core-d", {**body, "execute": True, "reason": "x",
                                        "expected_plan_hash": taken["plan_hash"]})
        empty = _plan(client, "core-e", no_hit)["plan"]
        sheet_by_page = _plan(client, "core-e", {"source_project_id": env["src"],
                                                 "selections": [{"doc_id": env["sheet"], "pages": [1]}]})["plan"]
    assert taken["eligible"] is False and "target project already exists" in taken["not_eligible"]
    assert out["executed"] is False and "already exists" in out["refused"]
    assert empty["eligible"] is False and empty["chunks_taken"] == 0
    assert empty["selections"][0]["pages_missing"] == [4, 6]
    # A sheet has no page locator: only "all" takes it.
    assert sheet_by_page["eligible"] is False
    assert projects_mod.get_project("core-e") is None


def test_document_from_another_project_is_not_found(env):
    other = projects_mod.create_project("other")["id"]
    foreign = projects_mod.add_document(other, "x.pdf", size=1)["id"]
    with TestClient(app) as client:
        plan = _plan(client, "core-f", {"source_project_id": env["src"],
                                        "selections": [{"doc_id": foreign, "pages": "all"},
                                                       {"doc_id": env["pdf"], "pages": [1]}]})["plan"]
    assert plan["eligible"] is False
    assert plan["selections"][0]["found"] is False and plan["selections"][1]["found"] is True


@pytest.mark.parametrize("selections", [
    [],
    [{"doc_id": "", "pages": [1]}],
    [{"doc_id": "a", "pages": []}],
    [{"doc_id": "a", "pages": [0]}],
    [{"doc_id": "a", "pages": [True]}],
    [{"doc_id": "a", "pages": "some"}],
    [{"doc_id": "a", "pages": [1]}, {"doc_id": "a", "pages": [2]}],
])
def test_bad_selection_is_a_400(env, selections):
    with TestClient(app) as client:
        r = client.post(URL.format("core-g"), json={"source_project_id": env["src"],
                                                    "selections": selections})
    assert r.status_code == 400, r.text
    assert projects_mod.get_project("core-g") is None


def test_non_admin_is_refused(env):
    app.dependency_overrides[require_api_key] = lambda: {"user_id": "u1", "role": "user"}
    with TestClient(app) as client:
        r = client.post(URL.format("core-h"), json={"source_project_id": env["src"],
                                                    "selections": [{"doc_id": env["pdf"], "pages": [1]}]})
    assert r.status_code in (401, 403)
