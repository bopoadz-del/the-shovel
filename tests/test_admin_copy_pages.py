"""Plan/execute copy of selected pages into a new project (additive only)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import page_copy
from app.core import projects as projects_mod
from app.dependencies import require_api_key
from app.main import app
from app.routers.admin import admin_copy_pages


def _url(target: str) -> str:
    path = next(r.path for r in app.routes if getattr(r, "endpoint", None) is admin_copy_pages)
    return path.replace("{target_project_id}", target)


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

    def add(name, pages):
        doc = projects_mod.add_document(src, name, size=10)
        texts = [f"{name} chunk {i} page {p}" for i, p in enumerate(pages)]
        emb = np.random.default_rng(len(texts)).random((len(texts), store.dim)).astype(np.float32)
        store.upsert_chunks(src, doc["id"], texts, emb, pages=pages,
                            ocr_flags=[p == 3 for p in pages])
        return doc["id"]

    # Start pages of the chunks: page 4 has no chunk of its own.
    pdf = add("spec.pdf", [1, 1, 2, 3, 3, 3, 5])
    sheet = add("bill.xlsx", [None, None, None])
    return {"store": store, "src": src, "pdf": pdf, "sheet": sheet}


def _fingerprint(store, src, doc_ids):
    rows = []
    for d in doc_ids:
        for r in store.doc_chunk_rows(src, d, with_vectors=True):
            rows.append((r["chunk_id"], r["page"], r["text"], r["embedding"].tobytes()))
    return hashlib.sha256(repr(rows).encode()).hexdigest()


def _post(client, target, body):
    r = client.post(_url(target), json=body)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.parametrize("pages,chunks,found,missing,ocr", [
    ([1, 3], 5, 2, [], 3),
    ([2, 4], 1, 1, [4], 0),
    ([5], 1, 1, [], 0),
])
def test_plan_counts_selected_pages_and_writes_nothing(env, pages, chunks, found, missing, ocr):
    body = {"source_project_id": env["src"], "selections": [{"doc_id": env["pdf"], "pages": pages}]}
    with TestClient(app) as client:
        out = _post(client, "core-a", body)
    plan = out["plan"]
    assert out["executed"] is False
    assert plan["eligible"] is True
    assert plan["chunks_taken"] == chunks
    assert plan["pages_found"] == found
    assert plan["selections"][0]["pages_missing"] == missing
    assert plan["selections"][0]["ocr_chunks_taken"] == ocr
    assert projects_mod.get_project("core-a") is None


@pytest.mark.parametrize("a,b", [
    ([1, 3], [2, 4]),
    ([1], [1, 2]),
])
def test_plan_hash_follows_the_selection(env, a, b):
    plan_a = page_copy.plan_page_copy(env["src"], "core-h", [{"doc_id": env["pdf"], "pages": a}])
    plan_b = page_copy.plan_page_copy(env["src"], "core-h", [{"doc_id": env["pdf"], "pages": b}])
    again = page_copy.plan_page_copy(env["src"], "core-h", [{"doc_id": env["pdf"], "pages": a}])
    assert plan_a["plan_hash"] != plan_b["plan_hash"]
    assert plan_a["plan_hash"] == again["plan_hash"]


@pytest.mark.parametrize("pages,want_pages,want_ocr", [
    ([3, 5], [3, 3, 3, 5], [True, True, True, False]),
    ([1, 2], [1, 1, 2], [False, False, False]),
])
def test_execute_copies_only_selected_pages_and_keeps_page(env, pages, want_pages, want_ocr):
    store = env["store"]
    before = _fingerprint(store, env["src"], [env["pdf"], env["sheet"]])
    sels = [{"doc_id": env["pdf"], "pages": pages}, {"doc_id": env["sheet"], "pages": "all"}]
    body = {"source_project_id": env["src"], "selections": sels,
            "profile": {"answer_path": "harnessed"}}
    with TestClient(app) as client:
        plan = _post(client, "core-b", body)["plan"]
        no_profile = _post(client, "core-b", {**body, "profile": None})["plan"]
        out = _post(client, "core-b", {**body, "execute": True, "reason": "core set",
                                        "expected_plan_hash": plan["plan_hash"]})
    # The profile is part of what the hash approves.
    assert no_profile["plan_hash"] != plan["plan_hash"]
    assert out["executed"] is True, out
    assert out["chunks_copied"] == len(want_pages) + 3
    by_src = {d["source_doc_id"]: d["doc_id"] for d in out["documents"]}
    rows = store.doc_chunk_rows("core-b", by_src[env["pdf"]], with_vectors=True)
    assert [r["page"] for r in rows] == want_pages
    assert [r["ocr"] for r in rows] == want_ocr
    assert all(f"page {p}" in r["text"] for r, p in zip(rows, want_pages))
    assert [r["page"] for r in store.doc_chunk_rows("core-b", by_src[env["sheet"]])] == [None] * 3
    new_doc = projects_mod.get_document(by_src[env["pdf"]])
    assert new_doc["project_id"] == "core-b"
    assert new_doc["chunk_count"] == len(want_pages)
    assert new_doc["metadata"]["copied_from"]["doc_id"] == env["pdf"]
    assert projects_mod.get_project("core-b")["profile"] == {"answer_path": "harnessed"}
    # The source is only read.
    assert _fingerprint(store, env["src"], [env["pdf"], env["sheet"]]) == before
    assert projects_mod.get_document(env["pdf"])["project_id"] == env["src"]


@pytest.mark.parametrize("hash_kind,reason,why", [
    ("stale", "core set", "plan_hash"),
    ("missing", "core set", "plan_hash"),
    ("live", "   ", "reason"),
])
def test_execute_refuses_and_writes_nothing(env, hash_kind, reason, why):
    sels = [{"doc_id": env["pdf"], "pages": [1]}]
    live = page_copy.plan_page_copy(env["src"], "core-c", sels)["plan_hash"]
    given = {"stale": "0" * 64, "missing": None, "live": live}[hash_kind]
    out = page_copy.execute_page_copy(env["src"], "core-c", sels,
                                      expected_plan_hash=given, reason=reason)
    assert out["executed"] is False
    assert why in out["refused"]
    assert projects_mod.get_project("core-c") is None


@pytest.mark.parametrize("pages", [[2], "all"])
def test_existing_target_is_refused(env, pages):
    projects_mod.create_project("already here", project_id="core-d")
    sels = [{"doc_id": env["pdf"], "pages": pages}]
    plan = page_copy.plan_page_copy(env["src"], "core-d", sels)
    out = page_copy.execute_page_copy(env["src"], "core-d", sels,
                                      expected_plan_hash=plan["plan_hash"], reason="x")
    assert plan["eligible"] is False
    assert plan["target_exists"] is True
    assert "target project already exists" in plan["not_eligible"]
    assert out["executed"] is False
    assert len(projects_mod.list_documents("core-d")) == 0


@pytest.mark.parametrize("which,pages,missing", [
    ("pdf", [4, 6], [4, 6]),
    ("pdf", [7], [7]),
    ("sheet", [1], [1]),
])
def test_selection_that_takes_no_chunk_is_not_eligible(env, which, pages, missing):
    plan = page_copy.plan_page_copy(env["src"], "core-e", [{"doc_id": env[which], "pages": pages}])
    assert plan["eligible"] is False
    assert plan["chunks_taken"] == 0
    assert plan["selections"][0]["pages_missing"] == missing


@pytest.mark.parametrize("pages", [[1], "all"])
def test_document_from_another_project_is_not_found(env, pages):
    other = projects_mod.create_project("other")["id"]
    foreign = projects_mod.add_document(other, "x.pdf", size=1)["id"]
    plan = page_copy.plan_page_copy(env["src"], "core-f", [
        {"doc_id": foreign, "pages": pages},
        {"doc_id": env["pdf"], "pages": [1]},
    ])
    assert plan["eligible"] is False
    assert plan["selections"][0]["found"] is False
    assert plan["selections"][1]["found"] is True


@pytest.mark.parametrize("selections,want", [
    ([{"doc_id": " a ", "pages": [3, 1, 3]}], [{"doc_id": "a", "pages": [1, 3]}]),
    ([{"doc_id": "b", "pages": "all"}], [{"doc_id": "b", "pages": "all"}]),
])
def test_normalise_selections_cleans_good_input(selections, want):
    assert page_copy._normalise_selections(selections) == want


@pytest.mark.parametrize("selections", [
    [],
    None,
    ["a"],
    [{"doc_id": "", "pages": [1]}],
    [{"doc_id": "a", "pages": []}],
    [{"doc_id": "a", "pages": [0]}],
    [{"doc_id": "a", "pages": [True]}],
    [{"doc_id": "a", "pages": "some"}],
    [{"doc_id": "a", "pages": [1]}, {"doc_id": "a", "pages": [2]}],
])
def test_bad_selection_is_a_400(env, selections):
    with pytest.raises(ValueError):
        page_copy._normalise_selections(selections)
    with TestClient(app) as client:
        r = client.post(_url("core-g"), json={"source_project_id": env["src"],
                                              "selections": selections or []})
    assert r.status_code in (400, 422), r.text
    assert projects_mod.get_project("core-g") is None


@pytest.mark.parametrize("role", ["user", "readonly"])
def test_non_admin_is_refused(env, role):
    app.dependency_overrides[require_api_key] = lambda: {"user_id": "u1", "role": role}
    with TestClient(app) as client:
        r = client.post(_url("core-h"), json={"source_project_id": env["src"],
                                              "selections": [{"doc_id": env["pdf"], "pages": [1]}]})
    assert r.status_code in (401, 403)
    assert projects_mod.get_project("core-h") is None


@pytest.mark.parametrize("call", [0, 1])
def test_store_is_the_process_store(env, call):
    assert page_copy._store() is env["store"]
