"""An owner-excluded document loses every chunk, keeps its row, and is never
ingested, indexed, retrieved or listed again.

Synthetic project and documents. Every decision is checked on two inputs: an
excluded document and a non-excluded one in the same project.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.core.rag.embeddings import get_embedder, reset_embedder_cache
from app.core.rag.vector_store import VectorStore, reset_store_cache

TEXT = "SYNTHWORD rate schedule for bulk excavation in rock"


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "fake")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    import app.core.db as db_mod
    import app.core.users as users_mod
    from app.core import projects

    importlib.reload(db_mod)
    importlib.reload(users_mod)
    users_mod._initialized = False
    projects._initialized = False
    reset_embedder_cache()
    reset_store_cache()
    projects.init_db()
    users_mod.ensure_user_exists("u1")
    pid = projects.create_project(name="Excl", client="C", user_id="u1")["id"]

    url = db_mod.get_database_url()
    embedder = get_embedder()
    store = VectorStore(db_path=url.replace("sqlite:///", ""), dim=embedder.dim)
    docs = {}
    for name, n in (("drop.txt", 3), ("keep_a.txt", 2), ("keep_b.txt", 1)):
        path = tmp_path / name
        path.write_text(f"{TEXT} {name}", encoding="utf-8")
        doc = projects.add_document(
            project_id=pid, original_name=name, file_path=str(path), size=path.stat().st_size,
        )
        texts = [f"{TEXT} {name} part {i}" for i in range(n)]
        store.upsert_chunks(pid, doc["id"], texts, embedder.encode(texts))
        projects.stamp_document_index(doc["id"], chunk_count=n, ingest_status="INDEXED")
        docs[name] = doc["id"]
    store._visibility_ready = None
    yield projects, store, embedder, pid, docs
    store.close()


def _counts(store, pid):
    return dict(store.count_by_doc(pid))


def _exclude(projects, doc_id):
    return projects.exclude_document(doc_id, by="owner", reason="unpriced copy", at="2026-11-13T15:05:00Z")


def test_excluding_deletes_only_that_documents_chunks_and_keeps_the_row(world):
    projects, store, _emb, pid, docs = world
    before = _counts(store, pid)
    plan = projects.plan_document_exclusion(docs["drop.txt"])
    assert plan["chunk_rows"] == 3 and len(plan["rows_to_delete"]) == 3
    assert plan["project_chunk_total"] == 6

    out = _exclude(projects, docs["drop.txt"])

    after = _counts(store, pid)
    assert out["chunks_deleted"] == 3
    assert after.get(docs["drop.txt"], 0) == 0
    assert after[docs["keep_a.txt"]] == before[docs["keep_a.txt"]] == 2
    assert after[docs["keep_b.txt"]] == before[docs["keep_b.txt"]] == 1
    row = projects.get_document(docs["drop.txt"])
    assert row is not None  # kept, as the record of the decision
    assert row["chunk_count"] == 0 and row["retrieval_visible"] is False
    assert row["metadata"]["excluded"] == {
        "by": "owner", "at": "2026-11-13T15:05:00Z", "reason": "unpriced copy",
    }
    assert projects.plan_document_exclusion(docs["drop.txt"])["project_chunk_total"] == 3
    kept = projects.get_document(docs["keep_a.txt"])
    assert kept["chunk_count"] == 2 and "excluded" not in (kept.get("metadata") or {})


def test_plan_hash_moves_when_the_chunk_rows_move(world):
    projects, store, embedder, pid, docs = world
    first = projects.plan_document_exclusion(docs["keep_a.txt"])["plan_hash"]
    again = projects.plan_document_exclusion(docs["keep_a.txt"])["plan_hash"]
    other = projects.plan_document_exclusion(docs["keep_b.txt"])["plan_hash"]
    store.upsert_chunks(pid, docs["keep_a.txt"], ["one", "two", "three"], embedder.encode(["one", "two", "three"]))
    moved = projects.plan_document_exclusion(docs["keep_a.txt"])["plan_hash"]
    assert first == again
    assert first != other
    assert first != moved


def test_is_excluded_reads_the_attribute_only():
    from app.core.ingest_status import is_excluded

    assert is_excluded({"metadata": {"excluded": {"by": "owner"}}}) is True
    assert is_excluded({"metadata": {"drive_file_id": "x"}}) is False
    assert is_excluded({"metadata": {"excluded": {}}}) is False
    assert is_excluded(None) is False
    assert is_excluded({"metadata": "not a dict"}) is False


def test_index_document_skips_an_excluded_document_and_not_another(world):
    projects, store, _emb, pid, docs = world
    from app.core import doc_index

    _exclude(projects, docs["drop.txt"])
    skipped = doc_index.index_document(pid, docs["drop.txt"])
    indexed = doc_index.index_document(pid, docs["keep_a.txt"])

    assert skipped["excluded"] is True and skipped["indexed"] == 0
    assert _counts(store, pid).get(docs["drop.txt"], 0) == 0
    assert projects.get_document(docs["drop.txt"])["chunk_count"] == 0
    assert indexed.get("excluded") is not True
    assert _counts(store, pid).get(docs["keep_a.txt"], 0) >= 1


def test_retrieval_never_returns_an_excluded_document(world):
    projects, store, embedder, pid, docs = world
    _exclude(projects, docs["drop.txt"])
    store._visibility_ready = None

    vec_ids = [c.doc_id for c in store._semantic_search(pid, embedder.encode([TEXT])[0], k=10)]
    bm25_ids = [c.doc_id for c in store.bm25_search(pid, "SYNTHWORD", k=10)]
    for ids in (vec_ids, bm25_ids):
        assert docs["keep_a.txt"] in ids
        assert docs["drop.txt"] not in ids


def test_documents_list_hides_an_excluded_document(world):
    projects, _store, _emb, pid, docs = world
    _exclude(projects, docs["drop.txt"])

    listed = {d["id"] for d in projects.list_documents(pid)}
    named = {d["id"] for d in projects.list_document_names(pid)}
    for ids in (listed, named):
        assert docs["drop.txt"] not in ids
        assert {docs["keep_a.txt"], docs["keep_b.txt"]} <= ids


def _reindex_deps(projects, calls):
    from scripts.doc_reindex_once import Deps

    def _download(drive_id):
        calls.append(("download", drive_id))
        return b"bytes", None

    return Deps(
        get_document=projects.get_document,
        drive_file_id_of=lambda doc: "drive-x",
        download=_download,
        write_transient=lambda raw, name: Path(name),
        discard_transient=lambda path: None,
        set_file_path=lambda doc_id, path: None,
        index_document=lambda *a, **k: calls.append(("index", a[1])) or {"status": "ok"},
        chunk_texts=lambda p, d: [],
        log=lambda m: None,
    )


def test_doc_reindex_once_refuses_an_excluded_document_before_download(world):
    projects, _store, _emb, pid, docs = world
    from scripts.doc_reindex_once import reindex_one

    _exclude(projects, docs["drop.txt"])
    calls: list = []
    refused = reindex_one(pid, docs["drop.txt"], force_ocr=False, heading_re=None,
                          deps=_reindex_deps(projects, calls))
    assert "excluded" in refused["error"] and calls == []

    reindex_one(pid, docs["keep_a.txt"], force_ocr=False, heading_re=None,
                deps=_reindex_deps(projects, calls))
    assert ("download", "drive-x") in calls and ("index", docs["keep_a.txt"]) in calls


def test_drive_worker_skips_an_excluded_row_before_download(world, tmp_path):
    projects, _store, _emb, pid, docs = world
    from scripts import p1b_ingest_drive_server as worker

    _exclude(projects, docs["drop.txt"])
    meta = {"id": "drive-x", "name": "drop.txt", "_drive_path": "drop.txt",
            "mimeType": "text/plain", "size": "10"}
    _rel, skipped = worker._ingest_file(
        meta, pid, tmp_path, "run", gdrive_service=None,
        existing_doc=projects.get_document(docs["drop.txt"]),
    )
    assert skipped == {"status": "error", "error": "SKIPPED_EXCLUDED"}

    # A non-excluded row goes past the gate (to the download, which this
    # synthetic service cannot serve) -- it is not refused as excluded.
    try:
        _rel, other = worker._ingest_file(
            meta, pid, tmp_path, "run", gdrive_service=None,
            existing_doc=projects.get_document(docs["keep_a.txt"]),
        )
    except Exception as exc:  # noqa: BLE001 - reaching the download is the point
        other = {"error": type(exc).__name__}
    assert other.get("error") != "SKIPPED_EXCLUDED"


def _route(**kw):
    from app.routers.admin import admin_exclude_document

    base = {"auth": {"role": "admin", "user_id": "u1"}, "plan_hash": None, "reason": ""}
    base.update(kw)
    return admin_exclude_document(**base)


def test_route_plans_without_writing_and_executes_only_with_the_live_hash(world):
    projects, store, _emb, pid, docs = world
    plan = _route(document_id=docs["drop.txt"], project_id=pid, execute=False)
    assert plan["executed"] is False and plan["plan"]["chunk_rows"] == 3

    wrong = _route(document_id=docs["drop.txt"], project_id=pid, execute=True,
                   plan_hash="0" * 64, reason="unpriced copy")
    no_reason = _route(document_id=docs["drop.txt"], project_id=pid, execute=True,
                       plan_hash=plan["plan"]["plan_hash"])
    assert wrong["executed"] is False and "plan_hash" in wrong["refused"]
    assert no_reason["executed"] is False and "reason" in no_reason["refused"]
    assert _counts(store, pid)[docs["drop.txt"]] == 3

    done = _route(document_id=docs["drop.txt"], project_id=pid, execute=True,
                  plan_hash=plan["plan"]["plan_hash"], reason="unpriced copy")
    assert done["executed"] is True and done["chunks_deleted"] == 3
    assert done["chunk_rows_after"] == 0 and done["chunk_count_after"] == 0
    assert done["project_chunk_total_after"] == 3
    assert done["excluded"]["by"] == "owner" and done["excluded"]["reason"] == "unpriced copy"


def test_route_refuses_a_document_of_another_project_and_a_non_admin(world):
    projects, _store, _emb, pid, docs = world
    with pytest.raises(HTTPException) as other_project:
        _route(document_id=docs["drop.txt"], project_id="someone-else", execute=False)
    with pytest.raises(HTTPException) as not_admin:
        _route(document_id=docs["drop.txt"], project_id=pid, execute=False,
               auth={"role": "user", "user_id": "u1"})
    assert other_project.value.status_code == 400
    assert not_admin.value.status_code == 403
