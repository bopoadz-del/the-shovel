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


SIZES = {"drop.txt": 3, "keep_a.txt": 2, "keep_b.txt": 1}
TOTAL = sum(SIZES.values())


@pytest.mark.parametrize("target", ["drop.txt", "keep_b.txt"])
def test_excluding_deletes_only_that_documents_chunks_and_keeps_the_row(world, target):
    projects, store, _emb, pid, docs = world
    others = [n for n in SIZES if n != target]
    before = _counts(store, pid)
    plan = projects.plan_document_exclusion(docs[target])
    assert plan["chunk_rows"] == SIZES[target] and len(plan["rows_to_delete"]) == SIZES[target]
    assert plan["project_chunk_total"] == TOTAL

    out = _exclude(projects, docs[target])

    after = _counts(store, pid)
    assert out["chunks_deleted"] == SIZES[target]
    assert after.get(docs[target], 0) == 0
    for name in others:
        assert after[docs[name]] == before[docs[name]] == SIZES[name]
        kept = projects.get_document(docs[name])
        assert kept["chunk_count"] == SIZES[name] and "excluded" not in (kept.get("metadata") or {})
    row = projects.get_document(docs[target])
    assert row is not None  # kept, as the record of the decision
    assert row["chunk_count"] == 0 and row["retrieval_visible"] is False
    assert row["metadata"]["excluded"] == {
        "by": "owner", "at": "2026-11-13T15:05:00Z", "reason": "unpriced copy",
    }
    assert projects.plan_document_exclusion(docs[target])["project_chunk_total"] == TOTAL - SIZES[target]


@pytest.mark.parametrize("missing", ["", None, "no-such-document"])
def test_plan_and_exclude_answer_none_for_a_missing_id_without_a_db_lookup_warning(world, missing):
    projects, store, _emb, pid, _docs = world
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a NULL-identity lookup warns; the guard must stop it first
        assert projects.plan_document_exclusion(missing) is None
        assert projects.exclude_document(missing, by="owner", reason="r", at="t") is None
    assert sum(_counts(store, pid).values()) == TOTAL


@pytest.mark.parametrize("target,other", [("keep_a.txt", "keep_b.txt"), ("keep_b.txt", "drop.txt")])
def test_plan_hash_moves_when_the_chunk_rows_move(world, target, other):
    projects, store, embedder, pid, docs = world
    first = projects.plan_document_exclusion(docs[target])["plan_hash"]
    again = projects.plan_document_exclusion(docs[target])["plan_hash"]
    other_hash = projects.plan_document_exclusion(docs[other])["plan_hash"]
    texts = [f"moved {i}" for i in range(SIZES[target] + 1)]
    store.upsert_chunks(pid, docs[target], texts, embedder.encode(texts))
    moved = projects.plan_document_exclusion(docs[target])["plan_hash"]
    assert first == again
    assert first != other_hash
    assert first != moved


def test_is_excluded_reads_the_attribute_only():
    from app.core.ingest_status import is_excluded

    assert is_excluded({"metadata": {"excluded": {"by": "owner"}}}) is True
    assert is_excluded({"metadata": {"drive_file_id": "x"}}) is False
    assert is_excluded({"metadata": {"excluded": {}}}) is False
    assert is_excluded(None) is False
    assert is_excluded({"metadata": "not a dict"}) is False


PAIRS = [("drop.txt", "keep_a.txt"), ("keep_a.txt", "keep_b.txt")]


@pytest.mark.parametrize("excluded,kept", PAIRS)
def test_index_document_skips_an_excluded_document_and_not_another(world, excluded, kept):
    projects, store, _emb, pid, docs = world
    from app.core import doc_index

    _exclude(projects, docs[excluded])
    skipped = doc_index.index_document(pid, docs[excluded])
    indexed = doc_index.index_document(pid, docs[kept])

    assert skipped["excluded"] is True and skipped["indexed"] == 0
    assert _counts(store, pid).get(docs[excluded], 0) == 0
    assert projects.get_document(docs[excluded])["chunk_count"] == 0
    assert indexed.get("excluded") is not True
    assert _counts(store, pid).get(docs[kept], 0) >= 1


@pytest.mark.parametrize("excluded,kept", PAIRS)
def test_retrieval_never_returns_an_excluded_document(world, excluded, kept):
    projects, store, embedder, pid, docs = world
    _exclude(projects, docs[excluded])
    store._visibility_ready = None

    vec_ids = [c.doc_id for c in store._semantic_search(pid, embedder.encode([TEXT])[0], k=10)]
    bm25_ids = [c.doc_id for c in store.bm25_search(pid, "SYNTHWORD", k=10)]
    for ids in (vec_ids, bm25_ids):
        assert docs[kept] in ids
        assert docs[excluded] not in ids


@pytest.mark.parametrize("excluded,kept", PAIRS)
def test_documents_list_hides_an_excluded_document(world, excluded, kept):
    projects, _store, _emb, pid, docs = world
    _exclude(projects, docs[excluded])

    listed = {d["id"] for d in projects.list_documents(pid)}
    named = {d["id"] for d in projects.list_document_names(pid)}
    for ids in (listed, named):
        assert docs[excluded] not in ids
        assert {docs[n] for n in SIZES if n != excluded} <= ids
        assert docs[kept] in ids


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


@pytest.mark.parametrize("excluded,kept", PAIRS)
def test_doc_reindex_once_refuses_an_excluded_document_before_download(world, excluded, kept):
    projects, _store, _emb, pid, docs = world
    from scripts.doc_reindex_once import reindex_one

    _exclude(projects, docs[excluded])
    calls: list = []
    refused = reindex_one(pid, docs[excluded], force_ocr=False, heading_re=None,
                          deps=_reindex_deps(projects, calls))
    assert "excluded" in refused["error"]
    assert calls == []

    reindex_one(pid, docs[kept], force_ocr=False, heading_re=None,
                deps=_reindex_deps(projects, calls))
    assert ("download", "drive-x") in calls and ("index", docs[kept]) in calls


@pytest.mark.parametrize("excluded,kept", PAIRS)
def test_drive_worker_skips_an_excluded_row_before_download(world, tmp_path, excluded, kept):
    projects, _store, _emb, pid, docs = world
    from scripts import p1b_ingest_drive_server as worker

    _exclude(projects, docs[excluded])
    meta = {"id": "drive-x", "name": excluded, "_drive_path": excluded,
            "mimeType": "text/plain", "size": "10"}
    _rel, skipped = worker._ingest_file(
        meta, pid, tmp_path, "run", gdrive_service=None,
        existing_doc=projects.get_document(docs[excluded]),
    )
    assert skipped == {"status": "error", "error": "SKIPPED_EXCLUDED"}

    # A non-excluded row goes past the gate (to the download, which this
    # synthetic service cannot serve) -- it is not refused as excluded.
    try:
        _rel, other = worker._ingest_file(
            meta, pid, tmp_path, "run", gdrive_service=None,
            existing_doc=projects.get_document(docs[kept]),
        )
    except Exception as exc:  # noqa: BLE001 - reaching the download is the point
        other = {"error": type(exc).__name__}
    assert other.get("error") != "SKIPPED_EXCLUDED"


def _route(**kw):
    from app.routers.admin import admin_exclude_document

    base = {"auth": {"role": "admin", "user_id": "u1"}, "plan_hash": None, "reason": ""}
    base.update(kw)
    return admin_exclude_document(**base)


@pytest.mark.parametrize("target", ["drop.txt", "keep_a.txt"])
def test_route_plans_without_writing_and_executes_only_with_the_live_hash(world, target):
    projects, store, _emb, pid, docs = world
    plan = _route(document_id=docs[target], project_id=pid, execute=False)
    assert plan["executed"] is False and plan["plan"]["chunk_rows"] == SIZES[target]

    wrong = _route(document_id=docs[target], project_id=pid, execute=True,
                   plan_hash="0" * 64, reason="unpriced copy")
    no_reason = _route(document_id=docs[target], project_id=pid, execute=True,
                       plan_hash=plan["plan"]["plan_hash"])
    assert wrong["executed"] is False and "plan_hash" in wrong["refused"]
    assert no_reason["executed"] is False and "reason" in no_reason["refused"]
    assert _counts(store, pid)[docs[target]] == SIZES[target]

    done = _route(document_id=docs[target], project_id=pid, execute=True,
                  plan_hash=plan["plan"]["plan_hash"], reason="unpriced copy")
    assert done["executed"] is True and done["chunks_deleted"] == SIZES[target]
    assert done["chunk_rows_after"] == 0 and done["chunk_count_after"] == 0
    assert done["project_chunk_total_after"] == TOTAL - SIZES[target]
    assert done["excluded"]["by"] == "owner" and done["excluded"]["reason"] == "unpriced copy"


@pytest.mark.parametrize("case,status", [
    ("unknown_document", 404),
    ("other_project", 400),
    ("not_admin", 403),
])
def test_route_refuses_unknown_document_other_project_and_non_admin(world, case, status):
    projects, store, _emb, pid, docs = world
    kw = {"document_id": docs["drop.txt"], "project_id": pid, "execute": True,
          "plan_hash": projects.plan_document_exclusion(docs["drop.txt"])["plan_hash"],
          "reason": "unpriced copy"}
    if case == "unknown_document":
        kw["document_id"] = "no-such-document"
    elif case == "other_project":
        kw["project_id"] = "someone-else"
    else:
        kw["auth"] = {"role": "user", "user_id": "u1"}
    with pytest.raises(HTTPException) as refused:
        _route(**kw)
    assert refused.value.status_code == status
    assert sum(_counts(store, pid).values()) == TOTAL  # nothing written on any refusal


@pytest.mark.parametrize("offset", [1, 2])
def test_documents_list_without_a_limit_is_every_document_whatever_the_offset(world, offset):
    """``limit=None`` is the legacy "all docs" listing: an offset only pages a
    LIMITED query, it never silently drops rows from the full list."""
    projects, _store, _emb, pid, docs = world
    listed = [d["id"] for d in projects.list_documents(pid, offset=offset)]
    assert sorted(listed) == sorted(docs.values())
    paged = [d["id"] for d in projects.list_documents(pid, limit=1, offset=offset)]
    assert paged == [listed[offset]]
