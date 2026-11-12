"""A spreadsheet's chunks survive the ledger write; a terminal type's do not.

The 2026-11-06 spreadsheet exclusion stamped every spreadsheet
UNSUPPORTED_TYPE terminal before evidence-beats-policy, and
``stamp_document_index`` deletes the chunks of any no-chunk status -- so a
re-extract that row-chunked a workbook threw the chunks away in the same
commit. The exclusion was reversed 2026-11-11. These tests drive the real
pair (``classify`` -> ``stamp_document_index``) on synthetic documents.
"""
from __future__ import annotations

import importlib

import pytest

from app.core import ingest_status as ist
from app.core.rag.embeddings import get_embedder, reset_embedder_cache
from app.core.rag.vector_store import VectorStore, reset_store_cache


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
    pid = projects.create_project(name="Sheets", client="C", user_id="u1")["id"]
    url = db_mod.get_database_url()
    embedder = get_embedder()
    store = VectorStore(db_path=url.replace("sqlite:///", ""), dim=embedder.dim)
    yield projects, store, embedder, pid
    store.close()


def _ingest(world, name: str, rows: list[str]):
    """Add a document, write its chunks, then stamp what ``classify`` decides."""
    projects, store, embedder, pid = world
    doc = projects.add_document(project_id=pid, original_name=name, size=4096)
    if rows:
        store.upsert_chunks(pid, doc["id"], rows, embedder.encode(rows))
    outcome = ist.classify(chunk_count=len(rows), extension="." + name.rsplit(".", 1)[1])
    projects.stamp_document_index(
        doc["id"], chunk_count=len(rows),
        ingest_status=outcome.status, ingest_status_reason=outcome.reason,
    )
    return doc["id"], outcome


@pytest.mark.parametrize("name,rows", [
    ("bill_of_quantities.xlsx", ["item 1.01 | excavation | m3 | 420 | 35.00",
                                 "item 1.02 | backfill | m3 | 180 | 22.50",
                                 "item 2.01 | blinding | m2 | 960 | 18.00"]),
    ("rates_schedule.xls", ["code A | labour | hr | 55.00",
                            "code B | plant | day | 1200.00"]),
])
def test_spreadsheet_round_trip_keeps_its_chunks(world, name, rows):
    projects, store, _emb, pid = world
    doc_id, outcome = _ingest(world, name, rows)

    assert outcome.status == ist.INDEXED
    assert store.count_by_doc(pid).get(doc_id, 0) == len(rows)
    stamped = projects.get_document(doc_id)
    assert stamped["ingest_status"] == ist.INDEXED
    assert stamped["chunk_count"] == len(rows)


@pytest.mark.parametrize("name", ["empty_sheet.xlsx", "blank_rates.xls"])
def test_spreadsheet_without_chunks_is_not_indexed(world, name):
    projects, store, _emb, pid = world
    doc_id, outcome = _ingest(world, name, [])

    assert outcome.status not in ist.INDEXED_STATUSES
    assert projects.get_document(doc_id)["ingest_status"] == ist.ZERO_CHUNK
    assert store.count_by_doc(pid).get(doc_id, 0) == 0


@pytest.mark.parametrize("name,rows", [
    ("site_boundary.kmz", ["Placemark | 55.27,25.20", "Polyline [9CA0]"]),
    ("survey_layer.kml", ["Hatch [771B]", "Point | 55.30,25.18", "LineString"]),
])
def test_genuinely_terminal_extension_still_loses_its_chunks(world, name, rows):
    projects, store, _emb, pid = world
    doc_id, outcome = _ingest(world, name, rows)

    assert outcome.status == ist.UNSUPPORTED_TYPE
    assert outcome.reason.endswith(ist.TERMINAL)
    assert store.count_by_doc(pid).get(doc_id, 0) == 0
    assert projects.get_document(doc_id)["chunk_count"] == 0
