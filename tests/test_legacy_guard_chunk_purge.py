"""Legacy guard cleanup: chunks the retired BOQ guard wrote as prose are purged.

Every test seeds its own documents and chunk rows. The purge plans by the old
writer's lead-in anchored at the start of the chunk text, deletes only on an
exact expected-count match, and re-derives each touched document's ledger.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.core import ingest_status as ist
from app.core.ingest_reconcile import (
    LEGACY_TOTAL_GUARD_PREFIX,
    p3_legacy_guard_chunk_purge,
)
from tests.test_ingest_reconcile import _reload_projects

ADMIN = {"user_id": "admin", "role": "admin"}


def _guard_text(name: str) -> str:
    return (
        f"{LEGACY_TOTAL_GUARD_PREFIX}'{name}': this is a Bill of Quantities, "
        "but NO verified total could be computed."
    )


def _seed(project_id: str, doc_id: str, texts: list[str]) -> None:
    import os

    import numpy as np
    from app.core.db import SessionLocal, engine
    from app.core.models import (
        EMBEDDING_DIM,
        make_rag_chunk_class,
        rag_chunk_class_for,
    )
    from app.core.rag.vector_store import ensure_table

    ns = (os.getenv("RAG_VECTOR_NAMESPACE", "v2") or "").strip()
    cls = rag_chunk_class_for(ns) or make_rag_chunk_class(ns, EMBEDDING_DIM, "test")
    ensure_table(cls.__table__, bind=engine)
    vec = np.zeros(int(cls.embedding_identity["dim"]), dtype=np.float32)
    now = datetime.now(timezone.utc).isoformat()
    with SessionLocal() as session:
        for i, body in enumerate(texts):
            session.add(cls(
                chunk_id=f"{doc_id}-{i}", project_id=project_id, doc_id=doc_id,
                chunk_index=i, text=body, embedding=vec, created_at=now,
            ))
        session.commit()


def _texts_of(doc_id: str) -> list[str]:
    from sqlalchemy import text
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    table = rag_chunk_table_name(_rag_vector_namespace())
    with SessionLocal() as session:
        rows = session.execute(
            text(f"SELECT text FROM {table} WHERE doc_id = :d ORDER BY chunk_index"),
            {"d": doc_id},
        ).all()
    return [r[0] for r in rows]


def _doc(pm, project_id: str, name: str, texts: list[str]) -> str:
    doc = pm.add_document(project_id, name, size=10)
    pm.stamp_document_index(
        doc["id"], chunk_count=len(texts), ingest_status=ist.INDEXED,
    )
    _seed(project_id, doc["id"], texts)
    return doc["id"]


# Two documents of different shape: a typed bill whose guard sat beside real
# rows, and a guard-only file whose sole chunk was the warning.
CASES = [
    pytest.param(
        "Bill A.pdf",
        ["| 1.1 | Excavation | m3 | 120 | 15.00 | 1800.00 |",
         "| 1.2 | Backfill | m3 | 80 | 9.50 | 760.00 |"],
        id="guard_beside_priced_rows",
    ),
    pytest.param("Scan B.pdf", [], id="guard_only"),
]


@pytest.mark.parametrize("name,content", CASES)
def test_plan_counts_then_execute_deletes_only_guard_chunks(
    monkeypatch, tmp_path, name, content,
):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("G", user_id="system", project_id="gproj")
    did = _doc(pm, "gproj", name, [_guard_text(name), *content])

    plan = p3_legacy_guard_chunk_purge(project_id="gproj")
    assert plan["guard_chunk_count"] == 1
    assert plan["documents"] == [did]
    assert plan["dry_run"] is True
    assert plan["deleted"] == 0
    assert len(_texts_of(did)) == 1 + len(content)  # plan writes nothing

    done = p3_legacy_guard_chunk_purge(
        project_id="gproj", execute=True, expected_count=1,
    )
    assert done["refused"] is None
    assert done["deleted"] == 1
    assert _texts_of(did) == content

    got = pm.get_document(did)
    assert got["chunk_count"] == len(content)
    expected_status = ist.classify(chunk_count=len(content), extension=".pdf").status
    assert got["ingest_status"] == expected_status
    assert done["restamped"] == [{
        "doc_id": did, "chunk_count": len(content), "ext": ".pdf",
        "ingest_status": expected_status,
    }]

    again = p3_legacy_guard_chunk_purge(project_id="gproj")
    assert again["guard_chunk_count"] == 0


@pytest.mark.parametrize(
    "body",
    [
        f"Note: the {LEGACY_TOTAL_GUARD_PREFIX}text appears mid-sentence here.",
        "verified-total guard for 'x': lower-case is document prose, not the lead-in",
    ],
)
def test_lead_in_must_start_the_chunk(monkeypatch, tmp_path, body):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("M", user_id="system", project_id="mproj")
    did = _doc(pm, "mproj", "notes.pdf", [body])

    plan = p3_legacy_guard_chunk_purge(project_id="mproj")
    assert plan["guard_chunk_count"] == 0
    done = p3_legacy_guard_chunk_purge(
        project_id="mproj", execute=True, expected_count=0,
    )
    assert done["deleted"] == 0
    assert _texts_of(did) == [body]


@pytest.mark.parametrize("expected", [None, 2])
def test_execute_refuses_unless_expected_equals_plan(
    monkeypatch, tmp_path, expected,
):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("R", user_id="system", project_id="rproj")
    did = _doc(pm, "rproj", "Bill R.pdf", [_guard_text("Bill R.pdf"), "| 1 | Item | 5.00 |"])

    done = p3_legacy_guard_chunk_purge(
        project_id="rproj", execute=True, expected_count=expected,
    )
    assert done["guard_chunk_count"] == 1
    assert done["deleted"] == 0
    assert done["refused"] == f"planned 1 != expected {expected}; nothing deleted"
    assert len(_texts_of(did)) == 2
    assert pm.get_document(did)["chunk_count"] == 2


@pytest.mark.parametrize("target,other", [("p1", "p2"), ("p2", "p1")])
def test_project_filter_scopes_plan_and_delete(monkeypatch, tmp_path, target, other):
    pm = _reload_projects(monkeypatch, tmp_path)
    ids = {}
    for pid, guards in (("p1", 1), ("p2", 2)):
        pm.create_project(pid, user_id="system", project_id=pid)
        ids[pid] = [
            _doc(pm, pid, f"{pid}-{i}.pdf", [_guard_text(f"{pid}-{i}.pdf"), "row"])
            for i in range(guards)
        ]
    want = len(ids[target])

    plan = p3_legacy_guard_chunk_purge(project_id=target)
    assert plan["guard_chunk_count"] == want
    assert plan["documents"] == sorted(ids[target])

    done = p3_legacy_guard_chunk_purge(
        project_id=target, execute=True, expected_count=want,
    )
    assert done["deleted"] == want
    for did in ids[target]:
        assert _texts_of(did) == ["row"]
    for did in ids[other]:
        assert len(_texts_of(did)) == 2


@pytest.mark.parametrize("execute", [False, True])
def test_admin_route_passes_through_plan_and_execute(monkeypatch, tmp_path, execute):
    from app.routers.admin import admin_corpus_legacy_guard_chunks

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("A", user_id="system", project_id="aproj")
    did = _doc(pm, "aproj", "Bill A.pdf", [_guard_text("Bill A.pdf"), "row"])

    out = admin_corpus_legacy_guard_chunks(
        project_id="aproj", execute=execute, expected_count=1, auth=ADMIN,
    )
    assert out["guard_chunk_count"] == 1
    assert out["dry_run"] is (not execute)
    assert out["deleted"] == (1 if execute else 0)
    assert len(_texts_of(did)) == (1 if execute else 2)


@pytest.mark.parametrize("role", ["user", "viewer"])
def test_admin_route_rejects_non_admin(monkeypatch, tmp_path, role):
    from fastapi import HTTPException
    from app.routers.admin import admin_corpus_legacy_guard_chunks

    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project("N", user_id="system", project_id="nproj")
    did = _doc(pm, "nproj", "Bill N.pdf", [_guard_text("Bill N.pdf")])

    with pytest.raises(HTTPException):
        admin_corpus_legacy_guard_chunks(
            project_id="nproj", execute=True, expected_count=1,
            auth={"user_id": "u", "role": role},
        )
    assert len(_texts_of(did)) == 1
