"""RQ-1 purge: existing noise chunks planned by class, deleted only on the plan hash.

Every test seeds its own project, documents and chunk rows.
"""
from __future__ import annotations

import pytest

from app.core import ingest_status as ist
from app.core.ingest_reconcile import LEGACY_TOTAL_GUARD_PREFIX
from app.core.noise_chunk_purge import classify_document_chunks, noise_chunk_purge
from tests.test_ingest_reconcile import _reload_projects
from tests.test_legacy_guard_chunk_purge import _doc, _texts_of

HEADER = "Item | Description | Unit | Qty"
ROW = "1.1 | Excavation | m3 | 120"


@pytest.mark.parametrize(
    "texts, classes",
    [
        # header re-stated as its own chunk after the document already said it
        ([f"{HEADER}\n{ROW}", HEADER], [None, "header_repeat"]),
        # the same label row as the FIRST statement is the header, kept
        ([HEADER, ROW], [None, None]),
        (["0 | 0 | 0\n0 | 0 | 0", ROW], ["empty_row", None]),
        ([f"{LEGACY_TOTAL_GUARD_PREFIX}'b.pdf': no total.", ROW], ["guard_sentence", None]),
    ],
)
def test_classify_document_chunks(texts, classes):
    assert classify_document_chunks(texts) == classes


def _project(monkeypatch, tmp_path, pid):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project(pid, user_id="system", project_id=pid)
    return pm


BILL = [HEADER + chr(10) + ROW, "0 | 0 | 0", HEADER, "2.1 | Backfill | m3 | 80"]
SPEC = ["Clause 3.1 Concrete grade C40.", ROW]


@pytest.mark.parametrize(
    "name, texts, counts, kept",
    [
        ("bill.xlsx", BILL,
         {"guard_sentence": 0, "empty_row": 1, "header_repeat": 1}, [BILL[0], BILL[3]]),
        ("spec.pdf", SPEC,
         {"guard_sentence": 0, "empty_row": 0, "header_repeat": 0}, SPEC),
    ],
)
def test_plan_writes_nothing_and_execute_needs_the_live_hash(
    monkeypatch, tmp_path, name, texts, counts, kept,
):
    pm = _project(monkeypatch, tmp_path, "nproj")
    did = _doc(pm, "nproj", name, texts)

    plan = noise_chunk_purge(project_id="nproj")["plan"]
    assert plan["counts"] == counts
    assert plan["total"] == len(texts) - len(kept)
    assert plan["documents"] == ([did] if kept != texts else [])
    assert len(_texts_of(did)) == len(texts)  # the plan writes nothing

    wrong = noise_chunk_purge(project_id="nproj", execute=True, plan_hash="0" * 64)
    assert wrong["executed"] is False and wrong["refused"]
    assert len(_texts_of(did)) == len(texts)

    done = noise_chunk_purge(
        project_id="nproj", execute=True, plan_hash=plan["plan_hash"],
    )
    assert done["executed"] is True
    assert done["deleted"] == len(texts) - len(kept)
    assert _texts_of(did) == kept
    if kept != texts:
        ext = name[name.rindex("."):]
        got = pm.get_document(did)
        assert got["chunk_count"] == len(kept)
        assert got["ingest_status"] == ist.classify(
            chunk_count=len(kept), extension=ext,
        ).status
    assert noise_chunk_purge(project_id="nproj")["plan"]["total"] == 0


@pytest.mark.parametrize("project_id, raises", [("", True), ("eproj", False)])
def test_project_id_is_required(monkeypatch, tmp_path, project_id, raises):
    _project(monkeypatch, tmp_path, "eproj")
    if raises:
        with pytest.raises(ValueError):
            noise_chunk_purge(project_id=project_id)
    else:
        assert noise_chunk_purge(project_id=project_id)["plan"]["total"] == 0


ADMIN = {"user_id": "admin", "role": "admin"}


@pytest.mark.parametrize("execute", [False, True])
def test_admin_route_plans_then_writes_on_hash(monkeypatch, tmp_path, execute):
    from app.routers.admin import admin_corpus_noise_chunks

    pm = _project(monkeypatch, tmp_path, "rproj")
    did = _doc(pm, "rproj", "w.xlsx", [ROW, "0 | 0"])
    from app.core import audit

    events: list = []
    monkeypatch.setattr(audit, "record", lambda event, **kw: events.append((event, kw)))
    planned = admin_corpus_noise_chunks(project_id="rproj", auth=ADMIN)["plan"]
    out = admin_corpus_noise_chunks(
        project_id="rproj", execute=execute, plan_hash=planned["plan_hash"], auth=ADMIN,
    )
    assert out["executed"] is execute
    assert len(_texts_of(did)) == (1 if execute else 2)
    # only a write is audited, with what it deleted
    assert [e for e, _ in events] == (["corpus.noise_chunks_purged"] if execute else [])
    if execute:
        assert events[0][1]["deleted"] == 1


@pytest.mark.parametrize("role", ["user", "viewer"])
def test_admin_route_rejects_non_admin(monkeypatch, tmp_path, role):
    from fastapi import HTTPException

    from app.routers.admin import admin_corpus_noise_chunks

    pm = _project(monkeypatch, tmp_path, "xproj")
    did = _doc(pm, "xproj", "w.xlsx", ["0 | 0"])
    with pytest.raises(HTTPException):
        admin_corpus_noise_chunks(
            project_id="xproj", execute=True, plan_hash="x",
            auth={"user_id": "u", "role": role},
        )
    assert len(_texts_of(did)) == 1
