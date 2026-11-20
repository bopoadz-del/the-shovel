"""RQ-3 backfill: doc_type / doc_role filled on existing rows, plan first,
written only on the live plan hash, compare-and-set per row.

Every test seeds its own project, documents and folder table.
"""
from __future__ import annotations

import json

import pytest

from app.core import doc_typing_backfill as bf
from tests.test_ingest_reconcile import _reload_projects

FOLDERS = {"Tender Pack": "tender", "Signed": "contract"}


@pytest.fixture(autouse=True)
def _folder_table(monkeypatch):
    monkeypatch.setenv("DOC_ROLE_FOLDERS", json.dumps(FOLDERS))


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"drive_path": "a/b.pdf"}, {"drive_path": "a/b.pdf"}),
        (json.dumps({"drive_path": "a/b.pdf"}), {"drive_path": "a/b.pdf"}),
        ("{not json", None),
        (json.dumps(["a", "b"]), None),
        (None, None),
        ("", None),
    ],
)
def test_metadata_reads_a_mapping_or_its_json(raw, expected):
    assert bf._metadata(raw) == expected


def _row(did, name, doc_type="document", doc_role="other", path=None):
    return {
        "id": did, "original_name": name, "doc_type": doc_type,
        "doc_role": doc_role, "metadata": {"drive_path": path} if path else None,
    }


@pytest.mark.parametrize(
    "row, changes",
    [
        # a default row in a tender folder: both columns are decided
        (_row("d1", "Tender BOQ.xlsx", path="Root/Tender Pack/Tender BOQ.xlsx"),
         [("doc_type", "document", "boq"), ("doc_role", "other", "tender")]),
        # the same name and folder, but someone already set both: left alone
        (_row("d2", "Tender BOQ.xlsx", doc_type="specification",
              doc_role="baseline_schedule", path="Root/Tender Pack/Tender BOQ.xlsx"),
         []),
        # a name and folder neither table knows: nothing to fill
        (_row("d3", "notes.txt", path="Root/Misc/notes.txt"), []),
        # a NULL column reads as its default and is filled
        (_row("d4", "Main Contract.pdf", doc_type=None, doc_role=None,
              path="Root/Signed/Main Contract.pdf"),
         [("doc_type", "document", "contract"), ("doc_role", "other", "contract")]),
    ],
)
def test_decide_changes_fills_only_defaults(row, changes):
    got = [(c["column"], c["old"], c["new"]) for c in bf.decide_changes([row])]
    assert got == changes
    assert all(c["doc_id"] == row["id"] for c in bf.decide_changes([row]))


def _project(monkeypatch, tmp_path, pid):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project(pid, user_id="system", project_id=pid)
    return pm


def _legacy_doc(pm, pid, name, path):
    """A row as the pre-RQ-3 writer left it: both columns at their default."""
    from sqlalchemy import update

    from app.core.db import SessionLocal
    from app.core.models import Document

    did = pm.add_document(pid, name, size=1, metadata={"drive_path": path})["id"]
    with SessionLocal() as s:
        s.execute(update(Document).where(Document.id == did)
                  .values(doc_type="document", doc_role="other"))
        s.commit()
    return did


def _cols(did):
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as s:
        d = s.get(Document, did)
        return d.doc_type, d.doc_role


@pytest.mark.parametrize(
    "name, path, counts, after",
    [
        ("Tender BOQ.xlsx", "Root/Tender Pack/Tender BOQ.xlsx",
         {"doc_type": {"boq": 1}, "doc_role": {"tender": 1}}, ("boq", "tender")),
        ("notes.txt", "Root/Misc/notes.txt",
         {"doc_type": {}, "doc_role": {}}, ("document", "other")),
    ],
)
def test_plan_writes_nothing_and_execute_needs_the_live_hash(
    monkeypatch, tmp_path, name, path, counts, after,
):
    pm = _project(monkeypatch, tmp_path, "tproj")
    did = _legacy_doc(pm, "tproj", name, path)

    plan = bf.doc_typing_backfill(project_id="tproj")["plan"]
    assert plan["counts"] == counts
    assert plan["documents_scanned"] == 1
    assert plan["documents"] == ([did] if after != ("document", "other") else [])
    assert _cols(did) == ("document", "other")  # the plan writes nothing

    wrong = bf.doc_typing_backfill(project_id="tproj", execute=True, plan_hash="0" * 64)
    assert wrong["executed"] is False and wrong["refused"]
    assert _cols(did) == ("document", "other")

    done = bf.doc_typing_backfill(project_id="tproj", execute=True, plan_hash=plan["plan_hash"])
    assert done["executed"] is True
    assert done["updated"] == plan["total"]
    assert done["skipped"] == []
    assert _cols(did) == after
    assert bf.doc_typing_backfill(project_id="tproj")["plan"]["total"] == 0


@pytest.mark.parametrize("moved", [False, True])
def test_a_row_set_after_the_plan_is_skipped_not_overwritten(monkeypatch, tmp_path, moved):
    from sqlalchemy import update

    from app.core.db import SessionLocal
    from app.core.models import Document

    pm = _project(monkeypatch, tmp_path, "cproj")
    did = _legacy_doc(pm, "cproj", "Tender BOQ.xlsx", "Root/Tender Pack/Tender BOQ.xlsx")
    real_plan = bf.plan_doc_typing

    def plan_then_someone_writes(session, project_id):
        plan = real_plan(session, project_id)
        if moved:  # a user confirms the type between the plan and the write
            with SessionLocal() as s:
                s.execute(update(Document).where(Document.id == did)
                          .values(doc_type="specification"))
                s.commit()
        return plan

    plan_hash = bf.doc_typing_backfill(project_id="cproj")["plan"]["plan_hash"]
    monkeypatch.setattr(bf, "plan_doc_typing", plan_then_someone_writes)
    done = bf.doc_typing_backfill(project_id="cproj", execute=True, plan_hash=plan_hash)
    assert done["executed"] is True
    if moved:
        assert [(c["column"], c["new"]) for c in done["skipped"]] == [("doc_type", "boq")]
        assert done["updated"] == 1
        assert _cols(did) == ("specification", "tender")
    else:
        assert done["skipped"] == []
        assert done["updated"] == 2
        assert _cols(did) == ("boq", "tender")


@pytest.mark.parametrize("project_id, raises", [("", True), ("eproj", False)])
def test_project_id_is_required(monkeypatch, tmp_path, project_id, raises):
    _project(monkeypatch, tmp_path, "eproj")
    if raises:
        with pytest.raises(ValueError):
            bf.doc_typing_backfill(project_id=project_id)
    else:
        assert bf.doc_typing_backfill(project_id=project_id)["plan"]["total"] == 0


ADMIN = {"user_id": "admin", "role": "admin"}


@pytest.mark.parametrize("execute", [False, True])
def test_admin_route_plans_then_writes_on_hash(monkeypatch, tmp_path, execute):
    from app.core import audit
    from app.routers.admin import admin_corpus_doc_typing

    pm = _project(monkeypatch, tmp_path, "rproj")
    did = _legacy_doc(pm, "rproj", "Tender BOQ.xlsx", "Root/Tender Pack/Tender BOQ.xlsx")
    events: list = []
    monkeypatch.setattr(audit, "record", lambda event, **kw: events.append((event, kw)))
    planned = admin_corpus_doc_typing(project_id="rproj", auth=ADMIN)["plan"]
    out = admin_corpus_doc_typing(
        project_id="rproj", execute=execute, plan_hash=planned["plan_hash"], auth=ADMIN,
    )
    assert out["executed"] is execute
    assert _cols(did) == (("boq", "tender") if execute else ("document", "other"))
    # only a write is audited, with what it updated
    assert [e for e, _ in events] == (["corpus.doc_typing_backfilled"] if execute else [])
    if execute:
        assert events[0][1]["updated"] == 2


@pytest.mark.parametrize("role", ["user", "viewer"])
def test_admin_route_rejects_non_admin(monkeypatch, tmp_path, role):
    from fastapi import HTTPException

    from app.routers.admin import admin_corpus_doc_typing

    pm = _project(monkeypatch, tmp_path, "xproj")
    did = _legacy_doc(pm, "xproj", "Tender BOQ.xlsx", "Root/Tender Pack/Tender BOQ.xlsx")
    with pytest.raises(HTTPException):
        admin_corpus_doc_typing(
            project_id="xproj", execute=True, plan_hash="x",
            auth={"user_id": "u", "role": role},
        )
    assert _cols(did) == ("document", "other")
