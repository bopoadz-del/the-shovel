"""Kind of standing: decided at indexing from doc_type + doc_role by a table,
stored as metadata["standing"]; existing rows get it by a hash-gated backfill
that writes that one field and touches no chunk.

Every test builds its own table, documents and project. Every kind, type,
role and folder below is invented for the test or comes from the shipped
table's vocabulary; no real project, document or value is named.
"""
from __future__ import annotations

import json

import pytest

from app.core import doc_standing as ds
from app.core import doc_standing_backfill as bf
from tests.test_ingest_reconcile import _reload_projects

TABLE = [
    {"kind": "contract_agreement", "doc_types": ["contract"], "doc_roles": ["contract"]},
    {"kind": "specification", "doc_types": ["specification"]},
    {"kind": "contractor_submittal", "doc_types": ["method_statement", "itp"]},
    {"kind": "contractor_submittal", "doc_types": ["drawing"], "doc_roles": ["contractor_submittal"]},
    {"kind": "contract_drawing", "doc_types": ["drawing"]},
    {"kind": "priced_boq", "doc_types": ["boq"]},
]
FOLDERS = {"Signed Pack": "contract", "Draft Pack": "draft", "Submittals": "contractor_submittal"}


@pytest.fixture(autouse=True)
def _tables(monkeypatch):
    monkeypatch.setenv("DOC_STANDING", json.dumps(TABLE))
    monkeypatch.setenv("DOC_ROLE_FOLDERS", json.dumps(FOLDERS))


# ── the rule ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "doc_type, doc_role, kind",
    [
        ("specification", "other", "specification"),
        ("Specification", "tender", "specification"),  # case-folded, any role
        ("drawing", "design", "contract_drawing"),
        ("boq", "other", "priced_boq"),
        ("method_statement", "other", "contractor_submittal"),
        ("itp", "reporting", "contractor_submittal"),
        # a contract is the signed agreement only in a signed-contract folder
        ("contract", "contract", "contract_agreement"),
        ("contract", "draft", "other"),
        ("contract", "other", "other"),
        # types the table does not name
        ("photo", "other", "other"),
        ("document", "other", "other"),
        (None, None, "other"),
    ],
)
def test_standing_for_reads_the_table(doc_type, doc_role, kind):
    assert ds.standing_for(doc_type, doc_role) == kind


@pytest.mark.parametrize(
    "doc_type, doc_role, kind",
    [
        # a shop drawing / a drawing in a contractor-submittal folder
        ("drawing", "contractor_submittal", "contractor_submittal"),
        # an ordinary contract drawing (any other role)
        ("drawing", "design", "contract_drawing"),
    ],
)
def test_a_drawing_role_decides_before_the_drawing_type(monkeypatch, doc_type, doc_role, kind):
    # The SHIPPED table puts the role-scoped drawing entry before the type-only
    # drawing entry, so the folder role decides first where one is known.
    monkeypatch.delenv("DOC_STANDING", raising=False)
    monkeypatch.delenv("DOC_STANDING_FILE", raising=False)
    assert ds.standing_for(doc_type, doc_role) == kind


@pytest.mark.parametrize(
    "table, kind",
    [
        # the first matching entry decides: a table that lists the type twice
        ([{"kind": "priced_boq", "doc_types": ["boq"]},
          {"kind": "specification", "doc_types": ["boq"]}], "priced_boq"),
        ([{"kind": "specification", "doc_types": ["boq"]},
          {"kind": "priced_boq", "doc_types": ["boq"]}], "specification"),
    ],
)
def test_first_matching_entry_decides(monkeypatch, table, kind):
    monkeypatch.setenv("DOC_STANDING", json.dumps(table))
    assert ds.standing_for("boq", "other") == kind


@pytest.mark.parametrize(
    "bad_entry",
    [
        {"kind": "governs_everything", "doc_types": ["boq"]},  # unknown kind
        {"kind": "other", "doc_types": ["boq"]},  # 'other' is the default, not an entry
        {"kind": "priced_boq", "doc_types": []},  # names no type
        "priced_boq",  # not a mapping
    ],
)
def test_bad_entries_are_dropped_not_obeyed(monkeypatch, bad_entry):
    monkeypatch.setenv("DOC_STANDING", json.dumps([bad_entry]))
    assert ds.standing_table() == []
    assert ds.standing_for("boq", "other") == "other"


@pytest.mark.parametrize("raw", ["{not json", json.dumps({"kind": "priced_boq"})])
def test_unreadable_inline_table_leaves_every_document_other(monkeypatch, raw):
    monkeypatch.setenv("DOC_STANDING", raw)
    assert ds.standing_for("boq", "other") == "other"


@pytest.mark.parametrize("doc_type, kind", [("specification", "specification"),
                                            ("boq", "priced_boq")])
def test_file_table_is_read_when_no_inline(monkeypatch, tmp_path, doc_type, kind):
    f = tmp_path / "standing.yaml"
    f.write_text(f"standing:\n  - kind: {kind}\n    doc_types: [{doc_type}]\n", encoding="utf-8")
    monkeypatch.delenv("DOC_STANDING", raising=False)
    monkeypatch.setenv("DOC_STANDING_FILE", str(f))
    assert ds.standing_for(doc_type, "other") == kind
    assert ds.standing_for("photo", "other") == "other"


@pytest.mark.parametrize("doc_type, kind", [("drawing", "contract_drawing"),
                                            ("method_statement", "contractor_submittal")])
def test_shipped_table_maps_known_types(monkeypatch, doc_type, kind):
    monkeypatch.delenv("DOC_STANDING", raising=False)
    monkeypatch.delenv("DOC_STANDING_FILE", raising=False)
    assert ds.standing_for(doc_type, "other") == kind


def test_shipped_table_names_only_known_kinds_and_known_types():
    """Every entry of the shipped table survives the loader, and every type it
    names is a type the type table can give a document."""
    import yaml

    import os

    cfg = os.path.dirname(ds._default_table_file())
    with open(ds._default_table_file(), encoding="utf-8") as f:
        raw = yaml.safe_load(f)["standing"]
    with open(os.path.join(cfg, "document_types.yaml"), encoding="utf-8") as f:
        known_types = {t["name"] for t in yaml.safe_load(f)["document_types"]}
    for entry in raw:
        assert entry["kind"] in ds.STANDING_KINDS and entry["kind"] != ds.OTHER
        assert set(entry["doc_types"]) <= known_types, entry


@pytest.mark.parametrize(
    "kind, words",
    [("contract_drawing", "contract drawing"),
     ("contractor_submittal", "contractor's submittal"),
     ("no_such_kind", "other document"),
     (None, "other document")],
)
def test_label_is_plain_words_and_never_a_guess(kind, words):
    assert ds.label_for(kind) == words


@pytest.mark.parametrize(
    "metadata, expected",
    [({"drive_path": "a/b.pdf"}, {"drive_path": "a/b.pdf", "standing": "priced_boq"}),
     (None, {"standing": "priced_boq"})],
)
def test_with_standing_copies_and_never_mutates(metadata, expected):
    before = dict(metadata) if metadata else metadata
    assert ds.with_standing(metadata, "boq", "other") == expected
    assert metadata == before


# ── at indexing ─────────────────────────────────────────────────────────────

def _project(monkeypatch, tmp_path, pid):
    pm = _reload_projects(monkeypatch, tmp_path)
    pm.create_project(pid, user_id="system", project_id=pid)
    return pm


def _stored(did):
    from app.core.db import SessionLocal
    from app.core.models import Document

    with SessionLocal() as s:
        d = s.get(Document, did)
        return (d.metadata_ or {}).get("standing"), d.metadata_


@pytest.mark.parametrize(
    "name, path, kind",
    [
        ("Main Contract Agreement.pdf", "Root/Signed Pack/Main Contract Agreement.pdf",
         "contract_agreement"),
        ("Main Contract Agreement.pdf", "Root/Draft Pack/Main Contract Agreement.pdf", "other"),
        ("Pump Room Specification.pdf", "Root/Misc/Pump Room Specification.pdf", "specification"),
        ("holiday photo.jpg", None, "other"),
    ],
)
def test_add_document_stores_the_kind(monkeypatch, tmp_path, name, path, kind):
    pm = _project(monkeypatch, tmp_path, "iproj")
    meta = {"drive_path": path} if path else None
    did = pm.add_document("iproj", name, size=1, metadata=meta)["id"]
    stored, full = _stored(did)
    assert stored == kind
    if path:
        assert full["drive_path"] == path  # the caller's metadata is kept


# ── the backfill ────────────────────────────────────────────────────────────

def _legacy_doc(pm, pid, name, path, doc_type=None):
    """A row as the pre-standing writer left it: no standing in metadata."""
    from sqlalchemy import update

    from app.core.db import SessionLocal
    from app.core.models import Document

    did = pm.add_document(pid, name, size=1, metadata={"drive_path": path})["id"]
    values = {"metadata_": {"drive_path": path}}
    if doc_type:
        values["doc_type"] = doc_type
    with SessionLocal() as s:
        s.execute(update(Document).where(Document.id == did).values(**values))
        s.commit()
    return did


def _chunk_count():
    from sqlalchemy import func, inspect, select

    from app.core.db import SessionLocal
    from app.core.models import RagChunk

    with SessionLocal() as s:
        if not inspect(s.get_bind()).has_table(RagChunk.__tablename__):
            return 0
        return s.execute(select(func.count()).select_from(RagChunk)).scalar()


@pytest.mark.parametrize(
    "docs, after, left_other",
    [
        ([("Pump Room Specification.pdf", "Root/Misc/Pump Room Specification.pdf"),
          ("Ground Floor Drawing.pdf", "Root/Misc/Ground Floor Drawing.pdf"),
          ("notes.txt", "Root/Misc/notes.txt")],
         {"contract_drawing": 1, "other": 1, "specification": 1}, 1),
        ([("Main Contract Agreement.pdf", "Root/Signed Pack/Main Contract Agreement.pdf"),
          ("Main Contract Agreement v2.pdf", "Root/Draft Pack/Main Contract Agreement v2.pdf")],
         {"contract_agreement": 1, "other": 1}, 1),
    ],
)
def test_plan_counts_per_kind_writes_nothing_execute_needs_live_hash(
    monkeypatch, tmp_path, docs, after, left_other,
):
    pm = _project(monkeypatch, tmp_path, "bproj")
    ids = [_legacy_doc(pm, "bproj", n, p) for n, p in docs]
    chunks_before = _chunk_count()

    plan = bf.doc_standing_backfill(project_id="bproj")["plan"]
    assert plan["field"] == "metadata.standing"
    assert plan["documents_scanned"] == len(docs)
    assert plan["after"] == after
    assert plan["left_other"] == left_other
    assert plan["total"] == len(docs)  # every legacy row lacks the field
    assert all(_stored(d)[0] is None for d in ids)  # the plan writes nothing

    wrong = bf.doc_standing_backfill(project_id="bproj", execute=True, plan_hash="0" * 64)
    assert wrong["executed"] is False and wrong["refused"]
    assert all(_stored(d)[0] is None for d in ids)

    done = bf.doc_standing_backfill(project_id="bproj", execute=True, plan_hash=plan["plan_hash"])
    assert done["executed"] is True and done["updated"] == len(docs) and done["skipped"] == []
    for d, (_, path) in zip(ids, docs):
        stored, full = _stored(d)
        assert stored in after
        assert full == {"drive_path": path, "standing": stored}  # one field added, nothing else
    assert _chunk_count() == chunks_before  # no chunk touched
    assert bf.doc_standing_backfill(project_id="bproj")["plan"]["total"] == 0  # idempotent


@pytest.mark.parametrize("retyped, new_kind", [("boq", "priced_boq"), ("itp", "contractor_submittal")])
def test_a_stale_kind_is_planned_when_the_type_moved(monkeypatch, tmp_path, retyped, new_kind):
    """The kind is derived, never user-set: when the type is filled later (the
    doc-typing backfill), the stored kind is re-planned from the new type."""
    pm = _project(monkeypatch, tmp_path, "sproj")
    did = pm.add_document("sproj", "untitled scan.pdf", size=1, metadata={"drive_path": "R/x.pdf"})["id"]
    assert _stored(did)[0] == "other"
    from sqlalchemy import update

    from app.core.db import SessionLocal
    from app.core.models import Document
    with SessionLocal() as s:
        s.execute(update(Document).where(Document.id == did).values(doc_type=retyped))
        s.commit()
    plan = bf.doc_standing_backfill(project_id="sproj")["plan"]
    assert [(c["doc_id"], c["old"], c["new"]) for c in plan["changes"]] == [(did, "other", new_kind)]


@pytest.mark.parametrize("moved", [False, True])
def test_a_row_moved_after_the_plan_is_skipped_not_overwritten(monkeypatch, tmp_path, moved):
    from app.core.db import SessionLocal
    from app.core.models import Document

    pm = _project(monkeypatch, tmp_path, "cproj")
    did = _legacy_doc(pm, "cproj", "Pump Room Specification.pdf", "R/Pump Room Specification.pdf")
    real_plan = bf.plan_doc_standing

    def plan_then_someone_writes(session, project_id):
        plan = real_plan(session, project_id)
        if moved:
            with SessionLocal() as s:
                d = s.get(Document, did)
                d.metadata_ = {**(d.metadata_ or {}), "standing": "priced_boq"}
                s.commit()
        return plan

    plan_hash = bf.doc_standing_backfill(project_id="cproj")["plan"]["plan_hash"]
    monkeypatch.setattr(bf, "plan_doc_standing", plan_then_someone_writes)
    done = bf.doc_standing_backfill(project_id="cproj", execute=True, plan_hash=plan_hash)
    assert done["executed"] is True
    if moved:
        assert [c["new"] for c in done["skipped"]] == ["specification"]
        assert done["updated"] == 0
        assert _stored(did)[0] == "priced_boq"
    else:
        assert done["skipped"] == [] and done["updated"] == 1
        assert _stored(did)[0] == "specification"


@pytest.mark.parametrize("other_pid", ["oproj2", "oproj3"])
def test_backfill_never_reaches_another_project(monkeypatch, tmp_path, other_pid):
    pm = _project(monkeypatch, tmp_path, "oproj")
    pm.create_project(other_pid, user_id="system", project_id=other_pid)
    mine = _legacy_doc(pm, "oproj", "Pump Room Specification.pdf", "R/Pump Room Specification.pdf")
    theirs = _legacy_doc(pm, other_pid, "Ground Floor Drawing.pdf", "R/Ground Floor Drawing.pdf")
    plan = bf.doc_standing_backfill(project_id="oproj")["plan"]
    assert [c["doc_id"] for c in plan["changes"]] == [mine]
    bf.doc_standing_backfill(project_id="oproj", execute=True, plan_hash=plan["plan_hash"])
    assert _stored(mine)[0] == "specification"
    assert _stored(theirs)[0] is None


@pytest.mark.parametrize("project_id, raises", [("", True), ("eproj", False)])
def test_project_id_is_required(monkeypatch, tmp_path, project_id, raises):
    _project(monkeypatch, tmp_path, "eproj")
    if raises:
        with pytest.raises(ValueError):
            bf.doc_standing_backfill(project_id=project_id)
    else:
        assert bf.doc_standing_backfill(project_id=project_id)["plan"]["total"] == 0


ADMIN = {"user_id": "admin", "role": "admin"}


@pytest.mark.parametrize("execute", [False, True])
def test_admin_route_plans_then_writes_on_hash(monkeypatch, tmp_path, execute):
    from app.core import audit
    from app.routers.admin import admin_corpus_doc_standing

    pm = _project(monkeypatch, tmp_path, "rproj")
    did = _legacy_doc(pm, "rproj", "Pump Room Specification.pdf", "R/Pump Room Specification.pdf")
    events: list = []
    monkeypatch.setattr(audit, "record", lambda event, **kw: events.append((event, kw)))
    planned = admin_corpus_doc_standing(project_id="rproj", auth=ADMIN)["plan"]
    out = admin_corpus_doc_standing(
        project_id="rproj", execute=execute, plan_hash=planned["plan_hash"], auth=ADMIN,
    )
    assert out["executed"] is execute
    assert _stored(did)[0] == ("specification" if execute else None)
    assert [e for e, _ in events] == (["corpus.doc_standing_backfilled"] if execute else [])
    if execute:
        assert events[0][1]["updated"] == 1


@pytest.mark.parametrize("role", ["user", "viewer"])
def test_admin_route_refuses_non_admin(monkeypatch, tmp_path, role):
    from fastapi import HTTPException

    from app.routers.admin import admin_corpus_doc_standing

    _project(monkeypatch, tmp_path, "nproj")
    with pytest.raises(HTTPException):
        admin_corpus_doc_standing(project_id="nproj", auth={"user_id": "u", "role": role})


@pytest.mark.parametrize(
    "doc_types, maps",
    [
        ("boq", True),  # a single type given as a bare string is one type
        (["boq"], True),
        (7, False),  # neither a string nor a list names no type: entry dropped
        ({"boq": 1}, False),
    ],
)
def test_entry_types_as_string_or_list(monkeypatch, doc_types, maps):
    monkeypatch.setenv("DOC_STANDING", json.dumps([{"kind": "priced_boq", "doc_types": doc_types}]))
    assert ds.standing_for("boq", "other") == ("priced_boq" if maps else "other")


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({"standing": "priced_boq", "drive_path": "a/b.pdf"}, "priced_boq"),
        ({"standing": "specification"}, "specification"),
        ({"drive_path": "a/b.pdf"}, None),  # never set
        ({"standing": ""}, None),  # empty is not a kind
        ({"standing": 3}, None),  # not a string is not a kind
        (None, None),
        ("standing", None),  # not a mapping
    ],
)
def test_stored_standing_reads_only_a_set_kind(metadata, expected):
    assert ds.stored_standing(metadata) == expected
