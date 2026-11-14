"""Include back: the reverse of an owner exclusion.

An excluded document is brought back by clearing ``metadata.excluded`` and
its tombstone; the row becomes indexable, listed and retrievable again once
re-indexed. Only a row that carries an owner exclusion is ever touched.

Synthetic project and documents (the ``world`` fixture of the exclusion
tests). Every decision is checked on two inputs.
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core import projects as projects_module

from tests.test_owner_excluded_document import (  # noqa: F401 - world is a fixture
    PAIRS,
    SIZES,
    TOTAL,
    _counts,
    _exclude,
    _reindex_deps,
    world,
)

AT = "2026-11-14T00:10:00Z"


def _include(projects, doc_id, reason="set needs it"):
    return projects.include_document(doc_id, by="owner", reason=reason, at=AT)


def _ghost(projects, doc_id):
    """Tombstone a row for a reason that is NOT an owner exclusion."""
    from app.core.models import Document

    projects.stamp_document_index(doc_id, chunk_count=0, ingest_status="TOMBSTONED",
                                  ingest_status_reason="ghost_not_stored")
    with projects.SessionLocal() as session:
        session.get(Document, doc_id).retrieval_visible = False
        session.commit()


@pytest.mark.parametrize("target", ["drop.txt", "keep_b.txt"])
def test_including_back_clears_the_exclusion_and_keeps_its_record(world, target):
    projects, store, _emb, pid, docs = world
    _exclude(projects, docs[target])
    stored = projects.get_document(docs[target])["metadata"]["excluded"]

    plan = projects.plan_document_inclusion(docs[target])
    assert plan["eligible"] is True and plan["not_eligible"] is None
    assert plan["excluded"] == stored and plan["status_after"] == "UNVERIFIED"

    out = _include(projects, docs[target])
    row = out["document"]
    assert out["changed"] is True
    assert "excluded" not in row["metadata"]
    assert row["metadata"]["included_back"] == {
        "by": "owner", "at": AT, "reason": "set needs it", "excluded": stored,
    }
    assert row["ingest_status"] == "UNVERIFIED" and row["ingest_status_reason"] is None
    assert row["retrieval_visible"] is True and row["chunk_count"] == 0
    from app.core.ingest_status import is_excluded

    assert is_excluded(row) is False
    assert docs[target] in {d["id"] for d in projects.list_documents(pid)}
    # The others are untouched.
    after = _counts(store, pid)
    for name in SIZES:
        if name != target:
            assert after[docs[name]] == SIZES[name]
            assert "included_back" not in (projects.get_document(docs[name]).get("metadata") or {})


@pytest.mark.parametrize("case", ["never_excluded", "ghost_tombstone"])
def test_a_row_without_an_owner_exclusion_is_never_touched(world, case):
    projects, store, _emb, pid, docs = world
    doc_id = docs["keep_a.txt"]
    if case == "ghost_tombstone":
        _ghost(projects, doc_id)
    before = projects.get_document(doc_id)

    plan = projects.plan_document_inclusion(doc_id)
    assert plan["eligible"] is False and "no owner exclusion" in plan["not_eligible"]
    assert plan["status_after"] == before["ingest_status"]

    out = _include(projects, doc_id)
    row = out["document"]
    assert out["changed"] is False
    assert row["ingest_status"] == before["ingest_status"]
    assert row["ingest_status_reason"] == before["ingest_status_reason"]
    assert row["retrieval_visible"] == before["retrieval_visible"]
    assert "included_back" not in (row.get("metadata") or {})


@pytest.mark.parametrize("target,other", PAIRS)
def test_an_included_back_document_is_indexed_again(world, target, other):
    projects, store, _emb, pid, docs = world
    from app.core import doc_index

    _exclude(projects, docs[target])
    _exclude(projects, docs[other])
    _include(projects, docs[target])

    back = doc_index.index_document(pid, docs[target])
    still_out = doc_index.index_document(pid, docs[other])
    assert back.get("excluded") is not True
    assert _counts(store, pid).get(docs[target], 0) >= 1
    assert still_out["excluded"] is True
    assert _counts(store, pid).get(docs[other], 0) == 0


@pytest.mark.parametrize("target,other", PAIRS)
def test_doc_reindex_once_accepts_an_included_back_document(world, target, other):
    projects, _store, _emb, pid, docs = world
    from scripts.doc_reindex_once import reindex_one

    _exclude(projects, docs[target])
    _exclude(projects, docs[other])
    _include(projects, docs[target])
    calls: list = []
    refused = reindex_one(pid, docs[other], force_ocr=False, heading_re=None,
                          deps=_reindex_deps(projects, calls))
    assert "excluded" in refused["error"] and calls == []

    reindex_one(pid, docs[target], force_ocr=False, heading_re=None,
                deps=_reindex_deps(projects, calls))
    assert ("download", "drive-x") in calls and ("index", docs[target]) in calls


@pytest.mark.parametrize("target,other", PAIRS)
def test_inclusion_plan_hash_moves_with_the_exclusion_state(world, target, other):
    projects, _store, _emb, _pid, docs = world
    never = projects.plan_document_inclusion(docs[target])["plan_hash"]
    _exclude(projects, docs[target])
    excluded = projects.plan_document_inclusion(docs[target])["plan_hash"]
    again = projects.plan_document_inclusion(docs[target])["plan_hash"]
    _include(projects, docs[target])
    included = projects.plan_document_inclusion(docs[target])["plan_hash"]
    other_hash = projects.plan_document_inclusion(docs[other])["plan_hash"]
    assert excluded == again
    assert len({never, excluded, included, other_hash}) == 4


@pytest.mark.parametrize("missing", ["", None, "no-such-document"])
def test_plan_and_include_answer_none_for_a_missing_id(world, missing):
    _projects, store, _emb, pid, _docs = world
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert projects_module.plan_document_inclusion(missing) is None
        assert projects_module.include_document(missing, by="owner", reason="r", at="t") is None
    assert sum(_counts(store, pid).values()) == TOTAL


def _route(**kw):
    from app.routers.admin import admin_include_document

    base = {"auth": {"role": "admin", "user_id": "u1"}, "plan_hash": None, "reason": ""}
    base.update(kw)
    return admin_include_document(**base)


@pytest.mark.parametrize("target", ["drop.txt", "keep_a.txt"])
def test_route_plans_without_writing_and_executes_only_with_the_live_hash(world, target):
    projects, _store, _emb, pid, docs = world
    _exclude(projects, docs[target])
    plan = _route(document_id=docs[target], project_id=pid, execute=False)
    assert plan["executed"] is False and plan["plan"]["eligible"] is True

    wrong = _route(document_id=docs[target], project_id=pid, execute=True,
                   plan_hash="0" * 64, reason="set needs it")
    no_reason = _route(document_id=docs[target], project_id=pid, execute=True,
                       plan_hash=plan["plan"]["plan_hash"])
    assert wrong["executed"] is False and "plan_hash" in wrong["refused"]
    assert no_reason["executed"] is False and "reason" in no_reason["refused"]
    assert "excluded" in projects.get_document(docs[target])["metadata"]

    done = _route(document_id=docs[target], project_id=pid, execute=True,
                  plan_hash=plan["plan"]["plan_hash"], reason="set needs it")
    assert done["executed"] is True and done["excluded_after"] is None
    assert done["included_back"]["reason"] == "set needs it"
    assert done["ingest_status"] == "UNVERIFIED" and done["retrieval_visible"] is True


@pytest.mark.parametrize("case", ["never_excluded", "ghost_tombstone"])
def test_route_refuses_a_row_without_an_owner_exclusion(world, case):
    projects, _store, _emb, pid, docs = world
    doc_id = docs["keep_b.txt"]
    if case == "ghost_tombstone":
        _ghost(projects, doc_id)
    before = projects.get_document(doc_id)
    plan = _route(document_id=doc_id, project_id=pid, execute=False)["plan"]
    out = _route(document_id=doc_id, project_id=pid, execute=True,
                 plan_hash=plan["plan_hash"], reason="set needs it")
    assert out["executed"] is False and "no owner exclusion" in out["refused"]
    row = projects.get_document(doc_id)
    assert row["ingest_status"] == before["ingest_status"]
    assert row["retrieval_visible"] == before["retrieval_visible"]


@pytest.mark.parametrize("case,status", [
    ("unknown_document", 404),
    ("other_project", 400),
    ("not_admin", 403),
])
def test_route_refuses_unknown_document_other_project_and_non_admin(world, case, status):
    projects, _store, _emb, pid, docs = world
    _exclude(projects, docs["drop.txt"])
    kw = {"document_id": docs["drop.txt"], "project_id": pid, "execute": True,
          "plan_hash": projects.plan_document_inclusion(docs["drop.txt"])["plan_hash"],
          "reason": "set needs it"}
    if case == "unknown_document":
        kw["document_id"] = "no-such-document"
    elif case == "other_project":
        kw["project_id"] = "someone-else"
    else:
        kw["auth"] = {"role": "user", "user_id": "u1"}
    with pytest.raises(HTTPException) as refused:
        _route(**kw)
    assert refused.value.status_code == status
    assert "excluded" in projects.get_document(docs["drop.txt"])["metadata"]  # nothing written
