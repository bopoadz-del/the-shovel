"""ML-3a: the ML loop block is admin-only at every door that names a block.

``ml_loop`` holds every project's turns, feedback and corrections. The
project-access check only sees the ``project_id`` values a request carries,
so ``export_tables`` with none returned every project's rows to any
signed-in user, and ``import_tables`` / ``record_*`` let them write rows.
The block is now in PRIVILEGED_BLOCKS; the host adapters reach it in
process and are unaffected.

Each test builds its own users, project and rows; each decision is shown
with two inputs.
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.blocks.ml_loop as ml_loop_mod
import app.core.privileges  # noqa: F401  dotted token so CI-M selects this file
from app.blocks.ml_loop import MLLoopBlock
from app.core import jwt_auth
from app.core import users as users_store
from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import MLLoopRepository, create_tables
from app.core.privileges import (
    PRIVILEGED_BLOCKS,
    caller_may_use_block,
    raise_if_privileged_steps,
    set_caller_role,
)


def _headers(role: str) -> dict:
    uid = f"ml3a-{role}-{uuid.uuid4().hex[:8]}"
    users_store.ensure_user_exists(uid, role=role, email=f"{uid}@local")
    return {"Authorization": f"Bearer {jwt_auth.create_token(uid)}"}


@pytest.fixture()
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c
    set_caller_role(None)


@pytest.fixture()
def store(tmp_path):
    """A fresh SQLite store installed as the block's process-wide settings,
    installed AFTER app startup so boot wiring cannot replace it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'ml.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine)
    MLLoopBlock.configure(MLLoopSettings(build_sha="sha-ml3a", session_factory=factory))
    ml_loop_mod._shared_block = None
    yield MLLoopRepository(factory)
    MLLoopBlock.configure(None)
    ml_loop_mod._shared_block = None
    engine.dispose()


def _turn(project_id: str, n: int) -> dict:
    return {
        "project_id": project_id, "conversation_id": f"c-{n}", "message_id": f"m{n}",
        "question": f"question {n}", "answer": f"answer {n}", "verdict": "refused",
        "route": "rag", "cited_ids": [],
    }


def _own_project(c: TestClient, headers: dict) -> str:
    r = c.post("/v1/projects", json={"name": f"ml3a {uuid.uuid4().hex[:6]}"}, headers=headers)
    assert r.status_code in (200, 201), r.text
    return r.json()["id"]


# ── the rule ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("role, allowed", [("user", False), ("admin", True)])
def test_ml_loop_is_admin_only_by_rule(role, allowed):
    assert "ml_loop" in PRIVILEGED_BLOCKS
    assert caller_may_use_block("ml_loop", role) is allowed
    # An unknown caller is not an admin: fails closed.
    assert caller_may_use_block("ml_loop", None) is False


@pytest.mark.parametrize("role, refused", [("user", True), ("admin", False)])
def test_orchestrator_steps_naming_ml_loop_follow_the_rule(role, refused):
    from fastapi import HTTPException

    steps = [{"block": "ml_loop", "input": {"operation": "export_tables"}}]
    if refused:
        with pytest.raises(HTTPException) as exc:
            raise_if_privileged_steps(steps, role)
        assert exc.value.status_code == 403
    else:
        raise_if_privileged_steps(steps, role)


# ── /v1/execute: read ───────────────────────────────────────────────────

@pytest.mark.parametrize("with_project", [False, True])
def test_non_admin_cannot_export_tables(client, store, with_project):
    user = _headers("user")
    store_project = f"other-{uuid.uuid4().hex[:6]}"
    MLLoopBlock().dispatch({"operation": "record_turn", "payload": _turn(store_project, 1)})
    payload = {"operation": "export_tables"}
    if with_project:
        payload["project_id"] = _own_project(client, user)
    r = client.post("/v1/execute", headers=user, json={"block": "ml_loop", "input": payload})
    assert r.status_code == 403, (r.status_code, r.text)
    assert "question 1" not in r.text


@pytest.mark.parametrize("operation", ["export_tables", "status"])
def test_admin_reaches_the_ml_loop(client, store, operation):
    admin = _headers("admin")
    project = f"p-{uuid.uuid4().hex[:6]}"
    MLLoopBlock().dispatch({"operation": "record_turn", "payload": _turn(project, 2)})
    r = client.post("/v1/execute", headers=admin,
                    json={"block": "ml_loop", "input": {"operation": operation}})
    assert r.status_code == 200, (r.status_code, r.text)
    if operation == "export_tables":
        assert "question 2" in r.text


# ── /v1/execute and /v1/chain: write ────────────────────────────────────

@pytest.mark.parametrize("operation, payload_of", [
    ("import_tables", lambda pid: {"tables": {"ml_turns": [_turn(pid, 3)]}}),
    ("record_turn", lambda pid: _turn(pid, 4)),
    ("record_feedback", lambda pid: {"project_id": pid, "conversation_id": "c-5",
                                     "message_id": "m5", "kind": "correct"}),
])
def test_non_admin_cannot_write_through_execute(client, store, operation, payload_of):
    user = _headers("user")
    pid = _own_project(client, user)
    r = client.post("/v1/execute", headers=user, json={
        "block": "ml_loop", "input": {"operation": operation, **payload_of(pid)},
    })
    assert r.status_code == 403, (r.status_code, r.text)
    for table in ("ml_turns", "ml_feedback"):
        assert store.rows(table, project_id=pid) == []


@pytest.mark.parametrize("role, expected", [("user", 403), ("admin", 200)])
def test_chain_step_naming_ml_loop_follows_the_rule(client, store, role, expected):
    headers = _headers(role)
    r = client.post("/v1/chain", headers=headers, json={
        "steps": [{"block": "ml_loop", "input": {"operation": "status"}}],
    })
    assert r.status_code == expected, (r.status_code, r.text)


# ── in process: the host adapters are unaffected ────────────────────────

@pytest.mark.parametrize("role", ["user", None])
def test_in_process_record_turn_still_writes_for_any_caller(tmp_path, store, role):
    from app.core.rag.dataset_log import DatasetRecord, append_dataset_record

    set_caller_role(role)
    try:
        project = f"inproc-{uuid.uuid4().hex[:6]}"
        question = f"what is the {uuid.uuid4().hex[:4]} slab thickness?"
        record = DatasetRecord(
            question=question, answer="250 mm", verdict="project_document",
            evidence_plan=("plan",), retrieved_ids=("d-1",), citations=({"doc_id": "d-1"},),
        )
        append_dataset_record(project, record, base=tmp_path)
        rows = store.rows("ml_turns", project_id=project)
        assert [r["question"] for r in rows] == [question]
    finally:
        set_caller_role(None)
