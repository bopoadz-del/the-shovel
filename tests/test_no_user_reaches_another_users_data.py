"""Cross-user isolation: A must not reach B's data.

Agent E (Isolation), Phase 1. Two plain users (A, B) and one admin.
Each owns a project, a synthetic document, a conversation, a
memory fact, and a workflow. A hitting B's ids must get 403/404, B's
objects stay put, and denial bodies must not carry B's canary text.
Live fixtures, if a later pass seeds them, are ``FIXTURE-e-<date>-…``
only — never ``FIXTURE-isolation-A`` or another agent's prefix.

This file is the failing-test-first ledger for high-risk UNPROVEN
id-bearing routes from docs/security/isolation_authz_gaps.md. A green
cell here is not a "secure" claim — it is one proven denial.
"""
from __future__ import annotations

import io
import json
import uuid
from typing import Any, Callable

import jwt
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests._dev_key import DEV_KEY
from tests._fixture_values import OTHER_PASSWORD, PASSWORD, WEAK_PASSWORD, WRONG_PASSWORD


_RUN = uuid.uuid4().hex[:8]
CANARY_B = f"CANARY-B-ISOLATION-{_RUN}-SECRET-RATE-0.1"
CANARY_A = f"CANARY-A-ISOLATION-{_RUN}-PUBLIC-RATE-9.9"
DENY = (403, 404)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _register(client: TestClient, email: str, password: str = PASSWORD) -> dict:
    r = client.post("/v1/users/register", json={"email": email, "password": password})
    assert r.status_code in (201, 409), r.text
    r = client.post("/v1/users/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    body = r.json()
    return {"token": body["token"], "id": body["user"]["id"], "email": email}


def _promote_admin(uid: str) -> None:
    from app.core.db import SessionLocal
    from app.core.models import User

    with SessionLocal() as db:
        db.get(User, uid).role = "admin"
        db.commit()


def _h(actor: dict, extra: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {actor['token']}"}
    if extra:
        headers.update(extra)
    return headers


def _upload(client: TestClient, actor: dict, pid: str, name: str, text: str) -> str:
    r = client.post(
        f"/v1/projects/{pid}/documents",
        headers=_h(actor),
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/plain")},
    )
    assert r.status_code == 201, r.text
    return r.json()["document"]["id"]


def _seed_conversation(pid: str, canary: str) -> str:
    from app.core import agent_memory
    from app.core.conversation_wbs import stage_conversation_wbs

    cid = f"ws-{pid}-{int(uuid.uuid4().int % 10**12)}"
    agent_memory.get_or_create_conversation(cid, "project-assistant", project_id=pid)
    agent_memory.append_message(cid, "user", f"what is the rate {canary}")
    agent_memory.append_message(
        cid, "assistant", f"The documented rate is {canary} and must stay private."
    )
    stage_conversation_wbs(
        cid,
        {
            "brief": "site fence",
            "activities": [
                {"id": "A1", "name": f"Fence {canary}", "duration_days": 5}
            ],
        },
    )
    return cid


@pytest.fixture
def world(client):
    """A + B + admin, each with project/doc/conversation/memory/workflow."""
    a = _register(client, f"iso-a-{_RUN}@example.com")
    b = _register(client, f"iso-b-{_RUN}@example.com")
    admin = _register(client, f"iso-admin-{_RUN}@example.com")
    _promote_admin(admin["id"])

    a_pid = client.post(
        "/v1/projects", json={"name": f"A private {_RUN}"}, headers=_h(a)
    ).json()["id"]
    b_pid = client.post(
        "/v1/projects", json={"name": f"B private {_RUN}"}, headers=_h(b)
    ).json()["id"]

    a_doc = _upload(client, a, a_pid, "a-notes.txt", CANARY_A * 20)
    b_doc = _upload(client, b, b_pid, "b-secret.txt", CANARY_B * 20)

    a_mem = client.post(
        f"/v1/projects/{a_pid}/memory",
        headers=_h(a),
        json={"key": "site_rate", "value": CANARY_A},
    )
    assert a_mem.status_code == 201, a_mem.text
    b_mem = client.post(
        f"/v1/projects/{b_pid}/memory",
        headers=_h(b),
        json={"key": "site_rate", "value": CANARY_B},
    )
    assert b_mem.status_code == 201, b_mem.text

    a_cid = _seed_conversation(a_pid, CANARY_A)
    b_cid = _seed_conversation(b_pid, CANARY_B)

    steps = [{"block": "formula_executor", "params": {}, "label": "Compute"}]
    a_wf = client.post(
        "/v1/workflows",
        headers=_h(a),
        json={"name": "A workflow", "steps": steps, "project_id": a_pid},
    )
    assert a_wf.status_code == 201, a_wf.text
    b_wf = client.post(
        "/v1/workflows",
        headers=_h(b),
        json={"name": "B workflow", "steps": steps, "project_id": b_pid},
    )
    assert b_wf.status_code == 201, b_wf.text

    from app.core import hydration_store

    hydration_store.record_run(
        run_date="2026-09-19",
        scope="project",
        project_id=b_pid,
        summary_md=f"B hydration {CANARY_B}",
        facts={"canary": CANARY_B},
        provider="test",
    )

    return {
        "a": a,
        "b": b,
        "admin": admin,
        "a_pid": a_pid,
        "b_pid": b_pid,
        "a_doc": a_doc,
        "b_doc": b_doc,
        "a_cid": a_cid,
        "b_cid": b_cid,
        "a_wf": a_wf.json()["id"],
        "b_wf": b_wf.json()["id"],
    }


def _assert_denied(resp, *, label: str) -> None:
    assert resp.status_code in DENY, f"{label}: expected 403/404, got {resp.status_code} {resp.text[:300]}"
    body = resp.text
    assert CANARY_B not in body, f"{label}: denial leaked B canary"
    assert "b-secret.txt" not in body.lower()
    assert "B-SECRET-FOLDER" not in body


def _assert_b_unchanged(client: TestClient, world: dict) -> None:
    b, b_pid, b_doc, b_cid, b_wf = (
        world["b"], world["b_pid"], world["b_doc"], world["b_cid"], world["b_wf"],
    )
    assert client.get(f"/v1/projects/{b_pid}", headers=_h(b)).status_code == 200
    docs = client.get(f"/v1/projects/{b_pid}/documents", headers=_h(b)).json()["documents"]
    assert any(d["id"] == b_doc for d in docs)
    mem = client.get(f"/v1/projects/{b_pid}/memory", headers=_h(b)).json()["facts"]
    assert any(CANARY_B in str(f.get("value")) for f in mem)
    assert client.get(f"/v1/workflows/{b_wf}", headers=_h(b)).status_code == 200
    msgs = client.get(
        f"/v1/agents/conversations/{b_cid}/messages", headers=_h(b)
    )
    assert msgs.status_code == 200
    assert CANARY_B in msgs.text


# ── high-risk id-bearing surfaces (A against B) ─────────────────────────────

def _cross(client: TestClient, world: dict, name: str):
    a, b_pid, b_doc, b_cid, b_wf = (
        world["a"], world["b_pid"], world["b_doc"], world["b_cid"], world["b_wf"],
    )
    a_pid = world["a_pid"]
    h = _h(a)
    if name == "project_get":
        return client.get(f"/v1/projects/{b_pid}", headers=h)
    if name == "project_patch":
        return client.patch(f"/v1/projects/{b_pid}", headers=h, json={"location": "x"})
    if name == "project_delete":
        return client.delete(f"/v1/projects/{b_pid}", headers=h)
    if name == "docs_list":
        return client.get(f"/v1/projects/{b_pid}/documents", headers=h)
    if name == "docs_search":
        return client.get(
            f"/v1/projects/{b_pid}/documents/search", headers=h, params={"q": "rate"}
        )
    if name == "doc_preview":
        return client.get(
            f"/v1/projects/{b_pid}/documents/{b_doc}/preview", headers=h
        )
    if name == "doc_preview_raw":
        return client.get(
            f"/v1/projects/{b_pid}/documents/{b_doc}/preview/raw", headers=h
        )
    if name == "doc_delete":
        return client.delete(
            f"/v1/projects/{b_pid}/documents/{b_doc}", headers=h
        )
    if name == "doc_patch":
        return client.patch(
            f"/v1/projects/{b_pid}/documents/{b_doc}",
            headers=h,
            json={"drive_file_id": "x" * 28},
        )
    if name == "doc_upload":
        return client.post(
            f"/v1/projects/{b_pid}/documents",
            headers=h,
            files={"file": ("x.txt", io.BytesIO(b"x"), "text/plain")},
        )
    if name == "memory_get":
        return client.get(f"/v1/projects/{b_pid}/memory", headers=h)
    if name == "memory_post":
        return client.post(
            f"/v1/projects/{b_pid}/memory",
            headers=h,
            json={"key": "stolen", "value": "x"},
        )
    if name == "memory_delete":
        return client.delete(f"/v1/projects/{b_pid}/memory/site_rate", headers=h)
    if name == "progress":
        return client.post(
            f"/v1/projects/{b_pid}/progress", headers=h, json={}
        )
    if name == "audit":
        return client.get(f"/v1/projects/{b_pid}/audit", headers=h)
    if name == "conversations_list":
        return client.get(f"/v1/projects/{b_pid}/conversations", headers=h)
    if name == "conversation_clear":
        return client.post(
            f"/v1/projects/{b_pid}/conversations/{b_cid}/clear", headers=h
        )
    if name == "conversation_export":
        return client.post(
            f"/v1/projects/{a_pid}/conversations/{b_cid}/export",
            headers=h,
            params={"format": "docx"},
        )
    if name == "conversation_export_on_b":
        return client.post(
            f"/v1/projects/{b_pid}/conversations/{b_cid}/export",
            headers=h,
            params={"format": "docx"},
        )
    if name == "conversation_export_schedule":
        return client.post(
            f"/v1/projects/{a_pid}/conversations/{b_cid}/export/schedule",
            headers=h,
        )
    if name == "export_schedule":
        return client.post(
            f"/v1/projects/{b_pid}/export/schedule",
            headers=h,
            json={"activities": [{"id": "A1", "name": "x", "duration_days": 1}]},
        )
    if name == "export_cost_boq":
        return client.post(
            f"/v1/projects/{a_pid}/export/cost-boq",
            headers=h,
            json={"document_id": b_doc},
        )
    if name == "price_boq":
        return client.post(
            f"/v1/projects/{a_pid}/price-boq",
            headers=h,
            json={"document_id": b_doc, "asset_type": "building"},
        )
    if name == "export_from_brief":
        return client.post(
            f"/v1/projects/{a_pid}/export/schedule-from-brief",
            headers=h,
            json={"brief": "fence", "conversation_id": b_cid},
        )
    if name == "export_from_document":
        return client.post(
            f"/v1/projects/{a_pid}/export/schedule-from-document",
            headers=h,
            json={"document_ids": [b_doc], "brief": "fence"},
        )
    if name == "export_from_boq":
        return client.post(
            f"/v1/projects/{a_pid}/export/schedule-from-boq",
            headers=h,
            json={"document_id": b_doc, "manhours_per_unit": {"concrete": 1.0}},
        )
    if name == "redlines":
        return client.post(
            f"/v1/projects/{b_pid}/documents/{b_doc}/redlines", headers=h
        )
    if name == "connectors":
        return client.get(f"/v1/projects/{b_pid}/connectors", headers=h)
    if name == "drive_import":
        return client.post(
            f"/v1/projects/{b_pid}/drive/import",
            headers=h,
            json={"file_id": "x", "name": "x.txt"},
        )
    if name == "drive_index":
        return client.post(
            f"/v1/projects/{b_pid}/drive/index-folder",
            headers=h,
            json={"folder_id": "x"},
        )
    if name == "drive_job":
        return client.get(
            f"/v1/projects/{b_pid}/drive/index-folder/job/not-a-real-job",
            headers=h,
        )
    if name == "drive_job_seeded":
        from app.routers import drive as drive_router

        jid = f"iso-job-{_RUN}"
        drive_router._DRIVE_FOLDER_JOBS[jid] = {
            "job_id": jid,
            "status": "queued",
            "project_id": b_pid,
            "folder_id": "B-SECRET-FOLDER",
            "created_at": 0,
        }
        return client.get(
            f"/v1/projects/{b_pid}/drive/index-folder/job/{jid}",
            headers=h,
        )
    if name == "rag_search":
        return client.post(
            "/v1/rag/search",
            headers=h,
            json={"query": "rate", "project_id": b_pid, "k": 5},
        )
    if name == "rag_gk":
        return client.get(
            "/v1/rag/gk-status", headers=h, params={"project_id": b_pid}
        )
    if name == "agent_messages":
        return client.get(
            f"/v1/agents/conversations/{b_cid}/messages", headers=h
        )
    if name == "agent_chat":
        return client.post(
            "/v1/agents/project-assistant/chat",
            headers=h,
            json={"message": "hi", "conversation_id": b_cid, "project_id": b_pid},
        )
    if name == "workflow_get":
        return client.get(f"/v1/workflows/{b_wf}", headers=h)
    if name == "workflow_delete":
        return client.delete(f"/v1/workflows/{b_wf}", headers=h)
    if name == "workflow_run":
        return client.post(
            f"/v1/workflows/{b_wf}/run", headers=h, json={"initial_input": "x"}
        )
    if name == "hydration_latest":
        return client.get(
            "/v1/hydration/latest",
            headers=h,
            params={"scope": "project", "project_id": b_pid},
        )
    if name == "hydration_history":
        return client.get(
            "/v1/hydration/history",
            headers=h,
            params={"scope": "project", "project_id": b_pid},
        )
    if name == "hydration_run":
        return client.post(
            "/v1/hydration/run",
            headers=h,
            json={"project_ids": [b_pid]},
        )
    if name == "feedback":
        return client.post(
            "/v1/feedback/route",
            headers=h,
            json={
                "message": "x",
                "correct_action": "chat",
                "project_id": b_pid,
            },
        )
    if name == "execute_rag":
        return client.post(
            "/v1/execute",
            headers=h,
            json={
                "block": "chat",
                "input": "what is the rate",
                "params": {"project_id": b_pid, "use_rag": True},
            },
        )
    if name == "execute_alias":
        return client.post(
            "/execute",
            headers=h,
            json={
                "block": "chat",
                "input": "what is the rate",
                "params": {"project_id": b_pid, "use_rag": True},
            },
        )
    if name == "chain_rag":
        return client.post(
            "/v1/chain",
            headers=h,
            json={
                "steps": [
                    {
                        "block": "chat",
                        "params": {"project_id": b_pid, "use_rag": True},
                    }
                ],
                "initial_input": "what is the rate",
            },
        )
    if name == "doc_search_b":
        return client.get(
            f"/v1/projects/{b_pid}/documents/search",
            headers=h,
            params={"q": CANARY_B[:20]},
        )
    if name == "doc_preview_swap":
        return client.get(
            f"/v1/projects/{a_pid}/documents/{b_doc}/preview", headers=h
        )
    if name == "doc_delete_swap":
        return client.delete(
            f"/v1/projects/{a_pid}/documents/{b_doc}", headers=h
        )
    if name == "redline_swap":
        return client.post(
            f"/v1/projects/{a_pid}/documents/{b_doc}/redlines", headers=h
        )
    if name == "workflow_stamp":
        return client.post(
            "/v1/workflows",
            headers=h,
            json={
                "name": "stamp B",
                "steps": [{"block": "formula_executor", "params": {}}],
                "project_id": b_pid,
            },
        )
    if name == "execute_document":
        return client.post(
            "/v1/execute",
            headers=h,
            json={
                "block": "chat",
                "input": "what is the rate",
                "params": {"document_id": b_doc, "use_rag": True},
            },
        )
    if name == "chain_document":
        return client.post(
            "/v1/chain",
            headers=h,
            json={
                "steps": [
                    {
                        "block": "chat",
                        "params": {"document_ids": [b_doc], "use_rag": True},
                    }
                ],
                "initial_input": "what is the rate",
            },
        )
    if name == "chat_stream_b_cid":
        return client.post(
            "/v1/chat/stream",
            headers=h,
            json={"message": "quote the rate", "conversation_id": b_cid},
        )
    if name == "chat_stream_b_docs":
        return client.post(
            "/v1/chat/stream",
            headers=h,
            json={
                "message": "extract schedule from the attached docs",
                "project_id": a_pid,
                "document_ids": [b_doc],
            },
        )
    if name == "agent_stream_b_docs":
        return client.post(
            "/v1/agents/project-assistant/chat/stream",
            headers=h,
            json={
                "message": "extract schedule from the attached docs",
                "project_id": a_pid,
                "document_ids": [b_doc],
            },
        )
    raise AssertionError(name)


CROSS_SURFACES = [
    "project_get",
    "project_patch",
    "project_delete",
    "docs_list",
    "docs_search",
    "doc_preview",
    "doc_preview_raw",
    "doc_delete",
    "doc_patch",
    "doc_upload",
    "memory_get",
    "memory_post",
    "memory_delete",
    "progress",
    "audit",
    "conversations_list",
    "conversation_clear",
    "conversation_export",
    "conversation_export_on_b",
    "conversation_export_schedule",
    "export_schedule",
    "export_cost_boq",
    "price_boq",
    "export_from_brief",
    "export_from_document",
    "export_from_boq",
    "redlines",
    "connectors",
    "drive_import",
    "drive_index",
    "drive_job",
    "drive_job_seeded",
    "rag_search",
    "rag_gk",
    "agent_messages",
    "agent_chat",
    "workflow_get",
    "workflow_delete",
    "workflow_run",
    "hydration_latest",
    "hydration_history",
    "hydration_run",
    "feedback",
    "execute_rag",
    "execute_alias",
    "chain_rag",
    "doc_search_b",
    "doc_preview_swap",
    "doc_delete_swap",
    "redline_swap",
    "workflow_stamp",
    "execute_document",
    "chain_document",
    "chat_stream_b_cid",
    "chat_stream_b_docs",
    "agent_stream_b_docs",
]


@pytest.mark.parametrize("surface", CROSS_SURFACES)
def test_a_denied_on_b_ids(client, world, surface):
    resp = _cross(client, world, surface)
    _assert_denied(resp, label=surface)
    _assert_b_unchanged(client, world)


# ── lists ───────────────────────────────────────────────────────────────────

def test_a_list_projects_excludes_b_private(client, world):
    rows = client.get("/v1/projects", headers=_h(world["a"])).json()["projects"]
    ids = {p["id"] for p in rows}
    names = {p.get("name") for p in rows}
    assert world["a_pid"] in ids
    assert world["b_pid"] not in ids
    assert not any(CANARY_B in str(n) for n in names)
    # master_corpus / approved GK may appear — that is one-to-all by design.


def test_a_list_workflows_excludes_b(client, world):
    rows = client.get("/v1/workflows", headers=_h(world["a"])).json()["workflows"]
    ids = {w["id"] for w in rows}
    assert world["a_wf"] in ids
    assert world["b_wf"] not in ids


# ── retrieval canary ────────────────────────────────────────────────────────

def test_retrieval_canary_never_crosses(client, world):
    a, a_pid, b_pid, b_cid = world["a"], world["a_pid"], world["b_pid"], world["b_cid"]
    h = _h(a)

    own = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": "rate", "project_id": a_pid, "k": 8},
    )
    assert own.status_code == 200, own.text
    assert CANARY_B not in own.text

    foreign = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": CANARY_B, "project_id": b_pid, "k": 8},
    )
    _assert_denied(foreign, label="rag B project")

    none = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": CANARY_B, "project_id": "no-such-project", "k": 8},
    )
    _assert_denied(none, label="rag missing project")

    mc = client.post(
        "/v1/rag/search",
        headers=h,
        json={"query": CANARY_B, "project_id": "master_corpus", "k": 8},
    )
    # master_corpus is one-to-all; B's private canary still must not appear.
    if mc.status_code == 200:
        assert CANARY_B not in mc.text
    else:
        assert mc.status_code in DENY + (401,)

    conv = client.get(
        f"/v1/agents/conversations/{b_cid}/messages", headers=h
    )
    _assert_denied(conv, label="B conversation messages")

    chat = client.post(
        "/v1/agents/project-assistant/chat",
        headers=h,
        json={"message": "quote the rate", "conversation_id": b_cid},
    )
    _assert_denied(chat, label="chat on B conversation")

    dropped = client.post(
        "/v1/chat",
        headers=h,
        json={"message": "quote the rate", "project_id": b_pid},
    )
    # /chat drops an unowned project_id rather than 404; B's canary still
    # must not appear (same fail-closed as project_ask).
    assert dropped.status_code not in (500,)
    assert CANARY_B not in dropped.text

    ask = client.post(
        "/v1/project/ask",
        headers=h,
        json={
            "session_id": f"iso-ask-{_RUN}",
            "request": "quote the documented rate",
            "project_id": b_pid,
        },
    )
    assert ask.status_code not in (500,)
    assert CANARY_B not in ask.text

    upload = client.post(
        "/v1/upload",
        headers=h,
        data={"project_id": b_pid},
        files={"file": ("x.txt", io.BytesIO(b"not-b"), "text/plain")},
    )
    assert upload.status_code in (200, 201) + DENY
    if upload.status_code in (200, 201):
        assert upload.json().get("document_id") in (None, "")
        docs = client.get(
            f"/v1/projects/{b_pid}/documents", headers=_h(world["b"])
        ).json()["documents"]
        assert all(d.get("original_name") != "x.txt" for d in docs)


# ── role: plain user vs admin / debug / governance / keys / monitoring ──────

ADMIN_PATHS = [
    ("GET", "/v1/admin/debug/doc-extract", {"project_id": "x", "document_id": "y"}),
    ("GET", "/v1/admin/debug/document-download", {"project_id": "x", "document_id": "y"}),
    ("POST", "/v1/admin/debug/doc-reindex", {"project_id": "x", "document_id": "y"}),
    ("GET", "/v1/admin/projects/archived", None),
    ("GET", "/v1/admin/training/list", None),
    ("GET", "/v1/admin/debug/pilot-preflight", None),
    ("GET", "/v1/admin/corpus/collections", None),
    ("GET", "/v1/admin/corpus/coverage", None),
    ("GET", "/v1/admin/drive/scan", None),
    ("GET", "/v1/admin/dead-letter", None),
    ("POST", "/v1/admin/corpus/delete-docs", {"doc_ids": ["x"], "project_id": "x"}),
    ("POST", "/v1/admin/debug/sweep-plaintext", None),
    ("GET", "/v1/governance", None),
    ("POST", "/v1/governance/purge", None),
    ("GET", "/debug/env", None),
    ("GET", "/v1/debug/env", None),
    ("GET", "/v1/auth/keys", None),
    ("POST", "/v1/auth/keys", None),
    ("POST", "/v1/auth/keys/revoke", {"api_key": DEV_KEY}),
    ("POST", "/v1/auth/keys/rotate", {"api_key": DEV_KEY}),
    ("GET", "/v1/metrics", None),
]


@pytest.mark.parametrize("method,path,params", ADMIN_PATHS)
def test_plain_user_denied_admin_debug_governance(client, world, method, path, params):
    h = _h(world["a"])
    if method == "GET":
        resp = client.get(path, headers=h, params=params or {})
    elif path.endswith("/doc-reindex"):
        resp = client.post(path, headers=h, params=params or {})
    else:
        resp = client.post(path, headers=h, json=params or {})
    assert resp.status_code in (401, 403, 404), (
        f"{method} {path}: expected deny, got {resp.status_code} {resp.text[:200]}"
    )
    assert CANARY_B not in resp.text
    # Status alone is not enough on /debug/env: the SPA catch-all used
    # to answer 200 HTML with no payload. Production must 404, and the
    # body must never carry process environment fields.
    if path.rstrip("/").endswith("/debug/env") or path in ("/debug/env", "/v1/debug/env"):
        body = resp.text
        assert "data_dir" not in body, f"{path} leaked data_dir: {body[:200]}"
        assert "DATA_DIR" not in body, f"{path} leaked DATA_DIR: {body[:200]}"
        assert "<!doctype html" not in body.lstrip().lower(), (
            f"{path} served the SPA shell ({resp.status_code})"
        )


def test_plain_user_cannot_probe_other_api_keys(client, world):
    h = _h(world["a"])
    # JWT user probing the well-known dev key must not learn its validity.
    for path, payload in (
        ("/v1/auth/validate", {"api_key": DEV_KEY}),
        ("/v1/auth/check", {"api_key": DEV_KEY, "block": "chat"}),
    ):
        resp = client.post(path, headers=h, json=payload)
        assert resp.status_code in (401, 403, 404), (
            f"{path}: expected deny, got {resp.status_code} {resp.text[:200]}"
        )
    usage = client.get("/v1/auth/usage", headers=h, params={"key": DEV_KEY})
    assert usage.status_code in (401, 403, 404)
    stolen = client.delete("/v1/auth/keys/" + DEV_KEY, headers=h)
    assert stolen.status_code in (401, 403, 404)


def test_plain_user_cannot_read_foreign_memory_cache_key(client, world):
    """Shared in-process cache must not let A get a key B set."""
    key = f"iso-cache-{_RUN}"
    set_b = client.post(
        "/v1/memory/set",
        headers=_h(world["b"]),
        json={"key": key, "value": CANARY_B},
    )
    assert set_b.status_code in (200, 201), set_b.text
    got = client.post(
        "/v1/memory/get",
        headers=_h(world["a"]),
        json={"key": key},
    )
    # Denial or empty miss — never B's value.
    if got.status_code in DENY:
        assert CANARY_B not in got.text
        return
    assert got.status_code == 200, got.text
    assert CANARY_B not in json.dumps(got.json())


# ── /v1/execute plain-user capability ───────────────────────────────────────

def test_execute_plain_user_privileged_blocks_blocked(client, world):
    from app.blocks import BLOCK_REGISTRY
    from app.core.privileges import PRIVILEGED_BLOCKS

    for actor in ("a", "b"):
        h = _h(world[actor])
        for path in ("/v1/execute", "/execute"):
            for block in sorted(PRIVILEGED_BLOCKS):
                if block not in BLOCK_REGISTRY:
                    assert block in PRIVILEGED_BLOCKS
                    continue
                r = client.post(
                    path, headers=h, json={"block": block, "input": "x"}
                )
                assert r.status_code == 403, (
                    actor, path, block, r.status_code, r.text[:200]
                )


def test_both_users_refused_privileged_agent_doors(client, world):
    """Agent F: both synthetic accounts see the same 404 / SSE error."""
    hats = ("self-coding", "external-mcp", "document-ingestion")
    for actor in ("a", "b"):
        h = _h(world[actor])
        names = {
            a["name"]
            for a in client.get("/v1/agents", headers=h).json()["agents"]
        }
        assert not names & set(hats), (actor, names)
        for hat in hats:
            r = client.get(f"/v1/agents/{hat}", headers=h)
            assert r.status_code == 404, (actor, hat, r.status_code, r.text[:200])
            chat = client.post(
                f"/v1/agents/{hat}/chat",
                headers=h,
                json={"message": "hi"},
            )
            assert chat.status_code == 404, (
                actor, hat, chat.status_code, chat.text[:200]
            )
            stream = client.post(
                "/v1/chat/stream",
                headers=h,
                json={"message": "list the files", "agent": hat},
            )
            events = [
                json.loads(line[5:])
                for line in stream.text.splitlines()
                if line.startswith("data:") and line[5:].strip().startswith("{")
            ]
            assert not any(
                e.get("type") == "route" and e.get("final") == hat for e in events
            ), (actor, hat, events)
            errors = [e for e in events if e.get("type") == "error"]
            assert errors and "not available" in errors[0]["message"], (
                actor, hat, events
            )


def test_execute_plain_user_nonprivileged_without_foreign_id_is_not_403(client, world):
    """Document current capability: translate (no tenant id) is allowed."""
    r = client.post(
        "/v1/execute",
        headers=_h(world["a"]),
        json={"block": "translate", "input": "hola", "params": {"target": "en"}},
    )
    assert r.status_code != 403, r.text
    assert CANARY_B not in r.text


def test_execute_plain_user_cannot_pass_foreign_project_id(client, world):
    r = client.post(
        "/v1/execute",
        headers=_h(world["a"]),
        json={
            "block": "chat",
            "input": "what is the rate",
            "params": {"project_id": world["b_pid"], "use_rag": True},
        },
    )
    _assert_denied(r, label="execute foreign project_id")
    _assert_b_unchanged(client, world)


# ── accounts ────────────────────────────────────────────────────────────────

def test_a_duplicate_email_answers_like_a_new_one_and_changes_nothing(client):
    """#978: the reply never reveals an existing account, and the second
    registration neither creates a second account nor resets the first."""
    email = f"dup-{_RUN}@example.com"
    first = client.post(
        "/v1/users/register", json={"email": email, "password": PASSWORD}
    )
    second = client.post(
        "/v1/users/register", json={"email": email, "password": OTHER_PASSWORD}
    )
    assert first.status_code == second.status_code == 201
    assert first.json() == second.json()
    assert client.post("/v1/users/login", json={"email": email, "password": OTHER_PASSWORD}).status_code == 401


def test_email_case_and_whitespace_collapse(client):
    email = f"Case-{_RUN}@Example.COM"
    r = client.post(
        "/v1/users/register",
        json={"email": f"  {email}  ", "password": PASSWORD},
    )
    assert r.status_code == 201
    again = client.post(
        "/v1/users/register",
        json={"email": email.lower(), "password": OTHER_PASSWORD},
    )
    assert again.status_code == 201
    # One account, under the collapsed address: the second registration did
    # not create another, and its password does not open the first.
    from app.core import users as users_store

    assert users_store.get_user_by_email(email.lower()) is not None
    assert client.post("/v1/users/login", json={"email": email.lower(), "password": OTHER_PASSWORD}).status_code == 401


def test_weak_password_rejected(client):
    r = client.post(
        "/v1/users/register",
        json={"email": f"weak-{_RUN}@example.com", "password": WEAK_PASSWORD},
    )
    assert r.status_code == 422


def test_login_errors_identical_for_missing_and_wrong_password(client):
    known = f"login-{_RUN}@example.com"
    client.post("/v1/users/register", json={"email": known, "password": PASSWORD})
    missing = client.post(
        "/v1/users/login",
        json={"email": f"no-such-{_RUN}@example.com", "password": PASSWORD},
    )
    wrong = client.post(
        "/v1/users/login",
        json={"email": known, "password": WRONG_PASSWORD},
    )
    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert missing.json() == wrong.json()


def test_resend_verification_does_not_enumerate(client):
    known = f"rs-{_RUN}@example.com"
    client.post("/v1/users/register", json={"email": known, "password": PASSWORD})
    a = client.post("/v1/users/resend-verification", json={"email": known})
    b = client.post(
        "/v1/users/resend-verification",
        json={"email": f"ghost-{_RUN}@example.com"},
    )
    assert a.status_code == 202
    assert b.status_code == 202
    assert a.json() == b.json()


def test_jwt_alg_none_and_tampered_rejected(client, world):
    none = jwt.encode({"user_id": world["b"]["id"]}, key="", algorithm="none")
    r = client.get("/v1/users/me", headers={"Authorization": f"Bearer {none}"})
    assert r.status_code == 401

    good = world["a"]["token"]
    tampered = good[:-2] + ("AA" if good[-2:] != "AA" else "BB")
    r2 = client.get("/v1/users/me", headers={"Authorization": f"Bearer {tampered}"})
    assert r2.status_code == 401


def test_revoked_style_deleted_user_token_stops(client, world):
    """Token for a deleted user_id must 401 (same class as revoked session)."""
    from app.core import jwt_auth

    ghost = jwt_auth.create_token("no-such-user-" + _RUN)
    r = client.get("/v1/users/me", headers={"Authorization": f"Bearer {ghost}"})
    assert r.status_code == 401


def test_jwt_signed_with_empty_or_other_secret_rejected(client, world):
    """Guessed/unset SECRET_KEY must not authenticate (HS256 only).

    PyJWT refuses to *encode* an empty HMAC key. The server-side property
    is the same: a token minted with a secret the process did not choose
    (empty string is not a usable HS256 key; a guessed string is) is 401.
    """
    payload = {"user_id": world["a"]["id"]}
    other = jwt.encode(payload, "not-the-server-secret", algorithm="HS256")
    r = client.get("/v1/users/me", headers={"Authorization": f"Bearer {other}"})
    assert r.status_code == 401


def test_login_before_verify_is_403(client):
    from app.core import users as users_store

    email = f"unverified-{_RUN}@example.com"
    users_store.create_user(email, PASSWORD, email_verified=False)
    r = client.post("/v1/users/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 403
    assert "verified" in r.text.lower()


def test_verify_token_mismatch_does_not_verify(client):
    from app.core import jwt_auth
    from app.core import users as users_store

    email = f"vtok-{_RUN}@example.com"
    user = users_store.create_user(email, PASSWORD, email_verified=False)
    session = jwt_auth.create_token(user["id"])
    garbage = client.get(
        "/v1/users/verify-email",
        params={"token": "not-a-token"},
        follow_redirects=False,
    )
    replay = client.get(
        "/v1/users/verify-email",
        params={"token": session},
        follow_redirects=False,
    )
    assert garbage.status_code in (302, 400, 401, 403, 404)
    assert replay.status_code in (302, 400, 401, 403, 404)
    assert users_store.get_user_by_email(email)["email_verified"] is False
    loc = (garbage.headers.get("location") or "") + (replay.headers.get("location") or "")
    if loc:
        assert "success" not in loc.lower() or "invalid" in loc.lower()


# ── input / spoofing ────────────────────────────────────────────────────────

def test_upload_path_traversal_does_not_escape(client, world):
    r = client.post(
        f"/v1/projects/{world['a_pid']}/documents",
        headers=_h(world["a"]),
        files={
            "file": (
                "../../etc/passwd.txt",
                io.BytesIO(b"not-a-secret"),
                "text/plain",
            )
        },
    )
    assert r.status_code in (201, 400), r.text
    if r.status_code == 201:
        stored = r.json()["document"]
        name = stored.get("original_name") or stored.get("stored_as") or ""
        assert ".." not in name
        assert "etc/passwd" not in name


def test_wrong_type_ids_do_not_leak(client, world):
    h = _h(world["a"])
    r = client.get("/v1/projects/" + "x" * 200, headers=h)
    assert r.status_code in DENY
    r2 = client.get("/v1/workflows/not-a-uuid", headers=h)
    assert r2.status_code in DENY


def test_spoofed_identity_headers_change_nothing(client, world):
    spoof = _h(
        world["a"],
        {
            "X-User-Id": world["b"]["id"],
            "X-User-Role": "admin",
            "X-Role": "admin",
            "X-Forwarded-User": world["b"]["email"],
            "X-Forwarded-Role": "admin",
        },
    )
    me = client.get("/v1/users/me", headers=spoof)
    assert me.status_code == 200
    assert me.json()["user_id"] == world["a"]["id"]
    assert me.json()["role"] != "admin" or world["a"]["id"] == world["admin"]["id"]
    foreign = client.get(f"/v1/projects/{world['b_pid']}", headers=spoof)
    _assert_denied(foreign, label="spoofed header project get")
    gov = client.get("/v1/governance", headers=spoof)
    assert gov.status_code in (401, 403, 404)
