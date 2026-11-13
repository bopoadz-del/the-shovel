"""ML-2: answer feedback lands in ``ml_feedback``; ML-1 follow-up: a dataset
turn carries its conversation and message ids.

Every test builds its own conversation and store; every decision is
exercised with two different inputs so a constant cannot pass it.
"""
from __future__ import annotations

import importlib
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.blocks.ml_loop as ml_loop_mod
from app.blocks.ml_loop import MLLoopBlock
from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import MLLoopRepository, create_tables


@pytest.fixture()
def store(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ml.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine)
    MLLoopBlock.configure(MLLoopSettings(build_sha="sha-ml2", session_factory=factory))
    ml_loop_mod._shared_block = None
    yield MLLoopRepository(factory)
    MLLoopBlock.configure(None)
    ml_loop_mod._shared_block = None
    engine.dispose()


@pytest.fixture()
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import app.core.agent_memory as am

    importlib.reload(am)
    from app.core import projects as projects_store

    # Project grant is not under test here: every project is readable and
    # owned by "owner"; the conversation row decides the rest.
    monkeypatch.setattr(projects_store, "can_access_project", lambda *_a, **_k: True)
    monkeypatch.setattr(projects_store, "storage_project_id", lambda pid: pid)
    monkeypatch.setattr(projects_store, "project_owner", lambda _pid: "owner")
    monkeypatch.setattr(projects_store, "_master_corpus_source", lambda _pid: None)
    return am


def _client(user_id: str, role: str = "user") -> TestClient:
    from app.dependencies import require_user
    from app.routers import feedback

    app = FastAPI()
    app.include_router(feedback.router)
    app.dependency_overrides[require_user] = lambda: {"user_id": user_id, "role": role}
    return TestClient(app)


def _seed(am, conversation_id: str, project_id: str, owner: str | None,
          answers: list[tuple[str, list[dict] | None]]) -> list[dict]:
    """A conversation with one user turn before each answer; returns the
    stored assistant rows in order."""
    am.get_or_create_conversation(conversation_id, "scout", project_id=project_id, owner_id=owner)
    stored = []
    for i, (text, provenance) in enumerate(answers):
        am.append_message(conversation_id, "user", f"question {i}")
        stored.append(am.append_message(conversation_id, "assistant", text, provenance=provenance))
    return stored


PROV_TWO_DOCS = [
    {"figure": "250 mm", "source": "project_document", "doc_id": "doc-a", "page": 3},
    {"figure": "C40", "source": "project_document", "doc_id": "doc-b", "page": 9},
    {"figure": "250 mm", "source": "project_document", "doc_id": "doc-a", "page": 4},
    {"figure": "12", "source": "calculator"},
]
PROV_ONE_DOC = [{"figure": "SAR 1,200", "source": "general_knowledge", "doc_id": "gk-7", "page": None}]


# ── a verdict lands with server-derived keys ────────────────────────────

@pytest.mark.parametrize("cid, pid, index, body, want_answer, want_cited", [
    ("conv-plain-1", "proj-1", -1, {"kind": "correct"}, 1, ["gk-7"]),
    ("ws-proj-2-1700000000000", "proj-2", 0,
     {"kind": "wrong_figure", "correct_value": 300.0, "unit": "mm"}, 0, ["doc-a", "doc-b"]),
])
def test_verdict_is_stored_with_message_and_cited_ids(
    store, memory, cid, pid, index, body, want_answer, want_cited,
):
    stored = _seed(memory, cid, pid, "owner", [
        ("The slab is 250 mm thick, grade C40.", PROV_TWO_DOCS),
        ("The rate is SAR 1,200 per m3.", PROV_ONE_DOC),
    ])
    resp = _client("owner").post("/v1/feedback/answer", json={
        "project_id": pid, "conversation_id": cid, "message_index": index, **body,
    })
    assert resp.status_code == 200, resp.text
    target = stored[want_answer]
    assert resp.json() == {
        "status": "recorded",
        "message_id": target["id"],
        "answer_head": target["content"][:80],
    }
    rows = store.rows("ml_feedback", project_id=pid)
    assert len(rows) == 1
    row = rows[0]
    assert row["message_id"] == target["id"]
    assert row["conversation_id"] == cid
    assert row["kind"] == body["kind"]
    assert row["cited_ids"] == want_cited
    assert row["user_id"] == "owner"
    assert row["build_sha"] == "sha-ml2"
    assert row["correct_value"] == body.get("correct_value")
    assert row["unit"] == body.get("unit")


@pytest.mark.parametrize("kinds", [
    ("wrong_document", "correct"),
    ("correct", "missing", "correct"),
])
def test_repeat_verdict_appends_a_row(store, memory, kinds):
    _seed(memory, "conv-r", "proj-r", "owner", [("Answer one.", None)])
    client = _client("owner")
    for kind in kinds:
        extra = {"note": "a note"} if kind in ("wrong_document", "missing") else {}
        assert client.post("/v1/feedback/answer", json={
            "project_id": "proj-r", "conversation_id": "conv-r", "kind": kind, **extra,
        }).status_code == 200
    rows = store.rows("ml_feedback", project_id="proj-r")
    assert [r["kind"] for r in rows] == list(kinds)
    assert len({r["message_id"] for r in rows}) == 1


# ── ownership ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("owner, caller, role, status", [
    ("owner", "owner", "user", 200),
    ("owner", "someone-else", "user", 404),
    ("owner", "an-admin", "admin", 200),
    (None, "someone-else", "user", 404),
    (None, "an-admin", "admin", 200),
])
def test_only_the_owner_or_an_admin_records(store, memory, owner, caller, role, status):
    _seed(memory, "conv-own", "proj-own", owner, [("An answer.", None)])
    resp = _client(caller, role).post("/v1/feedback/answer", json={
        "project_id": "proj-own", "conversation_id": "conv-own", "kind": "correct",
    })
    assert resp.status_code == status, resp.text
    assert len(store.rows("ml_feedback")) == (1 if status == 200 else 0)
    if status == 404:
        assert resp.json() == {"detail": "Conversation not found"}


@pytest.mark.parametrize("cid, pid", [
    ("conv-missing", "proj-x"),          # no conversation row at all
    ("conv-elsewhere", "proj-other"),    # row bound to another project
])
def test_unknown_or_foreign_conversation_is_404(store, memory, cid, pid):
    _seed(memory, "conv-elsewhere", "proj-home", "owner", [("An answer.", None)])
    resp = _client("owner").post("/v1/feedback/answer", json={
        "project_id": pid, "conversation_id": cid, "kind": "correct",
    })
    assert resp.status_code == 404
    assert store.rows("ml_feedback") == []


# ── message resolution and validation ───────────────────────────────────

@pytest.mark.parametrize("index", [2, -3])
def test_out_of_range_index_is_404(store, memory, index):
    _seed(memory, "conv-i", "proj-i", "owner", [("One.", None), ("Two.", None)])
    resp = _client("owner").post("/v1/feedback/answer", json={
        "project_id": "proj-i", "conversation_id": "conv-i", "message_index": index,
        "kind": "correct",
    })
    assert resp.status_code == 404
    assert store.rows("ml_feedback") == []


@pytest.mark.parametrize("body", [
    {"kind": "correct", "note": "fine"},
    {"kind": "wrong_figure", "note": "should be 300"},
    {"kind": "missing", "correct_value": 3.0},
    {"kind": "wrong_document", "unit": "mm"},
    {"kind": "missing", "note": "x" * 201},
    {"kind": "wrong_figure", "unit": "u" * 33},
    {"kind": "not_a_kind"},
])
def test_fields_must_match_the_kind(store, memory, body):
    _seed(memory, "conv-v", "proj-v", "owner", [("One.", None)])
    resp = _client("owner").post("/v1/feedback/answer", json={
        "project_id": "proj-v", "conversation_id": "conv-v", **body,
    })
    assert resp.status_code == 422
    assert store.rows("ml_feedback") == []


@pytest.mark.parametrize("settings", [
    MLLoopSettings(build_sha="sha-ml2", session_factory=None),   # not configured
    "disabled",
])
def test_store_not_succeeding_is_503(tmp_path, memory, settings):
    if settings == "disabled":
        engine = create_engine(f"sqlite:///{tmp_path / 'off.db'}")
        create_tables(engine)
        settings = MLLoopSettings(build_sha="sha-ml2", session_factory=sessionmaker(bind=engine),
                                  enabled=False)
    MLLoopBlock.configure(settings)
    ml_loop_mod._shared_block = None
    try:
        _seed(memory, "conv-s", "proj-s", "owner", [("One.", None)])
        resp = _client("owner").post("/v1/feedback/answer", json={
            "project_id": "proj-s", "conversation_id": "conv-s", "kind": "correct",
        })
        assert resp.status_code == 503
        assert resp.json() == {"detail": "feedback store unavailable"}
    finally:
        MLLoopBlock.configure(None)
        ml_loop_mod._shared_block = None


@pytest.mark.parametrize("kind, extra", [
    ("correct", {}),
    ("wrong_figure", {"correct_value": 7.5, "unit": "m3"}),
])
def test_feedback_never_changes_stored_messages(store, memory, kind, extra):
    _seed(memory, "conv-b", "proj-b", "owner", [("First.", PROV_ONE_DOC), ("Second.", PROV_TWO_DOCS)])
    before = json.dumps(memory.get_messages("conv-b", limit=200), sort_keys=True, default=str)
    assert _client("owner").post("/v1/feedback/answer", json={
        "project_id": "proj-b", "conversation_id": "conv-b", "kind": kind, **extra,
    }).status_code == 200
    after = json.dumps(memory.get_messages("conv-b", limit=200), sort_keys=True, default=str)
    assert after == before


# ── Ruling 2: a dataset turn carries its conversation and message ids ────

@pytest.mark.parametrize("ids", [
    {"conversation_id": "ws-proj-t-1700000000001", "message_id": "m-41"},
    {},
])
def test_dataset_turn_lands_its_ids_or_nulls(tmp_path, store, ids):
    from app.core.rag.dataset_log import DatasetRecord, append_dataset_record

    append_dataset_record("proj-t", DatasetRecord(
        question="How deep is the footing?", answer="1.2 m", verdict="project_document",
        citations=({"doc_id": "d-9"},), **ids,
    ), base=tmp_path)
    rows = store.rows("ml_turns", project_id="proj-t")
    assert len(rows) == 1
    assert rows[0]["conversation_id"] == ids.get("conversation_id")
    assert rows[0]["message_id"] == ids.get("message_id")


# ── index drift: the client's answer head must match the stored answer ───

@pytest.mark.parametrize("head, status, rows", [
    ("The slab is   250 mm\nthick, grade C40.", 200, 1),   # same text, other whitespace
    ("The rate is SAR 1,200 per m3.", 409, 0),             # the other answer
    ("   ", 409, 0),                                       # nothing to match
])
def test_answer_head_must_match_the_stored_answer(store, memory, head, status, rows):
    _seed(memory, "conv-h", "proj-h", "owner", [
        ("The slab is 250 mm thick, grade C40.", None),
        ("The rate is SAR 1,200 per m3.", None),
    ])
    resp = _client("owner").post("/v1/feedback/answer", json={
        "project_id": "proj-h", "conversation_id": "conv-h", "message_index": 0,
        "kind": "correct", "answer_head": head,
    })
    assert resp.status_code == status, resp.text
    assert len(store.rows("ml_feedback")) == rows
