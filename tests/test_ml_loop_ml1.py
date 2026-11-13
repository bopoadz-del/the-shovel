"""ML-1: the ML loop block's migration, boot wiring and host adapters.

Every test builds its own input; every decision is exercised with two
different inputs so a constant cannot pass it.
"""
from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from alembic.migration import MigrationContext
from alembic.operations import Operations

import app.blocks.ml_loop as ml_loop_mod
from app.blocks.ml_loop import MLLoopBlock
from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import TABLES, MLLoopRepository, create_tables

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "alembic" / "versions" / "0034_ml_loop_tables.py"


def _load_migration():
    spec = importlib.util.spec_from_file_location("ml_loop_migration_0034", MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def store(tmp_path):
    """A fresh SQLite store installed as the block's process-wide settings."""
    engine = create_engine(f"sqlite:///{tmp_path / 'ml.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine)
    MLLoopBlock.configure(MLLoopSettings(build_sha="sha-test", session_factory=factory))
    ml_loop_mod._shared_block = None
    yield MLLoopRepository(factory)
    MLLoopBlock.configure(None)
    ml_loop_mod._shared_block = None
    engine.dispose()


# ── migration ───────────────────────────────────────────────────────────

def _run(engine, fn_name: str) -> None:
    migration = _load_migration()
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            getattr(migration, fn_name)()


@pytest.mark.parametrize("cycles", [1, 3])
def test_migration_round_trip_creates_and_drops_exactly_the_ml_tables(tmp_path, cycles):
    engine = create_engine(f"sqlite:///{tmp_path / 'mig.db'}")
    for _ in range(cycles):  # every upgrade after a downgrade is clean
        _run(engine, "upgrade")
        assert set(inspect(engine).get_table_names()) == set(TABLES)
        _run(engine, "downgrade")
        assert inspect(engine).get_table_names() == []
    engine.dispose()


@pytest.mark.parametrize("name", sorted(TABLES))
def test_migration_matches_the_block_metadata(tmp_path, name):
    """Columns (name, nullability) and indexes of 0034 equal the module's."""
    migrated = create_engine(f"sqlite:///{tmp_path / 'a.db'}")
    _run(migrated, "upgrade")
    from_meta = create_engine(f"sqlite:///{tmp_path / 'b.db'}")
    create_tables(from_meta)
    a, b = inspect(migrated), inspect(from_meta)
    cols_a = {(c["name"], c["nullable"]) for c in a.get_columns(name)}
    cols_b = {(c["name"], c["nullable"]) for c in b.get_columns(name)}
    assert cols_a and cols_a == cols_b
    idx_a = {(i["name"], tuple(i["column_names"])) for i in a.get_indexes(name)}
    idx_b = {(i["name"], tuple(i["column_names"])) for i in b.get_indexes(name)}
    assert idx_a == idx_b
    migrated.dispose()
    from_meta.dispose()


@pytest.mark.parametrize("attr, expected", [
    ("revision", "0034"),        # no other file reuses the revision id
    ("down_revision", "0033"),   # no second head on 0033
])
def test_migration_continues_the_single_series(attr, expected):
    import re

    assert getattr(_load_migration(), attr) == expected
    pattern = re.compile(rf"^{attr}\b[^=\n]*=\s*\W{expected}\W", re.MULTILINE)
    claims = [
        p.name for p in sorted(MIGRATION.parent.glob("*.py"))
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert claims == [MIGRATION.name]


@pytest.mark.parametrize("calls", [2, 4])
def test_create_tables_is_idempotent(tmp_path, calls):
    engine = create_engine(f"sqlite:///{tmp_path / 'idem.db'}")
    for _ in range(calls):
        create_tables(engine)
    assert set(inspect(engine).get_table_names()) == set(TABLES)
    engine.dispose()


# ── shared entry point ──────────────────────────────────────────────────

@pytest.mark.parametrize("op, payload", [
    ("record_turn", {"project_id": "p-a", "question": "q1", "verdict": "refused"}),
    ("record_correction", {"project_id": "p-b", "formula_id": "f", "predicted": 1, "actual": 2}),
])
def test_record_without_a_store_is_not_configured(monkeypatch, op, payload):
    MLLoopBlock.configure(None)
    ml_loop_mod._shared_block = None
    result = ml_loop_mod.record(op, payload)
    assert result["status"] == "not_configured"
    assert result["stored"] is False


@pytest.mark.parametrize("op, exc", [
    ("record_turn", RuntimeError("store down")),
    ("record_correction", OSError("disk full")),
])
def test_record_never_raises(monkeypatch, store, op, exc):
    def boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(MLLoopBlock, "dispatch", boom)
    result = ml_loop_mod.record(op, {"project_id": "p"})
    assert result["status"] == "error"
    assert str(exc) in result["error"]


@pytest.mark.parametrize("flag", ["0", " False "])
def test_disabled_flag_stores_nothing(monkeypatch, tmp_path, flag):
    engine = create_engine(f"sqlite:///{tmp_path / 'off.db'}")
    create_tables(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setenv("ML_LOOP_ENABLED", flag)
    MLLoopBlock.configure(MLLoopSettings.from_env(session_factory=factory))
    ml_loop_mod._shared_block = None
    try:
        for question in ("first question", "second question"):
            result = ml_loop_mod.record(
                "record_turn", {"project_id": "p", "question": question, "verdict": "refused"},
            )
            assert result["status"] == "disabled"
        assert MLLoopRepository(factory).counts()["ml_turns"] == 0
    finally:
        MLLoopBlock.configure(None)
        engine.dispose()


def test_flag_absent_means_on(monkeypatch):
    monkeypatch.delenv("ML_LOOP_ENABLED", raising=False)
    assert MLLoopSettings.from_env(session_factory=lambda: None).writes_enabled is True
    monkeypatch.setenv("ML_LOOP_ENABLED", "off")
    assert MLLoopSettings.from_env(session_factory=lambda: None).writes_enabled is False


# ── adapter: dataset_log -> record_turn ─────────────────────────────────

@pytest.mark.parametrize("question, verdict, cites", [
    ("What is the slab thickness?", "project_document", [{"doc_id": "d-1"}, {"doc_id": "d-2"}]),
    ("Which grade of rebar applies?", "refused", []),
])
def test_dataset_log_forwards_each_turn(tmp_path, store, question, verdict, cites):
    from app.core.rag.dataset_log import DatasetRecord, append_dataset_record

    record = DatasetRecord(
        question=question, answer=f"answer to {question}", verdict=verdict,
        evidence_plan=("plan-a",), retrieved_ids=tuple(c["doc_id"] for c in cites),
        citations=tuple(cites),
    )
    path = append_dataset_record("proj-x", record, base=tmp_path)
    # The file stays the dataset of record.
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert line["question"] == question
    rows = store.rows("ml_turns", project_id="proj-x")
    assert len(rows) == 1
    row = rows[0]
    assert row["question"] == question
    assert row["verdict"] == verdict
    assert row["cited_ids"] == [c["doc_id"] for c in cites]
    assert row["retrieved_ids"] == [c["doc_id"] for c in cites]
    assert row["build_sha"] == "sha-test"


@pytest.mark.parametrize("questions", [("q one", "q two"), ("only one",)])
def test_dataset_log_file_write_survives_a_failing_loop(tmp_path, monkeypatch, store, questions):
    from app.core.rag.dataset_log import DatasetRecord, append_dataset_record

    monkeypatch.setattr(MLLoopBlock, "dispatch", lambda *a, **k: 1 / 0)
    for q in questions:
        path = append_dataset_record("p", DatasetRecord(question=q, answer="", verdict="refused"),
                                     base=tmp_path)
    lines = [json.loads(x)["question"] for x in path.read_text(encoding="utf-8").splitlines()]
    assert lines == list(questions)
    assert store.counts()["ml_turns"] == 0


@pytest.mark.parametrize("blank, real", [("   ", "real"), ("", "another question")])
def test_dataset_log_does_not_forward_an_empty_question(tmp_path, store, blank, real):
    from app.core.rag.dataset_log import DatasetRecord, append_dataset_record

    append_dataset_record("p", DatasetRecord(question=blank, answer="a", verdict="refused"),
                          base=tmp_path)
    append_dataset_record("p", DatasetRecord(question=real, answer="a", verdict="refused"),
                          base=tmp_path)
    assert [r["question"] for r in store.rows("ml_turns")] == [real]


# ── adapter: learning_engine -> record_correction / record_route_correction ─

def _engine(tmp_path):
    from app.blocks.learning_engine import LearningEngineBlock

    return LearningEngineBlock(config={"storage_path": str(tmp_path / "le.json")})


@pytest.mark.parametrize("formula_id, predicted, actual", [
    ("formula-alpha", 10.0, 12.5),
    ("formula-beta", 3.0, 2.0),
])
def test_learning_engine_forwards_numeric_corrections(tmp_path, store, formula_id, predicted, actual):
    le = _engine(tmp_path)
    result = asyncio.run(le._record_correction(
        {"project_id": "proj-c", "formula_id": formula_id, "predicted": predicted, "actual": actual},
        {},
    ))
    assert result["status"] == "success"
    rows = store.rows("ml_corrections", project_id="proj-c")
    assert [(r["formula_id"], r["predicted"], r["actual"]) for r in rows] == [
        (formula_id, predicted, actual)
    ]


@pytest.mark.parametrize("data", [{"formula_id": "f"}, {"predicted": 1, "actual": 2}])
def test_learning_engine_rejected_correction_is_not_forwarded(tmp_path, store, data):
    le = _engine(tmp_path)
    result = asyncio.run(le._record_correction({"project_id": "p", **data}, {}))
    assert result["status"] == "error"
    assert store.counts()["ml_corrections"] == 0


def _route_obs(text, action, corrected, original=None):
    return json.dumps({
        "text": text, "action": action, "score": 1.0, "source": "user_correction",
        "corrected": corrected, "original_action": original,
    })


@pytest.mark.parametrize("text, action, original", [
    ("draw me a gantt for the podium", "generate_schedule", "qto"),
    ("price the rebar in level 3", "boq_pricing", None),
])
def test_learning_engine_forwards_corrected_routes(tmp_path, store, text, action, original):
    le = _engine(tmp_path)
    le._record_pattern({
        "project_id": "proj-r", "category": "routing_decisions",
        "observation": _route_obs(text, action, True, original), "source": "feedback_route",
    }, {})
    rows = store.rows("ml_route_corrections", project_id="proj-r")
    assert [(r["message"], r["correct_action"], r["original_action"]) for r in rows] == [
        (text, action, original)
    ]


@pytest.mark.parametrize("category, corrected", [
    ("routing_decisions", False),     # the router's own log is not a correction
    ("friction_points", True),        # another category is not a route
])
def test_learning_engine_does_not_forward_other_patterns(tmp_path, store, category, corrected):
    le = _engine(tmp_path)
    le._record_pattern({
        "project_id": "p", "category": category,
        "observation": _route_obs("some text", "some_action", corrected),
    }, {})
    assert store.counts()["ml_route_corrections"] == 0


# ── adapter: /v1/feedback/route -> record_route_correction ──────────────

@pytest.mark.parametrize("message, correct, original", [
    ("summarise the contract delays", "contract_qa", "generate_wbs"),
    ("what is the pour volume", "construction_calc", None),
])
def test_feedback_route_lands_in_the_ml_store(tmp_path, monkeypatch, store, message, correct, original):
    from fastapi.testclient import TestClient

    from app.routers.auth import require_api_key
    from app.routers.feedback import router

    from fastapi import FastAPI

    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    from app.blocks import BLOCK_REGISTRY
    from app.blocks.learning_engine import LearningEngineBlock
    monkeypatch.setitem(BLOCK_REGISTRY, "learning_engine", LearningEngineBlock)
    LearningEngineBlock.reset_shared_instance_cache()

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[require_api_key] = lambda: {"auth_method": "api_key"}
    client = TestClient(app)
    body = {"message": message, "correct_action": correct, "project_id": "proj-f"}
    if original:
        body["original_action"] = original
    resp = client.post("/v1/feedback/route", json=body)
    assert resp.status_code == 200, resp.text
    rows = store.rows("ml_route_corrections", project_id="proj-f")
    assert [(r["message"], r["correct_action"], r["original_action"]) for r in rows] == [
        (message, correct, original)
    ]
    LearningEngineBlock.reset_shared_instance_cache()


# ── boundaries ──────────────────────────────────────────────────────────

ANSWER_PATH = (
    "app/agents/runtime.py",
    "app/agents/core/loop.py",
    "app/agents/core/streaming.py",
    "app/agents/driver/loop.py",
)


@pytest.mark.parametrize("rel", ANSWER_PATH)
def test_answer_path_does_not_import_the_ml_loop(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.fail(f"{rel} moved; update the boundary list")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert "ml_loop" not in (node.module or ""), rel
        elif isinstance(node, ast.Import):
            assert all("ml_loop" not in a.name for a in node.names), rel


def test_boot_configures_the_block_with_a_session_factory(monkeypatch, tmp_path):
    import app.main as main

    url = f"sqlite:///{tmp_path / 'boot.db'}"
    eng = create_engine(url)
    monkeypatch.setattr("app.core.db.get_engine", lambda: eng)
    monkeypatch.setattr("app.core.db.SessionLocal", sessionmaker(bind=eng))
    try:
        main._configure_ml_loop()
        assert MLLoopBlock._configured is not None
        assert MLLoopBlock._configured.store_configured
        assert set(TABLES) <= set(inspect(eng).get_table_names())
    finally:
        MLLoopBlock.configure(None)
        eng.dispose()


def test_boot_failure_leaves_the_block_not_configured(monkeypatch):
    import app.main as main

    def broken():
        raise RuntimeError("db unreachable")

    monkeypatch.setattr("app.core.db.get_engine", broken)
    try:
        main._configure_ml_loop()
        assert MLLoopBlock._configured is not None
        assert MLLoopBlock._configured.store_configured is False
    finally:
        MLLoopBlock.configure(None)
