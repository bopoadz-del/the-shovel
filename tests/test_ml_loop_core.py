"""ML-0 core: settings, repository, proposals and shadow, called directly.

Each test builds its own in-memory store and rows; two inputs per decision.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.ml_loop import proposals, shadow, store
from app.core.ml_loop.settings import MLLoopSettings, _env_build_sha
from app.core.ml_loop.store import (
    MLLoopRepository,
    _as_aware,
    _key_columns,
    utcnow,
    window_start,
)

NOW = datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def _repo() -> MLLoopRepository:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    store.metadata.create_all(engine)
    return MLLoopRepository(sessionmaker(bind=engine))


def _turn(project, verdict, route, n):
    return {"project_id": project, "message_id": f"m{n}", "question": "q", "answer": "a",
            "verdict": verdict, "route": route, "evidence_plan": [], "retrieved_ids": [],
            "cited_ids": [], "created_at": NOW - timedelta(hours=n)}


# ── settings ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("env,expected", [
    ({"GIT_SHA": "abc123"}, "abc123"),
    ({"SOURCE_VERSION": "def456"}, "def456"),
])
def test_env_build_sha_reads_the_injected_commit(monkeypatch, env, expected):
    for key in ("GIT_SHA", "SOURCE_VERSION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert _env_build_sha() == expected
    assert MLLoopSettings.from_env().build_sha == expected


def test_env_build_sha_is_none_when_not_injected(monkeypatch):
    for key in ("GIT_SHA", "SOURCE_VERSION"):
        monkeypatch.delenv(key, raising=False)
    assert _env_build_sha() is None


@pytest.mark.parametrize("flag,factory,enabled,writes", [
    ("0", object, False, False),
    ("", object, True, True),
    ("on", None, True, False),
])
def test_from_env_and_writes_enabled(monkeypatch, flag, factory, enabled, writes):
    monkeypatch.setenv("ML_LOOP_ENABLED", flag)
    settings = MLLoopSettings.from_env(session_factory=factory)
    assert (settings.enabled, settings.writes_enabled) == (enabled, writes)
    assert settings.with_session_factory(None).store_configured is False
    assert settings.with_session_factory(object).store_configured is True


# ── store ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (datetime(2026, 1, 1, 8, 0), datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)),
    (datetime(2026, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=4))),
     datetime(2026, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=4)))),
])
def test_as_aware_treats_a_naive_stamp_as_utc(value, expected):
    out = _as_aware(value)
    assert out == expected and out.tzinfo == expected.tzinfo


@pytest.mark.parametrize("table", ["ml_turns", "ml_shadow_predictions"])
def test_every_table_carries_the_key_columns(table):
    keys = [c.name for c in _key_columns()]
    assert keys == ["id", "project_id", "conversation_id", "message_id", "build_sha", "created_at"]
    assert set(keys) <= {c.name for c in store.TABLES[table].columns}


@pytest.mark.parametrize("days", [1, 7])
def test_window_start_and_utcnow(days):
    assert window_start(days, NOW) == NOW - timedelta(days=days)
    stamp = utcnow()
    assert stamp.tzinfo is not None and abs(stamp - datetime.now(timezone.utc)) < timedelta(minutes=1)


@pytest.mark.parametrize("n", [1, 3])
def test_repository_insert_defaults_created_at_and_returns_ids(n):
    repo = _repo()
    ids = [
        repo.insert("ml_route_corrections", {"project_id": "p", "message": "m",
                                             "correct_action": "b", "source": "s"})
        for _ in range(n)
    ]
    assert ids == list(range(1, n + 1))
    rows = repo.rows("ml_route_corrections")
    assert len(rows) == n and all(r["created_at"].tzinfo is not None for r in rows)


@pytest.mark.parametrize("rows,imported", [
    ([{"id": 99, "project_id": "p", "message": "m", "correct_action": "b", "source": "s",
       "created_at": "2026-11-01T00:00:00+00:00"}], 1),
    ([], 0),
])
def test_repository_import_rows_reassigns_ids(rows, imported):
    repo = _repo()
    assert repo.import_rows("ml_route_corrections", rows) == imported
    got = repo.rows("ml_route_corrections")
    assert [r["id"] for r in got] == list(range(1, imported + 1))
    if imported:
        assert got[0]["created_at"] == datetime(2026, 11, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("row", [
    {"project_id": "p", "bogus": 1},
    {"project_id": "p", "message": "m", "correct_action": "b", "source": "s", "other": "x"},
])
def test_repository_import_rows_refuses_unknown_columns(row):
    with pytest.raises(ValueError, match="unknown column"):
        _repo().import_rows("ml_route_corrections", [row])


# ── proposals ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("verdicts,refused_rate", [
    (["refused", "project_document"], 0.5),
    (["refused", "refused", "refused", "template"], 0.75),
])
def test_verdict_rates_by_route(verdicts, refused_rate):
    turns = [{"route": "r", "verdict": v} for v in verdicts] + [{"route": None, "verdict": "template"}]
    rates = proposals.verdict_rates_by_route(turns)
    assert rates["r"]["refused_rate"] == refused_rate and rates["r"]["total"] == len(verdicts)
    assert rates["(unrouted)"]["total"] == 1


@pytest.mark.parametrize("support,expected_kinds", [
    (2, ["route_rule", "retrieval_review"]),
    (5, []),
])
def test_build_proposals_respects_support(support, expected_kinds):
    settings = MLLoopSettings(min_proposal_support=support, refusal_review_rate=0.5)
    turns = [{"id": i, "route": "r", "verdict": "refused"} for i in range(3)]
    pairs = {"a->b": {"from_action": "a", "to_action": "b", "ids": [1, 2, 3]}}
    rates = proposals.verdict_rates_by_route(turns)
    out = proposals.build_proposals(rates, turns, pairs, settings)
    assert [p["kind"] for p in out] == expected_kinds


@pytest.mark.parametrize("project,turns", [("p1", 2), ("p2", 1)])
def test_propose_reads_the_window_for_one_project(project, turns):
    repo = _repo()
    repo.insert("ml_turns", _turn("p1", "refused", "r", 1))
    repo.insert("ml_turns", _turn("p1", "refused", "r", 2))
    repo.insert("ml_turns", _turn("p2", "template", "r", 3))
    repo.insert("ml_turns", _turn("p2", "refused", "r", 24 * 30))  # outside 7 days
    out = proposals.propose(repo, MLLoopSettings(), project_id=project, now=NOW)
    assert out["turns"] == turns and out["window_days"] == 7


# ── shadow ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rows,expected", [
    ([{"project_id": "p", "conversation_id": "c", "message_id": "m1", "correct_action": "a"},
      {"project_id": "p", "conversation_id": "c", "message_id": "m1", "correct_action": "b"}],
     {("p", "c", "m1"): "b"}),
    ([{"project_id": "p", "conversation_id": None, "message_id": None, "correct_action": "a"}], {}),
])
def test_corrected_label_by_message_keeps_the_latest(rows, expected):
    assert shadow._corrected_label_by_message(rows) == expected


@pytest.mark.parametrize("live,shadow_label,truth,expected", [
    ("a", "a", None, {"agree": 1, "judged": 0, "live_right": 0, "shadow_right": 0}),
    ("a", "b", "b", {"agree": 0, "judged": 1, "live_right": 0, "shadow_right": 1}),
])
def test_shadow_report_direct(live, shadow_label, truth, expected):
    repo = _repo()
    repo.insert("ml_shadow_predictions", {"project_id": "p", "conversation_id": "c", "message_id": "m",
                                          "predictor": "c1", "live_label": live,
                                          "shadow_label": shadow_label, "created_at": NOW})
    if truth is not None:
        repo.insert("ml_route_corrections", {"project_id": "p", "conversation_id": "c", "message_id": "m",
                                             "message": "t", "correct_action": truth, "source": "s",
                                             "created_at": NOW})
    got = shadow.shadow_report(repo, MLLoopSettings(), now=NOW)["predictors"]["c1"]
    assert {k: got[k] for k in expected} == expected
