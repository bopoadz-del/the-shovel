"""ML-0: the ML loop block's interface, store, proposals and shadow report.

Every test builds its own in-memory database and its own rows; every
decision is exercised with two inputs (one each side of the rule).
"""
from __future__ import annotations

import ast
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.blocks.ml_loop import MANIFEST, MLLoopBlock
from app.core.ml_loop.settings import MLLoopSettings
from app.core.ml_loop.store import metadata

REPO = Path(__file__).resolve().parents[1]


def _factory():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _block(factory=None, **kw) -> MLLoopBlock:
    settings = MLLoopSettings(session_factory=factory, **kw)
    return MLLoopBlock(config={"settings": settings})


def _run(block, op, payload=None):
    return asyncio.run(block.run(op, payload))


def _turn(project, verdict, route, n=1, **extra):
    return {
        "project_id": project, "conversation_id": f"c-{project}", "message_id": f"m{n}",
        "question": f"q{n}", "answer": f"a{n}", "verdict": verdict, "route": route,
        "cited_ids": [f"chunk-{n}"], **extra,
    }


# ── interface ────────────────────────────────────────────────────────────

ORDERED_OPERATIONS = (
    "record_turn", "record_feedback", "record_correction", "record_route_correction",
    "propose", "shadow_predict", "shadow_report", "status",
)


@pytest.mark.parametrize("operation", ["record_turn", "shadow_report"])
def test_manifest_lists_each_ordered_operation_with_a_schema(operation):
    assert operation in ORDERED_OPERATIONS
    assert MANIFEST["operations"][operation]["type"] == "object"
    assert MANIFEST["entry_point"] == "process"
    assert (MANIFEST["router_prefix"], MANIFEST["log_prefix"]) == ("/v1/ml", "ml_loop_")
    assert set(ORDERED_OPERATIONS) <= set(MANIFEST["operations"])
    assert MANIFEST["tables"] and all(t.startswith("ml_") for t in MANIFEST["tables"])


def test_block_is_registered_and_certified_against_process():
    from app.blocks import BLOCK_REGISTRY

    assert BLOCK_REGISTRY.get("ml_loop") is MLLoopBlock
    out = asyncio.run(MLLoopBlock().process({"operation": "status"}))
    assert out["status"] == "success" and out["operation"] == "status"
    flat = asyncio.run(_block(_factory()).process(
        {"operation": "record_turn", **_turn("p1", "refused", "rag")}
    ))
    assert flat["stored"] is True


@pytest.mark.parametrize("sha", ["sha-one", "sha-two"])
def test_configure_installs_process_settings_and_instance_settings_win(sha):
    try:
        MLLoopBlock.configure(MLLoopSettings(build_sha=sha))
        assert _run(MLLoopBlock(), "status")["build_sha"] == sha
        assert _run(_block(None, build_sha="own"), "status")["build_sha"] == "own"
    finally:
        MLLoopBlock.configure(None)
    assert _run(MLLoopBlock(), "status")["build_sha"] != sha


@pytest.mark.parametrize("op,payload,needle", [
    ("no_such_op", {}, "unknown operation"),
    ("record_turn", {"project_id": "p1", "verdict": "refused"}, "invalid payload"),
    ("record_feedback", {"project_id": "p1", "kind": "nonsense"}, "invalid payload"),
    ("record_feedback", {"project_id": "p1", "kind": "missing", "note": "x" * 201}, "invalid payload"),
])
def test_bad_calls_fail_with_an_error_and_store_nothing(op, payload, needle):
    block = _block(_factory())
    out = _run(block, op, payload)
    assert out["status"] == "error" and needle in out["error"]
    assert set(_run(block, "status")["row_counts"].values()) == {0}


@pytest.mark.parametrize("op,payload", [
    ("record_turn", _turn("p1", "refused", "rag")),
    ("record_route_correction", {"project_id": "p2", "message": "m", "correct_action": "b"}),
])
def test_without_a_session_factory_writes_report_not_configured(op, payload):
    block = _block(None)
    out = _run(block, op, payload)
    assert out == {"status": "not_configured", "operation": op, "stored": False}
    assert _run(block, "status")["store_configured"] is False


@pytest.mark.parametrize("enabled,expected_status,expected_rows", [
    (False, "disabled", 0),
    (True, "success", 1),
])
def test_disabled_switch_refuses_writes_but_status_still_reads(enabled, expected_status, expected_rows):
    block = _block(_factory(), enabled=enabled)
    assert _run(block, "record_turn", _turn("p1", "refused", "rag"))["status"] == expected_status
    assert _run(block, "status")["row_counts"]["ml_turns"] == expected_rows


# ── store ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("project,override,expected", [
    ("p1", None, ["sha-settings"]),
    ("p2", "sha-override", ["sha-override"]),
])
def test_rows_are_keyed_by_project_and_stamped_with_build_sha(project, override, expected):
    block = _block(_factory(), build_sha="sha-settings")
    _run(block, "record_turn", _turn(project, "refused", "rag", 1, build_sha=override))
    _run(block, "record_turn", _turn("other", "refused", "rag", 2, build_sha="sha-other"))

    assert _run(block, "status", {"project_id": project})["row_counts"]["ml_turns"] == 1
    rows = _run(block, "export_tables", {"project_id": project})["tables"]["ml_turns"]
    assert [r["build_sha"] for r in rows] == expected
    assert [r["conversation_id"] for r in rows] == [f"c-{project}"]


@pytest.mark.parametrize("cap", [10, 25])
def test_free_text_is_capped_by_settings(cap):
    block = _block(_factory(), max_text_chars=cap)
    _run(block, "record_turn", {**_turn("p1", "refused", "rag"), "question": "q" * 50})
    _run(block, "record_route_correction",
         {"project_id": "p1", "message": "m" * 50, "correct_action": "b"})
    tables = _run(block, "export_tables")["tables"]
    assert len(tables["ml_turns"][0]["question"]) == cap
    assert len(tables["ml_route_corrections"][0]["message"]) == cap


@pytest.mark.parametrize("formulas,expected", [
    (["f_a", "f_a", "f_b"], [1, 2, 1]),
    (["f_b", "f_c", "f_b", "f_b"], [1, 1, 2, 3]),
])
def test_record_correction_counts_samples_per_formula(formulas, expected):
    block = _block(_factory())
    counts = [
        _run(block, "record_correction", {
            "project_id": f"p{i % 2}", "formula_id": fid, "predicted": 10 + i, "actual": 11,
        })["sample_count"]
        for i, fid in enumerate(formulas)
    ]
    assert counts == expected


@pytest.mark.parametrize("project", ["p1", "p2"])
def test_export_import_round_trip_into_a_fresh_store(project):
    src = _block(_factory())
    _run(src, "record_turn", _turn(project, "refused", "rag"))
    _run(src, "record_feedback", {"project_id": project, "message_id": "m1", "kind": "wrong_figure",
                                  "correct_value": 12.5, "unit": "m3", "cited_ids": ["chunk-1"]})
    _run(src, "shadow_predict", {"project_id": project, "predictor": "cand",
                                 "live_label": "a", "shadow_label": "b"})
    exported = _run(src, "export_tables", {"project_id": project})["tables"]

    dst = _block(_factory())
    out = _run(dst, "import_tables", {"tables": exported})
    assert (out["imported"]["ml_turns"], out["imported"]["ml_feedback"]) == (1, 1)
    again = _run(dst, "export_tables")["tables"]

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "id"} for r in rows]

    for name in exported:
        assert strip(again[name]) == strip(exported[name])


@pytest.mark.parametrize("tables,needle", [
    ({"host_table": []}, "unknown table"),
    ({"ml_turns": [{"project_id": "p", "secret": 1}]}, "unknown column"),
])
def test_import_refuses_foreign_tables_and_columns(tables, needle):
    block = _block(_factory())
    out = _run(block, "import_tables", {"tables": tables})
    assert out["status"] == "error" and needle in out["error"]


# ── propose (read-only) ──────────────────────────────────────────────────

def _route_fix(block, project, original, correct, n):
    for i in range(n):
        _run(block, "record_route_correction", {
            "project_id": project, "message_id": f"{original}-{correct}-{i}",
            "message": f"private user text {i}", "original_action": original,
            "correct_action": correct,
        })


@pytest.mark.parametrize("support,expected_rules", [
    (3, [("act_x", "act_y", 3)]),
    (2, [("act_x", "act_y", 3), ("act_x", "act_z", 2)]),
])
def test_propose_emits_a_route_rule_only_with_enough_support(support, expected_rules):
    block = _block(_factory(), min_proposal_support=support)
    _route_fix(block, "p1", "act_x", "act_y", 3)
    _route_fix(block, "p2", "act_x", "act_z", 2)
    _route_fix(block, "p2", "act_w", "act_w", 4)  # a no-op correction never proposes
    before = _run(block, "status")["row_counts"]

    out = _run(block, "propose")
    rules = [p for p in out["proposals"] if p["kind"] == "route_rule"]
    assert [(r["from_action"], r["to_action"], r["support"]) for r in rules] == expected_rules
    assert "private user text" not in repr(out)
    assert _run(block, "status")["row_counts"] == before  # read-only


@pytest.mark.parametrize("rate,expected", [
    (0.5, [("route_a", 0.75, 3)]),
    (0.2, [("route_a", 0.75, 3), ("route_b", 0.25, 1)]),
])
def test_propose_flags_a_route_for_retrieval_review_by_refusal_rate(rate, expected):
    block = _block(_factory(), min_proposal_support=3, refusal_review_rate=rate)
    for i, verdict in enumerate(["refused", "refused", "refused", "project_document"]):
        _run(block, "record_turn", _turn("p1", verdict, "route_a", i))
    for i, verdict in enumerate(["refused", "project_document", "project_document", "project_document"]):
        _run(block, "record_turn", _turn("p2", verdict, "route_b", 10 + i))

    out = _run(block, "propose")
    reviews = [p for p in out["proposals"] if p["kind"] == "retrieval_review"]
    assert [(p["route"], p["refused_rate"], len(p["evidence"]["ids"])) for p in reviews] == expected
    assert _run(block, "propose", {"project_id": "p2"})["turns"] == 4


@pytest.mark.parametrize("window,expected", [
    (7, {"c->d": 1}),
    (60, {"a->b": 1, "c->d": 1}),
])
def test_propose_window_excludes_old_rows(window, expected):
    block = _block(_factory(), min_proposal_support=1)
    old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    _run(block, "import_tables", {"tables": {"ml_route_corrections": [{
        "project_id": "p1", "message": "m", "correct_action": "b", "original_action": "a",
        "source": "user_correction", "created_at": old,
    }]}})
    _route_fix(block, "p1", "c", "d", 1)
    assert _run(block, "propose", {"window_days": window})["route_corrections"] == expected


# ── shadow ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pairs,truths,expected", [
    ([("a", "a"), ("a", "b"), ("c", "b"), ("d", "e")], {"m1": "b", "m2": "c"},
     {"total": 4, "agree": 1, "judged": 2, "live_right": 1, "shadow_right": 1, "agreement_rate": 0.25}),
    ([("a", "a"), ("b", "b"), ("c", "x")], {"m2": "x"},
     {"total": 3, "agree": 2, "judged": 1, "live_right": 0, "shadow_right": 1, "agreement_rate": 0.6667}),
])
def test_shadow_predict_never_acts_and_report_scores_against_corrections(pairs, truths, expected):
    block = _block(_factory())
    for i, (live, shadow) in enumerate(pairs):
        out = _run(block, "shadow_predict", {
            "project_id": "p1", "conversation_id": "c1", "message_id": f"m{i}",
            "predictor": "cand_1", "live_label": live, "shadow_label": shadow,
        })
        assert (out["acted_on"], out["decision"]) == (False, live)
    _run(block, "shadow_predict", {"project_id": "p2", "predictor": "cand_2",
                                   "live_label": "x", "shadow_label": "y"})
    for mid, truth in truths.items():
        _run(block, "record_route_correction", {
            "project_id": "p1", "conversation_id": "c1", "message_id": mid,
            "message": "t", "correct_action": truth,
        })

    assert _run(block, "shadow_report")["predictors"]["cand_1"] == expected
    only = _run(block, "shadow_report", {"predictor": "cand_2"})["predictors"]
    assert list(only) == ["cand_2"] and only["cand_2"]["agreement_rate"] == 0.0


# ── isolation (Store readiness) ──────────────────────────────────────────

def _imports(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_block_code_imports_nothing_from_the_host_but_its_base_class():
    files = [REPO / "app/blocks/ml_loop.py", *(REPO / "app/core/ml_loop").glob("*.py")]
    allowed_host = {"app.core.universal_base"}
    for path in files:
        host = {m for m in _imports(path) if m.startswith("app.")}
        stray = {m for m in host if not m.startswith("app.core.ml_loop")} - allowed_host
        assert not stray, (path.name, stray)


def test_answer_path_does_not_import_ml_loop_internals():
    answer_path = (
        "app/agents/runtime.py", "app/agents/core/loop.py",
        "app/agents/core/streaming.py", "app/agents/driver/loop.py",
    )
    for rel in answer_path:
        path = REPO / rel
        assert path.exists(), f"answer-path module moved: {rel}"
        assert not any(m.startswith("app.core.ml_loop") for m in _imports(path)), rel


@pytest.mark.parametrize("operation", ["status", "not_an_operation"])
def test_perform_refuses_an_operation_without_a_handler(operation):
    with pytest.raises(ValueError, match="no handler"):
        MLLoopBlock()._perform(operation, None, None, None)
