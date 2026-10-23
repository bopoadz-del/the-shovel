"""The learning loop's WRITE side must actually be connected.

`learning_capture` opens by calling itself "the wiring" for a learning engine
that "NO runtime path ever fed". It was written, tested, gated behind
FORK_LEARNING_CAPTURE — and never called. Setting the flag in production would
have recorded nothing, silently, which is the worst shape for this particular
feature: the whole value is that the dataset accumulates quietly from day one,
so an empty dataset is not discovered until the day someone needs it.

These tests pin two things: that a completed container action reaches the
capture path, and that the READ side stays disconnected until it can state its
provenance.
"""
from __future__ import annotations

import pytest

from app.core import learning_capture

# ── the flag ─────────────────────────────────────────────────────────────

def test_capture_is_off_unless_the_flag_is_set(monkeypatch):
    monkeypatch.delenv("FORK_LEARNING_CAPTURE", raising=False)
    assert learning_capture.enabled() is False
    assert learning_capture.capture_from_action(
        "qa_qc_inspection", {"defects": [{"trade": "electrical"}]}
    ) == 0


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", "On"])
def test_the_flag_accepts_the_usual_spellings(monkeypatch, value):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", value)
    assert learning_capture.enabled() is True


# ── the dispatcher ───────────────────────────────────────────────────────

def test_a_known_action_reaches_its_capture_function(monkeypatch):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    seen = {}

    def _fake(result, project_id=""):
        seen["result"] = result
        seen["project_id"] = project_id
        return 3

    monkeypatch.setitem(
        learning_capture._CAPTURE_BY_ACTION, "qa_qc_inspection", _fake
    )
    n = learning_capture.capture_from_action(
        "qa_qc_inspection", {"defects": [{}]}, {"project_id": "p1"}
    )
    assert n == 3
    assert seen["project_id"] == "p1"


def test_an_unknown_action_is_a_no_op(monkeypatch):
    """Adding a container action must never silently start writing learning
    data whose shape no capture function understands."""
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    assert learning_capture.capture_from_action(
        "generate_carbon_report", {"anything": True}, {}
    ) == 0


def test_a_failed_action_is_not_recorded(monkeypatch):
    """A failed inspection produced no observation. Recording it would poison
    the defect statistics with the error path."""
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    assert learning_capture.capture_from_action(
        "qa_qc_inspection", {"status": "error", "error": "boom"}, {}
    ) == 0


def test_capture_never_raises_into_the_action(monkeypatch):
    """The action's own result must survive a broken learner."""
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")

    def _explode(result, project_id=""):
        raise RuntimeError("learner is broken")

    monkeypatch.setitem(
        learning_capture._CAPTURE_BY_ACTION, "qa_qc_inspection", _explode
    )
    assert learning_capture.capture_from_action(
        "qa_qc_inspection", {"defects": [{}]}, {}
    ) == 0


def test_project_id_falls_back_to_the_result(monkeypatch):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    seen = {}

    def _fake(result, project_id=""):
        seen["project_id"] = project_id
        return 1

    monkeypatch.setitem(
        learning_capture._CAPTURE_BY_ACTION, "qa_qc_inspection", _fake
    )
    learning_capture.capture_from_action(
        "qa_qc_inspection", {"defects": [{}], "project_id": "from-result"}, {}
    )
    assert seen["project_id"] == "from-result"


# ── the wiring itself ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_completed_container_action_reaches_capture(monkeypatch):
    """The regression that matters. Everything above can pass while the module
    has no call site at all — which is exactly the state it shipped in."""
    from app.containers.construction import ConstructionContainer

    calls = []
    monkeypatch.setattr(
        learning_capture,
        "capture_from_action",
        lambda action, result, params=None: calls.append((action, result)) or 0,
    )

    container = ConstructionContainer()

    async def _fake_inspection(input_data, params):
        return {"status": "success", "defects": [{"trade": "electrical"}]}

    monkeypatch.setattr(container, "qa_qc_inspection", _fake_inspection)

    result = await container.route("qa_qc_inspection", {}, {"project_id": "p1"})

    assert result["status"] == "success", result
    assert calls, "the container action never reached learning capture"
    assert calls[0][0] == "qa_qc_inspection"


@pytest.mark.asyncio
async def test_a_broken_learner_does_not_break_the_action(monkeypatch):
    from app.containers.construction import ConstructionContainer

    def _explode(action, result, params=None):
        raise RuntimeError("learning is down")

    monkeypatch.setattr(learning_capture, "capture_from_action", _explode)

    container = ConstructionContainer()

    async def _fake_inspection(input_data, params):
        return {"status": "success", "defects": []}

    monkeypatch.setattr(container, "qa_qc_inspection", _fake_inspection)

    result = await container.route("qa_qc_inspection", {}, {})
    assert result["status"] == "success", result


# ── the read side stays deferred ─────────────────────────────────────────

def test_no_runtime_path_consumes_learned_values_yet():
    """Deliberate, and the reason capture can ship immediately: nothing reads
    these numbers back, so switching capture on cannot change a single answer.

    When the read side does land it must carry provenance in the answer —
    "calibrated from 14 actuals on this project; default was 21 days" — never a
    silently shifted number. This test failing means that day has arrived, and
    the provenance requirement needs checking before it is deleted.
    """
    import pathlib
    import re

    readers = re.compile(
        r"get_calibrated_duration|get_lead_time_estimate|get_best_templates"
        r"|get_defect_rankings|get_template_recommendation"
    )
    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        if path.name in ("construction_learning.py", "learning_capture.py"):
            continue
        if readers.search(path.read_text(encoding="utf-8", errors="replace")):
            offenders.append(str(path.relative_to(app_dir.parent)))
    assert not offenders, (
        "the learning READ side is now connected in: "
        + ", ".join(offenders)
        + " — confirm each states its provenance in the answer"
    )
