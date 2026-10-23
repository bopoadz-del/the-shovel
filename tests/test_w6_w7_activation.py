"""W6/W7 — live activation, gated + noop-when-off.

W6: qa_qc_inspection defects + schedule delays feed the learning engine's
recorders when FORK_LEARNING_CAPTURE is on; nothing is recorded when off.
W7: dormant delegator agents get can_delegate ONLY when FORK_AGENT_DELEGATION is
on; explicit can_delegate always wins; default is unchanged.
"""
import pytest

from app.core import learning_capture as lc


class _FakeEngine:
    def __init__(self):
        self.defects = []
        self.durations = []

    def record_defect(self, **kw):
        self.defects.append(kw)
        return {"ok": True}

    def record_duration(self, **kw):
        self.durations.append(kw)
        return {"ok": True}


_QA_RESULT = {
    "inspection_type": "concrete",
    "defects": [
        {"description": "Crack visible", "severity": "major", "trade": "concrete"},
        {"description": "Honeycombing", "severity": "minor"},
    ],
}
_SCHED_RESULT = {"delay_analysis": {"activities": [
    {"name": "Excavation", "activity_type": "earthworks",
     "planned_duration": 10, "actual_duration": 14},
    {"name": "No actuals", "planned_duration": 5},   # dropped: no actual
]}}


# ── W6: learning capture ───────────────────────────────────────────────────

def test_capture_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("FORK_LEARNING_CAPTURE", raising=False)
    fake = _FakeEngine()
    monkeypatch.setattr(lc, "_engine", lambda: fake)
    assert lc.capture_qa_defects(_QA_RESULT, "p1") == 0
    assert lc.capture_schedule_durations(_SCHED_RESULT, "p1") == 0
    assert fake.defects == [] and fake.durations == []


def test_capture_qa_defects_when_on(monkeypatch):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    fake = _FakeEngine()
    monkeypatch.setattr(lc, "_engine", lambda: fake)
    n = lc.capture_qa_defects(_QA_RESULT, "p1")
    assert n == 2
    assert len(fake.defects) == 2
    assert fake.defects[0]["activity_type"] == "concrete"
    assert fake.defects[0]["severity"] == "major"
    assert fake.defects[0]["project_id"] == "p1"


def test_capture_schedule_durations_when_on(monkeypatch):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    fake = _FakeEngine()
    monkeypatch.setattr(lc, "_engine", lambda: fake)
    n = lc.capture_schedule_durations(_SCHED_RESULT, "p1")
    assert n == 1                                   # the no-actual activity is dropped
    assert fake.durations[0]["planned_days"] == 10
    assert fake.durations[0]["actual_days"] == 14


def test_capture_never_raises_on_bad_data(monkeypatch):
    monkeypatch.setenv("FORK_LEARNING_CAPTURE", "1")
    fake = _FakeEngine()
    monkeypatch.setattr(lc, "_engine", lambda: fake)
    assert lc.capture_qa_defects({"defects": "not-a-list"}, "p1") == 0
    assert lc.capture_qa_defects(None, "p1") == 0


# ── W7: agent delegation gate ──────────────────────────────────────────────

def test_explicit_can_delegate_always_wins(monkeypatch):
    from app.agents import runtime
    monkeypatch.delenv("FORK_AGENT_DELEGATION", raising=False)
    assert runtime._resolve_can_delegate({"can_delegate": "true"}) is True


def test_dormant_delegator_off_by_default(monkeypatch):
    from app.agents import runtime
    monkeypatch.delenv("FORK_AGENT_DELEGATION", raising=False)
    # can_delegate_when_enabled but flag off -> stays dormant (unchanged pilot).
    assert runtime._resolve_can_delegate({"can_delegate_when_enabled": "true"}) is False


def test_dormant_delegator_lights_up_when_flag_on(monkeypatch):
    from app.agents import runtime
    monkeypatch.setenv("FORK_AGENT_DELEGATION", "1")
    assert runtime._resolve_can_delegate({"can_delegate_when_enabled": "true"}) is True
    # a plain agent with neither field never delegates
    assert runtime._resolve_can_delegate({}) is False


def test_delegator_configs_declare_the_flag():
    """The 4 dormant delegators carry the flag-gated field so the master switch
    actually lights them up."""
    import pathlib
    cfg = pathlib.Path("app/agents/configs")
    for name in ("construction-pm", "document-analyst", "document-ingestion", "quantity-surveyor"):
        text = (cfg / f"{name}.md").read_text(encoding="utf-8")
        assert "can_delegate_when_enabled: true" in text, name
