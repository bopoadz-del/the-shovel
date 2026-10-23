"""Learning engine: hard-off local JSON + independently labeled trainable rows.

routing_mode stays auto. Own dispatch is never a training label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.blocks.learning_engine import (
    LearningEngineBlock,
    assert_learning_engine_hard_off,
    learning_engine_writes_enabled,
)
from app.core.learning.trainable import (
    INDEPENDENT_LABEL_SOURCES,
    is_independently_labeled,
)
from tests.conftest import requires_construction_kit


def test_hard_off_is_the_default(monkeypatch):
    monkeypatch.delenv("LEARNING_ENGINE_ENABLED", raising=False)
    assert learning_engine_writes_enabled() is False


def test_startup_assertion_refuses_a_local_json_write(tmp_path, monkeypatch):
    monkeypatch.delenv("LEARNING_ENGINE_ENABLED", raising=False)
    probe = tmp_path / "must_not_write.json"
    assert_learning_engine_hard_off(str(probe))
    assert not probe.exists()


def test_save_state_does_not_create_json_when_hard_off(tmp_path, monkeypatch):
    monkeypatch.delenv("LEARNING_ENGINE_ENABLED", raising=False)
    path = tmp_path / "le.json"
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(path))
    LearningEngineBlock.reset_shared_instance_cache()
    le = LearningEngineBlock()
    le._state = {"formulas": {"x": 1}, "history": [], "patterns": {}}
    le._save_state()
    assert not path.exists()


@pytest.mark.asyncio
async def test_own_dispatch_is_not_trainable(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    LearningEngineBlock.reset_shared_instance_cache()
    le = LearningEngineBlock()
    result = le._record_pattern(
        {
            "project_id": "p",
            "category": "routing_decisions",
            "source": "smart_orchestrator",
            "observation": json.dumps({
                "text": "extract BOQ totals from the cost sheet",
                "action": "boq_process",
                "score": 0.9,
                "source": "learned",
                "corrected": False,
            }),
        },
        {},
    )
    assert result["trainable"] is False
    row = le._state["patterns"]["p"]["routing_decisions"][0]
    assert row["trainable"] is False
    obs = json.loads(row["observation"])
    assert obs["trainable"] is False


@pytest.mark.asyncio
async def test_feedback_correction_is_trainable(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    LearningEngineBlock.reset_shared_instance_cache()
    le = LearningEngineBlock()
    result = le._record_pattern(
        {
            "project_id": "p",
            "category": "routing_decisions",
            "source": "feedback_route",
            "observation": json.dumps({
                "text": "how much will this project cost",
                "action": "estimate_costs",
                "score": 1.0,
                "source": "user_correction",
                "corrected": True,
            }),
        },
        {},
    )
    assert result["trainable"] is True
    obs = json.loads(le._state["patterns"]["p"]["routing_decisions"][0]["observation"])
    assert obs["trainable"] is True


@pytest.mark.asyncio
async def test_battery_grade_label_is_trainable(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    LearningEngineBlock.reset_shared_instance_cache()
    le = LearningEngineBlock()
    result = le._record_pattern(
        {
            "project_id": "p",
            "category": "routing_decisions",
            "source": "battery_grade",
            "observation": json.dumps({
                "text": "What does Schedule 10 of the contract contain?",
                "action": "project_assistant",
                "score": 1.0,
                "source": "battery_grade",
            }),
        },
        {},
    )
    assert result["trainable"] is True


def test_corrected_flag_on_own_dispatch_is_not_enough():
    assert is_independently_labeled(
        "smart_orchestrator",
        {"source": "learned", "corrected": True},
    ) is False


def test_independent_sources_are_the_named_ones():
    assert "battery_grade" in INDEPENDENT_LABEL_SOURCES
    assert "feedback_route" in INDEPENDENT_LABEL_SOURCES
    assert "user_correction" in INDEPENDENT_LABEL_SOURCES


def test_routing_mode_default_is_auto():
    from app.blocks.smart_orchestrator import SmartOrchestratorBlock

    so = SmartOrchestratorBlock()
    assert so.config.get("routing_mode", "auto") == "auto"


@requires_construction_kit
@pytest.mark.asyncio
async def test_auto_mode_does_not_record_dispatch_as_trainable(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    LearningEngineBlock.reset_shared_instance_cache()
    from app.blocks import BLOCK_REGISTRY

    so = BLOCK_REGISTRY["smart_orchestrator"]()
    await so.execute({"text": "extract BOQ totals from the cost sheet"}, {})
    le = LearningEngineBlock()
    decisions = []
    for _pid, by_cat in le._state.get("patterns", {}).items():
        decisions.extend(by_cat.get("routing_decisions", []))
    assert decisions == [] or all(not r.get("trainable") for r in decisions)


def test_runtime_training_rows_skip_own_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le.json"))
    LearningEngineBlock.reset_shared_instance_cache()
    le = LearningEngineBlock.shared_instance()
    le._record_pattern(
        {
            "project_id": "p",
            "category": "routing_decisions",
            "source": "smart_orchestrator",
            "observation": json.dumps({
                "text": "own dispatch sample",
                "action": "boq_process",
                "source": "learned",
            }),
        },
        {},
    )
    le._record_pattern(
        {
            "project_id": "p",
            "category": "routing_decisions",
            "source": "feedback_route",
            "observation": json.dumps({
                "text": "feedback sample should train",
                "action": "estimate_costs",
                "source": "user_correction",
                "corrected": True,
            }),
        },
        {},
    )
    from app.core.learning.router import _runtime_data_from_patterns

    rows = _runtime_data_from_patterns()
    texts = [r.text for r in rows]
    assert "own dispatch sample" not in texts
    assert "feedback sample should train" in texts


def test_env_example_keeps_the_engine_hard_off():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")
    assert "LEARNING_ENGINE_ENABLED=false" in text
