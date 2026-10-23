"""Tests for the auto-retrain hook in the hydration scheduler.

Contract:
- After each hydration pass, if new routing_decisions have accumulated
  since the last train_router run, retrain.
- Skip silently when nothing new has landed.
- Use prefer_corrected=True once corrected rows exceed the threshold.
- Failures in the retrain are logged but never abort the scheduler.
"""

from __future__ import annotations

import json

import pytest

from tests.conftest import construction_kit_markers

pytestmark = construction_kit_markers


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Standard isolation pattern — fresh DATA_DIR + learning_engine state
    plus module-level init flag resets."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEARNING_ENGINE_STORAGE", str(tmp_path / "le_state.json"))
    monkeypatch.setenv("LEARNING_ENGINE_ENABLED", "1")
    from app.core import agent_memory as _am
    from app.core import projects as _proj
    from app.core.learning import router as _router

    if hasattr(_am, "_initialized"):
        _am._initialized = False
    if hasattr(_proj, "_initialized"):
        _proj._initialized = False
    _router.invalidate_model_cache()
    yield tmp_path
    _router.invalidate_model_cache()


def _seed_routing_pattern(le, project_id: str, action: str, corrected: bool = False) -> None:
    le._record_pattern({
        "project_id": project_id,
        "category": "routing_decisions",
        "observation": json.dumps({
            "text": f"sample message for {action}",
            "action": action,
            "score": 1.0,
            "corrected": corrected,
        }),
        "source": "feedback_route",
    }, {})


@pytest.mark.asyncio
async def test_maybe_retrain_skips_when_no_new_patterns(isolated_data_dir):
    """No routing_decisions accumulated → retrain skipped, no model written."""
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router

    block = BLOCK_REGISTRY["learning_engine"]()
    # No routing_decisions at all
    await _maybe_retrain_router(block)
    assert "router" not in block._state.get("models", {})


@pytest.mark.asyncio
async def test_maybe_retrain_triggers_on_first_pattern(isolated_data_dir):
    """With at least one new routing_decision, retrain runs and stamps the
    snapshot count so subsequent passes can skip."""
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router

    block = BLOCK_REGISTRY["learning_engine"]()
    _seed_routing_pattern(block, "proj_a", "estimate_costs")

    await _maybe_retrain_router(block)

    router_meta = block._state.get("models", {}).get("router")
    assert router_meta is not None, "auto-retrain should have written model metadata"
    assert router_meta.get("patterns_seen_at_train") == 1


@pytest.mark.asyncio
async def test_maybe_retrain_skips_second_pass_with_no_new_patterns(isolated_data_dir):
    """First pass trains; second pass with identical pattern count skips."""
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router

    block = BLOCK_REGISTRY["learning_engine"]()
    _seed_routing_pattern(block, "proj_a", "estimate_costs")

    await _maybe_retrain_router(block)
    first_trained_at = block._state["models"]["router"]["trained_at"]

    # No new patterns added — second call should not retrain
    await _maybe_retrain_router(block)
    assert block._state["models"]["router"]["trained_at"] == first_trained_at, (
        "Identical pattern count should have skipped the retrain"
    )


@pytest.mark.asyncio
async def test_maybe_retrain_runs_again_when_patterns_grow(isolated_data_dir):
    """After a train, adding another pattern triggers a new retrain."""
    import time
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router

    block = BLOCK_REGISTRY["learning_engine"]()
    _seed_routing_pattern(block, "proj_a", "estimate_costs")
    await _maybe_retrain_router(block)
    first_trained_at = block._state["models"]["router"]["trained_at"]

    # Ensure time advances at least a tick — joblib's mtime resolution
    # otherwise makes trained_at look identical on fast machines.
    time.sleep(0.01)

    # Add a new pattern — should trigger retrain
    _seed_routing_pattern(block, "proj_a", "spec_analyze")
    await _maybe_retrain_router(block)

    second_trained_at = block._state["models"]["router"]["trained_at"]
    assert second_trained_at > first_trained_at, (
        "New pattern should have triggered a retrain"
    )
    assert block._state["models"]["router"]["patterns_seen_at_train"] == 2


@pytest.mark.asyncio
async def test_maybe_retrain_uses_prefer_corrected_above_threshold(isolated_data_dir):
    """When corrected rows exceed _MIN_TOTAL_SAMPLES, the retrain flips
    prefer_corrected on so the noisy "auto" rows get dropped."""
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router
    from app.core.learning.router import _MIN_TOTAL_SAMPLES

    block = BLOCK_REGISTRY["learning_engine"]()
    # Seed corrections across two classes (need ≥2 for LR to fit)
    for i in range(_MIN_TOTAL_SAMPLES + 5):
        _seed_routing_pattern(
            block, "proj_a",
            ("estimate_costs" if i % 2 == 0 else "extract_quantities"),
            corrected=True,
        )

    await _maybe_retrain_router(block)

    router_meta = block._state["models"]["router"]
    assert router_meta["prefer_corrected"] is True, (
        f"Above-threshold corrections should flip prefer_corrected on; "
        f"meta={router_meta.get('prefer_corrected')}"
    )
    # Only the corrected actions should be in the label distribution
    assert set(router_meta["label_distribution"].keys()).issubset(
        {"estimate_costs", "extract_quantities"}
    )


@pytest.mark.asyncio
async def test_maybe_retrain_handles_insufficient_classes(isolated_data_dir, caplog):
    """When prefer_corrected drops everything to one class, train_router
    returns insufficient_classes. The hook must surface this (warn log)
    without crashing the scheduler loop."""
    import logging
    from app.blocks import BLOCK_REGISTRY
    from app.core.hydration_scheduler import _maybe_retrain_router
    from app.core.learning.router import _MIN_TOTAL_SAMPLES

    block = BLOCK_REGISTRY["learning_engine"]()
    # All corrections go to ONE action — single-class scenario
    for i in range(_MIN_TOTAL_SAMPLES + 5):
        _seed_routing_pattern(block, "proj_a", "estimate_costs", corrected=True)

    with caplog.at_level(logging.WARNING):
        await _maybe_retrain_router(block)  # Must not raise

    # The warn log captures the non-success outcome
    assert any(
        "non-success" in rec.message or "insufficient" in rec.message.lower()
        for rec in caplog.records
    )


@pytest.mark.asyncio
async def test_run_one_pass_calls_auto_retrain(isolated_data_dir, monkeypatch):
    """End-to-end: _run_one_pass calls both hydrate AND the retrain hook.
    The previous hydration logic must still run; the retrain is additive."""
    from app.core.hydration_scheduler import _run_one_pass
    from app.blocks import BLOCK_REGISTRY

    # Stub the chat call inside hydration so it doesn't try real LLM
    from app.core.learning import hydration as _h
    async def fake_chat(prompt, max_tokens=600):
        return ("## ok", "offline_template")
    monkeypatch.setattr(_h, "_call_chat", fake_chat)

    block = BLOCK_REGISTRY["learning_engine"]()
    _seed_routing_pattern(block, "proj_a", "estimate_costs")

    await _run_one_pass()

    # Auto-retrain should have stamped the snapshot count
    fresh = BLOCK_REGISTRY["learning_engine"]()
    assert fresh._state.get("models", {}).get("router", {}).get(
        "patterns_seen_at_train"
    ) == 1


@pytest.mark.asyncio
async def test_run_one_pass_survives_retrain_failure(isolated_data_dir, monkeypatch):
    """If the retrain raises, _run_one_pass logs and continues — the
    scheduler must not die because of a router issue."""
    from app.core import hydration_scheduler as _sched
    from app.core.learning import hydration as _h

    async def fake_chat(prompt, max_tokens=600):
        return ("## ok", "offline_template")
    monkeypatch.setattr(_h, "_call_chat", fake_chat)

    async def explode(block):
        raise RuntimeError("simulated retrain failure")
    monkeypatch.setattr(_sched, "_maybe_retrain_router", explode)

    # Must complete without raising
    await _sched._run_one_pass()
