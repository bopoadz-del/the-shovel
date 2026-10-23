"""Tests for the W4 feedback surface — POST /v1/feedback/route.

The contract this surface exists to honor:

1. A correction lands as a routing_decisions pattern on learning_engine,
   tagged corrected=True, source=user_correction.
2. The next train_router call with prefer_corrected=True picks it up
   and weights "auto" rows out of the training set.

These tests prove both ends of that contract — the route writes what
it claims to write, and the router can consume what the route writes.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from tests.conftest import construction_kit_markers
from tests._dev_key import DEV_KEY

pytestmark = construction_kit_markers


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    """Same isolation pattern as test_router_ml/test_hydration."""
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


_AUTH = {"Authorization": "Bearer " + DEV_KEY}


def test_feedback_route_writes_corrected_pattern(isolated_data_dir):
    """Happy path: POST → pattern row appears in learning_engine state with
    corrected=True and source=user_correction."""
    from app.main import app
    from app.blocks import BLOCK_REGISTRY

    with TestClient(app) as client:
        r = client.post(
            "/v1/feedback/route",
            json={
                "message": "how much will this project cost",
                "correct_action": "estimate_costs",
                "project_id": "proj_fb",
                "original_action": "intelligent_workflow",
            },
            headers=_AUTH,
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "recorded"
        assert body["pattern_count"] == 1

    # Verify the row landed where the router will read it
    le = BLOCK_REGISTRY["learning_engine"]()
    patterns = le._state["patterns"]["proj_fb"]["routing_decisions"]
    assert len(patterns) == 1
    obs = json.loads(patterns[0]["observation"])
    assert obs["corrected"] is True
    assert obs["action"] == "estimate_costs"
    assert obs["original_action"] == "intelligent_workflow"
    assert obs["source"] == "user_correction"
    assert obs["score"] == 1.0  # corrections are ground truth, full weight


def test_feedback_route_optional_original_action(isolated_data_dir):
    """original_action is optional — callers that don't know what was
    originally picked can still submit a correction."""
    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/v1/feedback/route",
            json={
                "message": "any message",
                "correct_action": "spec_analyze",
                "project_id": "proj_fb",
            },
            headers=_AUTH,
        )
        assert r.status_code == 200, r.text


def test_feedback_route_validates_required_fields(isolated_data_dir):
    """Pydantic enforces non-empty message and correct_action."""
    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/v1/feedback/route",
            json={"message": "", "correct_action": "x", "project_id": "p"},
            headers=_AUTH,
        )
        assert r.status_code == 422
        r2 = client.post(
            "/v1/feedback/route",
            json={"message": "x", "correct_action": "", "project_id": "p"},
            headers=_AUTH,
        )
        assert r2.status_code == 422


def test_feedback_route_requires_auth(isolated_data_dir):
    from app.main import app

    with TestClient(app) as client:
        r = client.post(
            "/v1/feedback/route",
            json={"message": "x", "correct_action": "y", "project_id": "p"},
        )
        assert r.status_code == 401


def test_corrections_appear_with_label_quality_corrected(isolated_data_dir):
    """The contract between this route and the router: the row's
    label_quality flag flips to "corrected" because of the corrected=True
    blob field. This is what the prefer_corrected=True training mode reads."""
    from app.main import app
    from app.core.learning.router import _runtime_data_from_patterns

    with TestClient(app) as client:
        client.post(
            "/v1/feedback/route",
            json={
                "message": "show the rebar weights",
                "correct_action": "extract_quantities",
                "project_id": "proj_fb",
            },
            headers=_AUTH,
        )

    rows = _runtime_data_from_patterns()
    corrected = [r for r in rows if r.label_quality == "corrected"]
    assert len(corrected) == 1
    assert corrected[0].text == "show the rebar weights"
    assert corrected[0].label == "extract_quantities"
    assert corrected[0].source == "correction"


def test_multiple_corrections_accumulate(isolated_data_dir):
    """Three corrections in a row → three rows. Pattern_count grows
    monotonically; the route does not deduplicate (a duplicate is also
    a signal — same correction twice means high confidence)."""
    from app.main import app

    with TestClient(app) as client:
        for i in range(3):
            r = client.post(
                "/v1/feedback/route",
                json={
                    "message": f"message {i}",
                    "correct_action": "estimate_costs",
                    "project_id": "proj_fb",
                },
                headers=_AUTH,
            )
            assert r.status_code == 200
            # pattern_count grows with each submission
            assert r.json()["pattern_count"] == i + 1


def test_train_router_prefer_corrected_uses_only_corrections(isolated_data_dir):
    """The endgame this whole surface exists to enable: once enough
    corrections accumulate, train_router(prefer_corrected=True) drops
    the noisy seed+runtime "auto" rows and trains purely on user-validated
    data.

    Seeds corrections across multiple actions (sklearn needs ≥2 classes)
    and confirms the trained model's label distribution contains ONLY
    corrected actions — proving "auto" rows from ACTION_PATTERNS were
    excluded.
    """
    from app.main import app
    from app.core.learning.router import train, _MIN_TOTAL_SAMPLES

    # Spread corrections across two distinct actions so LR can fit
    actions = ["estimate_costs", "extract_quantities"]
    with TestClient(app) as client:
        for i in range(_MIN_TOTAL_SAMPLES + 10):
            client.post(
                "/v1/feedback/route",
                json={
                    "message": f"correction {i} for action {actions[i % 2]}",
                    "correct_action": actions[i % 2],
                    "project_id": "proj_fb",
                },
                headers=_AUTH,
            )

    result = train(prefer_corrected=True)
    assert result["status"] == "success"
    # Only corrected actions in the label distribution — seed rows for
    # all OTHER actions (boq_process, drawing_qto, etc.) were dropped.
    assert set(result["label_distribution"].keys()) == set(actions), (
        f"prefer_corrected should drop non-corrected actions; got: "
        f"{sorted(result['label_distribution'].keys())}"
    )


def test_train_router_handles_single_class_gracefully(isolated_data_dir):
    """When all corrections land on one action AND prefer_corrected drops
    everything else, train_router must NOT crash with sklearn's "needs ≥2
    classes" traceback. Returns insufficient_classes with a clear remediation
    message instead."""
    from app.main import app
    from app.core.learning.router import train, _MIN_TOTAL_SAMPLES

    with TestClient(app) as client:
        for i in range(_MIN_TOTAL_SAMPLES + 5):
            client.post(
                "/v1/feedback/route",
                json={
                    "message": f"correction {i}",
                    "correct_action": "estimate_costs",
                    "project_id": "proj_fb",
                },
                headers=_AUTH,
            )

    result = train(prefer_corrected=True)
    assert result["status"] == "insufficient_classes"
    assert result["single_class"] == "estimate_costs"
    assert "2 classes" in result["remediation"]
