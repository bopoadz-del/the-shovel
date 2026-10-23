"""End-to-end checks against REAL photographs, when they are present locally.

The synthetic tests in test_photo_observations.py pin the contract but cannot
tell whether the model is right about an image. These run the actual ONNX over
actual construction photographs.

They SKIP (never fail-open) when the fixtures or the weights are absent, so CI
needs no network and no 50 MB model. Fetch them with:

    python scripts/fetch_eval_photos.py

Measured behaviour for these specific images is recorded in
docs/PHOTO_INTELLIGENCE_EVAL.md. Deliberately, the assertions here do NOT pin
per-class confidences: those move with any re-bake, and a test that fails
whenever the model improves is a test people delete. What is pinned is the
contract — shape, tiering, no promotion, honest empties.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from app.containers.construction.photo_observations import (
    LOW_CONF_THRESHOLD,
    equipment_from_photos,
    quality_observations,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "photos"
WEIGHTS = Path("data/models/safety_world_v2.onnx")

pytestmark = pytest.mark.skipif(
    not FIXTURES.is_dir() or not any(FIXTURES.glob("*.jpg")),
    reason="eval photographs absent — run `python scripts/fetch_eval_photos.py` "
           "(network); these fixtures are deliberately not committed",
)


@pytest.fixture(scope="module")
def detector():
    if not WEIGHTS.is_file():
        pytest.skip(f"detector weights not found at {WEIGHTS}")
    os.environ.setdefault("SAFETY_WORLD_WEIGHTS", str(WEIGHTS))
    from app.blocks.safety_world_detector import default_detector

    det = default_detector()
    if det is None:
        pytest.skip("default_detector() returned None — SAFETY_WORLD_WEIGHTS not resolving")
    return det


def _analyse(detector, name):
    path = FIXTURES / name
    if not path.is_file():
        pytest.skip(f"fixture {name} not downloaded")
    return [{
        "photo": path.name,
        "detections": detector.detect(path, conf_threshold=LOW_CONF_THRESHOLD),
        "detector": "safety_world_v2",
        "safety_compliance": "unknown",
        "activities_detected": [], "headcount_estimate": 0, "progress_indicators": "",
    }]


def test_ladder_photo_yields_a_temporary_works_record(detector):
    """A photograph of a ladder produces a ladder record, categorised correctly.

    The strongest and least ambiguous detection in the whole eval set (0.962),
    so this is the one real-image assertion safe to make on class identity.
    """
    equipment = equipment_from_photos(_analyse(detector, "e04_ladder_peabody.jpg"))

    items = {e["item"]: e for e in equipment}
    assert "ladder" in items, f"expected a ladder record, got {list(items)}"
    assert items["ladder"]["category"] == "temporary_works"
    assert items["ladder"]["confidence_tier"] == "detected"
    assert items["ladder"]["seen_in_photos"] == ["e04_ladder_peabody.jpg"]


def test_every_observation_is_well_formed_and_traceable(detector):
    """Whatever comes out must carry its evidence: photo, class, confidence, tier."""
    analysis = _analyse(detector, "q11_cracked_tile_flooring.jpg")
    observations = quality_observations(analysis)

    assert observations, "expected at least one observation on this image"
    for obs in observations:
        assert obs["photo"] == "q11_cracked_tile_flooring.jpg"
        assert obs["detected_class"]
        assert LOW_CONF_THRESHOLD <= obs["confidence"] <= 1.0
        assert obs["confidence_tier"] in ("detected", "low_confidence")
        # tier and wording must agree — this is what stops a low-confidence
        # detection reading as a statement of fact about a client's concrete
        if obs["confidence_tier"] == "low_confidence":
            assert obs["observation"].startswith("possible ")
            assert obs["observation"].endswith("-- low confidence")
        else:
            assert obs["observation"].endswith(" detected")


def test_no_verdict_language_survives_the_real_pipeline(detector):
    """The observations-not-verdicts contract, enforced on real detector output."""
    banned = ("violation", "non-compliance", "non-compliant", "defect",
              "breach", "unsafe", "fail")
    for name in ("q11_cracked_tile_flooring.jpg", "e04_ladder_peabody.jpg"):
        analysis = _analyse(detector, name)
        for obs in quality_observations(analysis) + equipment_from_photos(analysis):
            text = obs["observation"].lower()
            assert not any(w in text for w in banned), obs["observation"]


def test_a_photo_with_nothing_detectable_yields_an_honest_empty(detector):
    """q08 produced zero detections in the eval — it must produce zero
    observations, not a hedge or a placeholder."""
    analysis = _analyse(detector, "q08_peeling_paint_bathroom.jpg")
    if analysis[0]["detections"]:
        pytest.skip("model now detects something in this image; empties are covered "
                    "deterministically in test_photo_observations.py")
    assert quality_observations(analysis) == []
    assert equipment_from_photos(analysis) == []
