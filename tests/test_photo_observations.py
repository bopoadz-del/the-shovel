"""Photo detections must reach the daily report — truthfully, or not at all.

The detector has worked for months and the daily report has had
`quality_observations` and `equipment` fields for just as long. Nothing carried
one to the other: `_extract_quality_observations` and
`_extract_equipment_from_photos` both returned `[]`, so every report said the
site was clear of workmanship issues and had no plant on it, whatever the
photographs showed. A confidently empty section is worse than a missing one.

These tests use synthetic detections deliberately: they pin the CONTRACT
(shape, tiering, dedup, honest degradation) and run without weights or network.
What they cannot check is whether the model is right about a photograph — that
is measured on real construction images in docs/PHOTO_INTELLIGENCE_EVAL.md,
because a passing shape test over a fabricated payload proves nothing about the
payload.
"""
from __future__ import annotations

import pytest

from app.containers.construction.photo_observations import (
    DETECTED_THRESHOLD,
    LOW_CONF_THRESHOLD,
    PLANT_PROMPTS,
    QUALITY_PROMPTS,
    equipment_from_photos,
    quality_observations,
    tier_for,
)


def _photo(name, detections, detector="safety_world_v2"):
    return {
        "photo": name,
        "activities_detected": [],
        "safety_compliance": "unknown",
        "headcount_estimate": 0,
        "progress_indicators": "",
        "detections": detections,
        "detector": detector,
    }


def _det(cls, conf, bbox=None):
    return {"class": cls, "confidence": conf, "bbox": bbox or [0, 0, 10, 10]}


# ── the tiering law is shared with the chat path, not reinvented ──────────────

def test_thresholds_match_the_chat_photo_path():
    """One photograph must not be described two ways by two surfaces.

    photo_observations duplicates these constants rather than importing a
    router into a container. This test is what makes the duplication safe: if
    chat_photos changes its thresholds, this fails instead of the two paths
    silently disagreeing.
    """
    from app.routers import chat_photos

    assert DETECTED_THRESHOLD == chat_photos._DETECTED_THRESHOLD
    assert LOW_CONF_THRESHOLD == chat_photos._LOW_CONF_THRESHOLD


def test_vocabulary_split_matches_the_baked_prompts():
    """Every class this module claims must exist in the detector's vocabulary.

    A typo here would silently drop a whole category — the class would never
    match a detection and the section would just stay empty.
    """
    import json
    from pathlib import Path

    prompts = json.loads(
        Path("app/blocks/safety_world_prompts.json").read_text(encoding="utf-8")
    )["prompts"]

    for label in QUALITY_PROMPTS:
        assert label in prompts, f"{label!r} is not in the baked vocabulary"
    for label in PLANT_PROMPTS:
        assert label in prompts, f"{label!r} is not in the baked vocabulary"
    # 13 quality prompts, not 12: "trip hazard on floor" sits among the
    # concrete prompts in the JSON but is a hazard, not workmanship.
    assert len(QUALITY_PROMPTS) == 13
    assert "trip hazard on floor" not in QUALITY_PROMPTS


# ── quality observations ──────────────────────────────────────────────────────

def test_quality_observation_carries_class_confidence_and_source_photo():
    out = quality_observations([_photo("IMG_1.jpg", [_det("crack in concrete wall", 0.62)])])

    assert len(out) == 1
    obs = out[0]
    assert obs["detected_class"] == "crack in concrete wall"
    assert obs["confidence"] == 0.62
    assert obs["confidence_tier"] == "detected"
    assert obs["photo"] == "IMG_1.jpg"          # evidence trail back to the image
    assert obs["observation"] == "crack in concrete wall detected"


def test_observations_never_read_as_verdicts():
    """The detector's docstring is binding: no judgement language, ever."""
    out = quality_observations([
        _photo("a.jpg", [_det("crack in concrete wall", 0.9), _det("rust on steel rebar", 0.8)]),
    ])
    banned = ("violation", "non-compliance", "non-compliant", "defect",
              "breach", "failure", "unsafe", "reject")
    for obs in out:
        text = obs["observation"].lower()
        for word in banned:
            assert word not in text, f"verdict language {word!r} in {obs['observation']!r}"


def test_no_detections_yields_empty_not_a_fabricated_observation():
    assert quality_observations([_photo("blank.jpg", [])]) == []
    assert equipment_from_photos([_photo("blank.jpg", [])]) == []


def test_non_quality_classes_are_not_reported_as_workmanship():
    """PPE and hazard classes belong to other sections, not this one."""
    out = quality_observations([
        _photo("a.jpg", [_det("hard hat", 0.9), _det("open excavation pit", 0.8),
                         _det("trip hazard on floor", 0.7)]),
    ])
    assert out == []


# ── the low-confidence tier must not be promoted ──────────────────────────────

@pytest.mark.parametrize("conf", [0.05, 0.17, 0.299])
def test_low_confidence_detection_is_flagged_not_promoted(conf):
    """A 0.05-0.30 detection is surfaced AS low confidence, never as a fact.

    This is the tier that would quietly become an assertion about a client's
    concrete if anyone rounded it up.
    """
    out = quality_observations([_photo("dim.jpg", [_det("concrete honeycomb voids", conf)])])

    assert len(out) == 1
    assert out[0]["confidence_tier"] == "low_confidence"
    assert out[0]["observation"] == "possible concrete honeycomb voids detected -- low confidence"
    assert "-- low confidence" in out[0]["observation"]


def test_below_the_low_confidence_floor_is_dropped_entirely():
    assert quality_observations([_photo("noise.jpg", [_det("peeling paint", 0.04)])]) == []


def test_tier_boundaries():
    assert tier_for(0.30) == "detected"
    assert tier_for(0.2999) == "low_confidence"
    assert tier_for(0.05) == "low_confidence"
    assert tier_for(0.0499) == "below_threshold"


# ── equipment ─────────────────────────────────────────────────────────────────

def test_equipment_deduped_across_photos_with_evidence_trail():
    """A crane standing all day is one item on site, not one per frame."""
    out = equipment_from_photos([
        _photo("1.jpg", [_det("crane", 0.41)]),
        _photo("2.jpg", [_det("crane", 0.77)]),
        _photo("3.jpg", [_det("crane", 0.55)]),
    ])

    assert len(out) == 1
    assert out[0]["item"] == "crane"
    assert out[0]["confidence"] == 0.77                      # strongest wins
    assert out[0]["seen_in_photos"] == ["1.jpg", "2.jpg", "3.jpg"]


def test_ladder_and_scaffolding_are_temporary_works_not_plant():
    out = {e["item"]: e["category"] for e in equipment_from_photos([
        _photo("a.jpg", [_det("crane", 0.5), _det("ladder", 0.5), _det("scaffolding", 0.5)]),
    ])}
    assert out == {"crane": "plant", "ladder": "temporary_works",
                   "scaffolding": "temporary_works"}


def test_equipment_output_is_not_padded_beyond_the_vocabulary():
    """Only the three detectable classes may ever appear.

    Excavators and telehandlers have no prompt string. Inferring one from
    context would be an invented observation on a client report.
    """
    out = equipment_from_photos([
        _photo("a.jpg", [_det("excavator", 0.95), _det("telehandler", 0.9),
                         _det("crane", 0.4)]),
    ])
    assert [e["item"] for e in out] == ["crane"]
