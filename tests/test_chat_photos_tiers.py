"""Tier-logic tests for app.routers.chat_photos.

The product contract is that the model produces OBSERVATIONS, never
"violations". These tests pin that contract by directly exercising
the tier helpers; the route handler itself is exercised in the
existing chat-photo bridge test.
"""
from __future__ import annotations

from app.routers.chat_photos import (
    _DETECTED_THRESHOLD,
    _LOW_CONF_THRESHOLD,
    _HAT_FRAGMENTS,
    _VEST_FRAGMENTS,
    _person_observation,
    _tier_for,
)

_FORBIDDEN = ("violation", "non-compliant", "non compliant", "breach")


def _check_no_violation_language(msg: str) -> None:
    low = msg.lower()
    for term in _FORBIDDEN:
        assert term not in low, f"forbidden term {term!r} in {msg!r}"


def test_vest_at_high_conf_returns_detected():
    out = _tier_for(
        [{"class": "yellow reflective safety vest", "confidence": 0.71}],
        _VEST_FRAGMENTS,
    )
    assert out["tier"] == "detected"
    assert out["max_confidence"] == 0.71
    assert "vest detected" in out["message"]
    _check_no_violation_language(out["message"])


def test_vest_at_low_conf_returns_low_confidence():
    out = _tier_for(
        [{"class": "high visibility vest", "confidence": 0.09}],
        _VEST_FRAGMENTS,
    )
    assert out["tier"] == "low_confidence"
    assert "low confidence" in out["message"]
    assert "possible" in out["message"]
    _check_no_violation_language(out["message"])


def test_vest_below_floor_returns_not_detected():
    # Anything below _LOW_CONF_THRESHOLD (0.05) gets filtered upstream
    # by the conf=0.05 predict call. Pass an empty list to exercise the
    # not_detected path.
    out = _tier_for([], _VEST_FRAGMENTS)
    assert out["tier"] == "not_detected"
    assert out["message"] == ""
    assert out["max_confidence"] == 0.0


def test_hat_tier_mirrors_vest_thresholds():
    high = _tier_for([{"class": "hard hat", "confidence": 0.55}], _HAT_FRAGMENTS)
    low = _tier_for([{"class": "yellow or white safety helmet", "confidence": 0.20}], _HAT_FRAGMENTS)
    assert high["tier"] == "detected"
    assert low["tier"] == "low_confidence"
    _check_no_violation_language(high["message"])
    _check_no_violation_language(low["message"])


def test_person_observation_fires_only_when_person_confident_and_vest_missing():
    detections = [
        {"class": "person", "confidence": 0.80},
        {"class": "yellow or white safety helmet", "confidence": 0.55},
    ]
    vest_tier = _tier_for(detections, _VEST_FRAGMENTS)  # not_detected
    obs = _person_observation(detections, vest_tier)
    assert obs == "no vest detected in image"
    _check_no_violation_language(obs)


def test_person_observation_suppressed_when_vest_detected():
    detections = [
        {"class": "person", "confidence": 0.80},
        {"class": "yellow reflective safety vest", "confidence": 0.45},
    ]
    vest_tier = _tier_for(detections, _VEST_FRAGMENTS)  # detected
    obs = _person_observation(detections, vest_tier)
    assert obs is None


def test_person_observation_suppressed_when_person_below_threshold():
    # Person at 0.20 is below _DETECTED_THRESHOLD (0.30) -- don't claim
    # "no vest detected in image" when we're not confident a person is
    # even there. This is the precision guard.
    detections = [{"class": "person", "confidence": 0.20}]
    vest_tier = _tier_for(detections, _VEST_FRAGMENTS)
    obs = _person_observation(detections, vest_tier)
    assert obs is None


def test_thresholds_are_what_the_product_spec_says():
    assert _DETECTED_THRESHOLD == 0.30
    assert _LOW_CONF_THRESHOLD == 0.05


def test_class_match_is_case_insensitive_and_substring():
    out = _tier_for([{"class": "ORANGE reflective Safety VEST", "confidence": 0.60}], _VEST_FRAGMENTS)
    assert out["tier"] == "detected"


# ────────── _other_observations: every non-vest/hat/person class surfaces ──────────

from app.routers.chat_photos import _other_observations


def test_other_observations_surfaces_qaqc_classes_above_threshold():
    """Concrete defects, hazards, equipment etc. must appear in observations
    instead of being silently dropped because they aren't PPE."""
    detections = [
        {"class": "porous holes in concrete surface", "confidence": 0.42},
        {"class": "crack in concrete wall", "confidence": 0.08},
        {"class": "crane", "confidence": 0.85},
    ]
    out = _other_observations(detections)
    assert "porous holes in concrete surface detected" in out
    assert "crane detected" in out
    assert "possible crack in concrete wall detected -- low confidence" in out
    # banned-word contract still applies
    for line in out:
        _check_no_violation_language(line)


def test_other_observations_excludes_vest_hat_person():
    """vest / hat / person are surfaced by their own tier functions --
    don't double-emit them through the generic path."""
    detections = [
        {"class": "yellow reflective safety vest", "confidence": 0.71},
        {"class": "yellow or white safety helmet", "confidence": 0.91},
        {"class": "person", "confidence": 0.80},
        {"class": "ladder", "confidence": 0.55},
    ]
    out = _other_observations(detections)
    assert out == ["ladder detected"]


def test_other_observations_sorted_by_confidence_descending():
    """Strongest signal first so the LLM / UI list has the most useful at top."""
    detections = [
        {"class": "ladder", "confidence": 0.40},
        {"class": "crane", "confidence": 0.80},
        {"class": "open excavation pit", "confidence": 0.55},
    ]
    out = _other_observations(detections)
    assert out == ["crane detected", "open excavation pit detected", "ladder detected"]


def test_other_observations_dedups_class_by_max_confidence():
    """Two boxes of the same class -> one observation, at the max confidence."""
    detections = [
        {"class": "ladder", "confidence": 0.10},
        {"class": "ladder", "confidence": 0.40},
        {"class": "ladder", "confidence": 0.22},
    ]
    out = _other_observations(detections)
    assert out == ["ladder detected"]


def test_other_observations_drops_below_low_floor():
    """Anything < 0.05 (the noise floor) is not surfaced even as 'possible'."""
    detections = [
        {"class": "rust on steel rebar", "confidence": 0.001},
        {"class": "missing handrail", "confidence": 0.04},
    ]
    out = _other_observations(detections)
    assert out == []


def test_people_counted_by_the_general_detector_are_an_observation():
    """The chat answer gets the photo's people even when the safety model
    saw nothing -- otherwise the attached photo reaches the chat empty."""
    from app.routers.chat_photos import _people_observation

    assert _people_observation(2, []) == "2 people visible"
    assert _people_observation(1, []) == "1 person visible"
    assert _people_observation(0, []) is None
    assert _people_observation(3, ["no vest detected on a person"]) is None
    _check_no_violation_language(_people_observation(4, []))
