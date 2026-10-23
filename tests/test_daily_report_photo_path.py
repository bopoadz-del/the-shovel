"""The daily report must carry photo observations — and say so when it cannot.

`daily_site_report` has always had `equipment` and `quality_observations`
fields; they were always empty because the extractors returned `[]`. These
tests pin the wired path end to end, with the detector stubbed so they need no
weights and no network.

The third state is the one that matters most: photos supplied but the detector
unavailable is NOT the same claim as photos analysed and nothing found. A
report that renders both as an empty section tells a site team the concrete is
clear when in fact nothing looked at it.
"""
from __future__ import annotations

import pytest

from app.containers.construction import ConstructionContainer


def _detections(*pairs):
    return [{"class": c, "confidence": conf, "bbox": [0, 0, 10, 10]} for c, conf in pairs]


@pytest.fixture
def container():
    return ConstructionContainer()


async def _report(container, photos, params=None):
    return await container.daily_site_report(
        {"photos": photos}, params or {"project_name": "Test", "date": "2026-08-12"}
    )


@pytest.mark.asyncio
async def test_detections_reach_the_report(container, monkeypatch):
    monkeypatch.setattr(
        container, "_run_construction_detector",
        lambda path: {"detections": _detections(
            ("crack in concrete wall", 0.62), ("crane", 0.55)),
            "detector": "safety_world_v2"},
    )

    report = await _report(container, ["/fake/site1.jpg"])

    quality = report["quality_observations"]
    assert [q["detected_class"] for q in quality] == ["crack in concrete wall"]
    assert quality[0]["observation"] == "crack in concrete wall detected"

    assert [e["item"] for e in report["equipment"]] == ["crane"]
    assert report["photo_intelligence"]["detector"] == "safety_world_v2"
    assert report["photo_intelligence"]["status"] == "ok"
    # Photos were analysed, so these sections are NOT flagged incomplete.
    assert "quality_observations" not in report["sections_incomplete"]


@pytest.mark.asyncio
async def test_photos_analysed_but_nothing_found_is_an_honest_empty(container, monkeypatch):
    monkeypatch.setattr(
        container, "_run_construction_detector",
        lambda path: {"detections": [], "detector": "safety_world_v2"},
    )

    report = await _report(container, ["/fake/clean.jpg"])

    assert report["quality_observations"] == []
    assert report["equipment"] == []
    # A clean site IS a valid finding — the section is complete, not missing.
    assert report["photo_intelligence"]["status"] == "ok"
    assert "quality_observations" not in report["sections_incomplete"]


@pytest.mark.asyncio
async def test_detector_unavailable_is_reported_not_silently_empty(container, monkeypatch):
    """The failure this test exists for: an empty section that reads as
    'nothing found on site' when the truth is 'nothing looked'."""
    monkeypatch.setattr(
        container, "_run_construction_detector",
        lambda path: {"detections": [], "detector": "unavailable"},
    )

    report = await _report(container, ["/fake/site1.jpg"])

    assert report["quality_observations"] == []
    assert report["photo_intelligence"]["detector"] == "unavailable"
    assert "unavailable" in report["photo_intelligence"]["status"]
    # And the reader is told the sections could not be produced.
    assert "quality_observations" in report["sections_incomplete"]
    assert "equipment" in report["sections_incomplete"]


@pytest.mark.asyncio
async def test_no_safety_observation_is_invented_when_nothing_was_analysed(container, monkeypatch):
    """`safety_compliance` defaults to 'unknown'; that must not become an issue.

    The previous implementation flagged anything != "compliant", so a photo
    nothing had analysed produced "Safety issues detected in photo analysis" on
    a client-facing report.
    """
    monkeypatch.setattr(
        container, "_run_construction_detector",
        lambda path: {"detections": [], "detector": "unavailable"},
    )

    report = await _report(container, ["/fake/site1.jpg"])

    assert report["safety_observations"] == []


@pytest.mark.asyncio
async def test_low_confidence_reaches_the_report_still_flagged(container, monkeypatch):
    monkeypatch.setattr(
        container, "_run_construction_detector",
        lambda path: {"detections": _detections(("rust on steel rebar", 0.11)),
                      "detector": "safety_world_v2"},
    )

    report = await _report(container, ["/fake/dim.jpg"])

    obs = report["quality_observations"]
    assert len(obs) == 1
    assert obs[0]["confidence_tier"] == "low_confidence"
    assert "-- low confidence" in obs[0]["observation"]
