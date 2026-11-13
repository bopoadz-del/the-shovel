"""A photo whose scan produced no observations still says what the scan found (S-P1 c).

An empty observation list made the composer emit a bare ``[attached: NAME]``
marker. The answer path cannot tell that marker from a file it never saw, so
the reply said the image could not be read. Each test builds its own image
bytes and its own detector output; every rule is pinned by two inputs that
must come out differently.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.rag.retriever import attachment_markers_carry_content
from app.main import app
from app.routers import chat_photos
from tests._dev_key import DEV_KEY

H = {"Authorization": "Bearer " + DEV_KEY}
JPEG = bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\x00\x10JFIF\x00" + b"\x00" * 32


@pytest.fixture(scope="module")
def client():
    mp = pytest.MonkeyPatch()
    mp.setenv("WARM_MODELS_BLOCKING", "1")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        mp.undo()


def _route_path(endpoint):
    for route in app.routes:
        routed = getattr(route, "endpoint", None)
        if routed is not None and (routed.__module__, routed.__qualname__) == (
            endpoint.__module__, endpoint.__qualname__,
        ):
            return route.path
    raise AssertionError(f"{endpoint.__qualname__} is not routed")


class _ScanBlock:
    """Stands in for the image block with a fixed scan result."""

    def __init__(self, result):
        self.result = result

    async def execute(self, inputs, params):
        return {"result": dict(self.result)}


def _observations(client, monkeypatch, result):
    monkeypatch.setattr("app.dependencies.get_block_instance", lambda name: _ScanBlock(result))
    r = client.post(
        _route_path(chat_photos.analyze_chat_photo),
        files={"file": ("site.jpg", JPEG, "image/jpeg")},
        headers=H,
    )
    assert r.status_code == 200, r.text
    return r.json()["observations"]


def _marker(name, observations):
    # The composer's marker shape (ChatComposer.tsx): payload only when the
    # server returned observations.
    if observations:
        return f"[attached: {name} | observations: {'; '.join(observations)}]"
    return f"[attached: {name}]"


@pytest.mark.parametrize(
    "scan,expected",
    [
        ({"safety_qaqc": [], "safety_qaqc_ran": True}, ["none detected"]),
        (
            {"safety_qaqc": [{"class": "excavation", "confidence": 0.62}],
             "safety_qaqc_ran": True},
            ["excavation detected"],
        ),
    ],
)
def test_ran_scan_with_and_without_detections(client, monkeypatch, scan, expected):
    obs = _observations(client, monkeypatch, scan)
    assert obs == expected
    assert attachment_markers_carry_content(_marker("site.jpg", obs))


@pytest.mark.parametrize(
    "scan,expected",
    [
        ({"safety_qaqc": [], "safety_qaqc_ran": True}, "none detected"),
        ({"safety_qaqc": [], "safety_qaqc_ran": False},
         "photo detector unavailable -- image not scanned"),
        ({"safety_qaqc": [], "safety_qaqc_ran": False, "safety_qaqc_error": "boom"},
         "photo detector failed on this image -- no observations"),
        ({"safety_qaqc": []}, "photo detector unavailable -- image not scanned"),
    ],
)
def test_empty_scan_names_whether_the_detector_ran(client, monkeypatch, scan, expected):
    assert _observations(client, monkeypatch, scan) == [expected]


@pytest.mark.parametrize(
    "scan,expected",
    [
        ({"safety_qaqc": [], "safety_qaqc_ran": True,
          "summary_by_class": {"person": 3}}, ["3 people visible"]),
        ({"safety_qaqc": [], "safety_qaqc_ran": True,
          "summary_by_class": {}}, ["none detected"]),
    ],
)
def test_empty_note_only_when_nothing_else_was_observed(client, monkeypatch, scan, expected):
    assert _observations(client, monkeypatch, scan) == expected


@pytest.mark.parametrize(
    "confidence,expected",
    [(0.01, ["none detected"]), (0.40, ["ladder detected"])],
)
def test_below_floor_detections_count_as_nothing_recognisable(
    client, monkeypatch, confidence, expected,
):
    scan = {"safety_qaqc": [{"class": "ladder", "confidence": confidence}],
            "safety_qaqc_ran": True}
    assert _observations(client, monkeypatch, scan) == expected


@pytest.mark.parametrize(
    "body,expected",
    [
        ({"safety_qaqc_ran": True}, "none detected"),
        ({"safety_qaqc_ran": "yes"}, "photo detector unavailable -- image not scanned"),
    ],
)
def test_unit_empty_scan_observation_two_inputs(body, expected):
    assert chat_photos._empty_scan_observation(body) == expected


class _EmptyDetector:
    def detect(self, path, conf_threshold=0.25):
        return []


@pytest.mark.parametrize(
    "detector,ran",
    [(_EmptyDetector(), True), (None, False)],
)
def test_image_block_reports_whether_the_safety_scan_ran(tmp_path, monkeypatch, detector, ran):
    import asyncio
    import random

    from PIL import Image

    from app.blocks import image as image_mod
    from app.blocks import safety_world_detector

    rng = random.Random(7)
    img = Image.new("RGB", (96, 96))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
                 for _ in range(96 * 96)])
    path = tmp_path / "scan.png"
    img.save(path)

    monkeypatch.setattr(safety_world_detector, "default_detector", lambda: detector)
    monkeypatch.setattr(image_mod, "_yolo_available", lambda: False)
    monkeypatch.setattr(image_mod, "_tesseract_ocr", lambda p: ("", 0.0))

    out = asyncio.run(image_mod.ImageBlock().process(
        str(path), {"mode": "safety_qaqc", "safety_qaqc_conf": 0.05}))
    assert out.get("status") == "success", out
    assert out["safety_qaqc"] == []
    assert out["safety_qaqc_ran"] is ran


@pytest.mark.parametrize("exists", [True, False])
def test_image_block_returns_the_source_error_for_a_missing_file(tmp_path, monkeypatch, exists):
    """A source that cannot be resolved ends the scan with its error; a real
    file is scanned. (The empty-scan note relies on the block's answer.)"""
    import asyncio

    from PIL import Image

    from app.blocks import image as image_mod
    from app.blocks import safety_world_detector

    path = tmp_path / "maybe.png"
    if exists:
        Image.new("RGB", (64, 64), (120, 80, 40)).save(path)
    monkeypatch.setattr(safety_world_detector, "default_detector", lambda: None)
    monkeypatch.setattr(image_mod, "_yolo_available", lambda: False)
    monkeypatch.setattr(image_mod, "_tesseract_ocr", lambda p: ("", 0.0))

    out = asyncio.run(image_mod.ImageBlock().process(str(path), {"mode": "safety_qaqc"}))
    if exists:
        assert out.get("safety_qaqc_ran") is False, out
    else:
        assert out == {"status": "error", "error": "Provide a valid file path or URL"}
