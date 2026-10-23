"""Smoke tests for app.blocks.safety_world_detector.

The detector is a thin wrapper around ultralytics' YOLO() applied to a
.pt produced by scripts/bake_world_model.py. The .pt has a fixed
class list (the baked prompts) and ultralytics exposes that as
``model.names`` -- we mirror it through ``detector.class_names``.

We mock ultralytics so the test doesn't need the real 30 MB checkpoint.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _fake_yolo_module(names: dict[int, str]):
    """Return a MagicMock that quacks like a loaded ultralytics YOLO model."""
    model = MagicMock()
    model.names = names

    boxes = MagicMock()
    boxes.cls = MagicMock()
    boxes.cls.tolist = lambda: [0.0, 1.0]
    boxes.conf = MagicMock()
    boxes.conf.tolist = lambda: [0.83, 0.41]
    boxes.xyxy = MagicMock()
    boxes.xyxy.tolist = lambda: [[10.0, 20.0, 100.0, 200.0],
                                 [50.0, 60.0, 150.0, 160.0]]
    boxes.__len__ = lambda self: 2

    result = MagicMock()
    result.boxes = boxes
    model.return_value = [result]
    return model


def test_class_names_mirror_baked_vocabulary(tmp_path: Path):
    fake_pt = tmp_path / "fake.pt"
    fake_pt.write_bytes(b"\x80\x00")  # contents irrelevant, mocked loader

    fake_model = _fake_yolo_module({0: "high visibility vest", 1: "hard hat"})
    with patch("app.blocks.safety_world_detector.YOLO", return_value=fake_model):
        from app.blocks.safety_world_detector import SafetyWorldDetector
        det = SafetyWorldDetector(fake_pt)
        assert det.class_names == ["high visibility vest", "hard hat"]


def test_detect_returns_prompt_strings(tmp_path: Path):
    fake_pt = tmp_path / "fake.pt"
    fake_pt.write_bytes(b"\x80\x00")
    fake_model = _fake_yolo_module({0: "high visibility vest", 1: "hard hat"})
    with patch("app.blocks.safety_world_detector.YOLO", return_value=fake_model):
        from app.blocks.safety_world_detector import SafetyWorldDetector
        det = SafetyWorldDetector(fake_pt)
        out = det.detect(tmp_path / "anything.jpg", conf_threshold=0.3)
    assert out == [
        {"class": "high visibility vest", "confidence": 0.83,
         "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"class": "hard hat", "confidence": 0.41,
         "bbox": [50.0, 60.0, 150.0, 160.0]},
    ]


def test_default_detector_returns_none_when_env_unset(monkeypatch, tmp_path):
    """No variable and no weights baked into this image: no detector."""
    monkeypatch.delenv("SAFETY_WORLD_WEIGHTS", raising=False)
    from app.blocks import safety_world_detector as swd
    monkeypatch.setattr(swd, "BAKED_WEIGHTS", (tmp_path / "absent.onnx",))
    monkeypatch.setattr(swd, "_LOADED", {})
    assert swd.default_detector() is None


def test_default_detector_uses_the_weights_baked_into_the_image(monkeypatch, tmp_path):
    """With SAFETY_WORLD_WEIGHTS unset (the ECS task never set it), the
    weights the image ships are used, and the model is built once, not per
    photo."""
    monkeypatch.delenv("SAFETY_WORLD_WEIGHTS", raising=False)
    from app.blocks import safety_world_detector as swd

    baked = tmp_path / "baked.onnx"
    baked.write_bytes(b"onnx")
    built = []

    class _Stub:
        def __init__(self, path):
            built.append(Path(path))

    monkeypatch.setattr(swd, "BAKED_WEIGHTS", (baked,))
    monkeypatch.setattr(swd, "_LOADED", {})
    monkeypatch.setattr(swd, "SafetyWorldDetector", _Stub)

    first, second = swd.default_detector(), swd.default_detector()

    assert isinstance(first, _Stub) and first is second
    assert built == [baked]


def test_default_detector_returns_none_when_file_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFETY_WORLD_WEIGHTS", str(tmp_path / "nope.pt"))
    from app.blocks.safety_world_detector import default_detector
    assert default_detector() is None
