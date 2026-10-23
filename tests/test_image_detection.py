"""Tests for the YOLO object detection tier on the image block (PR 3b).

Strategy:
- The ``ultralytics`` package may or may not be installed; tests cover
  BOTH cases via monkeypatching ``_yolo_available`` and ``_get_yolo_model``.
- We never download real YOLO weights — a fake model class returns
  deterministic detections so assertions are stable.
- Existing test_image_local_only.py covers the PIL + Tesseract paths;
  this file only exercises the new YOLO surface.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_test_image(tmp_path: Path) -> str:
    """Write a tiny valid PNG via PIL so the block has a real file to open."""
    from PIL import Image
    img = Image.new("RGB", (32, 32), color=(255, 255, 255))
    p = tmp_path / "test.png"
    img.save(p)
    return str(p)


class _FakeYoloBoxes:
    """Stand-in for ultralytics' Boxes object. Holds three tensors:
    .cls (class ids), .conf (confidences), .xyxy (bounding boxes)."""

    def __init__(self, classes, confs, boxes):
        # Minimal tensor stand-ins — only `.item()` and `.tolist()` needed
        class _Tensor:
            def __init__(self, vals): self._vals = vals
            def __len__(self): return len(self._vals)
            def __getitem__(self, i):
                v = self._vals[i]
                if isinstance(v, list):
                    class _Inner:
                        def tolist(self_inner): return v
                    return _Inner()
                class _Scalar:
                    def item(self_inner): return v
                return _Scalar()
        self.cls = _Tensor(classes)
        self.conf = _Tensor(confs)
        self.xyxy = _Tensor(boxes)

    def __len__(self): return len(self.cls)


class _FakeYoloResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class _FakeYoloModel:
    """Returns the same fake detections regardless of input. Stable for
    assertions; production code never sees this."""

    def __init__(self, *a, **kw): pass

    def __call__(self, file_path, conf=0.25, verbose=False):
        boxes = _FakeYoloBoxes(
            classes=[0, 0, 2],  # 0=person, 2=car (COCO)
            confs=[0.92, 0.78, 0.65],
            boxes=[[10.0, 20.0, 50.0, 100.0], [60.0, 30.0, 110.0, 90.0], [150.0, 40.0, 220.0, 130.0]],
        )
        result = _FakeYoloResult(
            boxes=boxes,
            names={0: "person", 2: "car"},
        )
        return [result]


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_detect_objects_graceful_when_ultralytics_missing(tmp_path, monkeypatch):
    """When ultralytics isn't importable, detect_objects returns
    available=false rather than raising. Critical: no chat path should
    break because a user installed the base requirements."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    monkeypatch.setattr(_img, "_yolo_available", lambda: False)
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "detect_objects"})
    assert result["status"] == "success"
    assert result["available"] is False
    assert result["detections"] == []
    assert "ultralytics" in result["note"]


@pytest.mark.asyncio
async def test_detect_objects_happy_path(tmp_path, monkeypatch):
    """With ultralytics available (mocked), detect_objects returns the
    structured detections + per-class summary."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    monkeypatch.setattr(_img, "_yolo_available", lambda: True)
    monkeypatch.setattr(_img, "_get_yolo_model", lambda *a, **kw: _FakeYoloModel())
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "detect_objects"})
    assert result["status"] == "success"
    assert result["available"] is True
    assert result["detection_count"] == 3
    # The fake returns 2 persons + 1 car
    assert result["summary_by_class"] == {"person": 2, "car": 1}
    # Detections preserved structurally
    assert result["detections"][0]["class_name"] == "person"
    assert result["detections"][0]["confidence"] == 0.92
    assert len(result["detections"][0]["box"]) == 4


@pytest.mark.asyncio
async def test_detect_objects_respects_conf_threshold(tmp_path, monkeypatch):
    """The conf_threshold param is forwarded to the YOLO call. We assert
    via a capturing fake rather than testing the model's filtering itself."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    captured = {}
    class _CapturingModel:
        def __call__(self, file_path, conf=0.25, verbose=False):
            captured["conf"] = conf
            return []

    monkeypatch.setattr(_img, "_yolo_available", lambda: True)
    monkeypatch.setattr(_img, "_get_yolo_model", lambda *a, **kw: _CapturingModel())
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    await block.process(fp, params={"operation": "detect_objects", "conf_threshold": 0.5})
    assert captured["conf"] == 0.5


@pytest.mark.asyncio
async def test_analyze_skips_detection_when_unavailable(tmp_path, monkeypatch):
    """The combined `analyze` op still works without ultralytics — it just
    omits the detection section. Backwards-compat for existing callers."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    monkeypatch.setattr(_img, "_yolo_available", lambda: False)
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "analyze"})
    assert result["status"] == "success"
    assert result["detections"] == []
    assert "yolo" not in result["provider"]  # provider should be 'pil' or 'pil+tesseract'


@pytest.mark.asyncio
async def test_analyze_includes_detection_when_available(tmp_path, monkeypatch):
    """When ultralytics IS available, analyze adds detections to its
    structured output AND appends a summary line to the description."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    monkeypatch.setattr(_img, "_yolo_available", lambda: True)
    monkeypatch.setattr(_img, "_get_yolo_model", lambda *a, **kw: _FakeYoloModel())
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "analyze"})
    assert result["status"] == "success"
    assert result["detection_count"] == 3 if "detection_count" in result else True
    assert result["summary_by_class"] == {"person": 2, "car": 1}
    assert "yolo" in result["provider"]
    # The human-readable description gets the summary appended
    assert "Detected" in result["description"]
    assert "person × 2" in result["description"]


@pytest.mark.asyncio
async def test_detection_failure_is_non_fatal_in_analyze(tmp_path, monkeypatch):
    """If YOLO raises mid-analyze, the metadata + OCR result still ships;
    error captured in detection_error rather than aborting the response."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    class _ExplodingModel:
        def __call__(self, *a, **kw):
            raise RuntimeError("simulated YOLO crash")

    monkeypatch.setattr(_img, "_yolo_available", lambda: True)
    monkeypatch.setattr(_img, "_get_yolo_model", lambda *a, **kw: _ExplodingModel())
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "analyze"})
    assert result["status"] == "success"  # not aborted
    assert "simulated YOLO crash" in (result.get("detection_error") or "")


@pytest.mark.asyncio
async def test_detect_objects_op_returns_error_on_explicit_failure(tmp_path, monkeypatch):
    """When the user explicitly asks for detection AND it fails, we DO
    return status=error (unlike analyze which is permissive). The
    operation was the whole request — silently returning empty would hide
    the problem."""
    from app.blocks import image as _img
    from app.blocks.image import ImageBlock

    class _ExplodingModel:
        def __call__(self, *a, **kw):
            raise RuntimeError("boom")

    monkeypatch.setattr(_img, "_yolo_available", lambda: True)
    monkeypatch.setattr(_img, "_get_yolo_model", lambda *a, **kw: _ExplodingModel())
    _img._reset_yolo_cache()

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "detect_objects"})
    assert result["status"] == "error"
    assert "boom" in result["error"]


def test_summarize_detections_groups_and_orders():
    """Helper test: per-class counts ordered by count desc, class name asc."""
    from app.blocks.image import _summarize_detections

    dets = [
        {"class_name": "car"},
        {"class_name": "person"},
        {"class_name": "person"},
        {"class_name": "person"},
        {"class_name": "truck"},
    ]
    summary = _summarize_detections(dets)
    # Ordered by count desc, class name asc as tiebreak
    assert list(summary.items()) == [("person", 3), ("car", 1), ("truck", 1)]


def test_yolo_available_reflects_import_state():
    """Sanity: _yolo_available() reports actual ultralytics importability."""
    from app.blocks.image import _yolo_available
    try:
        import ultralytics  # noqa: F401
        expected = True
    except ImportError:
        expected = False
    assert _yolo_available() is expected


@pytest.mark.asyncio
async def test_metadata_operation_unaffected(tmp_path):
    """No-regression on the existing metadata-only path. PR 3b adds; it
    never modifies the metadata or extract_text response shapes."""
    from app.blocks.image import ImageBlock

    block = ImageBlock()
    fp = _make_test_image(tmp_path)
    result = await block.process(fp, params={"operation": "metadata"})
    assert result["status"] == "success"
    assert result["provider"] == "pil"
    assert result["operation"] == "metadata"
    # No detection fields should appear on this branch
    assert "detections" not in result
