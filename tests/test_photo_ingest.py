"""Site photos are corpus material again (owner ruling 2026-11-06).

Pins the photo pipeline end to end: the extractor registry registers
.jpg/.jpeg, OCR text comes through, the YOLO-World summary rides along when
a detector is baked, a blank photo extracts to empty text (never an error),
and the bitmap is bounded before OCR. Spreadsheets stay out of the registry.
"""
from __future__ import annotations

import pytest

from app.core import doc_index
from app.core import text_extractors


class _Detector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, path, conf_threshold):
        return self._detections


def _patch_ocr(monkeypatch, result):
    async def _fake_process(self, file_path):
        return result

    monkeypatch.setattr("app.blocks.ocr.OCRBlock.process", _fake_process)


def _patch_detector(monkeypatch, detector):
    monkeypatch.setattr(
        "app.blocks.safety_world_detector.default_detector", lambda: detector
    )


def test_photo_pipeline_imports_resolve():
    # Also pins the mutation-gate killer-file anchor for these two modules.
    from app.core.doc_index import (  # noqa: F401
        _bounded_image,
        _safety_world_extract,
        extract_photo,
        image_ocr_max_pixels,
    )
    from app.core.text_extractors import extractor_for, formats  # noqa: F401


def test_photo_formats_are_registered_and_spreadsheets_are_not():
    formats = text_extractors.formats()
    assert ".jpg" in formats
    assert ".jpeg" in formats
    assert ".xls" not in formats
    assert ".xlsx" not in formats


def test_photo_extractor_reads_through_doc_index(monkeypatch, tmp_path):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": "JOB SITE EAST", "quality": {}})
    monkeypatch.setattr(doc_index, "_safety_world_extract", lambda p, name: "")
    text, meta = text_extractors.extractor_for(".jpg").read(str(photo), "site.jpg", False)
    assert "JOB SITE EAST" in text
    assert not meta


@pytest.mark.parametrize("ocr_text", ["GATE 4", "GATE 7"])
def test_photo_ocr_text_flows_into_extract_photo(monkeypatch, tmp_path, ocr_text):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": ocr_text, "quality": {}})
    _patch_detector(monkeypatch, None)
    text, meta = doc_index.extract_photo(str(photo), "site.jpg")
    assert text == ocr_text
    assert not meta


def test_photo_ocr_low_quality_is_recorded(monkeypatch, tmp_path):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": "faint", "quality": {"low_quality": True}})
    _patch_detector(monkeypatch, None)
    _, meta = doc_index.extract_photo(str(photo), "site.jpg")
    assert meta.get("ocr_low_quality") is True


@pytest.mark.parametrize("leaked", ["SHOULD NOT LEAK", "ALSO NOT LEAKED"])
def test_photo_ocr_failure_yields_blank_not_crash(monkeypatch, tmp_path, leaked):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "error", "text": leaked})
    _patch_detector(monkeypatch, None)
    text, meta = doc_index.extract_photo(str(photo), "site.jpg")
    assert text == ""
    assert not meta


def test_photo_with_no_ocr_and_no_detector_is_empty_but_supported(monkeypatch, tmp_path):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": "", "quality": {}})
    _patch_detector(monkeypatch, None)
    text, meta = doc_index.extract_photo(str(photo), "site.jpg")
    assert not text
    assert not meta


def test_photo_detector_summary_appends_to_ocr_text(monkeypatch, tmp_path):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": "SCAFFOLD BAY", "quality": {}})
    _patch_detector(
        monkeypatch,
        _Detector([{"class": "scaffold", "confidence": 0.9}, {"class": "worker", "confidence": 0.7}]),
    )
    text, _ = doc_index.extract_photo(str(photo), "site.jpg")
    assert "SCAFFOLD BAY" in text
    assert "scaffold (0.90)" in text
    assert "worker (0.70)" in text


def test_photo_detector_summary_is_the_whole_text_when_ocr_is_blank(monkeypatch, tmp_path):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_ocr(monkeypatch, {"status": "ok", "text": "", "quality": {}})
    _patch_detector(monkeypatch, _Detector([{"class": "crane", "confidence": 0.5}]))
    text, meta = doc_index.extract_photo(str(photo), "site.jpg")
    assert "crane (0.50)" in text
    assert not meta


@pytest.mark.parametrize(
    "env_value,expected",
    [("1000", 1000), ("garbage", 6_000_000), ("0", 6_000_000), ("", 6_000_000)],
)
def test_image_ocr_max_pixels(monkeypatch, env_value, expected):
    if env_value == "":
        monkeypatch.delenv("IMAGE_OCR_MAX_PIXELS", raising=False)
    else:
        monkeypatch.setenv("IMAGE_OCR_MAX_PIXELS", env_value)
    assert doc_index.image_ocr_max_pixels() == expected


@pytest.mark.parametrize("pixels,expect_bounded", [(3000 * 3000, True), (100 * 100, False)])
def test_bounded_image_scales_only_when_over_cap(monkeypatch, tmp_path, pixels, expect_bounded):
    from PIL import Image

    photo = tmp_path / "site.jpg"
    side = int(pixels**0.5)
    Image.new("RGB", (side, side), (200, 200, 200)).save(photo, "JPEG")
    monkeypatch.setenv("IMAGE_OCR_MAX_PIXELS", "6000000")
    with doc_index._bounded_image(str(photo)) as bounded:
        if expect_bounded:
            assert bounded != str(photo)
            with Image.open(bounded) as img:
                assert img.width * img.height <= 6_000_000
        else:
            assert bounded == str(photo)


@pytest.mark.parametrize("detector", [None, _Detector([])], ids=["no-detector", "no-detections"])
def test_safety_world_extract_blank(monkeypatch, tmp_path, detector):
    photo = tmp_path / "site.jpg"
    photo.write_bytes(b"\xff\xd8\xff")
    _patch_detector(monkeypatch, detector)
    assert doc_index._safety_world_extract(str(photo), "site.jpg") == ""
