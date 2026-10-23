"""Tests for OCR image-quality helpers — deskew + quality assessment.

Roadmap V2 · Epic 5. Pure PIL/numpy — no Tesseract needed.
"""

from PIL import Image, ImageDraw

from app.core.image_quality import (
    deskew,
    estimate_skew_angle,
    summarize_ocr_quality,
)


def _striped_page(size: int = 400, n_bars: int = 12) -> Image.Image:
    """A synthetic 'text page' — evenly spaced horizontal black bars on white."""
    img = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(img)
    gap = size // (n_bars + 1)
    for i in range(1, n_bars + 1):
        y = i * gap
        draw.rectangle([size * 0.15, y, size * 0.85, y + max(2, gap // 4)], fill=0)
    return img


# ── skew estimation ─────────────────────────────────────────────────────────

def test_level_page_has_no_skew():
    assert abs(estimate_skew_angle(_striped_page())) < 1.0


def test_skew_estimate_recovers_known_angle():
    skewed = _striped_page().rotate(6, resample=Image.BICUBIC,
                                    expand=True, fillcolor=255)
    # the correcting rotation should be roughly -6 degrees
    assert abs(estimate_skew_angle(skewed) - (-6)) < 2.5


def test_deskew_straightens_a_skewed_page():
    skewed = _striped_page().rotate(5, resample=Image.BICUBIC,
                                    expand=True, fillcolor=255)
    fixed, applied = deskew(skewed)
    assert abs(applied) > 0.5                      # a correction was applied
    assert abs(estimate_skew_angle(fixed)) < 2.5   # and it worked


def test_deskew_is_noop_on_level_page():
    fixed, applied = deskew(_striped_page())
    assert abs(applied) < 1.0


# ── OCR quality verdict ─────────────────────────────────────────────────────

def test_quality_no_words_is_low_quality():
    q = summarize_ocr_quality([])
    assert q["low_quality"] is True
    assert q["ocr_confidence"] == 0.0
    assert q["caveat"]


def test_quality_high_confidence_has_no_caveat():
    q = summarize_ocr_quality([0.95, 0.91, 0.93, 0.9])
    assert q["low_quality"] is False
    assert q["caveat"] is None
    assert q["ocr_confidence"] > 0.85
    assert q["words_measured"] == 4


def test_quality_low_confidence_flags_caveat():
    q = summarize_ocr_quality([0.3, 0.45, 0.4, 0.35])
    assert q["low_quality"] is True
    assert "unreliable" in q["caveat"].lower()


def test_quality_ignores_negative_confidences():
    # tesseract emits -1 for non-text boxes — those must not drag the mean down
    q = summarize_ocr_quality([-1, 0.9, -1, 0.92])
    assert q["words_measured"] == 2
    assert q["ocr_confidence"] > 0.85
