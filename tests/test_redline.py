"""Tests for redline / markup detection — colour-channel analysis.

Roadmap V2 · Epic 5 (Input quality handling). Pure PIL/numpy — no Tesseract
needed. Test images are synthesised programmatically so there are no external
fixtures.
"""

from PIL import Image, ImageDraw

from app.core.redline import detect_redlines


# ── synthetic test images ───────────────────────────────────────────────────

def _bw_drawing(size: int = 400) -> Image.Image:
    """A clean black-and-white 'drawing' — black lines/text on white, no colour."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # a frame + some interior linework, all pure black
    draw.rectangle([20, 20, size - 20, size - 20], outline=(0, 0, 0), width=3)
    for i in range(1, 8):
        y = 20 + i * (size - 40) // 8
        draw.line([20, y, size - 20, y], fill=(0, 0, 0), width=2)
    draw.text((40, 40), "PLAN VIEW - LEVEL 2", fill=(0, 0, 0))
    return img


def _marked_up_drawing(size: int = 400) -> Image.Image:
    """A B&W drawing with a red revision cloud and a blue annotation scribble."""
    img = _bw_drawing(size)
    draw = ImageDraw.Draw(img)
    # red markup — a revision cloud area, top-right
    draw.ellipse([260, 50, 360, 130], outline=(220, 20, 20), width=6)
    draw.line([270, 60, 350, 120], fill=(220, 20, 20), width=5)
    # blue markup — an annotation, bottom-left
    draw.line([60, 300, 160, 320], fill=(20, 30, 210), width=6)
    draw.line([60, 320, 160, 340], fill=(20, 30, 210), width=6)
    return img


def _greyscale_image(size: int = 200) -> Image.Image:
    """A pure-grey image — must NOT be mistaken for colour markup."""
    return Image.new("RGB", (size, size), (128, 128, 128))


# ── clean vs. marked-up ─────────────────────────────────────────────────────

def test_clean_bw_drawing_has_no_markup():
    result = detect_redlines(_bw_drawing())
    assert result["has_markup"] is False
    assert result["coverage"] < 0.01
    assert result["regions"] == []


def test_marked_up_drawing_is_detected():
    result = detect_redlines(_marked_up_drawing())
    assert result["has_markup"] is True
    assert result["coverage"] > 0.0
    assert len(result["regions"]) >= 1


def test_greyscale_is_not_markup():
    # mid-grey has zero saturation — it is not coloured ink
    result = detect_redlines(_greyscale_image())
    assert result["has_markup"] is False
    assert result["coverage"] < 0.01


def test_grayscale_mode_image_is_handled():
    # an "L"-mode image cannot contain colour — must not crash, must report clean
    grey = _bw_drawing().convert("L")
    result = detect_redlines(grey)
    assert result["has_markup"] is False


# ── region clustering & shape of result ─────────────────────────────────────

def test_result_has_expected_shape():
    result = detect_redlines(_marked_up_drawing())
    assert set(result.keys()) >= {"has_markup", "coverage", "regions"}
    assert isinstance(result["has_markup"], bool)
    assert isinstance(result["coverage"], float)
    for region in result["regions"]:
        assert set(region.keys()) >= {"bbox", "dominant_colour"}
        x0, y0, x1, y1 = region["bbox"]
        assert x0 < x1 and y0 < y1
        assert region["dominant_colour"] in {"red", "green", "blue"}


def test_red_and_blue_regions_are_separated():
    # the red cloud (top-right) and blue scribble (bottom-left) are far apart
    # and should cluster into distinct regions with the right dominant colours.
    result = detect_redlines(_marked_up_drawing())
    colours = {r["dominant_colour"] for r in result["regions"]}
    assert "red" in colours
    assert "blue" in colours


def test_red_region_bbox_is_top_right():
    result = detect_redlines(_marked_up_drawing())
    red = [r for r in result["regions"] if r["dominant_colour"] == "red"]
    assert red
    x0, y0, x1, y1 = red[0]["bbox"]
    # the red cloud was drawn around x 260-360, y 50-130
    assert x0 > 200 and y0 < 200


def test_coverage_is_a_fraction():
    result = detect_redlines(_marked_up_drawing())
    assert 0.0 <= result["coverage"] <= 1.0


def test_summarize_markup_caveat_for_marked_up():
    from app.core.redline import summarize_markup

    result = detect_redlines(_marked_up_drawing())
    summary = summarize_markup(result)
    assert summary["has_markup"] is True
    assert summary["caveat"]
    assert "markup" in summary["caveat"].lower() or "annotat" in summary["caveat"].lower()


def test_summarize_markup_no_caveat_for_clean():
    from app.core.redline import summarize_markup

    result = detect_redlines(_bw_drawing())
    summary = summarize_markup(result)
    assert summary["has_markup"] is False
    assert summary["caveat"] is None
