"""Tests for measured extraction confidence — Roadmap V2 · Epic 1."""

from app.core.confidence import assess_extraction_confidence

DRAWING_FIELDS = [
    "drawing_number", "revision", "scale", "title_block",
    "detected_disciplines", "measurements", "specifications",
]


def _rich_drawing():
    return {
        "sheets": [{"raw_text": "x" * 2400}],
        "total_pages": 1,
        "drawing_number": "A-101",
        "revision": "C",
        "scale": "1:100",
        "title_block": {"title": "Ground Floor Plan"},
        "detected_disciplines": ["architectural"],
        "measurements": [{"value": 12.5}],
        "specifications": [{"item": "concrete"}],
    }


def test_report_is_marked_as_measured():
    r = assess_extraction_confidence(_rich_drawing(), expected_fields=DRAWING_FIELDS)
    assert r["measured"] is True
    assert "signals" in r and "caveats" in r


def test_rich_extraction_scores_high_and_clean():
    r = assess_extraction_confidence(_rich_drawing(), expected_fields=DRAWING_FIELDS)
    assert r["overall"] > 0.9
    assert r["caveats"] == []


def test_empty_extraction_scores_low_with_caveats():
    r = assess_extraction_confidence(
        {"sheets": [], "total_pages": 1}, expected_fields=DRAWING_FIELDS
    )
    assert r["overall"] < 0.2
    assert r["caveats"]


def test_field_coverage_is_a_real_fraction():
    partial = {"text": "x" * 800, "drawing_number": "A-1", "revision": "", "scale": None}
    r = assess_extraction_confidence(
        partial, expected_fields=["drawing_number", "revision", "scale"]
    )
    assert r["signals"]["field_coverage"] == round(1 / 3, 3)


def test_ocr_quality_feeds_signal_and_caveat():
    ocrq = {"ocr_confidence": 0.42, "low_quality": True,
            "caveat": "Low-quality scan — extracted text may be unreliable."}
    r = assess_extraction_confidence(
        {"text": "x" * 800, "total_pages": 1}, ocr_quality=ocrq
    )
    assert r["signals"]["ocr_char_confidence"] == 0.42
    assert any("low-quality" in c.lower() for c in r["caveats"])


def test_confidence_is_not_a_constant():
    # the whole point of Epic 1 — different inputs must give different scores
    good = assess_extraction_confidence({"text": "x" * 4000, "total_pages": 1})
    poor = assess_extraction_confidence({"text": "x" * 80, "total_pages": 1})
    assert good["overall"] != poor["overall"]
    assert good["overall"] > poor["overall"]
    # and never the old hardcoded 0.7
    assert good["overall"] != 0.7 and poor["overall"] != 0.7


def test_construction_container_uses_measured_confidence():
    """The construction container's _calculate_confidence is now the real one."""
    from app.containers.construction import ConstructionContainer
    c = ConstructionContainer()
    conf = c._calculate_confidence(_rich_drawing())
    assert conf["measured"] is True
    assert conf["overall"] > 0.9
