"""Measured extraction confidence — Roadmap V2 · Epic 1.

Replaces the hardcoded confidence constants (`{"overall": 0.7}` /
`{"overall": 0.85, ...}`) with a report derived from real extraction signals.
A clean digital PDF and a poor scan now produce visibly different numbers.
"""

from typing import Any, Dict, List, Optional


def _ratio(n: float, d: float) -> float:
    return round(n / d, 3) if d else 0.0


def _is_filled(value: Any) -> bool:
    """True if a field actually carries information."""
    if value is None:
        return False
    if isinstance(value, (str, list, dict, tuple, set)):
        return len(value) > 0
    if isinstance(value, (int, float)):
        return value != 0
    return True


def assess_extraction_confidence(
    result: Dict[str, Any],
    expected_fields: Optional[List[str]] = None,
    ocr_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a confidence report from real extraction signals (each 0..1).

    Signals:
      text_extraction      — was meaningful text recovered (chars-per-page)
      field_coverage       — fraction of expected fields actually populated
      ocr_char_confidence  — mean OCR confidence, when OCR was the source

    `overall` is the mean of whichever signals are available — never a
    constant. `measured: True` marks the report as a real measurement.
    """
    caveats: List[str] = []
    signals: Dict[str, float] = {}

    # — text extraction — chars recovered relative to page count —
    sheets = result.get("sheets") or []
    if sheets:
        text_len = sum(len((s or {}).get("raw_text", "") or "") for s in sheets)
        pages = result.get("total_pages") or len(sheets) or 1
    else:
        text_len = len(result.get("text", "") or "")
        pages = result.get("total_pages") or 1
    text_score = min(1.0, _ratio(text_len, pages * 400))  # ~400 chars/page = solid
    signals["text_extraction"] = text_score
    if text_score < 0.3:
        caveats.append(
            "Very little text recovered — the document may be a scan or "
            "image-only and need OCR."
        )

    # — field coverage —
    if expected_fields:
        present = [f for f in expected_fields if _is_filled(result.get(f))]
        signals["field_coverage"] = _ratio(len(present), len(expected_fields))
        missing = [f for f in expected_fields if f not in present]
        if missing:
            caveats.append(f"Expected fields not populated: {', '.join(missing)}.")

    # — OCR character confidence (from the OCR block's quality report) —
    if ocr_quality is not None:
        ocr_conf = float(ocr_quality.get("ocr_confidence", 0) or 0)
        signals["ocr_char_confidence"] = round(ocr_conf, 3)
        if ocr_quality.get("caveat"):
            caveats.append(ocr_quality["caveat"])

    overall = round(sum(signals.values()) / len(signals), 3) if signals else 0.0

    return {
        "overall": overall,
        "signals": signals,
        "source_pages": pages,
        "measured": True,
        "caveats": caveats,
    }
