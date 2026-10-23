"""Image-quality helpers for OCR — deskew + quality assessment.

Roadmap V2 · Epic 5 (Input quality handling).

Pure functions over PIL images / confidence lists, so they unit-test without
a Tesseract install.
"""

from typing import List, Tuple

import numpy as np
from PIL import Image


def _rotate_binary(binary: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a 0/1 array by `angle` degrees, filling new area with background."""
    img = Image.fromarray((binary * 255).astype(np.uint8))
    rotated = img.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(rotated, dtype=np.float64) / 255.0


def estimate_skew_angle(
    image: Image.Image, max_angle: float = 10.0, step: float = 0.5
) -> float:
    """Estimate page skew via projection-profile variance.

    When a text page is level, the horizontal projection (sum of dark pixels
    per row) has sharp peaks at text lines and troughs between them — high
    variance. Skew flattens it. The correcting angle is the one that maximises
    that variance. Returns degrees (the angle to ROTATE BY to straighten).
    """
    gray = image.convert("L")
    w, h = gray.size
    if max(w, h) > 1000:  # downscale for speed
        scale = 1000.0 / max(w, h)
        gray = gray.resize((max(1, int(w * scale)), max(1, int(h * scale))))

    arr = np.asarray(gray, dtype=np.float64)
    binary = (arr < arr.mean()).astype(np.float64)  # dark text -> 1

    best_angle, best_score = 0.0, -1.0
    angle = -max_angle
    while angle <= max_angle + 1e-9:
        row_sums = _rotate_binary(binary, angle).sum(axis=1)
        score = float(np.var(row_sums))
        if score > best_score:
            best_score, best_angle = score, angle
        angle += step
    return round(best_angle, 2)


def deskew(image: Image.Image, max_angle: float = 10.0) -> Tuple[Image.Image, float]:
    """Return (deskewed_image, angle_applied). No-op when skew is negligible."""
    angle = estimate_skew_angle(image, max_angle=max_angle)
    if abs(angle) < 0.5:
        return image, 0.0
    fill = 255 if image.mode in ("L", "RGB") else None
    return image.rotate(
        angle, resample=Image.BICUBIC, expand=True, fillcolor=fill
    ), angle


def summarize_ocr_quality(
    word_confidences: List[float], low_threshold: float = 0.6
) -> dict:
    """Turn per-word OCR confidences (each 0..1) into a verdict for the user.

    This replaces the old hardcoded 0.85 — a low-quality scan now produces a
    visibly lower score and an explicit caveat instead of pretending the text
    is reliable.
    """
    valid = [c for c in word_confidences if c is not None and c >= 0]
    if not valid:
        return {
            "ocr_confidence": 0.0,
            "words_measured": 0,
            "low_quality": True,
            "caveat": "No text could be read from this image with confidence.",
        }
    mean = sum(valid) / len(valid)
    low = mean < low_threshold
    return {
        "ocr_confidence": round(mean, 3),
        "words_measured": len(valid),
        "low_quality": low,
        "caveat": (
            "Low-quality scan — extracted text may be unreliable; "
            "verify critical values against the source document."
            if low
            else None
        ),
    }
