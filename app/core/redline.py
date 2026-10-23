"""Redline / markup detection — colour-channel analysis for OCR inputs.

Roadmap V2 · Epic 5 (Input quality handling).

Construction redlines are coloured pen/markup (red, blue, green) drawn over an
otherwise black-and-white printed drawing. Printed linework is greyscale —
black ink on white paper has (near-)zero colour saturation. Coloured,
high-saturation pixels therefore indicate hand annotation.

This module identifies those pixels, clusters them into bounding boxes, and
returns a verdict so the OCR flow can FLAG annotated regions rather than
mangling them into the extracted text.

Pure functions over PIL images / numpy arrays — they unit-test without a
Tesseract install, matching the convention of `image_quality.py`.
"""

from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# A pixel counts as "coloured ink" when its HSV saturation and value both clear
# these thresholds. Mid-grey, black and white all have saturation ~0 and are
# excluded; faint scanner colour-noise is excluded by the value floor.
_SATURATION_THRESHOLD = 0.30
_VALUE_THRESHOLD = 0.20

# Below this fraction of coloured pixels the image is treated as a clean scan.
_COVERAGE_THRESHOLD = 0.001

# Minimum pixel area for a cluster to be reported as a region (drops specks).
_MIN_REGION_AREA = 40


def _colour_mask(image: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mask, rgb) — `mask` is a bool array of coloured-ink pixels.

    A greyscale-mode image can hold no colour, so its mask is all-False.
    """
    if image.mode == "L" or image.mode == "1":
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.zeros(rgb.shape[:2], dtype=bool), rgb

    hsv = np.asarray(image.convert("HSV"), dtype=np.float64) / 255.0
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = (saturation >= _SATURATION_THRESHOLD) & (value >= _VALUE_THRESHOLD)
    return mask, rgb


def _label_clusters(mask: np.ndarray) -> List[np.ndarray]:
    """Cluster True pixels into connected components (4-connectivity).

    A small flood-fill labeller — avoids a scipy dependency. Returns a list of
    (N, 2) arrays of [row, col] coordinates, one per component.
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters: List[np.ndarray] = []

    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            # iterative flood fill from this seed
            stack = [(sr, sc)]
            visited[sr, sc] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            clusters.append(np.asarray(coords, dtype=np.int64))
    return clusters


def _dominant_colour(rgb: np.ndarray, coords: np.ndarray) -> str:
    """Classify a cluster's mean RGB into 'red', 'green' or 'blue'."""
    pixels = rgb[coords[:, 0], coords[:, 1]].astype(np.float64)
    mean = pixels.mean(axis=0)  # R, G, B
    channel = int(np.argmax(mean))
    return ("red", "green", "blue")[channel]


def detect_redlines(
    image: Image.Image,
    coverage_threshold: float = _COVERAGE_THRESHOLD,
    min_region_area: int = _MIN_REGION_AREA,
) -> Dict:
    """Detect coloured markup / redlines on a drawing or scan.

    Identifies coloured (high-saturation, non-grey) pixels, clusters them into
    annotated regions, and returns a verdict:

        {
            "has_markup":   bool,    # any markup above the coverage floor
            "coverage":     float,   # fraction of image area that is coloured ink
            "regions": [             # one entry per clustered annotation
                {
                    "bbox": (x0, y0, x1, y1),   # pixel bounding box
                    "dominant_colour": "red" | "green" | "blue",
                    "pixels": int,              # coloured pixels in the cluster
                },
                ...
            ],
        }

    Greyscale-mode images and pure-grey images contain no colour and are
    always reported as clean.
    """
    mask, rgb = _colour_mask(image)
    total = mask.size
    coloured = int(mask.sum())
    coverage = round(coloured / total, 6) if total else 0.0

    if coverage < coverage_threshold:
        return {"has_markup": False, "coverage": coverage, "regions": []}

    regions: List[Dict] = []
    for coords in _label_clusters(mask):
        if len(coords) < min_region_area:
            continue
        rows, cols = coords[:, 0], coords[:, 1]
        regions.append(
            {
                # bbox as (x0, y0, x1, y1); +1 so the box is inclusive of edges
                "bbox": (
                    int(cols.min()),
                    int(rows.min()),
                    int(cols.max()) + 1,
                    int(rows.max()) + 1,
                ),
                "dominant_colour": _dominant_colour(rgb, coords),
                "pixels": int(len(coords)),
            }
        )

    # Largest annotations first — most relevant to the user.
    regions.sort(key=lambda r: r["pixels"], reverse=True)

    return {
        "has_markup": len(regions) > 0,
        "coverage": coverage,
        "regions": regions,
    }


def summarize_markup(result: Dict) -> Dict:
    """Turn a `detect_redlines` result into a verdict for the user.

    Mirrors `image_quality.summarize_ocr_quality` — a marked-up input gets an
    explicit caveat so the chat reply can flag it instead of presenting the
    annotated drawing as clean extracted data.
    """
    if not result.get("has_markup"):
        return {
            "has_markup": False,
            "coverage": result.get("coverage", 0.0),
            "region_count": 0,
            "caveat": None,
        }

    regions = result.get("regions", [])
    colours = sorted({r["dominant_colour"] for r in regions})
    colour_phrase = "/".join(colours) if colours else "coloured"
    return {
        "has_markup": True,
        "coverage": result.get("coverage", 0.0),
        "region_count": len(regions),
        "caveat": (
            f"This image contains {colour_phrase} markup / redlines "
            f"({len(regions)} annotated region(s)) — those areas were flagged, "
            "not merged into the extracted text; review them against the source."
        ),
    }
