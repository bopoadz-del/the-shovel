"""Phase 0 -- Grounding DINO survey of a folder of photos against the safety/QA-QC class registry.

Usage:
    python scripts/survey_photo_corpus.py <folder> <output_json>

Outputs per-class detection counts so the operator can decide which classes
have enough examples to ship as V1 active. Heavy dep (transformers + torch
+ ~1 GB model download) is only imported when the script actually runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "app" / "blocks" / "safety_world_prompts.json"

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_BOX_THRESHOLD = 0.35
_TEXT_THRESHOLD = 0.25


def _load_prompts() -> List[str]:
    """Load the live Safety Observation AI prompt vocabulary."""
    data = json.loads(_PROMPTS_PATH.read_text(encoding="utf-8"))
    return list(data["prompts"])


def detect_with_dino(image_path: Path, class_names: List[str]) -> List[Dict]:
    """Run Grounding DINO on one image with the given open-vocab class prompts.
    Returns [{class, confidence, bbox: [x1,y1,x2,y2]}, ...]."""
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    if not hasattr(detect_with_dino, "_cache"):
        proc = AutoProcessor.from_pretrained(_MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(_MODEL_ID).eval()
        detect_with_dino._cache = (proc, model)
    proc, model = detect_with_dino._cache

    image = Image.open(image_path).convert("RGB")
    # DINO prompt is the literal prompt phrases separated by periods.
    prompt = ". ".join(class_names) + "."
    inputs = proc(images=image, text=prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    results = proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=_BOX_THRESHOLD, text_threshold=_TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    name_lookup = {name: name for name in class_names}
    out = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        canon = name_lookup.get(label)
        if canon is None:
            # DINO's open-vocab decoder occasionally bleeds across adjacent
            # period-separated prompts, yielding labels like "no hardhat re
            # no plastic cap" that aren't any single requested class. Drop
            # them; counting them would be guesswork.
            continue
        out.append({"class": canon, "confidence": float(score), "bbox": [float(x) for x in box.tolist()]})
    return out


def survey_folder(folder: Path, output_json: Path) -> Dict:
    class_names = _load_prompts()
    per_class = {n: {"detections": 0, "photos_with_at_least_one": 0} for n in class_names}
    per_photo: List[Dict] = []

    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    for img in images:
        detections = detect_with_dino(img, class_names)
        by_class: Dict[str, int] = {}
        for d in detections:
            by_class[d["class"]] = by_class.get(d["class"], 0) + 1
        for name, count in by_class.items():
            per_class[name]["detections"] += count
            per_class[name]["photos_with_at_least_one"] += 1
        per_photo.append({"filename": img.name, "detections_by_class": by_class})

    report = {
        "folder": str(folder),
        "total_images": len(images),
        "model": _MODEL_ID,
        "box_threshold": _BOX_THRESHOLD,
        "text_threshold": _TEXT_THRESHOLD,
        "per_class": per_class,
        "per_photo": per_photo,
    }
    output_json.write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    p.add_argument("output_json", type=Path)
    args = p.parse_args()
    survey_folder(args.folder, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
