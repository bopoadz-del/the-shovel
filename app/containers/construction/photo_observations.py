"""Turn construction-detector output into daily-report observations.

The detector (`app/blocks/safety_world_detector.py`, YOLO-Worldv2 baked to
ONNX) returns raw detections:

    [{"class": "crack in concrete wall", "confidence": 0.62, "bbox": [...]}, ...]

`class` is the original prompt string from `app/blocks/safety_world_prompts.json`
(v4-hybrid, 33 prompts). This module splits that vocabulary by what a daily
report needs and applies the SAME confidence tiering the chat photo path
already uses, so one photograph cannot be described two different ways
depending on which surface the user came through.

OBSERVATIONS, NEVER VERDICTS
----------------------------
The detector's own docstring is binding: it "does NOT and MUST NOT label
anything as a 'violation', 'non-compliance', or 'PPE breach' -- those are
judgments the application layer makes with the operator in the loop."

So "crack in concrete wall detected (0.62)" is allowed and "structural defect"
is not. Nothing here infers cause, severity, or compliance. A crack is a thing
the camera saw; whether it matters is an engineer's call, and on a client
deliverable that distinction is the difference between a useful note and a
liability.

VOCABULARY SCOPE, NOT A BUG
---------------------------
The vocabulary is deliberately lopsided: 13 quality prompts against 3 for plant
and temporary works. Equipment recall is therefore low BY CONSTRUCTION -- an
excavator, telehandler, dumper or piling rig has no prompt string and cannot be
detected at any confidence. That is a vocabulary limit, not a failure of this
bridge, and it is not fixable here: the model is open-vocabulary, so widening
it means adding prompt strings and re-baking the ONNX.
"""
from __future__ import annotations

from typing import Any, Dict, List

# Mirrors app/routers/chat_photos.py. Duplicated deliberately rather than
# imported, because a container importing a router is a layering inversion --
# and pinned by tests/test_photo_observations.py, which fails if the two drift.
DETECTED_THRESHOLD = 0.30
LOW_CONF_THRESHOLD = 0.05

# Workmanship prompts, taken verbatim from safety_world_prompts.json.
# THIRTEEN, not twelve: "trip hazard on floor" sits between the concrete
# prompts in the JSON but is a hazard, not a workmanship observation, so it is
# excluded here and left to the safety path.
QUALITY_PROMPTS = (
    "crack in concrete wall",
    "crack in concrete floor",
    "porous holes in concrete surface",
    "rust on steel rebar",
    "exposed reinforcement bar",
    "water stain on wall",
    "peeling paint",
    "cracked floor tile",
    "missing grout between tiles",
    "uneven plaster surface",
    "dirty surface",
    "concrete honeycomb voids",
    "exposed aggregate in concrete surface",
)

# Plant and temporary works. Only three exist in the vocabulary, and
# "scaffolding" and "ladder" are temporary works rather than plant -- recorded
# with their category so the report does not call a ladder an item of plant.
PLANT_PROMPTS: Dict[str, str] = {
    "crane": "plant",
    "ladder": "temporary_works",
    "scaffolding": "temporary_works",
}


def tier_for(confidence: float) -> str:
    """`detected` at >= 0.30, `low_confidence` at >= 0.05, else `below_threshold`."""
    if confidence >= DETECTED_THRESHOLD:
        return "detected"
    if confidence >= LOW_CONF_THRESHOLD:
        return "low_confidence"
    return "below_threshold"


def phrase_for(label: str, confidence: float) -> str:
    """The report wording, matching the chat path's phrasing exactly."""
    if confidence >= DETECTED_THRESHOLD:
        return f"{label} detected"
    return f"possible {label} detected -- low confidence"


def _strongest_per_class(photo: Dict[str, Any], wanted) -> Dict[str, Dict[str, Any]]:
    """Best detection per class in one photo, ignoring anything under 0.05.

    One crack photographed from three angles is one observation, not three, so
    detections are collapsed per class and the highest confidence wins.
    """
    best: Dict[str, Dict[str, Any]] = {}
    for det in photo.get("detections") or []:
        label = (det.get("class") or "").strip()
        if not label or label not in wanted:
            continue
        try:
            conf = float(det.get("confidence") or 0.0)
        except (TypeError, ValueError):
            continue
        if conf < LOW_CONF_THRESHOLD:
            continue
        if conf > float(best.get(label, {}).get("confidence", 0.0)):
            best[label] = {"confidence": conf, "bbox": det.get("bbox")}
    return best


def quality_observations(photo_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Workmanship observations, one entry per class per photo.

    Empty when nothing was detected. Never invents an observation to make a
    report section look populated -- an empty list is the true answer and the
    report labels the section accordingly.
    """
    out: List[Dict[str, Any]] = []
    for photo in photo_analysis or []:
        for label, best in _strongest_per_class(photo, set(QUALITY_PROMPTS)).items():
            conf = best["confidence"]
            out.append({
                "source": "photo",
                "photo": photo.get("photo"),
                "observation": phrase_for(label, conf),
                "detected_class": label,
                "confidence": round(conf, 3),
                "confidence_tier": tier_for(conf),
                "bbox": best.get("bbox"),
            })
    out.sort(key=lambda o: -o["confidence"])
    return out


def equipment_from_photos(photo_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Plant and temporary works seen across the photo set.

    Deduped ACROSS photos, not just within one: a crane standing all day
    appears in every frame and is one item on site, not six. `seen_in_photos`
    keeps the evidence trail so a reader can go back to the source images.

    Only the three classes the vocabulary can detect are ever returned. The
    output is not padded with plant that has no prompt string.
    """
    merged: Dict[str, Dict[str, Any]] = {}
    for photo in photo_analysis or []:
        for label, best in _strongest_per_class(photo, set(PLANT_PROMPTS)).items():
            conf = best["confidence"]
            entry = merged.get(label)
            if entry is None:
                merged[label] = {
                    "item": label,
                    "category": PLANT_PROMPTS[label],
                    "source": "photo",
                    "observation": phrase_for(label, conf),
                    "confidence": round(conf, 3),
                    "confidence_tier": tier_for(conf),
                    "seen_in_photos": [photo.get("photo")],
                }
            else:
                entry["seen_in_photos"].append(photo.get("photo"))
                if conf > entry["confidence"]:
                    entry.update({
                        "confidence": round(conf, 3),
                        "confidence_tier": tier_for(conf),
                        "observation": phrase_for(label, conf),
                    })
    out = list(merged.values())
    out.sort(key=lambda e: -e["confidence"])
    return out
