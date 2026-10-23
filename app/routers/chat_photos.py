"""POST /v1/chat/analyze-photo -- Safety Observation AI v2.

This endpoint returns OBSERVATIONS, never violations. The model is
"Safety Observation AI v2" -- it tells you what it SEES (vest, hat,
person, crack, crane), with confidence tiers. It does NOT, and must
NOT, label anything as a "violation", "non-compliance", or "PPE
breach". Those are judgments the user makes with the operator in the
loop.

Chat-attached photos are question-context, not corpus material.
Routing them through ``/v1/projects/<pid>/documents`` forced
ownership checks that 404'd whenever a user opened a shared admin-
approved project. This endpoint:
  * accepts any signed-in user (no project ownership required)
  * writes the photo to an ephemeral temp file
  * runs YOLO-Worldv2 ONNX inference via the image block
  * returns observations + tiered vest/hat verdicts
  * deletes the temp file before returning

Confidence tiers (operator-defined, NEVER label as violations):
  conf >= 0.30        -> "vest detected"
  0.05 <= conf < 0.30 -> "possible vest detected -- low confidence"
  person present + no vest at any conf -> "no vest detected in image"
The hat tier mirrors the same thresholds.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import require_user

router = APIRouter()
logger = logging.getLogger(__name__)

_MAX_PHOTO_BYTES = 25 * 1024 * 1024  # 25 MB
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif"}

# Confidence tier thresholds. Held as module constants so the chat composer
# tag and the backend stay in lockstep -- change here, both update.
_DETECTED_THRESHOLD = 0.30
_LOW_CONF_THRESHOLD = 0.05

# Class-name fragments grouped per concept. Inference returns the original
# prompt string, so we group by substring rather than exact name. This
# survives prompt re-baking without code change as long as the substring
# stays present.
_VEST_FRAGMENTS = ("vest", "high visibility", "high-visibility")
_HAT_FRAGMENTS = ("hard hat", "helmet")
_PERSON_FRAGMENTS = ("person",)


def _matches(name: str, fragments) -> bool:
    n = (name or "").lower()
    return any(f in n for f in fragments)


def _tier_for(detections: List[Dict[str, Any]], fragments) -> Dict[str, Any]:
    """Tiered verdict for one concept (e.g. vest). Returns:
        {tier: "detected"|"low_confidence"|"not_detected",
         max_confidence: float, count: int, message: str}
    Never returns a violation verdict -- only what the model saw.
    """
    matches = [d for d in detections if _matches(d.get("class") or "", fragments)]
    if not matches:
        return {"tier": "not_detected", "max_confidence": 0.0, "count": 0, "message": ""}
    top_conf = max(float(d.get("confidence") or 0.0) for d in matches)
    label = fragments[0]
    if top_conf >= _DETECTED_THRESHOLD:
        return {"tier": "detected", "max_confidence": top_conf, "count": len(matches),
                "message": f"{label} detected"}
    if top_conf >= _LOW_CONF_THRESHOLD:
        return {"tier": "low_confidence", "max_confidence": top_conf, "count": len(matches),
                "message": f"possible {label} detected -- low confidence"}
    return {"tier": "not_detected", "max_confidence": top_conf, "count": 0, "message": ""}


def _person_observation(detections: List[Dict[str, Any]],
                        vest_tier: Dict[str, Any]) -> Optional[str]:
    """Add a 'no vest detected in image' observation when a person is
    present at a confident threshold AND no vest matched at any tier.
    Plain observation -- no 'violation' wording."""
    person_present = any(
        _matches(d.get("class") or "", _PERSON_FRAGMENTS)
        and float(d.get("confidence") or 0.0) >= _DETECTED_THRESHOLD
        for d in detections
    )
    if person_present and vest_tier["tier"] == "not_detected":
        return "no vest detected in image"
    return None


def _people_observation(person_count: int, observations: List[str]) -> Optional[str]:
    """How many people the general detector counted, when no other
    observation already speaks about people. The safety model may be absent
    or see nothing; the person count still tells the chat who is on site."""
    if person_count <= 0:
        return None
    if any("person" in o.lower() or "people" in o.lower() for o in observations):
        return None
    return "1 person visible" if person_count == 1 else f"{person_count} people visible"


# Classes whose surfacing is handled by the dedicated vest/hat/person logic
# above. Everything else above _LOW_CONF_THRESHOLD gets surfaced through the
# generic "other observations" path -- this is what makes concrete defects,
# excavation hazards, ladders, cranes etc. visible to the user instead of
# being silently dropped because they aren't PPE.
_HANDLED_FRAGMENTS = _VEST_FRAGMENTS + _HAT_FRAGMENTS + _PERSON_FRAGMENTS


def _other_observations(detections: List[Dict[str, Any]]) -> List[str]:
    """Surface every non-vest/hat/person class as a tiered observation.

    Groups detections by class, takes the max confidence per class, and
    emits 'X detected' (conf >= 0.30) or 'possible X detected -- low
    confidence' (0.05 <= conf < 0.30). Sorted by max confidence so the
    LLM sees the strongest signal first.

    Never returns 'violation'-style language for any class -- same
    observation-not-judgment contract as vest/hat.
    """
    by_class: Dict[str, float] = {}
    for d in detections:
        cls_name = (d.get("class") or "").strip()
        if not cls_name:
            continue
        if _matches(cls_name, _HANDLED_FRAGMENTS):
            continue
        conf = float(d.get("confidence") or 0.0)
        if conf < _LOW_CONF_THRESHOLD:
            continue
        if conf > by_class.get(cls_name, 0.0):
            by_class[cls_name] = conf
    out: List[str] = []
    for cls_name, conf in sorted(by_class.items(), key=lambda kv: -kv[1]):
        if conf >= _DETECTED_THRESHOLD:
            out.append(f"{cls_name} detected")
        else:
            out.append(f"possible {cls_name} detected -- low confidence")
    return out


@router.post("/v1/chat/analyze-photo")
async def analyze_chat_photo(
    file: UploadFile = File(...),
    auth: dict = Depends(require_user),
) -> Dict[str, Any]:
    original_name = (file.filename or "photo.jpg").strip() or "photo.jpg"
    original_name = os.path.basename(original_name.replace("\\", "/"))
    _, ext = os.path.splitext(original_name.lower())
    if ext not in _IMAGE_EXTS:
        raise HTTPException(400, f"Not an image extension: {ext or '<none>'}")

    data = await file.read()
    if len(data) > _MAX_PHOTO_BYTES:
        raise HTTPException(413, f"Photo exceeds {_MAX_PHOTO_BYTES // (1024*1024)} MB limit")
    if not data:
        raise HTTPException(400, "Empty file")

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix="chatphoto_")
    try:
        os.write(tmp_fd, data)
    finally:
        os.close(tmp_fd)

    try:
        from app.dependencies import get_block_instance
        image_block = get_block_instance("image")
        if image_block is None:
            raise HTTPException(503, "Image block not available")

        analysis = await image_block.execute(
            {"file_path": tmp_path},
            {"mode": "safety_qaqc", "prompt": "Safety Observation AI v2 scan",
             "safety_qaqc_conf": _LOW_CONF_THRESHOLD},
        )
        body = analysis.get("result", {}) or {}
        detections = body.get("safety_qaqc") or []
        coco = body.get("summary_by_class") or {}
        person_count = int(coco.get("person", 0))

        # Tiered observations -- never "violations".
        vest_tier = _tier_for(detections, _VEST_FRAGMENTS)
        hat_tier = _tier_for(detections, _HAT_FRAGMENTS)
        observations: List[str] = []
        if vest_tier["message"]:
            observations.append(vest_tier["message"])
        if hat_tier["message"]:
            observations.append(hat_tier["message"])
        no_vest_obs = _person_observation(detections, vest_tier)
        if no_vest_obs:
            observations.append(no_vest_obs)
        # Surface QA/QC + general-hazard classes too (crack, porous holes,
        # excavation, ladder, etc.). Vest/hat already handled above.
        observations.extend(_other_observations(detections))
        people = _people_observation(person_count, observations)
        if people:
            observations.insert(0, people)

        # Strongest-class summary for any caller that wants the raw top list.
        # Truncated to 8 entries; sorted by confidence descending.
        summary = None
        if detections:
            top = sorted(
                ({"class": d.get("class"),
                  "confidence": round(float(d.get("confidence") or 0.0), 3)}
                 for d in detections),
                key=lambda d: -d["confidence"],
            )[:8]
            summary = {"count": len(detections), "top": top}

        return {
            "filename": original_name,
            "size_bytes": len(data),
            "person_count": person_count,
            "safety_qaqc": summary,
            "observations": observations,
            "vest": vest_tier,
            "hat": hat_tier,
            "_product_name": "Safety Observation AI v2",
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            logger.warning(
                "swallowed %s in analyze_chat_photo() — continuing",
                "OSError", exc_info=True,
            )
