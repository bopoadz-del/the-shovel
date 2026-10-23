"""YOLO-Worldv2 detector with a pre-baked prompt vocabulary.

PRODUCT NAME: "Safety Observation AI v2". The model surfaces what it
SEES (vest, hat, person, crack...). It does NOT and MUST NOT label
anything as a "violation", "non-compliance", or "PPE breach" -- those
are judgments the application layer makes with the operator in the
loop.

The .onnx file at SAFETY_WORLD_WEIGHTS was produced by
``scripts/bake_world_model.py`` followed by an ONNX export. The bake
step calls ``YOLOWorld.set_classes(prompts)`` (CLIP runs there) then
saves; the export step writes a self-contained .onnx with the CLIP-
derived text vectors reparameterized into the classifier head. The
.onnx loads with plain ``YOLO()`` -- no CLIP import is needed at
inference time, which is why this module never touches CLIP.

Detections are returned with the original prompt strings (e.g.
"high visibility vest") rather than registry ids, because the class
list is defined by the prompt JSON, not by safety_classes.json.

Env vars
========
- ``SAFETY_WORLD_WEIGHTS``  path to the .onnx; if unset, the weights baked
  into the image (``BAKED_WEIGHTS``) are used when present, else no detector
- ``SAFETY_WORLD_CONF``     default confidence threshold (0.0-1.0); falls back to 0.05
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

# Render / non-root containers can't write ~/.config/Ultralytics.
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

logger = logging.getLogger(__name__)

# Populated on first use by _load_yolo(), NOT at import — see below. Kept as a
# module attribute so `patch("...safety_world_detector.YOLO", ...)` remains the
# seam tests hook, exactly as when this was a module-level import.
YOLO = None


def _load_yolo():
    """Import ultralytics on first detector construction, not at module import.

    `from ultralytics import YOLO` costs ~265 MB of RSS because it pulls in
    torch and cv2 — measured 2026-08-11, the single largest item in this app's
    boot. app/blocks/__init__.py imports every block eagerly, so that cost was
    paid on EVERY boot even where the feature is off: SAFETY_WORLD_WEIGHTS is
    unset on the Render service, so default_detector() returns None and no
    model is ever built. Deferring it here keeps the whole 265 MB out of any
    deployment that has not configured weights.

    Returns the YOLO class, or None when ultralytics is not installed — the
    same contract the module-level `YOLO = None` fallback provided.
    """
    global YOLO
    if YOLO is not None:  # already imported, or patched by a test
        return YOLO
    try:
        from ultralytics import YOLO as _YOLO
    except ImportError:
        return None
    YOLO = _YOLO
    return YOLO


class SafetyWorldDetector:
    def __init__(self, weights_path: Path) -> None:
        YOLO = _load_yolo()
        if YOLO is None:
            raise RuntimeError("ultralytics not installed")
        self._weights_path = Path(weights_path)
        self._model = YOLO(str(self._weights_path))
        # YOLO reads the baked class names directly from the .pt -- these
        # are the same prompt strings passed to set_classes() at bake time.
        self._names: Dict[int, str] = dict(self._model.names)
        # Load companion manifest if it exists, purely for provenance.
        manifest = self._weights_path.with_suffix(".prompts.json")
        if manifest.is_file():
            try:
                self._manifest = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                self._manifest = None
        else:
            self._manifest = None

    @property
    def weights_path(self) -> Path:
        return self._weights_path

    @property
    def class_names(self) -> List[str]:
        return [self._names[i] for i in sorted(self._names)]

    @property
    def manifest(self) -> Optional[dict]:
        return self._manifest

    def detect(self, file_path: Path, conf_threshold: float = 0.25) -> List[Dict]:
        """Run detection. Returns a list of dicts:
            [{class: "high visibility vest", confidence: 0.71, bbox: [x1,y1,x2,y2]}, ...]
        """
        results = self._model(str(file_path), conf=conf_threshold, verbose=False)
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        cls_indices = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        boxes = r.boxes.xyxy.tolist()

        out: List[Dict] = []
        for idx, conf, box in zip(cls_indices, confs, boxes):
            name = self._names.get(int(idx))
            if name is None:
                continue
            out.append({
                "class": name,
                "confidence": float(conf),
                "bbox": [float(x) for x in box],
            })
        return out


#: Where the image bakes the detector weights (the Dockerfile copies
#: data/models/safety_world_v2.onnx here, off the data volume). Used when
#: SAFETY_WORLD_WEIGHTS is unset: the Render service set the variable, the
#: ECS task definition never did, and every chat photo came back with no
#: observations while the weights sat in the image. A dev checkout opts in
#: with SAFETY_WORLD_WEIGHTS.
BAKED_WEIGHTS = (Path("/app/models/safety_world_v2.onnx"),)


def _weights_path() -> Optional[Path]:
    weights_env = os.getenv("SAFETY_WORLD_WEIGHTS")
    if weights_env:
        return Path(weights_env)
    return next((p for p in BAKED_WEIGHTS if p.is_file()), None)


#: One loaded model per weights file. Callers ask per photo; building the
#: ONNX model each time cost seconds and a fresh allocation per request.
_LOADED: Dict[Path, SafetyWorldDetector] = {}


def default_detector() -> Optional[SafetyWorldDetector]:
    weights = _weights_path()
    if weights is None:
        return None
    if weights in _LOADED:
        return _LOADED[weights]
    if not weights.is_file():
        logger.warning("SAFETY_WORLD_WEIGHTS=%s not found on disk", weights)
        return None
    try:
        detector = SafetyWorldDetector(weights)
    except Exception:
        logger.exception("failed to load SafetyWorldDetector from %s", weights)
        return None
    _LOADED[weights] = detector
    return detector
