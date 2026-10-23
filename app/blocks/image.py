"""Image Block — local-only image analysis (PIL metadata + Tesseract OCR + YOLO).

No external cloud vision API. All processing runs in-process on the host:

- ``metadata`` / default — PIL-based dimensions, mode, format, file size,
  megapixels, aspect ratio, dominant colour channel.
- ``extract_text`` — Tesseract OCR (already a project dependency) for
  text-bearing images. Returns the extracted text plus a confidence proxy.
- ``detect_objects`` — YOLOv8n object detection (PR 3b). Optional dep
  on ``ultralytics`` (``requirements-cv.txt``); returns
  ``{available: false}`` when missing rather than raising.
- ``construction`` / ``analyze`` — combines metadata + OCR + optional
  detection. Detection results appear under ``detections`` only when
  ultralytics is installed.

The block exposes ``provider`` so callers can see the local backend that
served the response (``pil``, ``tesseract``, ``pil+tesseract+yolo``).
"""

import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from app.core.universal_base import UniversalBlock
import logging

logger = logging.getLogger(__name__)

_MAX_IMAGE_BYTES = 15 * 1024 * 1024  # 15 MB — generous for local processing
# Leftover M16 ``trench_site.jpg`` is a 128×128 / ~884 B PIL stub. Treat
# that class of file as unreadable so the agent answers from stated
# trench dimensions instead of emitting an empty bubble.
_PLACEHOLDER_MAX_EDGE = 160
_PLACEHOLDER_MAX_BYTES = 2048

# Default YOLO model — COCO-trained, ~6 MB. Override via ``YOLO_MODEL`` env
# for custom fine-tunes (e.g. a PPE-aware model trained on site photos).
_DEFAULT_YOLO_MODEL = "yolov8n.pt"



async def _download_to_temp(url: str) -> str:
    """Download a URL to a temporary file and return its path."""
    import tempfile
    import httpx

    async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
        resp = await client.get(url)
        resp.raise_for_status()
    content_type = resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    suffix = "." + content_type.split("/")[-1] if "/" in content_type else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(resp.content)
        return f.name


def _pil_metadata(file_path: str) -> Dict:
    from PIL import Image
    from app.core.file_crypto import open_plaintext

    # Decrypt-on-read for encrypted-at-rest uploads; no-op for plaintext.
    # PIL needs a real path (it sniffs format from the bytes), and the
    # context manager keeps the temp file alive for the metadata pass below.
    with open_plaintext(file_path) as plain_path:
        img = Image.open(plain_path)
        width, height = img.size
        mode = img.mode
        fmt = img.format or Path(file_path).suffix.upper().lstrip(".")
        file_size = os.path.getsize(plain_path)

    info = {
        "width": width,
        "height": height,
        "mode": mode,
        "format": fmt,
        "file_size_bytes": file_size,
        "megapixels": round(width * height / 1_000_000, 2),
        "aspect_ratio": f"{width}:{height}",
    }

    if mode in ("RGB", "RGBA"):
        r, g, b = img.convert("RGB").split()
        info["dominant_channel"] = max(
            ("red", _avg(r)), ("green", _avg(g)), ("blue", _avg(b)),
            key=lambda x: x[1],
        )[0]

    return info


def _placeholder_image_error(file_path: str) -> Optional[Dict]:
    """Reject leftover-style stub JPEGs (128×128 / ~1 KB) as unreadable.

    YOLO unit tests write 32×32 PNGs (~100 B). Those must still process.
    A leftover site-photo stub is a JPEG around 128×128 and under 2 KB.
    """
    try:
        size = os.path.getsize(file_path)
    except OSError:
        return None
    width = height = 0
    plain_size = size
    fmt = ""
    try:
        from PIL import Image
        from app.core.file_crypto import open_plaintext

        with open_plaintext(file_path) as plain_path:
            img = Image.open(plain_path)
            width, height = img.size
            plain_size = os.path.getsize(plain_path)
            fmt = (img.format or Path(file_path).suffix.lstrip(".")).upper()
    except Exception:  # noqa: BLE001 — only leftover JPEGs are in scope
        return None
    leftover_stub = (
        fmt in {"JPEG", "JPG"}
        and 64 <= width <= _PLACEHOLDER_MAX_EDGE
        and 64 <= height <= _PLACEHOLDER_MAX_EDGE
        and plain_size <= _PLACEHOLDER_MAX_BYTES
    )
    if not leftover_stub:
        return None
    return {
        "status": "error",
        "error": (
            "Image is a placeholder / too small to analyze "
            f"({width}x{height}, {plain_size} bytes). "
            "Answer from stated dimensions; do not retry the photo."
        ),
        "placeholder": True,
        "width": width,
        "height": height,
        "file_size_bytes": plain_size,
    }


def _avg(channel) -> float:
    import numpy as np
    return float(np.array(channel).mean())


def _tesseract_ocr(file_path: str) -> Tuple[str, float]:
    """Run Tesseract OCR locally. Returns (text, mean_confidence_0_to_1)."""
    import pytesseract
    from PIL import Image
    from app.core.file_crypto import open_plaintext

    # Decrypt-on-read for encrypted-at-rest uploads; no-op for plaintext.
    # Both pytesseract calls must run INSIDE the with-block — PIL lazy-loads
    # pixel data, and the temp file gets removed when the context manager
    # exits, so the second call would otherwise hit a missing file.
    confidence = 0.0
    with open_plaintext(file_path) as plain_path:
        img = Image.open(plain_path)
        text = pytesseract.image_to_string(img).strip()
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c).isdigit() and int(c) >= 0]
            if confs:
                confidence = sum(confs) / len(confs) / 100.0
        except Exception:
            logger.debug(
                "swallowed %s in _tesseract_ocr() — continuing",
                "Exception", exc_info=True,
            )

    return text, confidence


# ── YOLO object detection (PR 3b) ─────────────────────────────────────────
# One model load per process — ultralytics' SentenceTransformer-equivalent
# pattern: instantiate once, infer many. The model is ~6MB for v8n and
# loads in ~1-2s on CPU.

_YOLO_MODEL_CACHE: Dict[str, Any] = {}
_YOLO_LOCK = Lock()


def _yolo_available() -> bool:
    """True when the ``ultralytics`` package is importable. Distinct from
    "model file exists" — ultralytics auto-downloads weights on first use
    from its own CDN (not Hugging Face), so callers can rely on
    ``available()`` to gate the operation without separately checking
    for the weights file."""
    try:
        import ultralytics  # noqa: F401
        return True
    except ImportError:
        return False


def _get_yolo_model(model_name: Optional[str] = None):
    """Return a process-cached ``YOLO`` model instance.

    First call loads the model (slow); subsequent calls return the cache.
    Raises ``RuntimeError`` when ultralytics isn't installed — callers
    should gate with :func:`_yolo_available` and degrade gracefully.
    """
    name = model_name or os.getenv("YOLO_MODEL") or _DEFAULT_YOLO_MODEL
    with _YOLO_LOCK:
        if name not in _YOLO_MODEL_CACHE:
            if not _yolo_available():
                raise RuntimeError(
                    "ultralytics not installed. Install with "
                    "`pip install -r requirements-cv.txt` to enable detection."
                )
            from ultralytics import YOLO
            _YOLO_MODEL_CACHE[name] = YOLO(name)
        return _YOLO_MODEL_CACHE[name]


def _reset_yolo_cache() -> None:
    """Drop cached YOLO models. Used by tests to swap fakes cleanly."""
    global _YOLO_MODEL_CACHE
    with _YOLO_LOCK:
        _YOLO_MODEL_CACHE = {}


def _yolo_detect(
    file_path: str,
    conf_threshold: float = 0.25,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run object detection on the image. Returns a list of detections
    each shaped ``{class_name, confidence, box: [x1, y1, x2, y2]}``.

    Empty list means "no objects above threshold detected" — not an
    error. Raises when ultralytics isn't installed; callers should
    gate with :func:`_yolo_available`.
    """
    from app.core.file_crypto import open_plaintext

    model = _get_yolo_model(model_name)
    # Uploaded files are Fernet-encrypted at rest; decrypt to a temp path
    # for ultralytics, which uses cv2.imread under the hood and would see
    # the encrypted ciphertext as a broken JPEG.
    with open_plaintext(file_path) as plain_path:
        results = model(str(plain_path), conf=conf_threshold, verbose=False)
    detections: List[Dict[str, Any]] = []
    for result in results:
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        if boxes is None:
            continue
        # ultralytics returns torch tensors; iterate per detection
        for i in range(len(boxes)):
            try:
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].tolist()
                detections.append({
                    "class_name": names.get(cls_id, str(cls_id)),
                    "confidence": round(conf, 3),
                    "box": [round(float(v), 1) for v in xyxy],
                })
            except Exception:
                # Skip malformed rows rather than aborting the whole batch
                continue
    return detections


def _summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Group detections by class for the human-readable summary line.

    e.g. ``[{class_name: "person", ...} × 3, {class_name: "car", ...}]`` →
    ``{"person": 3, "car": 1}``. Stable ordering by count desc, class asc."""
    counts: Dict[str, int] = {}
    for d in detections:
        cls = d.get("class_name") or "unknown"
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


class ImageBlock(UniversalBlock):
    """Local image analysis — PIL metadata + Tesseract OCR. No cloud calls."""

    auto_validate = False
    name = "image"
    version = "3.0"
    description = "Local image analysis — metadata + OCR (no cloud vision)"
    layer = 3
    tags = ["domain", "vision", "image", "local"]
    requires = []

    ui_schema = {
        "input": {
            "type": "image",
            "accept": [".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"],
            "placeholder": "Upload image to analyze...",
            "multiline": True,
        },
        "output": {
            "type": "text",
            "fields": [
                {"name": "description", "type": "markdown", "label": "Analysis"},
                {"name": "extracted_text", "type": "text", "label": "Text"},
                {"name": "metadata", "type": "object", "label": "Metadata"},
            ],
        },
        "quick_actions": [
            {"icon": "️", "label": "Metadata", "prompt": "Show image metadata"},
            {"icon": "", "label": "Construction", "prompt": "Analyze construction image"},
            {"icon": "", "label": "Extract Text", "prompt": "Extract all text"},
        ],
    }

    async def process(self, input_data: Any, params: Dict = None) -> Dict:
        params = params or {}
        operation = params.get("operation", "analyze")

        file_path, tmp_path = await self._resolve_source(input_data)
        if isinstance(file_path, dict):
            return file_path  # error dict

        try:
            file_size = os.path.getsize(file_path)
            if file_size > _MAX_IMAGE_BYTES:
                return {"status": "error", "error": f"Image too large (max {_MAX_IMAGE_BYTES // (1024*1024)} MB)"}

            stub = _placeholder_image_error(file_path)
            if stub:
                return stub

            if operation == "metadata":
                meta = _pil_metadata(file_path)
                return {"status": "success", "operation": "metadata", "provider": "pil", **meta}

            if operation == "extract_text":
                text, confidence = _tesseract_ocr(file_path)
                if not text:
                    return {
                        "status": "success",
                        "operation": "extract_text",
                        "provider": "tesseract",
                        "extracted_text": "",
                        "confidence": confidence,
                        "note": "No text detected by Tesseract.",
                    }
                return {
                    "status": "success",
                    "operation": "extract_text",
                    "provider": "tesseract",
                    "extracted_text": text,
                    "confidence": confidence,
                    "word_count": len(text.split()),
                }

            # ── PR 3b: object detection tier ────────────────────────────
            # Graceful no-op when ultralytics isn't installed — same
            # pattern as RAG's available() check. Operators get a clear
            # "install requirements-cv.txt" message rather than a stack
            # trace.
            if operation in ("detect_objects", "detect", "detection"):
                if not _yolo_available():
                    return {
                        "status": "success",
                        "operation": operation,
                        "provider": "yolo",
                        "available": False,
                        "detections": [],
                        "note": (
                            "Object detection unavailable: ultralytics not installed. "
                            "Install with `pip install -r requirements-cv.txt`."
                        ),
                    }
                conf = float(params.get("conf_threshold", 0.25))
                try:
                    detections = _yolo_detect(file_path, conf_threshold=conf)
                except Exception as e:
                    return {
                        "status": "error",
                        "operation": operation,
                        "provider": "yolo",
                        "error": f"YOLO detection failed: {e}",
                    }
                return {
                    "status": "success",
                    "operation": operation,
                    "provider": "yolo",
                    "available": True,
                    "model": os.getenv("YOLO_MODEL") or _DEFAULT_YOLO_MODEL,
                    "conf_threshold": conf,
                    "detections": detections,
                    "summary_by_class": _summarize_detections(detections),
                    "detection_count": len(detections),
                }

            # analyze / construction → combined local summary
            meta = _pil_metadata(file_path)
            ocr_text = ""
            ocr_conf = 0.0
            ocr_error = None
            try:
                ocr_text, ocr_conf = _tesseract_ocr(file_path)
            except Exception as e:
                ocr_error = str(e)

            # Optional detection tier — only invoked when ultralytics is
            # installed AND the operation hints at it (analyze/construction).
            # Failures are non-fatal so the metadata+OCR result still ships.
            detections: List[Dict[str, Any]] = []
            detection_error: Optional[str] = None
            yolo_used = False
            if _yolo_available():
                try:
                    detections = _yolo_detect(file_path)
                    yolo_used = True
                except Exception as e:
                    detection_error = str(e)

            # Optional safety/QA-QC tier — YOLO-Worldv2 with a baked
            # prompt vocabulary (see app/blocks/safety_world_prompts.json
            # and scripts/bake_world_model.py). Only runs when
            # mode=safety_qaqc is explicitly requested AND
            # SAFETY_WORLD_WEIGHTS env var points at a usable baked .pt.
            # Failures are non-fatal. CLIP is NOT imported here -- the
            # text vectors are reparameterized into the .pt at bake time.
            safety_qaqc: List[Dict[str, Any]] = []
            safety_qaqc_error: Optional[str] = None
            safety_qaqc_used = False
            if params.get("mode") == "safety_qaqc":
                from app.blocks.safety_world_detector import default_detector
                from app.core.file_crypto import open_plaintext
                detector = default_detector()
                if detector is not None:
                    try:
                        # Decrypt the at-rest-encrypted upload before
                        # handing the path to ultralytics — same reason
                        # _yolo_detect wraps with open_plaintext.
                        with open_plaintext(file_path) as plain_path:
                            safety_qaqc = detector.detect(
                                Path(plain_path),
                                conf_threshold=float(params.get("safety_qaqc_conf", 0.25)),
                            )
                        safety_qaqc_used = True
                    except Exception as e:
                        safety_qaqc_error = str(e)

            description = self._compose_local_summary(
                operation=operation, meta=meta, ocr_text=ocr_text, ocr_conf=ocr_conf, ocr_error=ocr_error,
            )

            # Append the detection summary to the human-readable description
            # so any consumer that only reads `description` still sees the
            # CV signal without parsing the structured field.
            if detections:
                summary = _summarize_detections(detections)
                top = ", ".join(f"{n} × {c}" for n, c in list(summary.items())[:5])
                description = f"{description}\n\n**Detected**: {top}"
            if safety_qaqc:
                safety_summary = ", ".join(
                    f"{d['class']} ({d['confidence']:.2f})" for d in safety_qaqc[:5]
                )
                description = f"{description}\n\n**Safety/QA-QC**: {safety_summary}"

            provider_parts = ["pil"]
            if ocr_text:
                provider_parts.append("tesseract")
            if yolo_used:
                provider_parts.append("yolo")
            if safety_qaqc_used:
                provider_parts.append("safety_qaqc")

            result = {
                "status": "success",
                "operation": operation,
                "provider": "+".join(provider_parts),
                "description": description,
                "extracted_text": ocr_text,
                "ocr_confidence": ocr_conf,
                "metadata": meta,
                "detections": detections,
                "summary_by_class": _summarize_detections(detections) if detections else {},
            }
            if detection_error:
                result["detection_error"] = detection_error
            if params.get("mode") == "safety_qaqc":
                result["safety_qaqc"] = safety_qaqc
                if safety_qaqc_error:
                    result["safety_qaqc_error"] = safety_qaqc_error
            return result

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    logger.debug(
                        "swallowed %s in process() — continuing",
                        "OSError", exc_info=True,
                    )

    async def _resolve_source(self, input_data: Any):
        """Return (file_path, tmp_path_or_None) or an error dict on failure."""
        url = None
        file_path = None

        if isinstance(input_data, str):
            if input_data.startswith("http"):
                url = input_data
            elif os.path.exists(input_data):
                file_path = input_data
            else:
                return {"status": "error", "error": "Provide a valid file path or URL"}, None
        elif isinstance(input_data, dict):
            file_path = input_data.get("file_path") or input_data.get("path")
            url = input_data.get("url")
            if not file_path and not url:
                raw = input_data.get("text") or input_data.get("input") or ""
                if isinstance(raw, str) and raw.startswith("http"):
                    url = raw
                elif raw and os.path.exists(raw):
                    file_path = raw
        else:
            return {"status": "error", "error": "Input must be a file path, URL, or {file_path, url}"}, None

        tmp_path = None
        if url and not file_path:
            try:
                file_path = await _download_to_temp(url)
                tmp_path = file_path
            except Exception as e:
                return {"status": "error", "error": f"Failed to download URL: {e}"}, None

        if not file_path:
            return {"status": "error", "error": "Provide file_path or url"}, None

        return file_path, tmp_path

    @staticmethod
    def _compose_local_summary(
        operation: str, meta: Dict, ocr_text: str, ocr_conf: float, ocr_error: str
    ) -> str:
        lines = []
        if operation == "construction":
            lines.append("**Construction image — local analysis**")
        else:
            lines.append("**Image — local analysis**")
        lines.append("")
        lines.append(
            f"- **Dimensions:** {meta.get('width')} × {meta.get('height')} px "
            f"({meta.get('megapixels')} MP, aspect {meta.get('aspect_ratio')})"
        )
        lines.append(f"- **Format:** {meta.get('format')} ({meta.get('mode')})")
        lines.append(f"- **File size:** {meta.get('file_size_bytes')} bytes")
        if "dominant_channel" in meta:
            lines.append(f"- **Dominant channel:** {meta['dominant_channel']}")

        lines.append("")
        if ocr_text:
            preview = ocr_text if len(ocr_text) <= 800 else ocr_text[:800] + "..."
            lines.append(f"**Extracted text** (Tesseract, confidence {ocr_conf:.2f}):")
            lines.append("")
            lines.append("```")
            lines.append(preview)
            lines.append("```")
        elif ocr_error:
            lines.append(f"_OCR unavailable: {ocr_error}_")
        else:
            lines.append("_No text detected._")

        return "\n".join(lines)
