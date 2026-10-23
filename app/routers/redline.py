"""Redline detection endpoint — HTTP layer for coloured markup analysis.

Roadmap V2 · Stream D — Part 2 (redline detection wiring).

POST /v1/projects/{project_id}/documents/{document_id}/redlines

Renders each page of a PDF and runs the colour-channel
redline detector (`app.core.redline.detect_redlines`) on it. Returns a
per-page breakdown and an aggregated has_markup verdict.

Authentication: Bearer JWT or API key (require_user dependency).
Ownership:      404-never-leak-existence pattern (same as projects.py).
"""

import io
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core import file_crypto
from app.core import projects as store
from app.core.redline import detect_redlines, summarize_markup
from app.dependencies import require_user

router = APIRouter()

# PDF page cap — bound the work for very large documents.
_MAX_PAGES = 20


@router.post("/v1/projects/{project_id}/documents/{document_id}/redlines")
async def detect_document_redlines(
    project_id: str,
    document_id: str,
    auth: dict = Depends(require_user),
) -> Dict[str, Any]:
    """Detect coloured markup / redlines on a project document.

    Runs on a PDF drawing (each page rendered to a raster image). Images are
    never project documents (docs/INGEST_EXCLUSION_RULE.md); a photo is
    attached in chat as question context instead. Returns a per-page
    breakdown and an aggregate verdict.

    Response shape::

        {
            "project_id":    str,
            "document_id":   str,
            "filename":      str,
            "has_markup":    bool,
            "total_regions": int,
            "caveat":        str | null,
            "pages": [
                {
                    "page":       int,        # 1-based
                    "has_markup": bool,
                    "coverage":   float,
                    "regions":    [{"bbox": [...], "dominant_colour": str, "pixels": int}, ...]
                },
                ...
            ]
        }
    """
    # ── ownership check ──────────────────────────────────────────────────────
    proj = store.get_project(project_id, user_id=auth["user_id"])
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")

    # ── document lookup ──────────────────────────────────────────────────────
    doc = store.get_document(document_id)
    if not doc or doc["project_id"] != project_id:
        raise HTTPException(404, f"Document '{document_id}' not found")

    original_name: str = doc.get("original_name") or ""
    file_path: Optional[str] = doc.get("file_path")
    if not file_path:
        raise HTTPException(
            404, "Document file is not available (no file path recorded)"
        )

    _, ext = os.path.splitext(original_name.lower())

    # ── dispatch by extension ────────────────────────────────────────────────
    if ext != ".pdf":
        raise HTTPException(
            400,
            f"Redline detection needs a PDF drawing (got '{ext or 'unknown'}').",
        )
    page_results = _analyse_pdf(file_path)

    # ── aggregate across pages ───────────────────────────────────────────────
    has_markup_any = any(p["has_markup"] for p in page_results)
    total_regions = sum(len(p["regions"]) for p in page_results)

    # Build a synthetic combined-result to obtain a consistent caveat string.
    all_regions: List[Dict] = []
    for p in page_results:
        all_regions.extend(p["regions"])
    combined = {
        "has_markup": has_markup_any,
        "coverage": max((p["coverage"] for p in page_results), default=0.0),
        "regions": all_regions,
    }
    summary = summarize_markup(combined)
    caveat: Optional[str] = summary.get("caveat")

    # Serialisable pages (bbox is a tuple → list for JSON)
    serialisable_pages = []
    for p in page_results:
        serialisable_pages.append({
            "page": p["page"],
            "has_markup": p["has_markup"],
            "coverage": p["coverage"],
            "regions": [
                {
                    "bbox": list(r["bbox"]),
                    "dominant_colour": r["dominant_colour"],
                    "pixels": r["pixels"],
                }
                for r in p["regions"]
            ],
        })

    return {
        "project_id": project_id,
        "document_id": document_id,
        "filename": original_name,
        "has_markup": has_markup_any,
        "total_regions": total_regions,
        "caveat": caveat,
        "pages": serialisable_pages,
    }


# ── per-type helpers ─────────────────────────────────────────────────────────

def _analyse_pdf(file_path: str) -> List[Dict]:
    """Render each page of a PDF to a PIL image and run detect_redlines."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise HTTPException(
            500, f"PDF rendering unavailable — PyMuPDF is not installed: {exc}"
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:
        raise HTTPException(
            500, f"Image processing unavailable — Pillow is not installed: {exc}"
        ) from exc

    results: List[Dict] = []
    with file_crypto.open_plaintext(file_path) as readable_path:
        try:
            doc = fitz.open(readable_path)
        except Exception as exc:
            raise HTTPException(500, f"Failed to open PDF: {exc}") from exc

        try:
            for page_num, page in enumerate(doc):
                if page_num >= _MAX_PAGES:
                    break
                try:
                    pix = page.get_pixmap()
                    pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
                    redline_result = detect_redlines(pil_img)
                    results.append({
                        "page": page_num + 1,
                        "has_markup": redline_result["has_markup"],
                        "coverage": redline_result["coverage"],
                        "regions": redline_result["regions"],
                    })
                except Exception:
                    # Skip pages that fail to render — don't 500 the whole request.
                    continue
        finally:
            doc.close()

    return results
