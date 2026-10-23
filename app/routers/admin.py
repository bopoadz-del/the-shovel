"""Admin-gated diagnostic endpoints — mounted in ALL environments.

These were originally in app/routers/debug.py but that router is only
mounted in development. The doc-extract / doc-reindex diagnostics are
the only way to inspect the production index without shell access, so
they need to live somewhere that the production app actually loads.

Admin gating via ``_require_admin`` is the only security boundary; the
endpoints never run unauthenticated and never run for non-admin users.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from pydantic import BaseModel

from app.dependencies import require_api_key
from app.core import file_crypto
from app.core.redis_client import get_redis_client

router = APIRouter()
logger = logging.getLogger(__name__)


def _require_admin(auth: Dict[str, Any]) -> None:
    if auth.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")


def _execute_recovering(conn, statement, params=None):
    """Execute ``statement``; ROLLBACK on failure so the connection stays usable.

    PostgreSQL aborts the open transaction on ``UndefinedTable`` (a missing
    ``chunks_*`` relation). Callers that swallow that error must roll back
    before the next statement, or ``SELECT COUNT(*) FROM documents`` raises
    ``InFailedSqlTransaction``.
    """
    try:
        return conn.execute(statement, params or {})
    except Exception:
        logger.warning(
            "swallowed %s after statement; rolling back so the txn stays usable",
            "Exception",
            exc_info=True,
        )
        try:
            conn.rollback()
        except Exception:
            logger.warning(
                "swallowed %s while rolling back a failed statement",
                "Exception",
                exc_info=True,
            )
        return None


@router.get("/v1/admin/debug/doc-extract")
def admin_doc_extract(
    project_id: str = Query(...),
    document_id: str = Query(...),
    re_extract: bool = Query(False, description="Run fresh extraction (don't read index)"),
    auth: dict = Depends(require_api_key),
):
    """Diagnostic report on what extraction produced for ``document_id``.

    Returns metadata, page count, text-layer character counts, indexed chunk
    counts + previews, and (when ``re_extract=true``) a fresh extraction so the
    caller can compare against what the index stored.
    """
    _require_admin(auth)

    from app.core import projects as _projects
    from app.core import doc_index as _doc_index

    doc = _projects.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.get("project_id") != project_id:
        raise HTTPException(status_code=400, detail="document does not belong to project")

    filename = doc.get("original_name", "")
    file_path = doc.get("file_path") or ""
    size = doc.get("size")
    ext = os.path.splitext(filename.lower())[1] if filename else ""

    response: Dict[str, Any] = {
        "document_id": document_id,
        "project_id": project_id,
        "filename": filename,
        "ext": ext,
        "size_bytes": size,
        "file_exists": bool(file_path and os.path.exists(file_path)),
    }

    if ext == ".pdf" and response["file_exists"]:
        try:
            import fitz  # PyMuPDF
            from app.core import file_crypto
            with file_crypto.open_plaintext(file_path) as readable_path:
                pdf = fitz.open(readable_path)
                response["pdf_page_count"] = pdf.page_count
                page_chars: List[int] = []
                for i, page in enumerate(pdf):
                    if i >= 10:
                        break
                    page_chars.append(len(page.get_text()))
                response["pdf_first_pages_chars"] = page_chars
                pdf.close()
        except Exception as exc:
            response["pdf_error"] = str(exc)

    # Indexed chunks (what RAG sees today) -- read from the vector store,
    # the only place chunk text lives. The doc_index entry carries counts
    # and flags only.
    from app.core.rag import retriever as _rag
    from app.core.rag.vector_store import get_store

    chunks: List[str] = []
    if _rag.available():
        store = get_store(dim=_rag.get_embedder().dim)
        chunks = [
            c.text for c in store.chunks_for_docs(project_id, [document_id], all_rows=True)
        ]
    index = _doc_index._load_index(project_id)  # noqa: SLF001 — diagnostic only
    if index and isinstance(index.get("documents"), list):
        for entry in index["documents"]:
            if entry.get("document_id") == document_id:
                if entry.get("ocr_low_quality"):
                    response["ocr_low_quality"] = True
                break

    response["indexed_chunk_count"] = len(chunks)
    response["indexed_chunks_avg_chars"] = (
        sum(len(c) for c in chunks) // len(chunks) if chunks else 0
    )
    response["indexed_chunks_preview"] = [
        {"i": i, "chars": len(c), "snippet": c[:200]}
        for i, c in enumerate(chunks[:3])
    ]

    if re_extract and response["file_exists"]:
        try:
            text, meta = _doc_index._extract_with_meta(file_path, filename)  # noqa: SLF001
            fresh_chunks = _doc_index.chunk_text(text)
            response["fresh_extraction"] = {
                "total_chars": len(text),
                "chunk_count": len(fresh_chunks),
                "avg_chars": sum(len(c) for c in fresh_chunks) // len(fresh_chunks) if fresh_chunks else 0,
                "meta": meta,
                "first_chunk_snippet": fresh_chunks[0][:200] if fresh_chunks else "",
            }
        except Exception as exc:
            response["fresh_extraction_error"] = str(exc)

    return response


@router.get("/v1/admin/debug/document-download")
def admin_document_download(
    project_id: str = Query(...),
    document_id: str = Query(...),
    auth: dict = Depends(require_api_key),
):
    """Download the original decrypted file bytes for any document.

    Admin-only diagnostic endpoint. Bypasses ownership checks so operators can
    recover fixture/source files from any project (including legacy/phantom
    rows that normal project-scoped endpoints reject).
    """
    _require_admin(auth)
    from app.core import projects as _projects

    doc = _projects.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.get("project_id") != project_id:
        raise HTTPException(status_code=400, detail="Document does not belong to project")

    filename = doc.get("original_name", "download")
    file_path = doc.get("file_path") or ""
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document file is not available")

    try:
        raw = file_crypto.read_document(file_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not read document: {exc}")

    import mimetypes
    media_type, _ = mimetypes.guess_type(filename)
    if not media_type:
        media_type = "application/octet-stream"

    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    return Response(content=raw, media_type=media_type, headers=headers)


# In-memory OCR/reindex job registry. Process-local — a worker restart
# drops the job and the next poll 404s (kick it off again). Same pattern
# as the training-scenario generator: a 370-page scan cannot finish inside
# the Render gateway window, so POST returns immediately and the caller
# polls GET /v1/admin/debug/doc-reindex/job/{job_id}.
_REINDEX_JOBS: Dict[str, Dict[str, Any]] = {}


def _pdf_needs_async_ocr(file_path: str, filename: str) -> bool:
    """True when admin reindex would exceed the HTTP gateway budget.

    A 370-page scan at 150 dpi is tens of minutes. Sync POST used to return
    in ~15s with cover-page chunks and call that success.
    """
    from app.core import doc_index as _doc_index

    pages = _doc_index._pdf_page_count(file_path)  # noqa: SLF001
    if pages > 40:
        return True
    return _doc_index._pdf_should_batch(file_path)  # noqa: SLF001


async def _doc_reindex_job(
    job_id: str,
    project_id: str,
    document_id: str,
    chunker: str,
    force_ocr: bool,
) -> None:
    import time as _time

    job = _REINDEX_JOBS[job_id]
    try:
        from app.core import doc_index as _doc_index

        result = await run_in_threadpool(
            _doc_index.index_document,
            project_id,
            document_id,
            chunker,
            force_ocr=force_ocr,
            stamp_as_indexed=True,
        )
        job.update(result)
        job["status"] = result.get("status") or "error"
        job["finished_at"] = _time.time()
    except Exception as exc:  # noqa: BLE001 — last-line safety
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["finished_at"] = _time.time()


@router.post("/v1/admin/debug/doc-reindex")
async def admin_doc_reindex(
    project_id: str = Query(...),
    document_id: str = Query(...),
    chunker: str = Query("default", pattern="^(default|finer)$"),
    force_ocr: bool = Query(
        True,
        description=(
            "OCR empty-text pages even when a leftover size gate would skip "
            "them. Required for scanned priced BOQs."
        ),
    ),
    auth: dict = Depends(require_api_key),
):
    """Re-run extraction + chunking + RAG indexing for a single document.

    ``chunker=finer`` uses the BOQ-aware char-level chunker (500-char target,
    50-char overlap, BOQ row boundaries preferred). Use it for BOQ / tender
    PDFs where the default 500-word chunker produces too-coarse chunks.

    ``force_ocr=true`` (the default) is the trigger that actually OCRs a
    scanned PDF whose text layer is cover-only. A 370-page scan is queued
    in-process and this endpoint returns ``status=ocr_started`` immediately
    — poll ``GET /v1/admin/debug/doc-reindex/job/{job_id}``. Success
    requires body OCR (``ocr_pages > 0``), not just a non-zero chunk count.
    """
    _require_admin(auth)
    import asyncio
    import time as _time
    import uuid as _uuid

    from app.core import doc_index as _doc_index
    from app.core import projects as _projects

    doc = _projects.get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    if doc.get("project_id") != project_id:
        raise HTTPException(status_code=400, detail="document does not belong to project")

    filename = doc.get("original_name") or ""
    file_path = doc.get("file_path") or ""
    ext = os.path.splitext(filename.lower())[1] if filename else ""

    if (
        force_ocr
        and ext == ".pdf"
        and file_path
        and os.path.exists(file_path)
        and _pdf_needs_async_ocr(file_path, filename)
    ):
        job_id = _uuid.uuid4().hex[:12]
        _REINDEX_JOBS[job_id] = {
            "job_id": job_id,
            "status": "ocr_started",
            "project_id": project_id,
            "document_id": document_id,
            "filename": filename,
            "chunker": chunker,
            "force_ocr": True,
            "started_at": _time.time(),
            "finished_at": None,
            "error": None,
        }
        asyncio.create_task(
            _doc_reindex_job(job_id, project_id, document_id, chunker, True)
        )
        return {
            "status": "ocr_started",
            "job_id": job_id,
            "status_url": f"/v1/admin/debug/doc-reindex/job/{job_id}",
            "project_id": project_id,
            "document_id": document_id,
            "page_count": _doc_index._pdf_page_count(file_path),  # noqa: SLF001
            "message": (
                "Scanned PDF queued for OCR. Poll status_url until status "
                "is ok (ocr_pages > 0) or error. A 15s ok with cover-only "
                "chunks is a failure — that path no longer reports success."
            ),
        }

    return await run_in_threadpool(
        _doc_index.index_document,
        project_id,
        document_id,
        chunker,
        force_ocr=force_ocr,
        stamp_as_indexed=True,
    )


@router.get("/v1/admin/debug/doc-reindex/job/{job_id}")
def admin_doc_reindex_job(job_id: str, auth: dict = Depends(require_api_key)):
    """Poll status of an async scanned-PDF reindex / OCR job."""
    _require_admin(auth)
    job = _REINDEX_JOBS.get(job_id)
    if not job:
        raise HTTPException(
            404, f"job '{job_id}' not found (worker may have restarted)"
        )
    return job


@router.get("/v1/admin/driver-mode")
def admin_driver_mode_status(auth: dict = Depends(require_api_key)):
    """Whether the platform honours the per-user switch (DRIVER_MODE=request),
    and which users have driver mode on."""
    _require_admin(auth)
    from app.agents import driver
    from app.core import users as _users
    return {"mode": driver.mode(),
            "users": [{"email": u["email"], "id": u["id"]} for u in _users.driver_mode_users()]}


@router.post("/v1/admin/driver-mode")
def admin_set_driver_mode(body: Dict[str, Any] = Body(...), auth: dict = Depends(require_api_key)):
    """Switch driver mode on or off for one user, by email."""
    _require_admin(auth)
    from app.core import users as _users
    email = str((body or {}).get("email") or "").strip()
    if not email:
        raise HTTPException(422, "Give the user's email.")
    user = _users.set_driver_mode(email, bool((body or {}).get("on")))
    if user is None:
        raise HTTPException(404, f"No user with email {email!r}.")
    return {"email": user["email"], "driver_mode": user["driver_mode"]}


@router.put("/v1/admin/projects/{project_id}/profile")
def admin_set_project_profile(project_id: str, profile: Dict[str, Any] = Body(...),
                              auth: dict = Depends(require_api_key)):
    """Set a project's profile (a JSON object; ``{}`` clears it). Driver mode
    gives it to the model on every turn of the project."""
    _require_admin(auth)
    from app.core import projects as _projects
    if not isinstance(profile, dict):
        raise HTTPException(422, "The profile is a JSON object.")
    updated = _projects.set_project_profile(project_id, profile)
    if updated is None:
        raise HTTPException(404, f"Project '{project_id}' not found")
    return {"status": "ok", "project_id": project_id, "profile": updated.get("profile")}


@router.post("/v1/admin/projects/{project_id}/approve")
def admin_approve_project(project_id: str, auth: dict = Depends(require_api_key)):
    """Make an existing project admin-visible (shows in the sidebar).

    Flips ``is_approved=True`` + ``origin='admin_drive_approved'`` on a project
    that already exists (e.g. one created by uploading documents), without
    re-importing from Drive. Idempotent.
    """
    _require_admin(auth)
    from app.core import projects as _projects
    if not _projects.approve_project(project_id):
        raise HTTPException(404, f"Project '{project_id}' not found")
    return {"status": "approved", "project_id": project_id}


@router.post("/v1/admin/documents/{old_id}/supersede")
def admin_supersede_document(
    old_id: str,
    new_id: str = Query(..., description="The document that replaces old_id"),
    auth: dict = Depends(require_api_key),
):
    """Retire a document in favour of a corrected copy. Never deletes.

    ``projects.supersede_document`` has existed for a while -- it hides the old
    row from retrieval (``retrieval_visible=false``) and points it at its
    replacement (``superseded_by``) -- but it was only reachable from inside
    ``add_document(reingest_of=...)``, which no route exposes. So the one
    safe, reversible way to replace a badly-indexed document could not be
    used from outside the process, and the alternatives were a hand-written
    UPDATE against production or a delete.

    Why it was needed: three documents carried a false ``CONTRACT DATA
    particulars`` label and outranked the real Contract Data on every contract
    question. They could not be re-indexed in place -- the server does not
    retain the source bytes of archive-ingested documents, so doc-reindex
    returns ZERO_CHUNK -- which left "upload a corrected copy, then retire the
    old row" as the only non-destructive fix.

    Guard rails: both documents must exist and must belong to the SAME
    project, so a document can never be retired in favour of one from another
    project. Reversible by setting ``retrieval_visible=true`` and clearing
    ``superseded_by``; the old row and its chunks are untouched.
    """
    _require_admin(auth)
    from app.core import audit
    from app.core import projects as _projects

    if old_id == new_id:
        raise HTTPException(400, "a document cannot supersede itself")
    old = _projects.get_document(old_id)
    new = _projects.get_document(new_id)
    if old is None:
        raise HTTPException(404, f"document '{old_id}' not found")
    if new is None:
        raise HTTPException(404, f"replacement document '{new_id}' not found")
    if old.get("project_id") != new.get("project_id"):
        raise HTTPException(
            400,
            "old and new documents belong to different projects; a document "
            "may only be superseded by one in the same project",
        )

    updated = _projects.supersede_document(old_id, new_id)
    if updated is None:
        raise HTTPException(500, "supersede failed")
    audit.record(
        "document.superseded",
        project_id=old.get("project_id"),
        user_id=auth.get("user_id"),
        old_document_id=old_id,
        new_document_id=new_id,
    )
    return {
        "status": "superseded",
        "old_document_id": old_id,
        "new_document_id": new_id,
        "project_id": old.get("project_id"),
        "retrieval_visible": updated.get("retrieval_visible"),
        "superseded_by": updated.get("superseded_by"),
    }


@router.get("/v1/admin/projects/archived")
def admin_list_archived_projects(auth: dict = Depends(require_api_key)):
    """List soft-archived (hidden) projects — so an admin can see what junk is
    eligible for a permanent purge. Read-only."""
    _require_admin(auth)
    from app.core.db import SessionLocal
    from app.core.models import Project
    out = []
    with SessionLocal() as s:
        for p in s.query(Project).filter(Project.status == "archived").all():
            out.append({"id": p.id, "name": p.name, "status": p.status})
    return {"archived": out, "count": len(out)}


@router.post("/v1/admin/projects/{project_id}/restore")
def admin_restore_archived_project(project_id: str,
                                   hide_from_sidebar: bool = False,
                                   auth: dict = Depends(require_api_key)):
    """Restore an ARCHIVED project to active — the missing undo for Delete.

    2026-07-26: the sidebar cleanup archived the REAL corpus projects
    (a client corpus project and the master-corpus source) along with the
    duplicate shells, and
    archive hides a project from retrieval — corpus-project chat silently
    degraded with no way back from the UI or API. Restore flips status to
    'active'; pass hide_from_sidebar=true to keep the row out of the sidebar
    (the correct end state for corpus/GK rows: retrievable, not listed).
    """
    _require_admin(auth)
    from app.core.db import SessionLocal
    from app.core.models import Project
    with SessionLocal() as s:
        p = s.get(Project, project_id)
        if not p:
            raise HTTPException(404, f"Project '{project_id}' not found")
        if p.status != "archived":
            raise HTTPException(409, f"'{project_id}' is not archived (status={p.status})")
        p.status = "active"
        if hide_from_sidebar:
            p.hidden_from_sidebar = True
        s.commit()
        from app.core import audit
        audit.record("project.restored", project_id=project_id,
                     user_id=auth.get("user_id"),
                     hidden_from_sidebar=hide_from_sidebar)
        return {"status": "restored", "project_id": project_id,
                "hidden_from_sidebar": bool(hide_from_sidebar)}


@router.post("/v1/admin/projects/{project_id}/purge")
def admin_purge_archived_project(project_id: str,
                                 auth: dict = Depends(require_api_key)):
    """PERMANENTLY purge an ARCHIVED project (its RAG chunks + row).

    Tightly guarded: only works on already-archived projects, and refuses the
    master corpus / backing / general-knowledge ids — so live RAG can never be
    destroyed through here (the never-delete-RAG rule still holds).
    """
    _require_admin(auth)
    from app.core import projects as _projects
    result = _projects.purge_archived_project(project_id)
    if result == "purged":
        return {"status": "purged", "project_id": project_id}
    code = {"protected": 403, "not_found": 404, "not_archived": 409}.get(result, 400)
    raise HTTPException(code, f"cannot purge '{project_id}': {result}")


@router.get("/v1/admin/training/list")
def admin_training_list(auth: dict = Depends(require_api_key)):
    """List all training-scenario JSONL files on the server."""
    _require_admin(auth)
    data_dir = os.getenv("DATA_DIR", "data")
    learn_dir = os.path.join(data_dir, "learning")
    if not os.path.isdir(learn_dir):
        return {"files": []}
    out = []
    for name in sorted(os.listdir(learn_dir)):
        path = os.path.join(learn_dir, name)
        if name.endswith(".jsonl") and os.path.isfile(path):
            line_count = 0
            try:
                with open(path, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
            except Exception:
                logger.warning(
                    "swallowed %s in admin_training_list() — continuing",
                    "Exception", exc_info=True,
                )
            out.append({
                "name": name,
                "size_bytes": os.path.getsize(path),
                "line_count": line_count,
            })
    return {"files": out}


@router.get("/v1/admin/training/download")
def admin_training_download(
    filename: str = Query(..., description="Filename inside DATA_DIR/learning/"),
    auth: dict = Depends(require_api_key),
):
    """Stream a training-scenario JSONL back to the caller. Sandboxed to
    DATA_DIR/learning so an attacker can't traverse to other paths."""
    _require_admin(auth)
    import os as _os
    # Path-traversal guard: filename must be a basename, no slashes.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "filename must be a plain basename")
    data_dir = _os.getenv("DATA_DIR", "data")
    full = _os.path.join(data_dir, "learning", filename)
    if not _os.path.isfile(full):
        raise HTTPException(404, "file not found")
    with open(full, "r", encoding="utf-8") as f:
        body = f.read()
    return {"filename": filename, "line_count": body.count("\n"),
            "size_bytes": len(body), "content": body}


@router.post("/v1/admin/debug/project-reindex")
def admin_project_reindex(
    project_id: str = Query(...),
    auth: dict = Depends(require_api_key),
):
    """Full rebuild of a project's doc_index from its current document list.

    Use this to clean up orphans left behind by deletes that pre-dated the
    auto-prune-on-delete fix, or to apply a chunker change to all docs at
    once. Slow — extracts text from every document.
    """
    _require_admin(auth)
    from app.core import doc_index as _doc_index
    result = _doc_index.index_project(project_id)
    if result.get("status") == "error":
        raise HTTPException(status_code=422, detail=result)
    return result


# ── Drive pipeline proof endpoints (Task T4) ───────────────────────────────

class _DriveProofRequest(BaseModel):
    project_id: str
    file_id: str | None = None


async def _resolve_drive_proof_file(
    project_id: str,
    file_id: str | None,
) -> Dict[str, Any]:
    """Return {file_id, name, error} for a Drive proof target.

    If ``file_id`` is supplied, use it directly. Otherwise walk the configured
    Drive folder for ``project_id`` and pick the first downloadable file.
    """
    from app.core import gdrive_service

    if file_id:
        return {"file_id": file_id, "name": "", "error": None}

    mapping = gdrive_service.parse_project_folder_map()
    folder_id = mapping.get(project_id)
    if not folder_id:
        return {
            "error": f"no Drive folder configured for project {project_id}",
        }

    files, walk_errors = await run_in_threadpool(
        gdrive_service.walk_folder, folder_id
    )
    for f_meta in files:
        if gdrive_service.is_downloadable(f_meta):
            return {
                "file_id": f_meta["id"],
                "name": f_meta.get("name", ""),
                "error": None,
            }
    err = walk_errors[0] if walk_errors else "no downloadable files found"
    return {"error": err}


@router.post("/v1/admin/drive/download-proof")
async def admin_drive_download_proof(
    req: _DriveProofRequest,
    auth: dict = Depends(require_api_key),
):
    """Prove the service account can download a real Drive file.

    Returns length + SHA-256 only; raw bytes never leave the server.
    """
    _require_admin(auth)
    from app.core import gdrive_service

    target = await _resolve_drive_proof_file(req.project_id, req.file_id)
    if target.get("error"):
        return {"ok": False, "error": target["error"]}

    file_id = target["file_id"]
    blob, dl_err = await run_in_threadpool(
        gdrive_service.download_file_bytes, file_id
    )
    if blob is None:
        return {"ok": False, "error": dl_err or "download failed"}

    import hashlib

    return {
        "ok": True,
        "file_id": file_id,
        "name": target.get("name", ""),
        "bytes": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "mime_type": "application/octet-stream",
    }


@router.post("/v1/admin/drive/ingest-proof")
async def admin_drive_ingest_proof(
    req: _DriveProofRequest,
    auth: dict = Depends(require_api_key),
):
    """Prove the service account can download + index a real Drive file.

    Downloads the file, writes it to DATA_DIR, registers it as a project
    document with Drive metadata, and runs incremental indexing.
    """
    _require_admin(auth)
    import hashlib
    import os

    from app.core import doc_index as _doc_index
    from app.core import file_crypto, gdrive_service, projects as _projects

    if not _projects.get_project(req.project_id):
        return {"ok": False, "error": f"project {req.project_id} not found"}

    target = await _resolve_drive_proof_file(req.project_id, req.file_id)
    if target.get("error"):
        return {"ok": False, "error": target["error"]}

    file_id = target["file_id"]
    blob, dl_err = await run_in_threadpool(
        gdrive_service.download_file_bytes, file_id
    )
    if blob is None:
        return {"ok": False, "error": dl_err or "download failed"}

    original_name = target.get("name") or file_id
    data_dir = os.getenv("DATA_DIR", "data")
    stored_as = f"{hashlib.sha256(file_id.encode()).hexdigest()[:8]}_{original_name}"
    stored_as = os.path.basename(stored_as.replace("\\", "/"))
    filepath = os.path.join(data_dir, stored_as)
    file_crypto.write_document(filepath, blob)

    try:
        doc = _projects.add_document(
            req.project_id,
            original_name,
            stored_as=stored_as,
            file_path=filepath,
            size=len(blob),
            content_sha256=hashlib.sha256(blob).hexdigest(),
            metadata={
                "drive_file_id": file_id,
                "source": "drive_admin_ingest_proof",
            },
        )
    except _projects.DuplicateContentError as exc:
        return {
            "ok": False,
            "error": "DUPLICATE_SHA",
            "existing_id": exc.existing_id,
            "file_id": file_id,
            "name": original_name,
        }

    idx_result = _doc_index.index_document(req.project_id, doc["id"])
    if idx_result.get("status") == "error":
        banner = idx_result.get(
            "banner",
            "ERROR: indexing produced 0 chunks — check extractor/OCR.",
        )
        return {
            "ok": False,
            "error": idx_result.get("error", "ZERO_CHUNK"),
            "banner": banner,
            "document_id": doc["id"],
            "file_id": file_id,
            "name": original_name,
        }

    return {
        "ok": True,
        "document_id": doc["id"],
        "chunk_count": idx_result.get("total_chunks", 0),
        "file_id": file_id,
        "name": original_name,
    }


# ── Training scenario generation (Task 1.4 / MEGA-2) ───────────────────────
#
# Long-running generation jobs are now ASYNC: POST returns a job_id
# immediately, the actual work runs in an asyncio background task, and the
# client polls GET /v1/admin/training/job/{job_id}. This decouples the
# 10-20 minute generator from any client-side connection (bridge, curl
# pipe) that might drop mid-flight.

# In-memory job registry. Process-local; if uvicorn restarts mid-job, the
# job is lost and the client will see a 404 on next poll — that's the
# correct signal for "kick a new one off". Persistent job state would
# need its own table; not worth it for an admin-only diagnostic tool.
_TRAINING_JOBS: Dict[str, Dict[str, Any]] = {}


async def _training_job_runner(
    job_id: str,
    project_id: str,
    questions_per_chunk: int,
    min_chunk_chars: int,
    max_chunks: int,
    provider_hint: str,
) -> None:
    """Background task: runs the full generator, updates job state."""
    import asyncio
    import json
    import os
    import time as _time

    from scripts.generate_training_scenarios import (
        iter_chunks_for_project,
        _generate_for_chunk,
        _validate_scenarios,
        _DEFAULT_PROMPT,
    )
    from app.blocks import BLOCK_REGISTRY

    job = _TRAINING_JOBS[job_id]
    try:
        chunks = list(iter_chunks_for_project(
            project_id, min_chars=min_chunk_chars, max_chunks=max_chunks
        ))
        if not chunks:
            job["status"] = "failed"
            job["error"] = f"no chunks in doc_index for project {project_id}"
            job["finished_at"] = _time.time()
            return
        job["total_chunks"] = len(chunks)

        rows: list = []
        skipped_chunks = 0
        skip_reasons: dict = {}
        per_chunk_timeout = 200.0

        for i, chunk in enumerate(chunks):
            reason = None
            try:
                pairs = await asyncio.wait_for(
                    _generate_for_chunk(chunk, questions_per_chunk, provider_hint),
                    timeout=per_chunk_timeout,
                )
            except asyncio.TimeoutError:
                pairs = []
                reason = "timeout"
            except Exception as exc:  # noqa: BLE001
                pairs = []
                reason = f"exc:{type(exc).__name__}"
            if not pairs:
                skipped_chunks += 1
                skip_reasons[reason or "no_pairs"] = skip_reasons.get(reason or "no_pairs", 0) + 1
            else:
                rows.extend(pairs)
            job["chunks_done"] = i + 1
            job["rows_generated"] = len(rows)
            job["chunks_skipped"] = skipped_chunks

        kept_rows, validation_report = _validate_scenarios(rows)

        data_dir = os.getenv("DATA_DIR", "data")
        out_dir = os.path.join(data_dir, "learning")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir,
            f"training_scenarios_{project_id}_{int(_time.time())}.jsonl",
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        job["status"] = "done"
        job["output_path"] = out_path
        job["rows_kept"] = len(kept_rows)
        job["validation"] = validation_report
        job["skip_reasons"] = skip_reasons
        job["finished_at"] = _time.time()
    except Exception as exc:  # noqa: BLE001 — last-line safety
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["finished_at"] = _time.time()


@router.post("/v1/admin/training/generate-scenarios")
async def admin_generate_training_scenarios(
    project_id: str = Query(...),
    questions_per_chunk: int = Query(3, ge=1, le=20),
    min_chunk_chars: int = Query(150, ge=50, le=2000),
    max_chunks: int = Query(200, ge=1, le=2000),
    provider_hint: str = Query("any"),
    auth: dict = Depends(require_api_key),
):
    """Kick off a Q&A generation job in the background. Returns a job_id
    for polling via GET /v1/admin/training/job/{job_id}.

    The job runs server-side and survives client disconnects — this is the
    reliability fix for the 10-20 minute generator (which used to time out
    bridge curl pipes mid-flight).
    """
    _require_admin(auth)

    import asyncio
    import time as _time
    import uuid as _uuid

    job_id = _uuid.uuid4().hex[:12]
    _TRAINING_JOBS[job_id] = {
        "job_id": job_id,
        "status": "running",
        "project_id": project_id,
        "questions_per_chunk": questions_per_chunk,
        "min_chunk_chars": min_chunk_chars,
        "max_chunks": max_chunks,
        "provider_hint": provider_hint,
        "started_at": _time.time(),
        "finished_at": None,
        "chunks_done": 0,
        "total_chunks": None,
        "rows_generated": 0,
        "chunks_skipped": 0,
        "output_path": None,
        "rows_kept": None,
        "error": None,
    }
    asyncio.create_task(_training_job_runner(
        job_id, project_id, questions_per_chunk, min_chunk_chars,
        max_chunks, provider_hint,
    ))
    return {
        "job_id": job_id,
        "status_url": f"/v1/admin/training/job/{job_id}",
        "status": "running",
    }


@router.get("/v1/admin/training/job/{job_id}")
def admin_training_job_status(job_id: str, auth: dict = Depends(require_api_key)):
    """Poll status of a generate-scenarios job."""
    _require_admin(auth)
    job = _TRAINING_JOBS.get(job_id)
    if not job:
        raise HTTPException(404, f"job '{job_id}' not found (worker may have restarted)")
    return job


# ── Synchronous (legacy) path — kept for tests / quick small jobs ────────

@router.post("/v1/admin/debug/migrate-sqlite")
def admin_migrate_sqlite(
    dry_run: bool = Query(True, description="Count rows only; no Postgres writes"),
    execute: bool = Query(False, description="Run idempotent migration (overrides dry_run)"),
    auth: dict = Depends(require_api_key),
):
    """Run SQLite→Postgres migration against the live DATA_DIR volume.

  One-off Render jobs cannot mount the service disk; this endpoint runs inside
  the web process so cutover dry-run / execute work on production.
    """
    _require_admin(auth)

    if execute:
        dry_run = False

    from pathlib import Path
    from sqlalchemy import create_engine

    from scripts.migrate_sqlite_to_pg import migrate

    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")
    if not data_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"DATA_DIR not found: {data_dir}")

    from app.core.db import to_psycopg_url
    engine = create_engine(to_psycopg_url(db_url), pool_pre_ping=True)
    try:
        counts = migrate(engine, data_dir, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 — operator diagnostic
        raise HTTPException(status_code=500, detail=f"migration failed: {exc}") from exc

    # Persist dry-run output for pilot-preflight when operators poll without re-running.
    if dry_run:
        log_path = data_dir / "pilot_dry_run.log"
        try:
            lines = [f"Migration summary (would migrate):"]
            for table, n in counts.items():
                lines.append(f"  {table}: {n}")
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            logger.warning(
                "swallowed %s in admin_migrate_sqlite() — continuing",
                "OSError", exc_info=True,
            )

    return {
        "dry_run": dry_run,
        "data_dir": str(data_dir),
        "counts": counts,
    }


def _drive_user_id(auth: Dict[str, Any]) -> str:
    """User id whose Google Drive connection these admin routes act on.

    `require_api_key` returns TWO different principal shapes. The JWT branch
    resolves a real user and includes `user_id`; the API-key branch returns
    the key record from `auth_manager`, which carries `user`/`tier`/`role`
    but NO `user_id`. `CEREBRUM_MASTER_KEY` is minted with `role: "admin"`,
    so a master-key caller passes `_require_admin` and then reached
    `auth["user_id"]` -> KeyError -> 500. Confirmed live 2026-08-02:
    GET /v1/admin/drive/scan returned 500 for the operator's own admin key.

    Drive tokens are stored PER USER, so a key principal genuinely has no
    Drive connection to act on — the honest answer is the 409 this route
    already documents for "not connected", not an unhandled 500.
    """
    user_id = auth.get("user_id")
    if not user_id:
        raise HTTPException(
            409,
            "Google Drive admin actions require a signed-in admin session. "
            "This request authenticated with an API key, which has no "
            "Google Drive connection of its own. Sign in as an admin user "
            "and connect Drive, then retry.",
        )
    return user_id


def _parse_vector_dim(coltype: Optional[str]) -> Optional[int]:
    """Width declared by a pgvector column type, e.g. ``vector(384)`` -> 384.

    Returns None for an absent column or an unsized ``vector`` — both mean
    "cannot assert a width", which callers must treat as unknown rather than
    as agreement.
    """
    if not coltype:
        return None
    m = re.match(r"^\s*vector\s*\(\s*(\d+)\s*\)\s*$", coltype)
    return int(m.group(1)) if m else None


@router.get("/v1/admin/debug/pilot-preflight")
def admin_pilot_preflight(auth: dict = Depends(require_api_key)):
    """Postgres schema + row-count snapshot for pilot cutover / re-index gates."""
    _require_admin(auth)

    import os
    from sqlalchemy import create_engine, text

    out: Dict[str, Any] = {
        "sentry_enabled": bool(os.getenv("SENTRY_DSN", "").strip()),
        "database_url_set": bool(os.getenv("DATABASE_URL", "").strip()),
        "data_dir": os.getenv("DATA_DIR", "data"),
    }

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        out["postgres"] = {"error": "DATABASE_URL unset"}
        return out

    try:
        from app.core.db import to_psycopg_url
        from app.core.models import rag_chunk_table_name
        from app.core.rag.embeddings import loaded_embedder_identity

        # Report the table the retriever ACTUALLY reads and writes (chunks_v2
        # by default) — NOT the retired legacy `chunks` table. The old code
        # read the dim off `chunks`, which has been empty since the v2
        # migration, and compared it to a hardcoded `vector(256)`. So this
        # gate answered True from a dead table: it would have stayed green
        # with the live store missing, empty, or the wrong width. Same
        # false-green class as the admin corpus count fixed earlier in this
        # file — that one showed 0 chunks for every project and "invited a
        # destructive re-index click".
        namespace = os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
        active_table = rag_chunk_table_name(namespace)

        engine = create_engine(to_psycopg_url(db_url))
        with engine.connect() as conn:
            def _embedding_type(table: str):
                return conn.execute(
                    text(
                        """
                        SELECT format_type(a.atttypid, a.atttypmod)
                        FROM pg_attribute a
                        JOIN pg_class t ON a.attrelid = t.oid
                        WHERE t.relname = :table
                          AND a.attname = 'embedding'
                          AND NOT a.attisdropped
                        """
                    ),
                    {"table": table},
                ).scalar()

            active_type = _embedding_type(active_table)
            legacy_type = _embedding_type("chunks")

            counts = {}
            tables = [
                "users",
                "projects",
                "documents",
                "chunks",
                "conversations",
                "messages",
            ]
            if active_table not in tables:
                tables.append(active_table)
            for table in tables:
                # rag_chunk_table_name validates the identifier, so the
                # interpolation below cannot carry a hostile namespace.
                # Missing tables (fresh namespace) must not abort the
                # Postgres transaction — see _execute_recovering.
                counted = _execute_recovering(
                    conn, text(f"SELECT COUNT(*) FROM {table}")
                )
                counts[table] = counted.scalar() if counted is not None else None

            # Dim agreement is only assertable when the embedder is already
            # warm: `loaded_embedder_identity` never triggers the slow cold
            # load, so a cold cache reports None ("unknown"). Unknown is
            # reported as None, never as a green boolean — an operator gate
            # must not read "ok" from an absent measurement.
            identity = loaded_embedder_identity()
            expected_dim = identity.get("dim") if identity else None
            actual_dim = _parse_vector_dim(active_type)
            if expected_dim is None or actual_dim is None:
                dim_ok = None
            else:
                dim_ok = actual_dim == expected_dim

            # Storage headroom. Bulk re-ingest is gated on this and there was
            # no way to read it without shell access — the sandboxed code
            # block has DATABASE_URL scrubbed, by design.
            try:
                db_size_mb = round(
                    (conn.execute(
                        text("SELECT pg_database_size(current_database())")
                    ).scalar() or 0) / 1048576, 1
                )
            except Exception:  # noqa: BLE001 — diagnostic only
                db_size_mb = None

            # Corpus integrity. A restored / bulk-inserted corpus carries
            # document rows whose FILE never reached this host: extraction
            # cannot run, re-index is a no-op, and the row is invisible in
            # every count that only looks at `documents`. 2026-08-19: 95 of
            # 232 Master Folder docs had zero chunks and ALL 232 had no file
            # on disk, which no existing gate reported. Counted here so the
            # gap is visible before someone concludes "ingestion is broken".
            integrity: Dict[str, Any] = {}
            try:
                rows = conn.execute(
                    text(
                        f"""
                        SELECT d.id, d.file_path, COALESCE(c.n, 0) AS n_chunks
                        FROM documents d
                        LEFT JOIN (
                            SELECT doc_id, COUNT(*) AS n
                            FROM {active_table} GROUP BY doc_id
                        ) c ON c.doc_id = d.id
                        """
                    )
                ).all()
                missing = sum(
                    1 for r in rows
                    if not (r.file_path and os.path.exists(r.file_path))
                )
                zero = sum(1 for r in rows if not r.n_chunks)
                integrity = {
                    "documents_total": len(rows),
                    "documents_zero_chunk": zero,
                    "documents_missing_file": missing,
                }
            except Exception as exc:  # noqa: BLE001 — diagnostic only
                integrity = {"error": f"{type(exc).__name__}: {exc}"}

            out["postgres"] = {
                "active_chunk_table": active_table,
                "active_chunk_embedding_type": active_type,
                "active_chunk_embedding_dim": actual_dim,
                "legacy_chunks_embedding_type": legacy_type,
                "embedder_dim": expected_dim,
                "embedder_loaded": identity is not None,
                "embedding_dim_ok": dim_ok,
                "row_counts": counts,
                "db_size_mb": db_size_mb,
                "corpus_integrity": integrity,
            }
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        out["postgres"] = {"error": f"{type(exc).__name__}: {exc}"}

    data_dir = out["data_dir"]
    dry_log = os.path.join(data_dir, "pilot_dry_run.log")
    if os.path.isfile(dry_log):
        try:
            with open(dry_log, encoding="utf-8") as f:
                out["sqlite_dry_run_log"] = f.read()[-8000:]
        except Exception as exc:  # noqa: BLE001
            out["sqlite_dry_run_log_error"] = str(exc)

    return out


@router.post("/v1/admin/debug/sentry-smoke")
def admin_sentry_smoke(auth: dict = Depends(require_api_key)):
    """Raise a tagged test exception so Sentry capture can be verified pre-cutover."""
    _require_admin(auth)

    import os

    if not os.getenv("SENTRY_DSN", "").strip():
        raise HTTPException(
            status_code=503,
            detail="SENTRY_DSN not configured — set env and redeploy before smoke test",
        )

    import sentry_sdk

    event_id = sentry_sdk.capture_exception(
        RuntimeError("pilot sentry-smoke: intentional test exception (safe to ignore)")
    )
    return {
        "status": "captured",
        "event_id": event_id,
        "message": "Check Sentry Issues for pilot sentry-smoke event",
    }


@router.get("/v1/admin/corpus/collections")
def admin_corpus_collections(
    folder_breakdown: bool = Query(
        True,
        description="Include top-folder breakdown for project_ids whose document "
                    "count exceeds folder_breakdown_min (default 50).",
    ),
    folder_breakdown_min: int = Query(
        50,
        ge=1,
        description="Minimum doc count before a project_id gets the folder breakdown.",
    ),
    auth: dict = Depends(require_api_key),
):
    """Per-project_id corpus inventory.

    For each project_id in the corpus, returns:
      * documents — count of rows in the `documents` table.
      * chunks    — count of rows in the `chunks` (RAG vector store) table.
      * by_top_folder — first '/'-segment of `original_name`, sorted by doc
        count desc. Only emitted for project_ids with >= folder_breakdown_min
        documents (avoids 200 single-doc tiles for ad-hoc projects).

    Read-only. Issues one COUNT(*) per project + one GROUP BY for folder
    breakdown — bounded by the number of distinct project_ids. Designed
    for a 70 GB master-corpus source where the operator needs to see
    what's actually in there grouped by source folder.

    Works on both Postgres production and SQLite dev / pilot.
    """
    _require_admin(auth)

    import os
    from sqlalchemy import create_engine, text

    db_url = os.getenv("DATABASE_URL", "").strip()
    if not db_url:
        # SQLite fallback — match the app's default resolution
        data_dir = os.getenv("DATA_DIR", "data")
        db_path = os.path.join(data_dir, "the_fork.db")
        db_url = f"sqlite:///{db_path}"

    from app.core.db import to_psycopg_url
    from app.core.models import rag_chunk_table_name
    # Count against the SAME table the retriever reads/writes — the namespaced
    # store (chunks_v2 by default), NOT the retired legacy `chunks` table. The
    # old code counted `chunks`, which is empty post-v2-migration, so the admin
    # page showed 0 chunks for every project even though retrieval worked. That
    # false zero is what "invited a destructive re-index click" (see below).
    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )
    engine = create_engine(to_psycopg_url(db_url))
    collections: List[Dict[str, Any]] = []

    with engine.connect() as conn:
        # Distinct project_ids — union of documents + chunks tables. A
        # project_id can exist in chunks without documents (legacy
        # imports), so we collect both sides.
        try:
            project_ids = {
                row[0]
                for row in conn.execute(text("SELECT DISTINCT project_id FROM documents"))
            }
        except Exception as exc:  # noqa: BLE001 — diagnostic
            raise HTTPException(
                status_code=500,
                detail=f"Corpus query failed: {type(exc).__name__}: {exc}",
            )
        # The namespaced chunk table may not exist yet in a fresh DB —
        # the documents side already covers approved projects. Must
        # ROLLBACK on UndefinedTable (Postgres) before the next query.
        chunk_projects = _execute_recovering(
            conn, text(f"SELECT DISTINCT project_id FROM {chunks_table}")
        )
        if chunk_projects is not None:
            project_ids.update(row[0] for row in chunk_projects)

        for pid in sorted(project_ids):
            doc_count = conn.execute(
                text("SELECT COUNT(*) FROM documents WHERE project_id = :pid"),
                {"pid": pid},
            ).scalar() or 0
            chunk_rows = _execute_recovering(
                conn,
                text(f"SELECT COUNT(*) FROM {chunks_table} WHERE project_id = :pid"),
                {"pid": pid},
            )
            chunk_count = 0
            if chunk_rows is not None:
                chunk_count = chunk_rows.scalar() or 0

            entry: Dict[str, Any] = {
                "project_id": pid,
                "documents": int(doc_count),
                "chunks": int(chunk_count),
            }

            if folder_breakdown and doc_count >= folder_breakdown_min:
                # Group documents by first '/' segment of original_name.
                # SQLite has instr(); Postgres has strpos(). PR #91's first
                # cut used instr() and 500'd on prod Postgres. Do the
                # grouping in Python — bounded by docs-per-project (a few
                # thousand at most for the largest corpus), so a single
                # full scan + Counter is fine and dialect-portable.
                from collections import Counter
                name_rows = conn.execute(
                    text("SELECT original_name FROM documents WHERE project_id = :pid"),
                    {"pid": pid},
                ).fetchall()
                buckets: "Counter[str]" = Counter()
                for (name,) in name_rows:
                    if name and "/" in name:
                        buckets[name.split("/", 1)[0]] += 1
                    else:
                        buckets["(no folder)"] += 1
                entry["by_top_folder"] = [
                    {"folder": folder, "docs": n}
                    for folder, n in buckets.most_common(50)
                ]

            collections.append(entry)

    # Pilot master-corpus alias: the canonical project_id
    # (master_corpus) is a read-only view over the backing Drive-folder
    # corpus (projects_folder). The admin inventory must never show the alias
    # as 0 chunks — that invited a destructive re-index click in a live run. Reflect
    # the source counts under the alias so the admin page and reconciliation
    # script agree with the API's resolved chunk_count.
    from app.core.projects import (
        MASTER_CORPUS_PROJECT_ID,
        MASTER_CORPUS_SOURCE_PROJECT_ID,
    )

    by_pid = {c["project_id"]: c for c in collections}
    source = by_pid.get(MASTER_CORPUS_SOURCE_PROJECT_ID)
    if source and MASTER_CORPUS_PROJECT_ID not in by_pid:
        alias_entry = dict(source)
        alias_entry["project_id"] = MASTER_CORPUS_PROJECT_ID
        alias_entry["source_project_id"] = MASTER_CORPUS_SOURCE_PROJECT_ID
        alias_entry["is_master_corpus_alias"] = True
        collections.append(alias_entry)

    # Largest first so the eye lands on the master-corpus source immediately
    # when it exists.
    collections.sort(key=lambda c: (-c["chunks"], -c["documents"], c["project_id"]))

    return {
        "collections": collections,
        "total_project_ids": len(collections),
        "total_documents": sum(c["documents"] for c in collections),
        "total_chunks": sum(c["chunks"] for c in collections),
    }


# ──────────────────────────────────────────────────────────────────────────
# /v1/admin/corpus/bulk-insert — operator-driven migration endpoint
# ──────────────────────────────────────────────────────────────────────────
#
# Accepts a JSON payload of projects/documents/chunks and inserts them
# into the production tables. Designed for operator loads (the original
# master-corpus migration, a single-contract volume backfill) where direct
# psycopg from outside the Render perimeter is blocked by the pgsql
# ipAllowList default.
#
# 2026-08-05 contract fix: chunk writes used to go through the static
# ``RagChunk`` model, which maps the RETIRED legacy ``chunks`` table.
# Production retrieval reads the namespaced table (``chunks_v2`` via
# RAG_VECTOR_NAMESPACE) — so every chunk loaded through this endpoint
# after the namespace split was invisible to retrieval: a silent no-op
# for the endpoint's stated purpose. Chunks now go through
# ``VectorStore.upsert_chunks`` on the ACTIVE namespace, which also
# enforces the embedding-dimension and model-identity guards. Semantics
# changed with it: chunk writes are a per-document idempotent REPLACE
# (send whole documents; a retry re-sends the doc), no longer additive
# ON CONFLICT DO NOTHING per chunk row. Projects/documents keep the old
# additive semantics.

class _BulkProjectRow(BaseModel):
    id: str
    name: str
    user_id: str = "system"
    status: str = "active"
    created_at: Optional[str] = None  # ISO; defaults to now


class _BulkDocumentRow(BaseModel):
    id: str
    project_id: str
    original_name: str
    doc_type: str = "document"
    doc_role: str = "other"
    size: int = 0
    uploaded_at: Optional[str] = None
    file_path: Optional[str] = None


class _BulkChunkRow(BaseModel):
    chunk_id: str
    project_id: str
    doc_id: str
    chunk_index: int
    text: str
    embedding: List[float]
    created_at: Optional[str] = None


class _BulkInsertRequest(BaseModel):
    projects: List[_BulkProjectRow] = []
    documents: List[_BulkDocumentRow] = []
    chunks: List[_BulkChunkRow] = []
    # None -> the ACTIVE namespace (RAG_VECTOR_NAMESPACE, prod "v2"). Explicit
    # "" would target the retired legacy table — allowed only deliberately.
    namespace: Optional[str] = None
    # Sizeless document rows are REJECTED by default. The pilot corpus was
    # loaded through this endpoint with the size field dropped in the payload
    # transform (the source rag_backfill_*.json carry a real `size` per file),
    # which is why every document rendered as "0 B" in the UI. A silent default
    # made a data-integrity bug look like a UI bug for weeks. Operators who
    # genuinely mean to register metadata-only rows opt in explicitly.
    allow_missing_size: bool = False


@router.post("/v1/admin/corpus/bulk-insert")
def admin_corpus_bulk_insert(
    req: _BulkInsertRequest,
    auth: dict = Depends(require_api_key),
):
    """Idempotent bulk insert of projects + documents + chunks.

    Insert order respects FK: projects -> documents -> chunks. Projects and
    documents use existing-row-skip semantics (a partial batch can be
    re-sent without dup-key errors; skipped rows are NOT counted).

    Chunks go through ``VectorStore.upsert_chunks`` on the request's
    namespace (default: the ACTIVE retrieval namespace) — a per-document
    idempotent REPLACE. Send complete documents; re-sending a document
    re-writes exactly its own chunks. The store's embedding-dimension and
    model-identity guards apply: a payload whose vectors don't match the
    namespace's embedder returns 400 instead of writing unsearchable rows.
    """
    _require_admin(auth)

    from datetime import datetime, timezone
    import numpy as np

    from app.core.db import SessionLocal
    from app.core.models import Document, Project, rag_chunk_table_name
    from app.core.rag.vector_store import get_store

    # Resolve + validate the namespace BEFORE any row is written. A
    # request-supplied "" would select the RETIRED legacy ``chunks`` table:
    # refused outright -- a request never chooses the retired namespace.
    if req.namespace is not None and not req.namespace.strip():
        raise HTTPException(
            status_code=400,
            detail="namespace '' is the retired legacy chunks table; omit it to use the active namespace",
        )
    try:
        namespace_table = rag_chunk_table_name(
            req.namespace if req.namespace is not None
            else os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    store = get_store(namespace=req.namespace) if req.namespace is not None else get_store()

    # Reject sizeless document rows BEFORE writing anything — a document that
    # claims 0 bytes renders as "0 B" forever and no downstream check catches
    # it. Whole-payload rejection (not per-row skip) so a partial corpus is
    # never half-loaded with half the sizes missing.
    if not req.allow_missing_size:
        sizeless = [d.id for d in req.documents if (d.size or 0) <= 0]
        if sizeless:
            preview = ", ".join(sizeless[:10])
            more = f" (+{len(sizeless) - 10} more)" if len(sizeless) > 10 else ""
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{len(sizeless)} document row(s) have size <= 0: {preview}"
                    f"{more}. A sizeless document renders as '0 B' and cannot be "
                    f"distinguished from a failed ingest. Populate `size` from "
                    f"the source payload, or resend with allow_missing_size=true "
                    f"to register metadata-only rows deliberately. Nothing was "
                    f"written."
                ),
            )

    now_iso = datetime.now(timezone.utc).isoformat()
    counts: Dict[str, int] = {"projects": 0, "documents": 0, "chunks": 0,
                              "projects_seen": 0, "documents_seen": 0,
                              "chunks_seen": 0}

    with SessionLocal() as session:
        # ── projects (FK target) ───────────────────────────────────────
        for p in req.projects:
            counts["projects_seen"] += 1
            if session.get(Project, p.id) is not None:
                continue
            session.add(Project(
                id=p.id, name=p.name, status=p.status,
                aconex_connected=False, user_id=p.user_id,
                created_at=p.created_at or now_iso,
            ))
            counts["projects"] += 1
        session.flush()

        # ── documents ──────────────────────────────────────────────────
        for d in req.documents:
            counts["documents_seen"] += 1
            if session.get(Document, d.id) is not None:
                continue
            session.add(Document(
                id=d.id, project_id=d.project_id,
                original_name=d.original_name,
                stored_as=None, file_path=d.file_path,
                doc_type=d.doc_type, doc_role=d.doc_role,
                size=d.size,
                uploaded_at=d.uploaded_at or now_iso,
            ))
            counts["documents"] += 1
        session.flush()

        session.commit()

    # ── chunks — per-document idempotent upsert into the ACTIVE namespace ──
    # Grouped so each document is one ``upsert_chunks`` call (replace-all for
    # that doc). Ordered by chunk_index so text order survives the transport.
    by_doc: Dict[tuple, List[_BulkChunkRow]] = {}
    for c in req.chunks:
        counts["chunks_seen"] += 1
        by_doc.setdefault((c.project_id, c.doc_id), []).append(c)

    # Validate EVERY embedding's dimension before the first write, so a bad
    # payload is rejected whole — never some docs written and others not.
    for (project_id, doc_id), rows in by_doc.items():
        for r in rows:
            if len(r.embedding) != store.dim:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"embedding dim {len(r.embedding)} != store dim "
                        f"{store.dim} (doc {doc_id}) — payload was embedded "
                        f"with a different model than namespace "
                        f"{namespace_table!r} uses; nothing was written"
                    ),
                )

    verified: Dict[str, int] = {}
    for (project_id, doc_id), rows in by_doc.items():
        rows.sort(key=lambda r: r.chunk_index)
        # PostgreSQL rejects NUL bytes in text — same scrub as extraction.
        texts = [r.text.replace("\x00", "") for r in rows]
        emb = np.asarray([r.embedding for r in rows], dtype=np.float32)
        try:
            written = store.upsert_chunks(project_id, doc_id, texts, emb)
        except ValueError as exc:
            # Residual store-side guard (model identity) — surfaced, not 500.
            raise HTTPException(status_code=400, detail=str(exc))
        counts["chunks"] += written
        verified[doc_id] = store.count_by_doc(project_id).get(doc_id, 0)

    return {
        "status": "ok",
        "counts": counts,
        "namespace_table": namespace_table,
        "verified_chunks_by_doc": verified,
        # Non-zero only when the operator opted into metadata-only rows.
        "documents_without_size": sum(
            1 for d in req.documents if (d.size or 0) <= 0
        ),
    }


class _RepairSizesRequest(BaseModel):
    # None = every project. Scoped by default in the caller's hands, because a
    # corpus-wide repair stats every file on disk.
    project_id: Optional[str] = None
    # Report-only by default: an operator sees the damage before writing.
    dry_run: bool = True


@router.post("/v1/admin/corpus/repair-document-sizes")
def admin_repair_document_sizes(
    req: _RepairSizesRequest,
    auth: dict = Depends(require_api_key),
):
    """Recompute recorded document sizes from the bytes on disk.

    Repairs corpora already loaded by the pre-fix code paths — the rows behind
    the "every document shows 0 B" report. Measures PLAINTEXT length (so
    encrypted-at-rest files report their real size, not the inflated
    ciphertext), fixes rows that disagree with disk, and reports the rows it
    could not repair because the file is gone rather than inventing a number.
    """
    _require_admin(auth)

    from app.core import projects as _p

    result = _p.repair_document_sizes(req.project_id, dry_run=req.dry_run)
    return {"status": "ok", **result}


class _DeleteDocsRequest(BaseModel):
    doc_ids: List[str]
    # Safety scope: every doc must belong to this project_id or it is skipped.
    # Required so a bad id-list can't reach across corpora.
    project_id: str
    # "export" returns the docs' text + metadata WITHOUT deleting (the restore
    # bundle); "delete" removes each document row + its chunks (document-level,
    # no project cascade). Default is the safe read-only export.
    mode: str = "export"


@router.post("/v1/admin/corpus/delete-docs")
def admin_corpus_delete_docs(
    req: _DeleteDocsRequest,
    auth: dict = Depends(require_api_key),
):
    """Export or delete a specific list of documents from one corpus.

    Purpose: prune project-specific content that was indexed into a general
    corpus (e.g. company procedure folders in the master-corpus source) without
    touching the rest. Operates on an explicit ``doc_ids`` list scoped to one
    ``project_id`` — it never deletes a whole project and never cascades.

    Two modes:
      * ``export`` (default, read-only) — returns each doc's metadata and its
        concatenated chunk text. Use this to save a restore bundle BEFORE
        deleting; re-ingest is also possible from the source (the file_path
        points at the original).
      * ``delete`` — removes each document row and its chunks via the same
        ``projects.delete_document`` used by the normal delete path (chunk
        cleanup included). Files on disk are only touched if they exist on
        this host; Drive-sourced docs (``G:\\...`` paths) have no server file,
        so only the index row + chunks go.

    A doc whose ``project_id`` does not match the request scope is skipped, not
    deleted. Admin-only.
    """
    _require_admin(auth)

    mode = (req.mode or "export").strip().lower()
    if mode not in ("export", "delete"):
        raise HTTPException(status_code=400, detail="mode must be 'export' or 'delete'")
    if not req.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids is empty")
    if len(req.doc_ids) > 5000:
        raise HTTPException(status_code=400, detail="doc_ids capped at 5000 per call")

    from app.core import projects as _projects

    # Resolve + scope-check every id first (both modes need this).
    in_scope: List[Dict[str, Any]] = []
    not_found: List[str] = []
    wrong_project: List[str] = []
    for did in req.doc_ids:
        doc = _projects.get_document(did)
        if not doc:
            not_found.append(did)
            continue
        if doc.get("project_id") != req.project_id:
            wrong_project.append(did)
            continue
        in_scope.append(doc)

    if mode == "export":
        texts: Dict[str, List[str]] = {}
        try:
            from app.core.rag import retriever as _rag, vector_store as _vs
            if _rag.available():
                store = _vs.get_store()
                texts = store.doc_chunk_texts(
                    req.project_id, [d["id"] for d in in_scope]
                )
        except Exception as exc:  # noqa: BLE001 — export must still return metadata
            texts = {}
        docs_out = []
        for d in in_scope:
            chunks = texts.get(d["id"], [])
            docs_out.append({
                "id": d["id"],
                "name": d.get("original_name") or d.get("name"),
                "file_path": d.get("file_path"),
                "size": d.get("size"),
                "chunk_count": len(chunks),
                "text": "\n".join(chunks),
            })
        return {
            "mode": "export",
            "project_id": req.project_id,
            "requested": len(req.doc_ids),
            "in_scope": len(in_scope),
            "not_found": not_found,
            "wrong_project": wrong_project,
            "docs": docs_out,
        }

    # mode == "delete"
    deleted: List[str] = []
    failed: List[Dict[str, str]] = []
    files_removed = 0
    index_pruned = 0
    for d in in_scope:
        try:
            # Stored file + row + chunks + index entry (projects.purge_document).
            out = _projects.purge_document(d["id"])
            deleted.append(d["id"])
            files_removed += int(out["file_removed"])
            index_pruned += int(out["index_pruned"])
        except Exception as exc:  # noqa: BLE001
            failed.append({"id": d["id"], "error": str(exc)})
    return {
        "mode": "delete",
        "project_id": req.project_id,
        "requested": len(req.doc_ids),
        "deleted": len(deleted),
        "files_removed": files_removed,
        "index_pruned": index_pruned,
        "failed": failed,
        "not_found": not_found,
        "wrong_project": wrong_project,
    }


@router.post("/v1/admin/corpus/reconcile")
def admin_corpus_reconcile(
    execute: bool = Query(
        False,
        description="When true, repair mismatched chunk project_ids; "
                    "otherwise return a read-only report."
    ),
    sample_limit: int = Query(
        20, ge=0, le=1000,
        description="Maximum individual mismatches to return in the report."
    ),
    auth: dict = Depends(require_api_key),
):
    """Detect (and optionally repair) chunks stored under the wrong project_id.

    The master-corpus migration wrote chunks via ``/v1/admin/corpus/bulk-insert``
    using a per-document destination project_id. If the manifest mapped a
    document to project A but the chunk rows were stamped with project B,
    direct search on project A returns 0 results while project B returns chunks
    whose doc_id belongs to project A. This endpoint makes that mismatch
    visible and corrects it by aligning ``chunks.project_id`` with
    ``documents.project_id``.

    Always runs as a dry-run unless ``execute=true`` is passed, so operators
    can review the report before mutating the corpus. No chunks are deleted:
    only the ``project_id`` column is updated.
    """
    _require_admin(auth)

    from sqlalchemy import text
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    # Reconcile the SAME table the retriever reads — the namespaced store
    # (chunks_v2 by default), NOT the retired legacy `chunks` table. This
    # queried `chunks` literally, which has 0 rows post-v2-migration, so the
    # endpoint reported a clean corpus no matter how badly the real store was
    # mismatched. Same false-zero class as the admin corpus count fixed above.
    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )

    mismatches: List[Dict[str, Any]] = []
    dangling: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {}

    with SessionLocal() as session:
        # 1) Chunks whose project_id disagrees with their parent document.
        mismatch_rows = session.execute(
            text(
                f"""
                SELECT c.chunk_id, c.doc_id,
                       c.project_id AS current_project_id,
                       d.project_id AS correct_project_id
                FROM {chunks_table} c
                JOIN documents d ON c.doc_id = d.id
                WHERE c.project_id != d.project_id
                ORDER BY c.project_id, d.project_id, c.doc_id
                """
            )
        ).all()

        for row in mismatch_rows:
            cur = row.current_project_id
            corr = row.correct_project_id
            summary.setdefault(cur, {"chunks": 0, "mismatched_to": {}})
            summary.setdefault(corr, {"chunks": 0, "mismatched_from": {}})
            summary[cur]["mismatched_to"][corr] = (
                summary[cur]["mismatched_to"].get(corr, 0) + 1
            )
            summary[corr]["mismatched_from"][cur] = (
                summary[corr]["mismatched_from"].get(cur, 0) + 1
            )
            if len(mismatches) < sample_limit:
                mismatches.append({
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "current_project_id": cur,
                    "correct_project_id": corr,
                })

        # 2) Chunks whose doc_id has no parent document at all.
        dangling_rows = session.execute(
            text(
                f"""
                SELECT c.chunk_id, c.project_id, c.doc_id
                FROM {chunks_table} c
                LEFT JOIN documents d ON c.doc_id = d.id
                WHERE d.id IS NULL
                ORDER BY c.project_id, c.doc_id
                """
            )
        ).all()
        for row in dangling_rows:
            if len(dangling) < sample_limit:
                dangling.append({
                    "chunk_id": row.chunk_id,
                    "project_id": row.project_id,
                    "doc_id": row.doc_id,
                })

        repaired = 0
        if execute and mismatch_rows:
            # Align chunks.project_id with documents.project_id. This is a
            # single bulk update; unique-constraint collisions would indicate
            # duplicate chunks and will raise so the operator can inspect.
            result = session.execute(
                text(
                    f"""
                    UPDATE {chunks_table}
                    SET project_id = (
                        SELECT project_id FROM documents
                        WHERE id = {chunks_table}.doc_id
                    )
                    WHERE doc_id IN (
                        SELECT doc_id FROM {chunks_table} c2
                        JOIN documents d2 ON c2.doc_id = d2.id
                        WHERE c2.project_id != d2.project_id
                    )
                    AND project_id != (
                        SELECT project_id FROM documents
                        WHERE id = {chunks_table}.doc_id
                    )
                    """
                )
            )
            session.commit()
            repaired = result.rowcount or 0

    return {
        "dry_run": not execute,
        "execute": execute,
        # Name the table that was actually reconciled. Without this the caller
        # cannot tell a genuinely clean corpus from a scan of the wrong table.
        "chunks_table": chunks_table,
        "mismatches_sample": mismatches,
        "mismatches_total": len(mismatch_rows),
        "dangling_sample": dangling,
        "dangling_total": len(dangling_rows),
        "repaired": repaired,
        "summary": summary,
        "coverage": _corpus_coverage(),
    }


def _corpus_coverage(project_id: str | None = None) -> Dict[str, Any]:
    from app.core.ingest_reconcile import coverage_truth

    try:
        return coverage_truth(project_id=project_id)
    except Exception:
        logger.warning("coverage_truth failed", exc_info=True)
        return {"error": "coverage_unavailable"}


@router.get("/v1/admin/corpus/coverage")
def admin_corpus_coverage(
    project_id: Optional[str] = Query(None),
    auth: dict = Depends(require_api_key),
):
    """Queryable ingest coverage: status histogram, tombstones, OCR_DEGRADED,
    orphan chunks (report-only), embedding-model mismatch, and S11 office
    census (pptx / xlsx / pdf by status, thin rate, missing source,
    extractor_version).

    Read-only. Does not delete vectors or documents.
    """
    _require_admin(auth)
    return _corpus_coverage(project_id)


# ──────────────────────────────────────────────────────────────────────────
# PR A — admin-approved projects from Drive
#
# Two endpoints to support the operator's auto-detection + admin-approval
# architecture:
#
#   GET  /v1/admin/drive/scan
#     Walks the admin's connected Drive. Returns a cascaded folder tree
#     with file counts. Detection only — does NOT create projects,
#     does NOT index anything. The admin UI renders this as checkboxes
#     for "approve as project" actions.
#
#   POST /v1/admin/projects/approve-from-drive
#     Takes a Drive folder_id + project name. Creates a project row
#     with is_approved=True. Queues a background recursive import of
#     every supported file under that folder into the new project
#     (reuses the existing drive_index_folder helper logic).
# ──────────────────────────────────────────────────────────────────────────


class _DriveScanFolder(BaseModel):
    folder_id: str
    name: str
    direct_file_count: int
    subfolder_count: int
    is_candidate: bool  # True iff direct_file_count > 0
    children: List["_DriveScanFolder"] = []


_DriveScanFolder.model_rebuild()


@router.get("/v1/admin/drive/scan")
async def admin_drive_scan(
    max_depth: int = Query(2, ge=1, le=3,
                            description="How deep to descend; 2 covers the typical "
                                        "<root>/<container>/<project> layout."),
    auth: dict = Depends(require_api_key),
):
    """Detection-only Drive scan.

    Returns the folder tree from the admin's My Drive root to ``max_depth``.
    Each node carries ``direct_file_count`` (files immediately inside the
    folder) so the UI can mark candidates (>0 files) and the admin can
    decide which folders represent real projects.

    Cost is bounded by depth + Drive's listing pagination — for the
    pilot corpus (~10 top-level folders, ~50 sub-folders) this returns
    in under a second.
    """
    _require_admin(auth)

    from app.core import drive_auth
    from app.blocks.google_drive import GoogleDriveBlock

    drive_user_id = _drive_user_id(auth)
    try:
        access_token = await drive_auth.get_access_token(drive_user_id)
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected for this admin.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")

    drive = GoogleDriveBlock()

    FOLDER_MIME = "application/vnd.google-apps.folder"

    async def _list(folder_id: Optional[str]) -> List[Dict[str, Any]]:
        # GoogleDriveBlock.process expects query string + opts; folder_id
        # filters to direct children. Limit 200 per folder is plenty for
        # the operator's structure; the cap protects against bombs.
        resp = await drive.process("", {
            "operation": "list",
            "access_token": access_token,
            "limit": 200,
            "folder_id": folder_id,
        })
        if resp.get("status") != "success":
            raise HTTPException(502, resp.get("error", "Drive list failed."))
        return resp.get("files", [])

    async def _walk(folder_id: Optional[str], depth: int) -> List[Dict[str, Any]]:
        items = await _list(folder_id)
        folders = [i for i in items if i.get("mime_type") == FOLDER_MIME]
        files = [i for i in items if i.get("mime_type") != FOLDER_MIME]

        out: List[Dict[str, Any]] = []
        for f in folders:
            entry: Dict[str, Any] = {
                "folder_id": f.get("id"),
                "name": f.get("name", ""),
                "direct_file_count": 0,  # computed if we recurse
                "subfolder_count": 0,
                "is_candidate": False,
                "children": [],
            }
            if depth + 1 <= max_depth:
                # Recurse one level — collect direct children to compute
                # counts. The recursion result becomes this entry's children
                # only when those children themselves have nested folders
                # (depth + 2 <= max_depth); otherwise we still set the
                # counts but children stay empty.
                child_items = await _list(f.get("id"))
                child_folders = [c for c in child_items if c.get("mime_type") == FOLDER_MIME]
                child_files = [c for c in child_items if c.get("mime_type") != FOLDER_MIME]
                entry["direct_file_count"] = len(child_files)
                entry["subfolder_count"] = len(child_folders)
                entry["is_candidate"] = entry["direct_file_count"] > 0

                if depth + 2 <= max_depth:
                    # Go one more level for nested project structures.
                    deeper = await _walk(f.get("id"), depth + 1)
                    # _walk recurses through the listing — but we already
                    # listed this folder above. To avoid a second list call,
                    # build children from child_folders directly + recurse on each.
                    nested = []
                    for cf in child_folders:
                        nested_items = await _list(cf.get("id"))
                        nested_files = [c for c in nested_items if c.get("mime_type") != FOLDER_MIME]
                        nested_folders = [c for c in nested_items if c.get("mime_type") == FOLDER_MIME]
                        nested.append({
                            "folder_id": cf.get("id"),
                            "name": cf.get("name", ""),
                            "direct_file_count": len(nested_files),
                            "subfolder_count": len(nested_folders),
                            "is_candidate": len(nested_files) > 0,
                            "children": [],
                        })
                    entry["children"] = nested
                    _ = deeper  # placeholder; real depth-3 walk above is fine
            out.append(entry)
        return out

    root_items = await _list(None)
    root_folders = [i for i in root_items if i.get("mime_type") == FOLDER_MIME]
    root_files = [i for i in root_items if i.get("mime_type") != FOLDER_MIME]

    tree: List[Dict[str, Any]] = []
    for f in root_folders:
        # Manually compute first-level counts + (optionally) recurse for
        # depth-2 children to populate the cascade.
        sub_items = await _list(f.get("id"))
        sub_folders = [c for c in sub_items if c.get("mime_type") == FOLDER_MIME]
        sub_files = [c for c in sub_items if c.get("mime_type") != FOLDER_MIME]

        entry: Dict[str, Any] = {
            "folder_id": f.get("id"),
            "name": f.get("name", ""),
            "direct_file_count": len(sub_files),
            "subfolder_count": len(sub_folders),
            "is_candidate": len(sub_files) > 0,
            "children": [],
        }
        if max_depth >= 2:
            children: List[Dict[str, Any]] = []
            for sf in sub_folders:
                grand_items = await _list(sf.get("id"))
                grand_folders = [c for c in grand_items if c.get("mime_type") == FOLDER_MIME]
                grand_files = [c for c in grand_items if c.get("mime_type") != FOLDER_MIME]
                children.append({
                    "folder_id": sf.get("id"),
                    "name": sf.get("name", ""),
                    "direct_file_count": len(grand_files),
                    "subfolder_count": len(grand_folders),
                    "is_candidate": len(grand_files) > 0,
                    "children": [],
                })
            entry["children"] = children
        tree.append(entry)

    return {
        "max_depth": max_depth,
        "root_file_count": len(root_files),
        "candidates_total": sum(1 for f in tree if f["is_candidate"]),
        "tree": tree,
    }


class _ApproveFromDriveRequest(BaseModel):
    folder_id: str
    name: str
    max_files: int = 500
    max_depth: int = 6
    role: str = "other"


@router.post("/v1/admin/projects/approve-from-drive", status_code=201)
async def admin_approve_from_drive(
    req: _ApproveFromDriveRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(require_api_key),
):
    """Create a project row + queue recursive Drive-folder import.

    Flow:
      1. Slug the supplied name → project_id.
      2. Insert project row with is_approved=True, user_id=admin.
      3. Queue async import using the existing drive_index_folder logic
         — walks the Drive folder up to ``max_depth``, imports each
         supported file as a project document, kicks off doc-index +
         RAG indexing for each.

    Returns immediately with project_id and a status of "queued";
    the import progresses in the background. The admin can hit
    /v1/admin/corpus/collections later to see when chunks land.
    """
    _require_admin(auth)

    if not req.folder_id or not req.folder_id.strip():
        raise HTTPException(400, "folder_id is required")
    if not req.name or not req.name.strip():
        raise HTTPException(400, "name is required")

    import re
    from fastapi import BackgroundTasks
    from app.core import drive_auth, projects as _projects_mod

    # Verify Drive auth before doing any DB writes — fail fast.
    drive_user_id = _drive_user_id(auth)
    try:
        access_token = await drive_auth.get_access_token(drive_user_id)
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected for this admin.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")

    name = req.name.strip()
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48] or "project"

    # Ensure the slug is unique — append a short suffix if it collides.
    existing = _projects_mod.get_project(slug)
    if existing is not None:
        suffix = 2
        while _projects_mod.get_project(f"{slug}_{suffix}") is not None:
            suffix += 1
        slug = f"{slug}_{suffix}"

    project = _projects_mod.create_project(
        name=name,
        user_id=drive_user_id,
        is_approved=True,
        project_id=slug,
        origin="admin_drive_approved",
    )

    # Queue the recursive import as a background task.
    background_tasks.add_task(
        _run_drive_folder_import,
        project_id=slug, user_id=drive_user_id,
        folder_id=req.folder_id, max_files=req.max_files,
        max_depth=req.max_depth, role=req.role,
    )

    return {
        "status": "queued",
        "project": project,
        "import": {
            "folder_id": req.folder_id,
            "max_files": req.max_files,
            "max_depth": req.max_depth,
            "role": req.role,
        },
    }


# Wire BackgroundTasks injection — separate signature so FastAPI sees it.
@router.post("/v1/admin/projects/approve-from-drive/_bg")
async def _admin_approve_from_drive_bg(
    req: _ApproveFromDriveRequest,
    auth: dict = Depends(require_api_key),
):
    # Hidden alias — kept so the import-target dep gets BackgroundTasks
    # injected without bloating the canonical handler signature. Not for
    # external use.
    raise HTTPException(410, "Use /v1/admin/projects/approve-from-drive")


async def _run_drive_folder_import(
    *,
    project_id: str,
    user_id: str,
    folder_id: str,
    max_files: int,
    max_depth: int,
    role: str,
) -> None:
    """Background worker: walk the Drive folder + import every file
    into ``project_id``. Reuses the same helpers the per-project
    drive_index_folder route uses, so a file lands encrypted-at-rest +
    eagerly indexed for RAG identically to a user-initiated import.
    """
    import logging
    log = logging.getLogger(__name__)
    log.info("approve-from-drive: starting import project=%s folder=%s",
             project_id, folder_id)
    try:
        # Delay-import to avoid pulling Drive deps at module-load time.
        from app.routers import drive as drive_router
        from app.core import drive_auth
        from fastapi import BackgroundTasks as _BG

        access_token = await drive_auth.get_access_token(user_id)
        _bg = _BG()
        await drive_router._walk_drive_folder_into_project(
            project_id=project_id,
            user_id=user_id,
            access_token=access_token,
            folder_id=folder_id,
            max_files=max_files,
            max_depth=max_depth,
            role=role,
            background_tasks=_bg,
        )
        # Drain indexing tasks inline so the worker doesn't exit before
        # chunks are written. This keeps the admin "documents/chunks" counters
        # consistent once the import finishes.
        for task in _bg.tasks:
            if asyncio.iscoroutinefunction(task.func):
                await task.func(*task.args, **task.kwargs)
            else:
                task.func(*task.args, **task.kwargs)
    except Exception as exc:
        log.exception("approve-from-drive: import failed project=%s folder=%s: %s",
                      project_id, folder_id, exc)


# ── Dead-letter queue (Task 14) ───────────────────────────────────────────

@router.get("/v1/admin/dead-letter")
async def admin_dead_letter(auth: dict = Depends(require_api_key)):
    """List failed arq jobs from Redis result keys.

    Scans ``arq:result:*`` for entries with an ``"e"`` (error) field and
    returns them as dead-letter records. If Redis is unavailable the list
    is empty and an error flag is returned so the admin UI degrades safely.
    """
    _require_admin(auth)

    client = await get_redis_client()
    if client is None:
        return {"dead_letter": [], "error": "Redis not available"}

    try:
        keys = await client.keys("arq:result:*")
        failed: List[Dict[str, Any]] = []
        for key in keys:
            raw = await client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if data.get("e"):
                failed.append({
                    "job_id": key.split(":")[-1],
                    "function": data.get("f"),
                    "args": data.get("a"),
                    "error": data.get("e"),
                    "timestamp": data.get("t"),
                })
    except Exception as exc:  # noqa: BLE001 — graceful degradation for Redis errors
        logger.warning("Redis dead-letter scan failed (%s); returning empty list", exc)
        return {"dead_letter": [], "error": "Redis scan failed"}

    return {"dead_letter": failed}


import asyncio  # for asyncio.iscoroutinefunction used above


@router.post("/v1/admin/debug/sweep-plaintext")
def admin_sweep_plaintext(
    max_age_seconds: int = 3600,
    auth: dict = Depends(require_api_key),
):
    """Run the decrypted-temp sweeper on demand and return its counts.

    The sweeper also runs at boot and hourly (see main.lifespan); this
    endpoint exists so an operator — or a live gate — can verify the
    reaping behaviour deterministically instead of waiting for the timer.
    """
    _require_admin(auth)
    from app.core.file_crypto import sweep_stale_plaintext

    return {"status": "ok", **sweep_stale_plaintext(max_age_seconds)}


@router.post("/v1/admin/knowledge/documents", status_code=201)
async def admin_add_knowledge_document(
    response: Response,
    file: UploadFile = File(...),
    auth: dict = Depends(require_api_key),
):
    """Add a reference work to the general-knowledge layer (admin only).

    Codes, standards and contract-form guides are knowledge every project
    reads, not one project's record. The file is stored in the configured
    general-knowledge project with ``provenance: admin_knowledge`` -- never
    ``user_upload``, which the layered RAG files under the uploader's own
    session layer. The same content twice returns the existing document
    (200), not a copy.

    A reference work is uploaded ONCE, whole, up to its own cap
    (``ADMIN_KNOWLEDGE_MAX_UPLOAD_MB``). This route only stores it: the file
    is streamed to disk and hashed block by block (memory does not grow with
    the file), the row is recorded as pending, and the answer returns at
    once. Extraction and embedding run in the ingest task
    (``app.core.knowledge_ingest``), never inside the live web process.
    """
    _require_admin(auth)
    import hashlib
    import uuid

    from app.core import compressed, projects as store, upload_limits
    from app.core.ingest_status import TEXT_BEARING_EXTS
    from app.core.knowledge_ingest import pending_indexing_metadata
    from app.core.system_projects import primary_general_knowledge_project
    from app.core.users import SYSTEM_USER_ID, ensure_user_exists

    gk = primary_general_knowledge_project()
    if not gk:
        raise HTTPException(409, "No general-knowledge project is configured.")

    name = os.path.basename((file.filename or "").replace("\\", "/")).strip()
    if not name or name in (".", ".."):
        raise HTTPException(400, "Invalid filename")
    if compressed.is_compressed(name, compressed.head_of(file.file)):
        raise HTTPException(415, compressed.COMPRESSED_UPLOAD_DETAIL)
    if os.path.splitext(name.lower())[1] not in TEXT_BEARING_EXTS:
        raise HTTPException(415, "Only text-bearing formats go into the knowledge base.")

    if not store.get_project(gk):
        from sqlalchemy.exc import IntegrityError

        ensure_user_exists(SYSTEM_USER_ID, role="admin")
        try:
            store.create_project("General Knowledge", user_id=SYSTEM_USER_ID,
                                 project_id=gk, origin="admin_drive_approved")
        except IntegrityError:
            # The row exists already: the boot seed creates the same project
            # concurrently, or get_project hides it. Either way it is there.
            logger.info("general-knowledge project %s already exists", gk)

    data_dir = os.getenv("DATA_DIR", "./data")
    os.makedirs(data_dir, exist_ok=True)
    stored_as = f"{str(uuid.uuid4())[:8]}_{name}"
    filepath = os.path.join(data_dir, stored_as)
    hasher = hashlib.sha256()
    try:
        size = file_crypto.write_document_stream(
            filepath, file.file,
            max_bytes=upload_limits.admin_knowledge_max_bytes(),
            hasher=hasher,
        )
    except file_crypto.UploadTooLarge as exc:
        raise HTTPException(413, f"File too large. Max is {exc.limit} bytes.") from exc

    sha = hasher.hexdigest()
    existing = store.find_document_by_sha(gk, sha)
    if existing:
        os.remove(filepath)
        response.status_code = 200
        return {"status": "exists", "document": existing}

    doc = store.add_document(gk, name, stored_as, filepath, size, content_sha256=sha,
                             metadata={"provenance": "admin_knowledge",
                                       "uploader_id": auth.get("user_id"),
                                       **pending_indexing_metadata()})
    return {"status": "pending", "document": doc}
