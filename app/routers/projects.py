"""Project API — create projects, attach documents, gated progress tracking.

Roadmap V2 · Part 0:
  0.1  Project entity
  0.2  Readiness gate — progress tracking refuses to run on an unready project
  0.3  Execution-intent model — attaching a document stores it and runs NOTHING;
       analysis happens only when explicitly requested.
"""

import io
import math
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.core import audit, doc_index, file_crypto, projects as store
from app.blocks import BLOCK_REGISTRY, get_block
from app.dependencies import (
    require_user,
    block_instances,
    _create_block_instance,
)
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

DATA_DIR = os.getenv("DATA_DIR", "./data")
# Cap + accepted formats come from app.core.upload_limits so this route and
# /upload can never drift apart again (they had different caps AND different
# extension sets). Read through the function so a deployment-time env change
# takes effect — the old module-level int bound the limit at import.
from app.core import upload_limits


def _max_doc_upload_size() -> int:
    return upload_limits.max_document_bytes()
# Pilot guardrail: approved Drive projects with this many or fewer indexed
# documents are treated as incomplete shells and suppressed from non-admin
# project lists so pilot users land on the master corpus instead.
PILOT_INCOMPLETE_SHELL_DOC_THRESHOLD = int(
    os.getenv("PILOT_INCOMPLETE_SHELL_DOC_THRESHOLD", "1")
)
# Project-detail returns only the first page of documents (newest first); the
# rest load on demand via GET .../documents. Keeps a 2700-doc corpus from
# serializing every row on every workspace open.
_DOC_FIRST_PAGE = int(os.getenv("PROJECT_DETAIL_DOC_PAGE", "100"))
try:
    os.makedirs(DATA_DIR, exist_ok=True)
except PermissionError:
    import tempfile
    DATA_DIR = tempfile.gettempdir()

# A project document is persisted and indexed: the RAG takes text formats
# only (ingest_status.TEXT_BEARING_EXTS, docs/INGEST_EXCLUSION_RULE.md).
from app.core.ingest_status import TEXT_BEARING_EXTS as ALLOWED_DOC_EXTENSIONS  # noqa: E402


def _owned_or_404(
    project_id: str,
    user_id: str,
    *,
    read_only: bool = False,
    role: str = "user",
    doc_limit: Optional[int] = None,
    doc_offset: int = 0,
):
    """Load a project the caller can access, or 404 (never leak existence).

    PR D — when ``read_only=True``, non-owners are also allowed to load
    admin-approved platform projects (origin='admin_drive_approved' +
    is_approved=True). Used by the read-only GET handler so users can
    open shared platform projects without owning them. Mutating
    handlers must use the default (owner-only).

    2026-07-24 — read-only access also honors ``role="admin"``: the list
    endpoint shows admins EVERY project, so the detail page must open them
    too (the operator hit "Project not found" clicking an eval-fixture
    project owned by the seeding account). Mutating handlers are
    unaffected — they never pass ``role``.

    2026-07-25 — document upload uses the SAME extended access rule
    (``read_only=True`` + caller role). The pilot works on shared projects:
    anyone who can OPEN a project must be able to add documents to it —
    the old owner-only check made the upload button 404 with "Project not
    found" on every project the caller didn't personally create, which
    testers reported as "upload is broken". Destructive mutations
    (delete/clear/archive) stay owner-only.

    ``doc_limit``/``doc_offset`` paginate the returned documents (default None
    = all, for mutating callers that need the full set).
    """
    proj = store.get_project(
        project_id, user_id=user_id, include_admin_approved=read_only,
        doc_limit=doc_limit, doc_offset=doc_offset,
    )
    if proj is None and read_only and (role or "").lower() == "admin":
        proj = store.get_project(
            project_id, doc_limit=doc_limit, doc_offset=doc_offset,
        )
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    return proj


# ── request models ──────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    name: str
    client: Optional[str] = None
    location: Optional[str] = None


class UpdateProjectRequest(BaseModel):
    location: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str


class CreateProjectFromDriveRequest(BaseModel):
    """PR C — user-facing variant of approve-from-drive.

    Lets any authenticated user create a personal project from a Drive
    folder they own (not just admins from a scanned cascade). The
    created row is owned by the calling user and stamped
    origin='user_drive_import' so admins can distinguish it from their
    own 'admin_drive_approved' rows and from blank 'user_create' rows.
    """
    folder_id: str
    name: str
    client: Optional[str] = None
    max_files: int = 500
    max_depth: int = 6
    role: str = "other"


class ProgressRequest(BaseModel):
    planned_percent: float = 0
    actual_percent: float = 0
    contract_value: float = 0
    reporting_period: Optional[str] = None
    activities: List[Dict[str, Any]] = []
    photos: List[str] = []


class UpdateDocumentRequest(BaseModel):
    """Seed a Drive file id on an existing document without re-uploading.

    Master Corpus rag_backfill stubs have ``drive_file_id: null``. The
    service-account name-search cannot see anyone-with-link files, so an
    owner/admin PATCH is how we attach the live id (e.g. <doc-id> →
    ``1ExampleDriveFileId001xxxxxxxxxxx``). Preview then hydrates via
    ``files.get`` / public download.
    """
    drive_file_id: str


# ── projects ────────────────────────────────────────────────────────────────

@router.post("/v1/projects", status_code=201)
async def create_project(req: CreateProjectRequest, auth: dict = Depends(require_user)):
    """Create a project. Documents and analytics hang off this entity."""
    name = (req.name or "").strip()
    if not name:
        raise HTTPException(400, "Project name is required")
    proj = store.create_project(
        name, (req.client or "").strip() or None, user_id=auth["user_id"],
        location=(req.location or "").strip() or None,
    )
    await audit.arecord("project.created", project_id=proj["id"], name=name, user_id=auth["user_id"])
    return proj


@router.patch("/v1/projects/{project_id}")
async def update_project(project_id: str, req: UpdateProjectRequest,
                         auth: dict = Depends(require_user)):
    """Update a project's confirmed metadata (currently: site location).

    The location set here is what ``daily_site_report`` reads for weather — it is
    NEVER inferred from a chat message. Setting to empty clears it.
    """
    _owned_or_404(project_id, auth["user_id"])
    updated = store.set_project_location(
        project_id, (req.location or "").strip() or None)
    if updated is None:
        raise HTTPException(404, "Project not found")
    await audit.arecord("project.updated", project_id=project_id, user_id=auth["user_id"])
    return updated


@router.post("/v1/projects/from-drive", status_code=201)
async def create_project_from_drive(
    req: CreateProjectFromDriveRequest,
    background_tasks: BackgroundTasks,
    auth: dict = Depends(require_user),
):
    """Create a user-owned project seeded from a Drive folder.

    Mirrors /v1/admin/projects/approve-from-drive but:
      * Open to any authenticated user (no admin gate).
      * Project row is owned by the caller, not the admin.
      * origin='user_drive_import' (not 'admin_drive_approved').

    Validates Drive auth eagerly (fails fast 409 if not connected),
    slugs the project name with a uniqueness suffix, persists the row,
    and queues the recursive Drive folder walk as a background task —
    the import progresses asynchronously; the response returns once
    the row is on disk so the UI can navigate into the new project
    immediately.
    """
    import re
    from app.core import drive_auth
    from app.routers.admin import _run_drive_folder_import

    folder_id = (req.folder_id or "").strip()
    name = (req.name or "").strip()
    if not folder_id:
        raise HTTPException(400, "folder_id is required")
    if not name:
        raise HTTPException(400, "name is required")

    try:
        await drive_auth.get_access_token(auth["user_id"])
    except drive_auth.DriveNotConnected:
        raise HTTPException(409, "Google Drive is not connected for your account.")
    except drive_auth.DriveAuthError as e:
        raise HTTPException(409, f"{e} Reconnect Google Drive.")

    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:48] or "project"
    if store.get_project(slug) is not None:
        suffix = 2
        while store.get_project(f"{slug}_{suffix}") is not None:
            suffix += 1
        slug = f"{slug}_{suffix}"

    proj = store.create_project(
        name=name,
        client=(req.client or "").strip() or None,
        user_id=auth["user_id"],
        is_approved=True,
        project_id=slug,
        origin="user_drive_import",
    )
    await audit.arecord(
        "project.created_from_drive",
        project_id=proj["id"], name=name, user_id=auth["user_id"],
        folder_id=folder_id,
    )

    background_tasks.add_task(
        _run_drive_folder_import,
        project_id=slug, user_id=auth["user_id"],
        folder_id=folder_id, max_files=req.max_files,
        max_depth=req.max_depth, role=req.role,
    )

    return {
        "status": "queued",
        "project": proj,
        "import": {
            "folder_id": folder_id,
            "max_files": req.max_files,
            "max_depth": req.max_depth,
            "role": req.role,
        },
    }


@router.get("/v1/projects")
async def list_projects(auth: dict = Depends(require_user)):
    """List projects visible to the caller.

    PR D visibility model:
      * Admins see every project — they need the full picture to
        approve, re-index, and delete.
      * Non-admins see their own projects PLUS the admin-curated
        platform projects (origin='admin_drive_approved' AND
        is_approved=True). Other users' personal projects stay hidden.
      * ``is_approved=False`` rows never appear for non-owners — the
        column is reserved for future "detected but pending" candidates.
    """
    role = (auth.get("role") or "user").lower()
    if role == "admin":
        rows = store.list_projects()  # full set
    else:
        rows = store.list_projects(
            user_id=auth["user_id"], include_admin_approved=True,
        )
        # Pilot: hide incomplete approved shells from non-admins so they
        # gravitate to the Master Corpus.
        rows = [
            r for r in rows
            if not (
                r.get("origin") == "admin_drive_approved"
                and not r.get("is_master_corpus")
                and r.get("document_count", 0) <= PILOT_INCOMPLETE_SHELL_DOC_THRESHOLD
            )
        ]
    return {"projects": rows}


@router.get("/v1/projects/{project_id}")
async def get_project(project_id: str, auth: dict = Depends(require_user)):
    """Project detail — documents + the computed readiness gate.

    Documents are enriched with the live ``chunk_count`` and ``panel_status``.
    The badge states the reason stored on the row. A count of zero is not
    treated as an extraction failure.
    """
    proj = _owned_or_404(
        project_id, auth["user_id"], read_only=True,
        role=auth.get("role") or "user", doc_limit=_DOC_FIRST_PAGE,
    )
    # `documents` is now the first page only; `document_count` is the true total
    # (a cheap COUNT). Gate enrichment on the TOTAL, not the page length.
    doc_count = proj.get("document_count") or 0
    proj["documents_truncated"] = len(proj.get("documents") or []) < doc_count

    # Live chunk counts come from the vector store, not the legacy doc_index
    # JSON blob. Bulk-inserted chunks (the master-corpus migration) never touch
    # doc_index, so counting from the ``chunks`` table is the only accurate
    # source. Both the project total and the per-document counts are cheap
    # indexed queries (COUNT / GROUP BY), so they run for every project size.
    resolved_id = store._master_corpus_source(project_id) or project_id
    chunk_store = None
    try:
        from app.core.rag.embeddings import get_embedder
        from app.core.rag.vector_store import get_store
        embedder = get_embedder()
        chunk_store = get_store(dim=embedder.dim)
        indexed_chunks = chunk_store.count(resolved_id)
        proj["chunk_count"] = indexed_chunks
        proj["indexed_chunks"] = indexed_chunks
        proj["has_indexed_chunks"] = indexed_chunks > 0
    except Exception:
        # Vector store may not be configured in all test environments.
        proj["chunk_count"] = None
        proj["indexed_chunks"] = None
        proj["has_indexed_chunks"] = None

    # Per-document chunk_count for the first page (same page the endpoint
    # returns). A healthy bulk-migrated doc must never show 0 here.
    try:
        if chunk_store is not None:
            chunk_counts = chunk_store.count_by_doc(resolved_id)
            for doc in proj.get("documents", []) or []:
                doc["chunk_count"] = chunk_counts.get(doc.get("id"), 0)
                doc["searchable"] = bool(doc["chunk_count"])
                _idx_meta = (doc.get("metadata") or {}).get("indexing")
                if _idx_meta:
                    doc["indexing_status"] = _idx_meta
                # Live counts override the ledger. Recompute the note so a
                # file that now has chunks does not keep a not-indexed label.
                store.attach_panel_status(doc)
    except Exception:
        # Enrichment is best-effort — never break the project load on it.
        logger.warning(
            "swallowed %s in get_project() — continuing",
            "Exception", exc_info=True,
        )

    return proj


@router.get("/v1/projects/{project_id}/documents")
async def list_project_documents(
    project_id: str,
    offset: int = 0,
    limit: int = 100,
    auth: dict = Depends(require_user),
):
    """A page of a project's documents (newest first).

    The workspace gets page 1 with the project detail and loads further pages
    here on demand, so a large corpus never serializes every row at once.
    """
    # Access check (read-only: platform projects are visible to all; admins
    # paginate any project they can open — the detail endpoint already honors
    # role, so pagination must too or the workspace breaks after page 1).
    # doc_limit=0 makes the check itself cheap — no document rows loaded.
    proj = _owned_or_404(
        project_id, auth["user_id"], read_only=True,
        role=auth.get("role", "user"), doc_limit=0,
    )
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    return {
        "documents": store.list_documents(
            project_id, limit=limit, offset=offset, newest_first=True,
        ),
        "total": proj.get("document_count") or store.count_documents(project_id),
        "offset": offset,
        "limit": limit,
    }


# ── inline document preview ─────────────────────────────────────────────────
# Render generated / uploaded artifacts (schedules, BOQs, procurement lists,
# reports) in the right-panel WITHOUT forcing a download. Read-only; shared and
# master-corpus projects are previewable like the read GET.

# Per-sheet caps so a huge workbook can't serialize megabytes of cells into the
# preview payload. The frontend notes truncation to the user.
PREVIEW_MAX_ROWS = 200
PREVIEW_MAX_COLS = 40
# First N chars of extracted text (docx/txt/md) — enough to preview, bounded.
PREVIEW_TEXT_CHARS = 20_000

_TABLE_EXTS = {".xlsx", ".xls", ".csv"}
_TEXT_EXTS = {".docx", ".doc", ".txt", ".md"}


def _cell_str(value: Any) -> str:
    """Stringify a cell for the JSON payload. None and pandas NaN → ""."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value)


def _preview_citeable_owner_ids(workspace_project_id: str) -> set[str]:
    """Project ids whose documents chat may cite — and therefore preview.

    Opening a Sources cite always hits
    ``/v1/projects/{workspace}/documents/{cited_doc}/preview``. The cited
    file is often Master Corpus or general-knowledge, not a row owned by
    the workspace. Restricting preview to the URL project's own rows 404s
    that fetch even though RAG already exposed the document.
    """
    owners: set[str] = set()
    if workspace_project_id:
        owners.add(workspace_project_id)
        alias_src = store._master_corpus_source(workspace_project_id)
        if alias_src:
            owners.add(alias_src)
    mc_alias = getattr(store, "MASTER_CORPUS_PROJECT_ID", None)
    if mc_alias:
        owners.add(mc_alias)
    mc_src = (getattr(store, "MASTER_CORPUS_SOURCE_PROJECT_ID", None) or "").strip()
    if mc_src:
        owners.add(mc_src)
    try:
        from app.core.rag.retriever import _general_knowledge_project_ids
        owners.update(_general_knowledge_project_ids())
    except Exception:
        logger.debug(
            "preview citeable-owner lookup skipped GK ids", exc_info=True,
        )
    return {pid for pid in owners if pid}


def _resolve_document_file(
    project_id: str, document_id: str, user_id: str, *, role: str = "user",
) -> Tuple[Dict[str, Any], str, Optional[str], Optional[HTTPException]]:
    """Access-check and resolve a document and its stored file.

    Returns ``(doc, ext, file_path, missing)``. When no original is stored,
    ``file_path`` is None and ``missing`` is the clear 404 that explains why
    (empty blob, missing, Drive not configured, Drive fetch failed...); the
    caller raises it or falls back to the indexed text.

    Read-only access — non-owners may preview shared / master-corpus
    documents like the read GET does. ``doc_limit=0`` keeps the project
    access check cheap (no document rows serialized — critical for the
    ~2,700-doc master corpus). The master-corpus alias resolves to its
    backing source id before the ownership match, because ``get_document``
    returns the SOURCE project_id, not the alias.

    A document cited from Master Corpus / GK is previewable from the
    workspace that cited it. A private project the caller does not have
    open stays 404.

    File bytes: P1B / Master Corpus rows often have a stale ``file_path``
    (the platform keeps no copy of an original) and ``size=0``. After the
    ownership check, hydrate from the row's Google Drive file id rather
    than 404-ing on disk.
    """
    _owned_or_404(
        project_id, user_id, read_only=True, role=role, doc_limit=0,
    )
    doc = store.get_document(document_id)
    owner = (doc or {}).get("project_id")
    allowed = _preview_citeable_owner_ids(project_id)
    if not doc or owner not in allowed:
        raise HTTPException(
            404, f"Document '{document_id}' not found in project '{project_id}'"
        )
    _, ext = os.path.splitext((doc.get("original_name") or "").lower())
    fp, blob_state = store.materialize_document_file(doc)
    if not fp:
        if blob_state == "empty":
            missing = HTTPException(
                404,
                "Document file is empty (0 bytes) and cannot be previewed",
            )
        elif blob_state == "missing":
            missing = HTTPException(404, "Document file is not available for preview")
        else:
            missing = HTTPException(
                404,
                f"Document file is not available for preview ({blob_state})",
            )
        return doc, ext, None, missing
    refreshed = store.get_document(document_id) or doc
    return refreshed, ext, fp, None


def _resolve_preview_document(
    project_id: str, document_id: str, user_id: str, *, role: str = "user",
) -> Tuple[Dict[str, Any], str, str]:
    """``_resolve_document_file`` for callers that need the original: a
    document with no stored file is its clear 404, never a 500."""
    doc, ext, fp, missing = _resolve_document_file(
        project_id, document_id, user_id, role=role,
    )
    if missing is not None:
        raise missing
    return doc, ext, fp


def _table_preview(file_path: str, ext: str) -> Dict[str, Any]:
    """Build a ``{"kind":"table", "sheets":[...]}`` payload from a spreadsheet.

    Caps each sheet at PREVIEW_MAX_ROWS x PREVIEW_MAX_COLS and flags truncation.
    Legacy ``.xls`` needs xlrd (not installed) — pandas raises and the caller
    converts that to a 422.
    """
    sheets: List[Dict[str, Any]] = []
    truncated = False

    if ext == ".csv":
        import csv as _csv

        raw = file_crypto.read_document(file_path)
        text = raw.decode("utf-8", errors="replace")
        rows: List[List[str]] = []
        for i, row in enumerate(_csv.reader(io.StringIO(text))):
            if i >= PREVIEW_MAX_ROWS:
                truncated = True
                break
            if len(row) > PREVIEW_MAX_COLS:
                truncated = True
            rows.append([_cell_str(c) for c in row[:PREVIEW_MAX_COLS]])
        sheets.append({"name": "Sheet1", "rows": rows})

    elif ext == ".xlsx":
        import openpyxl

        with file_crypto.open_plaintext(file_path) as readable_path:
            wb = openpyxl.load_workbook(
                readable_path, data_only=True, read_only=True
            )
            try:
                for name in wb.sheetnames:
                    ws = wb[name]
                    rows = []
                    for r, row in enumerate(ws.iter_rows(values_only=True)):
                        if r >= PREVIEW_MAX_ROWS:
                            truncated = True
                            break
                        if len(row) > PREVIEW_MAX_COLS:
                            truncated = True
                        rows.append([_cell_str(c) for c in row[:PREVIEW_MAX_COLS]])
                    sheets.append({"name": name, "rows": rows})
            finally:
                wb.close()

    else:  # .xls — best effort via pandas; raises without xlrd → 422 upstream.
        import pandas as pd

        with file_crypto.open_plaintext(file_path) as readable_path:
            frames = pd.read_excel(readable_path, sheet_name=None, header=None)
        for name, df in frames.items():
            if df.shape[0] > PREVIEW_MAX_ROWS or df.shape[1] > PREVIEW_MAX_COLS:
                truncated = True
            clipped = df.iloc[:PREVIEW_MAX_ROWS, :PREVIEW_MAX_COLS]
            rows = [[_cell_str(c) for c in record] for record in clipped.values.tolist()]
            sheets.append({"name": str(name), "rows": rows})

    return {"kind": "table", "sheets": sheets, "truncated": truncated}


def _text_preview(file_path: str, ext: str) -> Dict[str, Any]:
    """Build a ``{"kind":"text", "text":...}`` payload (first PREVIEW_TEXT_CHARS).

    Legacy binary ``.doc`` is not readable by python-docx and raises → 422.
    """
    if ext in {".txt", ".md"}:
        raw = file_crypto.read_document(file_path)
        text = raw.decode("utf-8", errors="replace")
    else:  # .docx / .doc
        import docx

        with file_crypto.open_plaintext(file_path) as readable_path:
            document = docx.Document(readable_path)
            text = "\n".join(p.text for p in document.paragraphs)
    truncated = len(text) > PREVIEW_TEXT_CHARS
    return {"kind": "text", "text": text[:PREVIEW_TEXT_CHARS], "truncated": truncated}


_INDEXED_ONLY_NOTE = (
    "The original file is not stored on the platform. This is the text the "
    "platform indexed from it, which is what answers were drawn from."
)


def _indexed_text(owner_project_id: str, document_id: str) -> str:
    """The text the platform indexed for a document, chunks in order.

    Read under the document's OWNING project: a Master Corpus / general
    knowledge document's chunks are not stored under the workspace that cited
    it. Empty when nothing was indexed or the chunk store is unavailable."""
    try:
        from app.core.rag import vector_store as _vs

        chunks = _vs.get_store().doc_chunk_texts(owner_project_id, [document_id])
    except Exception:  # noqa: BLE001 — no chunk store is "nothing indexed"
        logger.debug("indexed text lookup failed", exc_info=True)
        return ""
    return "\n\n".join(t for t in chunks.get(document_id, []) or [] if t)


def _display_filename(project_id: str, doc: Dict[str, Any]) -> str:
    """The name a user sees for a document: its stored name. Another
    project's documents are never served, and shared ones are cleaned at
    source, so the stored name is the one to show."""
    name = str(doc.get("original_name") or doc.get("id") or "document")
    return os.path.basename(name.replace("\\", "/")) or "document"


def _content_disposition(filename: str) -> str:
    """RFC 6266 attachment header that survives non-ASCII filenames."""
    from urllib.parse import quote

    ascii_name = filename.encode("ascii", "replace").decode("ascii").replace('"', "'")
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{quote(filename)}'


@router.get("/v1/projects/{project_id}/documents/{document_id}/preview")
async def preview_document(
    project_id: str, document_id: str, auth: dict = Depends(require_user)
):
    """Render-friendly JSON preview of a stored document, by extension.

    * .xlsx/.xls/.csv → {"kind":"table", "sheets":[...]}
    * .pdf            → {"kind":"pdf"} (raw bytes at the sibling /preview/raw)
    * .docx/.doc/.txt/.md → {"kind":"text", "text": ...}
    * anything else   → {"kind":"unsupported", "ext": ext}

    A malformed / unreadable file returns 422 (never 500).
    """
    doc, ext, fp, missing = _resolve_document_file(
        project_id, document_id, auth["user_id"],
        role=auth.get("role") or "user",
    )
    if fp is None:
        text = _indexed_text(str(doc.get("project_id") or project_id), document_id)
        if not text.strip():
            raise missing
        return {
            "kind": "text",
            "text": text[:PREVIEW_TEXT_CHARS],
            "truncated": len(text) > PREVIEW_TEXT_CHARS,
            "indexed_only": True,
            "note": _INDEXED_ONLY_NOTE,
            "size": int(doc.get("size") or 0),
            "has_file": False,
        }
    try:
        if ext in _TABLE_EXTS:
            payload: Dict[str, Any] = _table_preview(fp, ext)
        elif ext == ".pdf":
            payload = {"kind": "pdf"}
        elif ext in _TEXT_EXTS:
            payload = _text_preview(fp, ext)
        else:
            payload = {"kind": "unsupported", "ext": ext}
        payload["size"] = int(doc.get("size") or 0)
        payload["has_file"] = True
        return payload
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — bad file must 422, never 500.
        raise HTTPException(
            422, f"Could not render a preview for this document: {exc}"
        )


@router.get("/v1/projects/{project_id}/documents/{document_id}/preview/raw")
async def preview_document_raw(
    project_id: str, document_id: str, auth: dict = Depends(require_user)
):
    """Raw decrypted PDF bytes so the frontend can embed the document.

    Returns the bytes in-memory (NOT a FileResponse over an open_plaintext temp
    path — that temp file is deleted when the context manager exits, before
    Starlette can stream it, which fails whenever encryption-at-rest is on). The
    upload size cap keeps buffering the whole file in memory safe for v1.
    """
    _doc, ext, fp = _resolve_preview_document(
        project_id, document_id, auth["user_id"],
        role=auth.get("role") or "user",
    )
    if ext != ".pdf":
        raise HTTPException(400, "Raw preview is available only for PDF documents")
    try:
        raw = file_crypto.read_document(fp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not read the document: {exc}")
    return Response(content=raw, media_type="application/pdf")


@router.get("/v1/projects/{project_id}/documents/{document_id}/download")
async def download_document(
    project_id: str, document_id: str, auth: dict = Depends(require_user)
):
    """The document as a file: its original bytes, or -- when no original is
    stored -- the text the platform indexed, as ``<name> (indexed text).txt``.

    Same access rule as the preview: the workspace's own documents and the
    ones it may cite. A cited document from another layer keeps its filename
    scrubbed, exactly as the Sources panel shows it."""
    doc, _ext, fp, missing = _resolve_document_file(
        project_id, document_id, auth["user_id"],
        role=auth.get("role") or "user",
    )
    filename = _display_filename(project_id, doc)
    if fp is None:
        text = _indexed_text(str(doc.get("project_id") or project_id), document_id)
        if not text.strip():
            raise missing
        return Response(
            content=text.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": _content_disposition(f"{filename} (indexed text).txt")},
        )
    try:
        raw = file_crypto.read_document(fp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"Could not read the document: {exc}")
    import mimetypes

    media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    return Response(
        content=raw,
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition(filename)},
    )


@router.delete("/v1/projects/{project_id}")
async def delete_project(project_id: str, auth: dict = Depends(require_user)):
    """Delete a project: its document records, facts, AND files on disk."""
    # Resolve pilot master-corpus alias before the ownership check so admins
    # can delete the shared corpus project from the UI.
    resolved_id = store._master_corpus_source(project_id) or project_id
    # Admins see every project in the list, so they must be able to load (and
    # then delete) any of them — look up unscoped for admins. Without this an
    # admin gets a 404 for a project they don't own, BEFORE the admin-bypass
    # below ever runs (the "frozen projects / not found" bug). Non-admins keep
    # the per-user scoped lookup.
    is_admin = auth.get("role") == "admin"
    proj = store.get_project(
        project_id,
        user_id=None if is_admin else auth["user_id"],
        include_admin_approved=True,
    )
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    if proj.get("user_id") != auth["user_id"] and not is_admin:
        raise HTTPException(403, "Admin or project owner required")
    # SOFT delete: archive the project — hide it from listings, detail,
    # ownership gates and retrieval — WITHOUT removing the row. `chunks` is
    # ON DELETE CASCADE on project_id, so a hard delete would destroy the
    # project's RAG chunks. The operator principle is "delete the UI, never
    # the RAG; build on it only", so documents, files on disk, and chunks all
    # stay and the project is restorable (set status back to 'active').
    store.archive_project(resolved_id)
    await audit.arecord("project.archived", project_id=resolved_id,
                 user_id=auth["user_id"])
    return {
        "status": "archived",
        "project_id": resolved_id,
    }


@router.post("/v1/projects/{project_id}/conversations/{conversation_id}/clear")
async def clear_project_conversation(
    project_id: str,
    conversation_id: str,
    auth: dict = Depends(require_user),
):
    """Wipe one conversation's messages + facts without deleting the
    conversation row. Lets the operator escape a thread poisoned by
    prior hallucinated tool-skip / fabricated-table turns without
    creating a new project.

    Owner-only. Cross-project conversation IDs are rejected with 404 to
    avoid info-leak via timing.
    """
    # Resolve pilot master-corpus alias so the shared corpus can be cleared
    # from the workspace UI.
    resolved_id = store._master_corpus_source(project_id) or project_id
    proj = store.get_project(
        project_id, user_id=auth["user_id"], include_admin_approved=True
    )
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    if proj.get("user_id") != auth["user_id"] and auth.get("role") != "admin":
        raise HTTPException(403, "Admin or project owner required")
    from app.core import agent_memory

    # Workspace conversation IDs are deterministic: ws-{project_id} for the
    # original single thread, ws-{project_id}-{suffix} for the multi-session
    # chat history (Quarry parity). Accept both the alias and the backing
    # project id for the master corpus; reject any other workspace prefix
    # before we let the call near agent_memory.
    allowed_prefixes = tuple(
        p for pid in {project_id, resolved_id} for p in (f"ws-{pid}", f"ws-{pid}-")
    )
    if conversation_id.startswith("ws-") and not (
        conversation_id in {f"ws-{project_id}", f"ws-{resolved_id}"}
        or conversation_id.startswith(tuple(p for p in allowed_prefixes if p.endswith("-")))
    ):
        raise HTTPException(404, "Conversation not found")

    # For non-workspace conversation IDs, confirm the stored row (if any)
    # belongs to this project (alias or source id).
    conv = agent_memory.get_conversation(conversation_id)
    if conv is not None and conv.get("project_id") not in (None, "", project_id, resolved_id):
        raise HTTPException(404, "Conversation not found")

    cleared = agent_memory.clear_conversation(conversation_id)
    await audit.arecord(
        "conversation.cleared",
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=auth["user_id"],
        messages=cleared["messages"],
        facts=cleared["facts"],
    )
    return {
        "status": "cleared",
        "conversation_id": conversation_id,
        **cleared,
    }


def _session_scope(project_id: str, user_id: str) -> tuple[set[str], Optional[str]]:
    """Project ids a session may be bound to, and the project row owner.

    Uses the row-only access check. Loading the project's documents here
    made every history refresh walk the corpus.
    """
    if not store.can_access_project(
        project_id, user_id=user_id, include_admin_approved=True
    ):
        raise HTTPException(404, f"Project '{project_id}' not found")
    storage_id = store.storage_project_id(project_id)
    return {project_id, storage_id}, store.project_owner(storage_id)


def _public_session(conv: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": conv["id"],
        "title": conv.get("title"),
        "created_at": conv.get("created_at"),
        "updated_at": conv.get("updated_at"),
    }


def _session_for_caller(
    project_id: str,
    conversation_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """The session when it belongs to this project and this caller, else 404."""
    from app.core import agent_memory

    project_ids, owner_id = _session_scope(project_id, user_id)
    conv = agent_memory.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    on_project = conv.get("project_id") in project_ids or any(
        agent_memory.workspace_id_binds(conversation_id, pid) for pid in project_ids
    )
    if not on_project or not agent_memory.session_visible_to(conv, user_id, owner_id):
        raise HTTPException(404, "Conversation not found")
    return conv


@router.get("/v1/projects/{project_id}/conversations")
async def list_project_conversations(
    project_id: str,
    auth: dict = Depends(require_user),
):
    """Chat sessions on this project that belong to the caller, newest first.

    Anyone who can open the project can list. They see sessions they wrote.
    A session stored before owner tracking (NULL owner) is listed only for
    the project row owner. Another user's sessions are omitted.
    """
    from app.core import agent_memory

    project_ids, owner_id = _session_scope(project_id, auth["user_id"])
    rows = agent_memory.list_visible_sessions(
        project_ids, auth["user_id"], owner_id, limit=50,
    )
    return {"conversations": [_public_session(conv) for conv in rows]}


@router.patch("/v1/projects/{project_id}/conversations/{conversation_id}")
async def rename_project_conversation(
    project_id: str,
    conversation_id: str,
    body: RenameConversationRequest,
    auth: dict = Depends(require_user),
):
    """Rename a session the caller owns. The stored name is collapsed
    whitespace, at most 80 characters, and cannot contain HTML.
    """
    from app.core import agent_memory

    _session_for_caller(project_id, conversation_id, auth["user_id"])
    try:
        updated = agent_memory.rename_conversation(conversation_id, body.title)
    except agent_memory.SessionTitleError as exc:
        if exc.code == "HTML":
            detail = "Title cannot contain HTML"
        elif exc.code == "long":
            detail = "Title is too long"
        else:
            detail = "Title cannot be empty"
        raise HTTPException(400, detail) from exc
    if updated is None:
        raise HTTPException(404, "Conversation not found")
    await audit.arecord(
        "conversation.renamed",
        project_id=project_id,
        conversation_id=conversation_id,
        user_id=auth["user_id"],
    )
    return _public_session(updated)


# ── documents — store only, no analysis (Roadmap V2 · 0.3) ──────────────────

@router.post("/v1/projects/{project_id}/documents", status_code=201)
async def add_document(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    role: Optional[str] = Form(None),
    auth: dict = Depends(require_user),
):
    """Attach a document to a project.

    This STORES and CLASSIFIES the file — it runs no pipeline, no blocks, no
    analysis. Attaching a file is not the same as asking for analysis; run a
    block explicitly (via /v1/execute) when you actually want results.
    """
    # Anyone who can OPEN the project can add documents to it (shared pilot
    # projects + admin access) — see _owned_or_404's 2026-07-25 note.
    proj = _owned_or_404(
        project_id, auth["user_id"], read_only=True,
        role=auth.get("role", "user"), doc_limit=0,
    )

    original_name = (file.filename or "unknown").strip()
    if not original_name or original_name in (".", ".."):
        raise HTTPException(400, "Invalid filename")
    original_name = os.path.basename(original_name.replace("\\", "/"))
    _, ext = os.path.splitext(original_name.lower())
    from app.core import compressed

    if compressed.is_compressed(original_name, compressed.head_of(file.file)):
        raise HTTPException(415, compressed.COMPRESSED_UPLOAD_DETAIL)
    if ext not in ALLOWED_DOC_EXTENSIONS:
        raise HTTPException(
            415,
            f"File type '{ext}' is not ingestible: the knowledge base takes text "
            f"formats only (docs/INGEST_EXCLUSION_RULE.md). Drawings, photos, "
            f"video and CAD can still be attached to a chat for the session.",
        )

    max_size = _max_doc_upload_size()
    # Reject oversize uploads BEFORE materialising the file — this used to read
    # the whole document into memory (and copy it again to encrypt), so a large
    # BIM model OOM'd the shared worker and dropped every concurrent user, not
    # just the uploader. write_document_stream copies in 1 MB blocks and aborts
    # the moment the cap is exceeded, removing the partial file.
    file.file.seek(0, 2)
    declared_size = file.file.tell()
    file.file.seek(0)
    if declared_size > max_size:
        raise HTTPException(
            413,
            f"File too large ({declared_size} bytes). Max is {max_size} bytes.",
        )

    file_id = str(uuid.uuid4())[:8]
    stored_as = f"{file_id}_{original_name}"
    filepath = os.path.join(DATA_DIR, stored_as)
    # Persist the document — encrypted at rest iff DATA_ENCRYPTION_KEY is set
    # (opt-in; plaintext otherwise — see app/core/file_crypto.py). The recorded
    # `size` is the original plaintext size, not the (larger) ciphertext size.
    try:
        size = file_crypto.write_document_stream(
            filepath, file.file, max_bytes=max_size,
        )
    except file_crypto.UploadTooLarge as exc:
        # Reachable when the declared size understated the stream (chunked
        # transfer encoding sends no length).
        raise HTTPException(
            413, f"File too large. Max is {exc.limit} bytes.",
        ) from exc

    if role is not None and role not in store.VALID_ROLES:
        raise HTTPException(
            400, f"Invalid role '{role}'. Allowed: {sorted(store.VALID_ROLES)}"
        )

    # Layered RAG (Stage 4): stamp provenance so indexing routes this file to
    # the uploader's user_session layer, not the authoritative Main corpus. Only
    # this interactive upload endpoint sets it; seeds/Drive imports do not, so
    # they keep classifying as shared_domain / project_record. Inert until
    # RAG_LAYERED is on (classify reads it only then).
    doc = store.add_document(
        project_id, original_name, stored_as, filepath, size, role=role,
        metadata={"provenance": "user_upload", "uploader_id": auth["user_id"]},
    )
    await audit.arecord("document.added", project_id=project_id,
                 document_id=doc["id"], name=original_name, size=size, user_id=auth["user_id"])
    # Only the admin path adds to the project's knowledge base (owner ruling,
    # app/core/privileges.py). A user's upload is stored, not indexed.
    from app.core import privileges

    if privileges.caller_may_add_to_project_rag(auth.get("role")):
        # Index under the id the document was actually STORED under. For the
        # master-corpus alias that is the backing corpus, not the virtual id —
        # indexing under the alias would put the chunks in a different project
        # from the document row they describe.
        background_tasks.add_task(
            doc_index.maybe_eager_index, store.storage_project_id(project_id), doc["id"],
        )
    else:
        doc = store.update_document_metadata(doc["id"], {"indexing": {
            "status": "not_indexed",
            "detail": privileges.PROJECT_RAG_ADMIN_ONLY_DETAIL,
        }}) or doc

    # V2 inline safety + QA/QC detection for image uploads — runs PIL +
    # COCO YOLO + the fine-tuned safety_qaqc detector and surfaces a
    # compact summary the frontend can show to the user without a
    # separate /v1/execute round-trip. Failures are non-fatal: the upload
    # still succeeds, the result just won't carry safety_qaqc data.
    safety_summary = None
    if ext in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}:
        try:
            from app.dependencies import get_block_instance
            image_block = get_block_instance("image")
            if image_block is not None:
                analysis = await image_block.execute(
                    {"file_path": filepath},
                    {"mode": "safety_qaqc", "prompt": "construction safety + QA/QC scan"},
                )
                body = analysis.get("result", {}) or {}
                detections = body.get("safety_qaqc") or []
                if detections:
                    safety_summary = {
                        "count": len(detections),
                        "top": [
                            {
                                "class": d.get("class"),
                                "confidence": round(float(d.get("confidence") or 0.0), 3),
                            }
                            for d in detections[:8]
                        ],
                    }
        except Exception:
            # Detection is best-effort; never fail the upload over it.
            logger.warning(
                "swallowed %s in add_document() — continuing",
                "Exception", exc_info=True,
            )

    response: Dict[str, Any] = {
        "status": "stored",
        "message": (
            f"Added '{original_name}' — classified as {doc['doc_type']} "
            f"(role: {doc['doc_role']})."
            + (f" Detected {safety_summary['count']} safety/QA-QC issue(s)."
               if safety_summary else " No analysis was run; ask in chat to analyze it.")
        ),
        "document": doc,
        "readiness": store.compute_readiness(project_id),
    }
    if safety_summary:
        response["safety_qaqc"] = safety_summary
    return response


# ── gated progress tracking (Roadmap V2 · 0.2) ──────────────────────────────

@router.post("/v1/projects/{project_id}/progress")
async def project_progress(
    project_id: str, req: ProgressRequest, auth: dict = Depends(require_user)
):
    """Run the progress tracker for a project — but only if the project is ready.

    Until the project has a baseline schedule, daily and weekly reports, and
    Aconex connected, this returns a structured 'not_ready' response naming
    exactly what is missing — never fabricated all-zero numbers.
    """
    proj = _owned_or_404(project_id, auth["user_id"])

    readiness = proj["readiness"]
    if not readiness["ready"]:
        return {
            "status": "not_ready",
            "project_id": project_id,
            "message": (
                "Project is not ready for progress tracking. "
                "Load the missing items, then try again."
            ),
            "missing": readiness["missing"],
            "readiness": readiness,
        }

    container = block_instances.get("construction")
    if container is None:
        construction_cls = get_block("construction")
        if construction_cls is None:
            raise HTTPException(
                status_code=503,
                detail="construction kit not enabled — set CEREBRUM_DOMAIN_KITS",
            )
        container = _create_block_instance(construction_cls)
        block_instances["construction"] = container

    params = req.model_dump()
    if not params.get("reporting_period"):
        params.pop("reporting_period", None)

    tracker = await container.progress_tracker({}, params)
    return {
        "status": "success",
        "project_id": project_id,
        "readiness": readiness,
        "tracker": tracker,
    }


# ── project memory (Roadmap V2 · Epic 3) ────────────────────────────────────

class FactRequest(BaseModel):
    key: str
    value: str
    source_document: Optional[str] = None
    confidence: Optional[float] = None


@router.get("/v1/projects/{project_id}/memory")
async def get_memory(
    project_id: str, q: Optional[str] = None, auth: dict = Depends(require_user)
):
    """List the durable facts known about a project (optionally keyword-filtered)."""
    _owned_or_404(project_id, auth["user_id"])
    facts = store.search_facts(project_id, q) if q else store.list_facts(project_id)
    return {"project_id": project_id, "facts": facts, "count": len(facts)}


@router.post("/v1/projects/{project_id}/memory", status_code=201)
async def add_memory(
    project_id: str, req: FactRequest, auth: dict = Depends(require_user)
):
    """Add or correct a project fact (manual entry / correction)."""
    _owned_or_404(project_id, auth["user_id"])
    key = (req.key or "").strip()
    if not key:
        raise HTTPException(400, "Fact key is required")
    return store.set_fact(
        project_id, key, req.value,
        source_document=req.source_document, confidence=req.confidence,
    )


@router.delete("/v1/projects/{project_id}/memory/{key}")
async def delete_memory(
    project_id: str, key: str, auth: dict = Depends(require_user)
):
    """Forget a single project fact."""
    _owned_or_404(project_id, auth["user_id"])
    if not store.delete_fact(project_id, key):
        raise HTTPException(404, f"Fact '{key}' not found for project '{project_id}'")
    return {"status": "deleted", "project_id": project_id, "key": key}


# ── data governance (Roadmap V2 · Epic 6) ───────────────────────────────────

@router.patch("/v1/projects/{project_id}/documents/{document_id}")
async def update_document(
    project_id: str,
    document_id: str,
    req: UpdateDocumentRequest,
    auth: dict = Depends(require_user),
):
    """Set ``drive_file_id`` on an existing document. Owner or admin only.

    Does not re-upload or re-index. Preview hydrate uses the id on the
    next cite-click (SA ``files.get`` + media, then public download).
    """
    from app.core.gdrive_service import is_valid_drive_file_id

    is_admin = auth.get("role") == "admin"
    proj = store.get_project(
        project_id,
        user_id=None if is_admin else auth["user_id"],
        include_admin_approved=True,
    )
    if not proj:
        raise HTTPException(404, f"Project '{project_id}' not found")
    if proj.get("user_id") != auth["user_id"] and not is_admin:
        raise HTTPException(403, "Admin or project owner required")

    drive_file_id = (req.drive_file_id or "").strip()
    if not is_valid_drive_file_id(drive_file_id):
        raise HTTPException(422, "drive_file_id is not a valid Drive file id")

    resolved_id = store.storage_project_id(project_id)
    doc = store.get_document(document_id)
    if not doc or doc.get("project_id") not in {project_id, resolved_id}:
        raise HTTPException(
            404, f"Document '{document_id}' not found in project '{project_id}'"
        )

    updated = store.update_document_metadata(document_id, {
        "drive_file_id": drive_file_id,
        "drive_resolved_from": "patch",
    })
    if updated is None:
        raise HTTPException(
            404, f"Document '{document_id}' not found in project '{project_id}'"
        )
    await audit.arecord(
        "document.drive_file_id_set",
        project_id=resolved_id,
        document_id=document_id,
        user_id=auth["user_id"],
    )
    return updated


@router.delete("/v1/projects/{project_id}/documents/{document_id}")
async def delete_document(
    project_id: str, document_id: str, auth: dict = Depends(require_user)
):
    """Delete a single document — its record and its file on disk."""
    _owned_or_404(project_id, auth["user_id"])
    doc = store.get_document(document_id)
    if not doc or doc.get("project_id") != project_id:
        raise HTTPException(
            404, f"Document '{document_id}' not found in project '{project_id}'"
        )
    # Stored file + row + chunks + index entry, in one place (projects.purge_document).
    purged = store.purge_document(document_id)
    file_removed = purged["file_removed"]
    index_pruned = purged["index_pruned"]
    await audit.arecord("document.deleted", project_id=project_id,
                 document_id=document_id, file_removed=file_removed,
                 index_pruned=index_pruned, user_id=auth["user_id"])
    return {
        "status": "deleted",
        "document_id": document_id,
        "file_removed": file_removed,
        "index_pruned": index_pruned,
    }


@router.get("/v1/projects/{project_id}/audit")
async def project_audit(
    project_id: str, limit: int = 100, auth: dict = Depends(require_user)
):
    """The audit trail for a project — uploads, deletions, purges."""
    _owned_or_404(project_id, auth["user_id"])
    return {
        "project_id": project_id,
        "entries": audit.read_audit(limit, project_id),
    }


@router.get("/v1/governance")
async def governance_status(auth: dict = Depends(require_user)):
    """Where client data lives and how long it is kept (Roadmap V2 · Epic 6)."""
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    retention = int(os.getenv("DATA_RETENTION_DAYS", "0") or "0")
    return {
        "data_directory": os.getenv("DATA_DIR", "./data"),
        "retention_days": retention if retention > 0 else "indefinite",
        "audit_logging": True,
        "delete_on_request": True,
        "policy_document": "docs/DATA_GOVERNANCE.md",
    }


@router.post("/v1/governance/purge")
async def governance_purge(auth: dict = Depends(require_user)):
    """Purge documents older than DATA_RETENTION_DAYS (no-op if unset)."""
    if auth["role"] != "admin":
        raise HTTPException(403, "Admin only")
    days = int(os.getenv("DATA_RETENTION_DAYS", "0") or "0")
    if days <= 0:
        return {"status": "skipped",
                "reason": "DATA_RETENTION_DAYS is not set", "purged": 0}
    purged = store.purge_documents_older_than(days)
    files_removed = 0
    # Audit each purged document individually — the summary "governance.purge"
    # entry below records only a count, not the IDs. Without per-row entries
    # there's no forensic trail of which specific documents the bulk purge
    # removed (the "BOQ disappeared with no explanation" failure mode).
    for doc in purged:
        await audit.arecord(
            "document.deleted",
            project_id=doc.get("project_id"),
            document_id=doc.get("id"),
            name=doc.get("original_name"),
            reason="governance_purge",
            retention_days=days,
            user_id=auth["user_id"],
        )
        fp = doc.get("file_path")
        if fp and os.path.exists(fp):
            try:
                os.remove(fp)
                files_removed += 1
            except OSError:
                logger.warning(
                    "swallowed %s in governance_purge() — continuing",
                    "OSError", exc_info=True,
                )
    await audit.arecord("governance.purge",
                 documents_purged=len(purged), files_removed=files_removed, user_id=auth["user_id"])
    return {
        "status": "purged",
        "documents_purged": len(purged),
        "files_removed": files_removed,
    }
