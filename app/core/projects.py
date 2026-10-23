"""Project entity — groups documents and gates project-level analytics.

Roadmap V2 · Part 0.1 (Project entity) + 0.2 (readiness gate).

A Project is the backbone the platform was missing: documents are no longer
processed in isolation, and project-level analytics (progress tracking, earned
value) stay inert until the project is genuinely set up.

SQLAlchemy-backed via app.core.db — unified The Fork schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, delete, func, or_, select, text as sqla_text
from sqlalchemy.exc import IntegrityError

from app.core.db import SessionLocal, engine, get_database_url
from app.core.ingest_status import (
    EXTRACT_FAILED,
    EXTRACTOR_VERSION,
    INDEXED,
    INDEXED_STATUSES,
    NO_CHUNK_STATUSES,
    QUARANTINED,
    TOMBSTONED,
    UNSUPPORTED_TYPE,
    ZERO_CHUNK,
)
from app.core.models import Document, IngestionJob, Project, ProjectFact
from app.core.system_projects import GENERAL_KNOWLEDGE_PROJECT_DEFAULT, general_knowledge_env, primary_general_knowledge_project

import logging

logger = logging.getLogger(__name__)


class DuplicateContentError(ValueError):
    """Ingest refused: this sha256 already has a documents row.

    Hard-delete is forbidden. Pass ``reingest_of=<old_id>`` to add a new
    row and hide the old one (``retrieval_visible=false``).
    """

    def __init__(self, existing_id: str, content_sha256: str):
        self.existing_id = existing_id
        self.content_sha256 = content_sha256
        super().__init__(
            f"DUPLICATE_SHA existing_id={existing_id}"
        )



# ── pilot master-corpus alias ───────────────────────────────────────────────
# Per-project Drive approval/indexing is not pilot-ready. Expose the existing
# full-drive corpus (currently stored under project_id "projects_folder") as a
# single admin-visible pilot project without duplicating chunks or re-importing
# Drive. This is a temporary alias; proper per-project trees come post-pilot.
MASTER_CORPUS_PROJECT_ID = os.getenv("MASTER_CORPUS_PROJECT_ID", "master_corpus")
MASTER_CORPUS_SOURCE_PROJECT_ID = os.getenv(
    "MASTER_CORPUS_SOURCE_PROJECT_ID", "projects_folder"
)
MASTER_CORPUS_NAME = os.getenv("MASTER_CORPUS_NAME", "Master Corpus")

# Boot-seeded general-knowledge origin. Seeded general-knowledge rows were
# created with this before the shared-platform grant existed.
_SYSTEM_SEED_ORIGIN = "system_seed"
_ADMIN_APPROVED_ORIGIN = "admin_drive_approved"


#: Default for ``RAG_GENERAL_KNOWLEDGE_PROJECTS`` when the env var is unset.
#: The retriever reads it from here rather than carrying its own copy.
DEFAULT_GENERAL_KNOWLEDGE_PROJECTS = GENERAL_KNOWLEDGE_PROJECT_DEFAULT


def general_knowledge_project_ids() -> frozenset[str]:
    """Configured always-on general-knowledge project ids.

    Same ``RAG_GENERAL_KNOWLEDGE_PROJECTS`` env that ``knowledge_seed``,
    layered RAG, and the retriever use. Default is the product's
    general-knowledge project (``app.core.system_projects``). Empty /
    whitespace entries are dropped.
    """
    raw = general_knowledge_env()
    return frozenset(p.strip() for p in raw.split(",") if p.strip())


def _is_shared_platform_grant(project: Any) -> bool:
    """Approved platform-shared row for ``include_admin_approved`` readers.

    Drive-approved origin, boot-seeded GK origin, or an id listed in
    ``RAG_GENERAL_KNOWLEDGE_PROJECTS``. Fail-closed on ``is_approved=False``.
    """
    if not bool(getattr(project, "is_approved", True)):
        return False
    origin = getattr(project, "origin", "user_create") or "user_create"
    if origin in (_ADMIN_APPROVED_ORIGIN, _SYSTEM_SEED_ORIGIN):
        return True
    pid = getattr(project, "id", None)
    return bool(pid) and pid in general_knowledge_project_ids()


def _master_corpus_source(project_id: Optional[str]) -> Optional[str]:
    """Return the backing project_id for a master-corpus alias, if any."""
    if project_id == MASTER_CORPUS_PROJECT_ID:
        return MASTER_CORPUS_SOURCE_PROJECT_ID
    return None


def ui_project_id(project_id: Optional[str]) -> Optional[str]:
    """The project id export URLs and owner gates must use.

    Chat remaps ``MASTER_CORPUS_PROJECT_ID`` (``master_corpus``) to
    ``MASTER_CORPUS_SOURCE_PROJECT_ID`` (the backing source project) so RAG
    hits the backing corpus. That remapped id is not a user-visible
    project: ``get_project(<source id>)`` 404s for a signed-in user
    on Master Corpus. Reverse-map the source back to the
    alias. Any other id is returned unchanged.
    """
    if project_id and project_id == MASTER_CORPUS_SOURCE_PROJECT_ID:
        return MASTER_CORPUS_PROJECT_ID
    return project_id


# ── document roles that feed the readiness gate ─────────────────────────────
ROLE_BASELINE = "baseline_schedule"
ROLE_DAILY = "daily_report"
ROLE_WEEKLY = "weekly_report"
ROLE_OTHER = "other"
VALID_ROLES = {ROLE_BASELINE, ROLE_DAILY, ROLE_WEEKLY, ROLE_OTHER}

_lock = threading.Lock()
_initialized = False
# Tracks WHICH database the schema was created in. `_initialized` alone is a
# global boolean, but the database is per-DATA_DIR / per-DATABASE_URL: after
# init runs against one database, `_ensure_db` short-circuits for every other
# one, so a later switch silently gets NO schema ("no such table: projects").
# Sibling stores (doc_index, hydration_store, rag.budget) already track the
# URL; these did not. Found via an order-dependent test failure, but the bug
# is real wherever the database can change after first use.
_initialized_for_url: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_sqlite_parent_dir() -> None:
    url = get_database_url()
    if url.startswith("sqlite:///"):
        parent = os.path.dirname(url[len("sqlite:///") :])
        if parent:
            os.makedirs(parent, exist_ok=True)


def _profile_of(project: Project) -> Optional[Dict[str, Any]]:
    raw = getattr(project, "profile", None)
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        logger.warning("project %s has a profile that is not JSON; ignoring it", project.id)
        return None
    return value if isinstance(value, dict) else None


def _project_as_dict(project: Project) -> Dict[str, Any]:
    return {
        "id": project.id,
        "name": project.name,
        "client": project.client,
        "location": getattr(project, "location", None),
        "profile": _profile_of(project),
        "status": project.status,
        "aconex_connected": bool(project.aconex_connected),
        "user_id": project.user_id,
        "created_at": project.created_at,
        "is_approved": bool(getattr(project, "is_approved", True)),
        "origin": getattr(project, "origin", "user_create") or "user_create",
        "is_master_corpus": False,
    }


# Drive file id aliases actually written by P1B / rag_render / Drive OAuth /
# audit. Live rows also use camelCase and a JSON *string* in the JSONB column.
# The platform keeps no copy of an original file (docs/INGEST_EXCLUSION_RULE.md):
# Google Drive is the source of truth, so the Drive id is the only pointer.
_DRIVE_ID_ALIASES = (
    "drive_file_id", "driveFileId", "drive_id", "driveId",
)


def coerce_document_metadata(meta: Any) -> Dict[str, Any]:
    """Return metadata as a dict. Live JSONB sometimes stores a JSON string."""
    if meta is None:
        return {}
    if isinstance(meta, dict):
        return meta
    if isinstance(meta, str):
        try:
            parsed = json.loads(meta)
        except json.JSONDecodeError:
            logger.warning("document metadata is not valid JSON", exc_info=True)
            return {}
        if isinstance(parsed, dict):
            return parsed
        # Double-encoded JSONB string (audit rows, CAST-as-text inserts).
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except json.JSONDecodeError:
                logger.warning(
                    "document metadata is double-encoded but not valid JSON",
                    exc_info=True,
                )
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}
    return {}


def _first_nonempty_str(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_document_source_pointers(doc: Dict[str, Any]) -> Dict[str, str]:
    """Resolve where a document's original lives: its Google Drive file id.

    Reads the key P1B Drive ingest writes (``drive_file_id``) and the
    rag_render / audit aliases, so a RAG-citable row without the exact name
    still resolves.
    """
    meta = coerce_document_metadata(doc.get("metadata"))
    drive_id = _first_nonempty_str(
        doc.get("drive_file_id"),
        *[meta.get(alias) for alias in _DRIVE_ID_ALIASES],
    )
    return {"drive_file_id": drive_id}


def _path_looks_present(path: str) -> bool:
    return bool(path) and os.path.isfile(path) and os.path.getsize(path) > 0


# A zero chunk count is not an extraction failure. These two stored statuses
# are the ones that mean the extract itself produced nothing.
_EXTRACT_FAILURE_DETAIL = (
    "Extraction returned no usable text; the assistant cannot read this document"
)
_STORED_NOT_IN_INDEX = (
    "This file is stored and has not been added to the project knowledge base."
)
_STATUS_DETAIL = {
    ZERO_CHUNK: _EXTRACT_FAILURE_DETAIL,
    EXTRACT_FAILED: _EXTRACT_FAILURE_DETAIL,
    UNSUPPORTED_TYPE: "This file type is not added to the project knowledge base.",
    TOMBSTONED: "This file is no longer in the project.",
    QUARANTINED: "This file is quarantined and is not in the project knowledge base.",
}


def _indexing_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    recorded = meta.get("indexing") if isinstance(meta.get("indexing"), dict) else None
    if recorded:
        return recorded
    direct = doc.get("indexing_status")
    return direct if isinstance(direct, dict) else {}


def panel_status_for(doc: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """The Documents-panel note for a file project search does not hold.

    The detail is the reason already stored on the row (the index skip, or
    the ingest status and its reason). A missing chunk count never becomes
    an extraction failure on its own.
    """
    if int(doc.get("chunk_count") or 0) > 0 or doc.get("searchable"):
        return None
    indexing = _indexing_record(doc)
    ingest = str(doc.get("ingest_status") or "").strip()
    reason = str(doc.get("ingest_status_reason") or "").strip()
    status = str(indexing.get("status") or "").strip()
    detail = str(indexing.get("detail") or "").strip()
    if status and status not in INDEXED_STATUSES:
        return {
            "label": "Not indexed",
            "detail": detail or reason or _STATUS_DETAIL.get(ingest, _STORED_NOT_IN_INDEX),
            "status": status,
        }
    if ingest in NO_CHUNK_STATUSES:
        return {
            "label": "Not indexed",
            "detail": reason or _STATUS_DETAIL.get(ingest, _STORED_NOT_IN_INDEX),
            "status": ingest,
        }
    return {
        "label": "Not indexed",
        "detail": reason or _STORED_NOT_IN_INDEX,
        "status": ingest or "not_indexed",
    }


def attach_panel_status(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Set or clear ``panel_status`` from the document's current fields."""
    note = panel_status_for(doc)
    if note:
        doc["panel_status"] = note
    else:
        doc.pop("panel_status", None)
    return doc


def _document_as_dict(document: Document) -> Dict[str, Any]:
    out = {
        "id": document.id,
        "project_id": document.project_id,
        "original_name": document.original_name,
        "stored_as": document.stored_as,
        "file_path": document.file_path,
        "doc_type": document.doc_type,
        "doc_role": document.doc_role,
        "size": document.size,
        "uploaded_at": document.uploaded_at,
        "content_sha256": document.content_sha256,
    }
    meta = coerce_document_metadata(getattr(document, "metadata_", None))
    if getattr(document, "metadata_", None) is not None:
        out["metadata"] = meta
    out["ingest_status"] = getattr(document, "ingest_status", None)
    out["ingest_status_reason"] = getattr(document, "ingest_status_reason", None)
    out["chunk_count"] = int(getattr(document, "chunk_count", 0) or 0)
    out["superseded_by"] = getattr(document, "superseded_by", None)
    out["retrieval_visible"] = bool(
        True if getattr(document, "retrieval_visible", None) is None
        else document.retrieval_visible
    )
    out["extractor_version"] = getattr(document, "extractor_version", None)
    out["drive_md5"] = getattr(document, "drive_md5", None)
    pointers = extract_document_source_pointers(out)
    local = _path_looks_present(out.get("file_path") or "")
    out["has_remote_source"] = bool(pointers["drive_file_id"])
    out["has_file"] = local or out["has_remote_source"]
    attach_panel_status(out)
    return out


def _fact_as_dict(fact: ProjectFact) -> Dict[str, Any]:
    return {
        "id": fact.id,
        "project_id": fact.project_id,
        "key": fact.key,
        "value": fact.value,
        "source_document": fact.source_document,
        "confidence": fact.confidence,
        "updated_at": fact.updated_at,
    }


def init_db() -> None:
    """Create the schema if absent. Idempotent — safe to call on every startup.

    Also runs lightweight in-place column patches for legacy SQLite
    databases that were created before recent migrations landed. Prod
    Postgres applies the same changes via Alembic; this branch keeps
    local dev / fresh test environments self-healing without requiring
    an explicit `alembic upgrade head` run.
    """
    global _initialized, _initialized_for_url
    if _initialized and _initialized_for_url == get_database_url():
        return
    with _lock:
        # The URL this run initialises: read once, so a DATA_DIR change mid-way
        # cannot mark another database initialised.
        url = get_database_url()
        if _initialized and _initialized_for_url == url:
            return
        from app.core.users import init_db as init_users_db

        init_users_db()
        _ensure_sqlite_parent_dir()
        Project.__table__.create(bind=engine, checkfirst=True)
        Document.__table__.create(bind=engine, checkfirst=True)
        ProjectFact.__table__.create(bind=engine, checkfirst=True)
        IngestionJob.__table__.create(bind=engine, checkfirst=True)
        _patch_legacy_columns()
        _initialized = True
        _initialized_for_url = url


def _patch_legacy_columns() -> None:
    """Add columns to legacy tables when they're missing.

    SQLite only — Postgres deployments are managed by Alembic. This
    function exists because checkfirst=True on Table.create() does NOT
    add new columns to existing tables; we have to ALTER manually for
    dev environments + tests.

    Currently handles:
      * projects.is_approved (Alembic 0004)
      * projects.origin       (Alembic 0005)
      * documents.metadata    (Alembic 0009)
      * documents.superseded_by / retrieval_visible / extractor_version (0017)
    """
    url = get_database_url()
    if not url.startswith("sqlite"):
        return  # Postgres is migration-managed.
    try:
        with engine.connect() as conn:
            cols = {row[1] for row in conn.execute(
                sqla_text("PRAGMA table_info(projects)")
            )}
            if "is_approved" not in cols:
                conn.execute(sqla_text(
                    "ALTER TABLE projects "
                    "ADD COLUMN is_approved BOOLEAN NOT NULL DEFAULT 1"
                ))
                conn.commit()
            if "origin" not in cols:
                conn.execute(sqla_text(
                    "ALTER TABLE projects "
                    "ADD COLUMN origin TEXT NOT NULL DEFAULT 'user_create'"
                ))
                conn.commit()
            if "location" not in cols:  # 0010 (F4 — daily_site_report weather)
                conn.execute(sqla_text(
                    "ALTER TABLE projects ADD COLUMN location TEXT"
                ))
                conn.commit()
            if "profile" not in cols:  # 0028 (driver mode: the project profile)
                conn.execute(sqla_text("ALTER TABLE projects ADD COLUMN profile TEXT"))
            if "identity" not in cols:  # 0031 (the project's identity set)
                conn.execute(sqla_text("ALTER TABLE projects ADD COLUMN identity TEXT"))
                conn.commit()

            doc_cols = {row[1] for row in conn.execute(
                sqla_text("PRAGMA table_info(documents)")
            )}
            if "metadata" not in doc_cols:
                conn.execute(sqla_text(
                    "ALTER TABLE documents ADD COLUMN metadata TEXT"
                ))
                conn.commit()
            if "superseded_by" not in doc_cols:
                conn.execute(sqla_text(
                    "ALTER TABLE documents ADD COLUMN superseded_by TEXT"
                ))
                conn.commit()
            if "retrieval_visible" not in doc_cols:
                conn.execute(sqla_text(
                    "ALTER TABLE documents ADD COLUMN retrieval_visible "
                    "BOOLEAN NOT NULL DEFAULT 1"
                ))
                conn.commit()
            if "extractor_version" not in doc_cols:
                conn.execute(sqla_text(
                    "ALTER TABLE documents ADD COLUMN extractor_version TEXT"
                ))
                conn.commit()
            if "drive_md5" not in doc_cols:
                conn.execute(sqla_text(
                    "ALTER TABLE documents ADD COLUMN drive_md5 TEXT"
                ))
                conn.commit()
    except Exception:
        # Don't crash boot on a dev-environment patch failure — the
        # next call to a feature that needs the column will surface
        # the real error with a clearer stack.
        import logging
        logging.getLogger(__name__).warning(
            "legacy column patch skipped", exc_info=True,
        )


def _ensure_db() -> None:
    if not _initialized or _initialized_for_url != get_database_url():
        init_db()


# ── classification ──────────────────────────────────────────────────────────

def classify_doc_type(filename: str) -> str:
    """Coarse document-type guess from the filename (display only)."""
    n = (filename or "").lower()
    _, ext = os.path.splitext(n)
    if ext in {".xer", ".mpp"} or "primavera" in n or "p6" in n:
        return "schedule"
    if ext == ".ifc":
        return "bim"
    if ext in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        return "photo"
    if "boq" in n or "bill of quant" in n:
        return "boq"
    if "contract" in n or "agreement" in n:
        return "contract"
    if "spec" in n:
        return "specification"
    if "drawing" in n or ext == ".dwg":
        return "drawing"
    if "schedule" in n or "programme" in n or "program" in n:
        return "schedule"
    return "document"


def classify_doc_role(filename: str) -> str:
    """Map a filename to a readiness role. Conservative — defaults to 'other'."""
    n = (filename or "").lower()
    if "baseline" in n:
        return ROLE_BASELINE
    if "daily" in n:
        return ROLE_DAILY
    if "weekly" in n:
        return ROLE_WEEKLY
    return ROLE_OTHER


# ── projects ────────────────────────────────────────────────────────────────

def _stable_project_id_for_name(name: str) -> str:
    """Deterministic id so parallel ingest workers converge without UNIQUE(name)."""
    digest = hashlib.sha256(name.strip().lower().encode("utf-8")).hexdigest()[:16]
    return f"p_{digest}"


def _find_active_project_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return the oldest active project with this exact name, if any."""
    _ensure_db()
    with SessionLocal() as session:
        row = session.execute(
            select(Project)
            .where(Project.name == name)
            .where(Project.status != "archived")
            .order_by(Project.created_at.asc())
            .limit(1)
        ).scalar_one_or_none()
    if not row:
        return None
    return _project_as_dict(row)


def _identity_changed(project_id: Optional[str]) -> None:
    """Mark the project's stored identity set (app.core.project_identity) stale
    after its record, profile or documents changed: the next turn derives it
    once and stores it. One small UPDATE in this thread -- no background
    thread, which a forked ingest child could inherit mid-lock (CI: a child
    process hung 120 s)."""
    if not project_id:
        return
    try:
        with SessionLocal() as session:
            row = session.get(Project, project_id)
            if row is None:
                return
            try:
                stored = json.loads(row.identity) if row.identity else {}
            except (TypeError, ValueError):
                stored = {}
            added = stored.get("added") if isinstance(stored, dict) else None
            # Admin-added members survive; the derived part is redone on read.
            row.identity = json.dumps({"stale": True, "added": added or []})
            session.commit()
    except Exception:  # noqa: BLE001 -- logged; the next turn still derives a set
        logger.warning("project %s: identity set not marked stale", project_id, exc_info=True)


def create_project(
    name: str,
    client: Optional[str] = None,
    user_id: str = "system",
    *,
    location: Optional[str] = None,
    is_approved: bool = True,
    project_id: Optional[str] = None,
    origin: str = "user_create",
) -> Dict[str, Any]:
    """Create a project row.

    PR A: ``is_approved`` defaults to True for both user-created and
    admin-created flows. The approve-from-Drive endpoint also passes
    True explicitly. A future "detected but pending" code path can
    pass False to create a candidate row that's hidden from the user
    rail until an admin flips it.

    PR B: ``origin`` records how the row was created. The admin page
    filters on origin='admin_drive_approved' so user-created rows
    don't appear in the admin's approved list. Allowed values:
    'user_create' (default), 'admin_drive_approved', 'user_drive_import'.

    ``project_id`` lets a caller pre-supply the id (used by the
    approve-from-Drive flow so the slug is human-friendly instead of
    a random hex).
    """
    _ensure_db()
    pid = project_id or str(uuid.uuid4())[:8]
    with _lock:
        with SessionLocal() as session:
            session.add(
                Project(
                    id=pid,
                    name=name,
                    client=client,
                    location=(location or "").strip() or None,
                    status="active",
                    aconex_connected=False,
                    user_id=user_id,
                    is_approved=is_approved,
                    origin=origin,
                    created_at=_now(),
                )
            )
            session.commit()
    _identity_changed(pid)
    return get_project(pid)


def set_project_location(project_id: str, location: Optional[str]) -> Optional[Dict[str, Any]]:
    """Set (or clear, with None) a project's confirmed site location. Returns the
    updated project dict, or None if the project does not exist."""
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if project is None:
                return None
            project.location = (location or "").strip() or None
            session.commit()
    _identity_changed(project_id)
    return get_project(project_id)


def set_project_profile(project_id: str, profile: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Set (or clear, with None or {}) a project's profile. Returns the updated
    project dict, or None if the project does not exist."""
    _ensure_db()
    clean = {str(k): v for k, v in (profile or {}).items() if v not in (None, "")}
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if project is None:
                return None
            project.profile = json.dumps(clean) if clean else None
            session.commit()
    _identity_changed(project_id)
    return get_project(project_id)


def get_or_create_project(
    name: str,
    client: Optional[str] = None,
    user_id: str = "system",
    *,
    is_approved: bool = True,
    project_id: Optional[str] = None,
    origin: str = "user_create",
) -> Tuple[Dict[str, Any], bool]:
    """Ensure exactly one project row for this ingest target (id-primary).

    ``projects.name`` is not UNIQUE, so parallel shard workers that each call
    ``create_project(folder_name)`` would mint separate random ids and split
    the corpus. This helper:

    1. Uses a stable ``project_id`` (caller-supplied, else derived from name).
    2. Returns an existing row by id, else by name (oldest active — resume).
    3. Creates with that primary key; on ``IntegrityError`` re-fetches.

    Returns ``(project_dict, created)``.
    """
    _ensure_db()
    pid = (project_id or "").strip() or _stable_project_id_for_name(name)

    existing = get_project(pid)
    if existing:
        return existing, False

    by_name = _find_active_project_by_name(name)
    if by_name:
        # Legacy / pre-shard rows: keep ingesting into the oldest name match
        # rather than creating a second project under the stable id.
        return by_name, False

    try:
        created = create_project(
            name,
            client=client,
            user_id=user_id,
            is_approved=is_approved,
            project_id=pid,
            origin=origin,
        )
        return created, True
    except IntegrityError:
        # Another worker won the insert race on the same primary key.
        raced = get_project(pid) or _find_active_project_by_name(name)
        if raced is None:
            raise
        return raced, False


def list_projects(
    user_id: Optional[str] = None,
    *,
    include_admin_approved: bool = False,
    include_hidden: bool = False,
) -> List[Dict[str, Any]]:
    """List projects.

    PR D — visibility model:
      * When ``user_id`` is None: every row (admin / internal use only).
      * When ``user_id`` is set + ``include_admin_approved=False``: rows
        owned by the caller only (legacy behaviour).
      * When ``user_id`` is set + ``include_admin_approved=True``: rows
        owned by the caller PLUS approved platform-shared rows:
        origin='admin_drive_approved', origin='system_seed' (boot-seeded
        GK), or an id listed in ``RAG_GENERAL_KNOWLEDGE_PROJECTS``.
        ``is_approved=False`` rows are hidden from non-owners regardless
        of origin — defensive against future "detected but not yet
        approved" rows that could otherwise leak.

    Pilot: if the master-corpus source project is visible to the caller,
    the virtual ``master_corpus`` alias is appended to the list.
    """
    from sqlalchemy import or_, and_

    _ensure_db()
    with SessionLocal() as session:
        # Soft-archived projects are hidden from every listing (the "Delete"
        # action archives rather than deletes — see archive_project).
        stmt = (
            select(Project)
            .where(Project.status != "archived")
            .order_by(Project.created_at.desc())
        )
        # Layered RAG (Stage 5): hidden rows (RAG corpora / GK / eval) stay
        # retrievable but drop out of the sidebar. include_hidden=True is for
        # internal enumeration / admin tooling only.
        if not include_hidden:
            stmt = stmt.where(Project.hidden_from_sidebar.is_(False))
        if user_id is not None:
            if include_admin_approved:
                shared = [
                    Project.origin.in_(
                        (_ADMIN_APPROVED_ORIGIN, _SYSTEM_SEED_ORIGIN)
                    ),
                ]
                gk_ids = general_knowledge_project_ids()
                if gk_ids:
                    shared.append(Project.id.in_(gk_ids))
                stmt = stmt.where(
                    or_(
                        Project.user_id == user_id,
                        and_(
                            Project.is_approved.is_(True),
                            or_(*shared),
                        ),
                    )
                )
            else:
                stmt = stmt.where(Project.user_id == user_id)
        rows = session.scalars(stmt).all()
    out = []
    for project in rows:
        p = _project_as_dict(project)
        p["readiness"] = compute_readiness(p["id"])
        p["document_count"] = len(list_documents(p["id"]))
        out.append(p)

    # Expose the pilot master-corpus alias when the backing corpus is visible.
    master = get_project(
        MASTER_CORPUS_PROJECT_ID,
        user_id=user_id,
        include_admin_approved=include_admin_approved,
    )
    if master is not None:
        master["is_master_corpus"] = True
        master["document_count"] = len(list_documents(MASTER_CORPUS_SOURCE_PROJECT_ID))
        # Pilot: the master corpus is the canonical starting point, so it
        # always appears first regardless of creation date.
        out.insert(0, master)

    return out


def _accessible_project(project_id: str, user_id: Optional[str],
                        include_admin_approved: bool) -> Optional[tuple]:
    """``(Project row, source id)`` when the caller may read the project, else None.

    The access rule of ``get_project``, without loading its documents or
    readiness -- an access check needs the row only.
    """
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    with SessionLocal() as session:
        project = session.get(Project, source_id)
    if not project:
        return None
    # Soft-archived projects are treated as gone everywhere they're read
    # (UI detail, ownership gates, retrieval scoping) — but the row + its RAG
    # chunks stay in the DB. See archive_project.
    if getattr(project, "status", "active") == "archived":
        return None
    is_alias = source_id != project_id
    if user_id is not None and project.user_id != user_id:
        if is_alias:
            # The master-corpus alias is treated as an admin-approved platform
            # project: visible to any authenticated user when the caller asks
            # for platform projects, otherwise owner-only.
            allowed = include_admin_approved and bool(
                getattr(project, "is_approved", True)
            )
        else:
            allowed = include_admin_approved and _is_shared_platform_grant(project)
        if not allowed:
            return None
    return project, source_id


def can_access_project(project_id: str, user_id: Optional[str] = None,
                       include_admin_approved: bool = False) -> bool:
    """Whether ``get_project`` would return the project, at the cost of one row.

    Live 2026-10-07: chat's access gate called ``get_project``, which lists
    every document and checks each one's file on disk (6,585 on the master
    corpus) -- 1-2 s on the event loop per message, just to say yes.
    """
    return _accessible_project(project_id, user_id, include_admin_approved) is not None


def get_project(
    project_id: str,
    user_id: Optional[str] = None,
    *,
    include_admin_approved: bool = False,
    doc_limit: Optional[int] = None,
    doc_offset: int = 0,
) -> Optional[Dict[str, Any]]:
    """Load a project the caller can access.

    PR D — non-owners may also read approved platform-shared projects
    when ``include_admin_approved=True`` (Drive-approved origin,
    boot-seeded ``system_seed``, or a configured GK id).
    ``is_approved=False`` rows stay owner-only regardless of origin
    (defensive — admins shouldn't leak detected-but-pending candidates
    to users). The physical master-corpus source id stays owner-only
    here; ``get_project_accessible`` remaps that path.

    Pilot: a virtual master-corpus project (default ``master_corpus``)
    is backed by the existing full-drive corpus (default ``projects_folder``).
    It appears as a first-class project without duplicating chunks.
    """
    found = _accessible_project(project_id, user_id, include_admin_approved)
    if found is None:
        return None
    project, source_id = found
    proj = _project_as_dict(project)
    # Virtual master-corpus project: expose alias id/name but keep the source
    # corpus behind it.
    if source_id != project_id:
        proj["id"] = project_id
        proj["name"] = MASTER_CORPUS_NAME
        proj["origin"] = "admin_drive_approved"
        proj["is_approved"] = True
        proj["is_master_corpus"] = True
    # Paginate documents: a large corpus (2713-doc master) must not serialize
    # every row on every load. doc_limit=None preserves the legacy "all docs"
    # behaviour for the many internal callers; the detail endpoint passes a
    # first-page limit. document_count is a cheap COUNT, NOT len() of the page.
    proj["documents"] = list_documents(
        source_id,
        limit=doc_limit,
        offset=doc_offset,
        newest_first=doc_limit is not None,
    )
    proj["document_count"] = count_documents(source_id)
    proj["readiness"] = compute_readiness(source_id)
    return proj


def project_owner(project_id: str) -> Optional[str]:
    """Return the user_id that owns the project, or None if the project doesn't exist."""
    _ensure_db()
    with SessionLocal() as session:
        project = session.get(Project, project_id)
    return project.user_id if project else None


def _is_platform_project(project: Dict[str, Any]) -> bool:
    """A platform/corpus project an admin may read cross-tenant on the chat/RAG
    DATA path: system/seed-owned, an admin-approved shared project, or the
    master-corpus alias. A real end-user's private ``user_create`` project is
    NOT platform — it must never be readable by another account here.
    """
    from app.core.users import SYSTEM_USER_ID
    return (
        project.get("user_id") == SYSTEM_USER_ID
        or (project.get("origin") or "user_create") == "admin_drive_approved"
        or bool(project.get("is_master_corpus"))
    )


def get_project_accessible(project_id: str, user_id: Optional[str] = None):
    """Open-access resolution for READ/chat surfaces (2026-07-26).

    The chat tenant gates used owner-only get_project(pid, user_id=...),
    which silently DROPPED the project for admins and for admin-approved
    shared platform projects -- the caller could OPEN the project in the UI
    (#267 read rule) and upload to it (#277) but chat lost all RAG context
    on it (zero injected sources, no-op search tool). Third instance of the
    same asymmetry class; this helper is the single rule for all of them:
    owner -> admin-approved shared -> master-corpus source id (same ACL
    as the alias) -> admin on PLATFORM projects only.
    Returns the project dict or None; archived projects stay invisible on
    every path.

    SECURITY (legacy-admin tenancy fix): ``require_user`` maps a legacy
    API key to the singleton ``system`` user for *identity* only. Authority
    (``role``) comes from the key record — only ``CEREBRUM_MASTER_KEY`` is
    minted with ``role="admin"``. A plain API key must not inherit admin
    from the system user row. The admin fallthrough below is therefore
    reachable by genuine admins and the master key, not every legacy key.
    The admin cross-tenant read is therefore scoped to PLATFORM projects
    (``_is_platform_project``) — system/seed-owned corpora and admin-approved
    shared projects — and NEVER a real end-user's private project. Genuine
    admin cross-tenant operations on arbitrary projects go through the
    explicit, audited ``/v1/admin/*`` endpoints (which call ``get_project``
    directly), not this chat data-path helper. Owner reads and the
    admin-approved-shared grant are unchanged (handled by the scoped
    ``get_project`` call above).
    """
    if not user_id:
        # user_id=None means UNSCOPED in get_project — an anonymous caller
        # must never resolve a project through this helper (fail closed).
        return None
    proj = get_project(project_id, user_id=user_id,
                       include_admin_approved=True, doc_limit=0)
    if proj is not None:
        return proj
    # S13 / WATCH-1: the backing corpus id is the same ACL as the
    # master-corpus alias. get_project(source) stays owner-only so the
    # physical id stays hidden from the project picker;
    # this data-path helper must still let a master-corpus member
    # search/chat against the source id. A 404 here was being read as
    # "auth failed" or "retrieval empty".
    alias_id = ui_project_id(project_id)
    if alias_id and alias_id != project_id:
        aliased = get_project(
            alias_id, user_id=user_id,
            include_admin_approved=True, doc_limit=0,
        )
        if aliased is not None:
            source = get_project(project_id, doc_limit=0)
            return source if source is not None else aliased
    try:
        from app.core import users as users_store
        u = users_store.get_user_by_id(user_id)
        if u and (u.get("role") or "").lower() == "admin":
            # Admin fallthrough — but ONLY for platform/corpus projects, so a
            # legacy-key SYSTEM admin (or any promoted admin) cannot read a
            # real user's private project through the chat gate.
            candidate = get_project(project_id, doc_limit=0)
            if candidate is not None and _is_platform_project(candidate):
                return candidate
    except Exception:  # noqa: BLE001 -- fail closed on lookup errors
        logger.warning(
            "admin platform-project fallthrough failed for project_id=%r",
            project_id,
            exc_info=True,
        )
        return None
    return None


def archive_project(project_id: str) -> bool:
    """Soft-delete: hide the project from listings, detail, ownership gates and
    retrieval WITHOUT removing the row. `chunks.project_id` is ON DELETE
    CASCADE, so keeping the row is what preserves the RAG — the operator
    principle 'delete the UI, never the RAG; build on it only'. Reversible:
    set status back to 'active' to restore. Returns False if not found."""
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return False
            project.status = "archived"
            session.commit()
            return True


def set_hidden_from_sidebar(project_id: str, hidden: bool = True) -> bool:
    """Hide (or unhide) a project from the sidebar WITHOUT archiving it — the
    row and its RAG chunks are untouched and stay retrievable. Returns False if
    the project doesn't exist. Reversible: pass hidden=False to restore."""
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return False
            project.hidden_from_sidebar = bool(hidden)
            session.commit()
            return True


def approve_project(project_id: str) -> bool:
    """Make an EXISTING project admin-visible: set ``is_approved=True`` and
    ``origin='admin_drive_approved'`` so the sidebar (which filters on that
    origin) lists it for all users — the same visibility a Drive-approved
    project gets, without re-importing. Idempotent. Returns False if not found.
    """
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return False
            project.is_approved = True
            project.origin = "admin_drive_approved"
            session.commit()
            return True


def delete_project(project_id: str) -> bool:
    """HARD delete — removes the row and (via ON DELETE CASCADE) its documents
    AND RAG chunks. Reserved for genuine admin cleanup; the user-facing Delete
    action uses archive_project so the RAG is never destroyed."""
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return False
            session.delete(project)
            session.commit()
            return True


# Projects that may NEVER be hard-purged regardless of status — the master
# corpus, its backing source, and the shared general-knowledge base.
_PURGE_PROTECTED_IDS = {
    MASTER_CORPUS_PROJECT_ID,
    MASTER_CORPUS_SOURCE_PROJECT_ID,
    primary_general_knowledge_project(),
}


def purge_archived_project(project_id: str) -> str:
    """PERMANENTLY remove an ARCHIVED project — its RAG chunks AND its row.

    SAFETY (never-delete-RAG rule still holds for everything live):
      * refuses protected master/backing/general-knowledge ids outright;
      * refuses any project whose status is not 'archived' — so active
        projects and the master corpus can never be purged through here.
    Returns one of: 'protected' | 'not_found' | 'not_archived' | 'purged'.
    """
    if project_id in _PURGE_PROTECTED_IDS:
        return "protected"
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return "not_found"
            if getattr(project, "status", "active") != "archived":
                return "not_archived"
    # Clear the vector store first, then hard-delete the row (cascade clears
    # the chunks table). Both, belt-and-suspenders.
    try:
        from app.core import doc_index
        doc_index.purge_project_index(project_id)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger(__name__).warning(
            "purge_archived_project: index purge failed for %s: %s", project_id, exc)
    delete_project(project_id)
    return "purged"


def set_aconex(project_id: str, connected: bool) -> bool:
    """Mark whether this project has a CDE feed (live OAuth or managed outside)."""
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            project = session.get(Project, project_id)
            if not project:
                return False
            project.aconex_connected = connected
            session.commit()
            return True


# ── documents ───────────────────────────────────────────────────────────────

def storage_project_id(project_id: str) -> str:
    """The project id rows must actually be WRITTEN under.

    ``MASTER_CORPUS_PROJECT_ID`` is a VIRTUAL alias: it is injected into
    listings on the fly and deliberately has no ``projects`` row (see
    ``_is_master_corpus_alias``). Every reader already resolves it —
    ``get_project``, ``list_documents``, ``count_documents`` and friends all
    open with ``_master_corpus_source(project_id) or project_id``.

    ``add_document`` did not, and it is a WRITER. So uploading to the Master
    Corpus passed the permission check (the reader resolved the alias and found
    the backing corpus) and then tried to insert a Document row whose
    ``project_id`` foreign key pointed at an id with no project behind it:

        sqlite3.IntegrityError: FOREIGN KEY constraint failed

    An unhandled exception mid-request means the response never completes, and
    a request that dies without a response is reported by the browser as
    ``TypeError: Failed to fetch`` — with no status, because there is no
    response to read a status from. Uploading to the Master Corpus was
    therefore impossible, and said so in the least diagnosable way available.

    Readers may keep using the alias; only writes need resolving.
    """
    return _master_corpus_source(project_id) or project_id


def resolve_document_size(size: int, file_path: Optional[str]) -> int:
    """The size to record for a document — measured from disk when not supplied.

    Live incident: the Master Corpus listed every one of its ~227 documents as
    "0 B" in the UI. ``size`` defaulted to 0 and NOTHING ever checked it against
    the bytes actually on disk, so a caller that forgot the argument (or a
    migration payload whose transform dropped the field) silently registered a
    document whose recorded size was a lie. The UI rendered that lie faithfully.

    Policy: a caller-supplied positive size wins (it is the plaintext length the
    writer just measured, and is correct even when the file is encrypted at
    rest). Otherwise, if a readable ``file_path`` exists, measure it. If the
    file is missing or unreadable the size stays 0 — a metadata-only row is a
    legitimate thing to register (bulk-inserted corpora reference files that
    live on an operator's drive, not ours) — but it is logged, because a
    document that claims zero bytes is nearly always a defect upstream.
    """
    try:
        if int(size or 0) > 0:
            return int(size)
    except (TypeError, ValueError):
        pass  # unparsable size is treated as "not supplied"
    if not file_path:
        return 0
    try:
        from app.core import file_crypto

        measured = file_crypto.plaintext_size(file_path)
        if measured > 0:
            return measured
        logger.warning(
            "document at %s measures 0 bytes on disk — registering size 0",
            file_path,
        )
        return 0
    except (OSError, ValueError) as exc:
        logger.warning(
            "could not measure size of %s (%s); registering size 0", file_path, exc,
        )
        return 0


def add_document(
    project_id: str,
    original_name: str,
    stored_as: Optional[str] = None,
    file_path: Optional[str] = None,
    size: int = 0,
    role: Optional[str] = None,
    content_sha256: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    reingest_of: Optional[str] = None,
    drive_md5: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a document under a project. Storing only — runs no analysis.

    A non-empty ``content_sha256`` that already exists in this project is
    refused unless ``reingest_of`` names the old row. That path creates a
    new id and hides the old one (``retrieval_visible=false``). Hard
    delete is never used.
    """
    _ensure_db()
    # Writes go to the BACKING corpus, never the virtual alias — see
    # storage_project_id. Without this, uploading to the Master Corpus violated
    # the project_id foreign key and killed the request.
    project_id = storage_project_id(project_id)
    # A recorded size must describe the bytes that exist, not the default of
    # whichever caller forgot to pass one — see resolve_document_size.
    size = resolve_document_size(size, file_path)
    if content_sha256 and not reingest_of:
        existing = find_document_by_sha(project_id, content_sha256)
        if existing is not None:
            raise DuplicateContentError(existing["id"], content_sha256)
    if reingest_of:
        old_row = get_document(reingest_of)
        if old_row is None:
            raise ValueError(f"reingest_of={reingest_of} is not a documents row")
        old_sha = old_row.get("content_sha256")
        if content_sha256 and old_sha and old_sha != content_sha256:
            logger.warning(
                "reingest sha differs old_id=%s old_sha=%s new_sha=%s",
                reingest_of, old_sha, content_sha256,
            )
    did = str(uuid.uuid4())[:8]
    doc_type = classify_doc_type(original_name)
    doc_role = role if role in VALID_ROLES else classify_doc_role(original_name)
    with _lock:
        with SessionLocal() as session:
            session.add(
                Document(
                    id=did,
                    project_id=project_id,
                    original_name=original_name,
                    stored_as=stored_as,
                    file_path=file_path,
                    doc_type=doc_type,
                    doc_role=doc_role,
                    size=size,
                    uploaded_at=_now(),
                    content_sha256=content_sha256,
                    metadata_=metadata,
                    retrieval_visible=True,
                    drive_md5=drive_md5,
                )
            )
            session.commit()
    if reingest_of:
        supersede_document(reingest_of, did)
    with SessionLocal() as session:
        document = session.get(Document, did)
    assert document is not None
    _identity_changed(project_id)
    return _document_as_dict(document)


def create_ingestion_job(project_id: str, document_id: str):
    """Persist a pending ingestion job for the worker queue.

    Returns the created job row so callers can pass ``job.id`` to the queue.
    """
    from app.core.db import SessionLocal
    from app.core.models import IngestionJob

    _ensure_db()
    # Same alias-vs-row hazard as add_document: IngestionJob.project_id is a
    # foreign key, and the alias has no row to point at.
    project_id = storage_project_id(project_id)
    db = SessionLocal()
    try:
        job = IngestionJob(project_id=project_id, document_id=document_id, status="pending")
        db.add(job)
        db.commit()
        db.refresh(job)
        return job
    finally:
        db.close()


def set_document_size_if_zero(doc_id: str, size: int) -> None:
    """Record ``size`` only when the row still claims 0 bytes.

    Master Corpus / P1B rows often land as ``size=0`` after the local file is
    deleted (or never existed). Once preview hydrates real bytes we know the
    length — write it so the UI stops rendering "0 B" / "no file".
    """
    if int(size or 0) <= 0 or not doc_id:
        return
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None or int(document.size or 0) > 0:
                return
            document.size = int(size)
            session.commit()


def _preview_cache_path(doc: Dict[str, Any]) -> str:
    data_dir = os.getenv("DATA_DIR", "./data")
    cache_dir = os.path.join(data_dir, "preview_cache")
    os.makedirs(cache_dir, exist_ok=True)
    ext = os.path.splitext((doc.get("original_name") or "").lower())[1][:20]
    raw_id = "".join(
        c for c in str(doc.get("id") or "unknown") if c.isalnum() or c in "-_"
    )[:32] or "unknown"
    return os.path.join(cache_dir, f"{raw_id}{ext}")


def _local_plaintext_size(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        from app.core import file_crypto

        return int(file_crypto.plaintext_size(path) or 0)
    except (OSError, ValueError):
        logger.warning("could not read plaintext size for %s", path, exc_info=True)
        return 0


def _resolve_drive_id_by_filename(doc: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Fill a missing drive_file_id from an exact Drive filename match.

    Live Master Corpus cites (e.g. ``<doc-id>``) were inserted by
    ``rag_render_bulk_ingest`` with ``drive_file_id: null`` and a stale
    Windows ``file_path``. The PDF still exists on Drive under
    ``original_name``. Persist the id on success so later hydrates skip
    the lookup.
    """
    name = str(doc.get("original_name") or "").strip()
    if not name:
        return None, "document has no original_name"
    try:
        from app.core import gdrive_service

        file_id, err = gdrive_service.find_file_id_by_exact_name(name)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    if not file_id:
        return None, err or f"no Drive file named {name}"
    _persist_resolved_drive_id(doc, file_id, "original_name")
    logger.info(
        "resolved drive_file_id for doc %s from filename %r",
        doc.get("id"), name,
    )
    return file_id, None


def _resolve_drive_id_from_ledger(doc: Dict[str, Any]) -> Optional[str]:
    """Fill a missing drive_file_id from the ledger itself.

    The rag_backfill stubs carry the chunks a chat cites but no Drive id; the
    same original was later ingested from Drive (P1B) into another row that
    has one. When every other row with this exact ``original_name`` that
    carries a Drive id carries the SAME id, that is the original. Two
    different ids is ambiguous: no guess. Persisted like the filename resolve.
    """
    name = str(doc.get("original_name") or "").strip()
    doc_id = str(doc.get("id") or "")
    if not name:
        return None
    _ensure_db()
    with SessionLocal() as session:
        rows = session.scalars(
            select(Document).where(Document.original_name == name, Document.id != doc_id)
        ).all()
        ids = {
            extract_document_source_pointers(_document_as_dict(row))["drive_file_id"]
            for row in rows
        } - {""}
    if len(ids) != 1:
        return None
    file_id = ids.pop()
    _persist_resolved_drive_id(doc, file_id, "ledger_twin")
    logger.info("resolved drive_file_id for doc %s from its ledger twin", doc_id)
    return file_id


def _persist_resolved_drive_id(
    doc: Dict[str, Any], file_id: str, source: str,
) -> None:
    """Write ``drive_file_id`` onto the row after a successful resolve."""
    doc_id = str(doc.get("id") or "")
    if not doc_id or not file_id:
        return
    try:
        update_document_metadata(doc_id, {
            "drive_file_id": file_id,
            "drive_resolved_from": source,
        })
    except Exception:
        logger.info(
            "could not persist resolved drive_file_id for doc %s",
            doc_id, exc_info=True,
        )


def _fetch_remote_document_bytes(
    doc: Dict[str, Any],
) -> Tuple[Optional[bytes], Optional[str], Optional[str]]:
    """Fetch a document's original from Google Drive, its source of truth.

    The platform keeps no copy of originals (docs/INGEST_EXCLUSION_RULE.md):
    the ledger row's Drive file id is the only way back to the file.
    Returns ``(bytes, source, error)``. ``error`` is a logging-safe reason
    when bytes are missing — never a secret.
    """
    pointers = extract_document_source_pointers(doc)
    drive_id = pointers["drive_file_id"]
    drive_error: Optional[str] = None

    if not drive_id:
        # rag_backfill_client_clean_all stubs: size=0, G:\ path, null
        # drive_file_id. The ledger usually already knows the id (the same
        # original ingested from Drive into another row); else resolve the
        # live Drive file by exact original_name. Either way the id is
        # persisted so the next preview is cheap.
        drive_id = _resolve_drive_id_from_ledger(doc) or ""
        if not drive_id:
            resolved_id, resolve_err = _resolve_drive_id_by_filename(doc)
            if resolved_id:
                drive_id = resolved_id
            elif resolve_err:
                drive_error = resolve_err

    if drive_id:
        try:
            from app.core import gdrive_service

            # files.get (supportsAllDrives) confirms the id before media.
            # Name-search cannot see anyone-with-link files; files.get can.
            _meta, meta_err = gdrive_service.get_file_metadata(drive_id)
            if meta_err:
                logger.info(
                    "Drive files.get miss for doc %s: %s", doc.get("id"), meta_err,
                )
            blob, err = gdrive_service.download_file_bytes(drive_id)
            if blob:
                return blob, "drive", None
            drive_error = err or "Drive download returned no bytes"
            logger.info(
                "Drive fallback failed for doc %s: %s", doc.get("id"), drive_error,
            )
            # SA media failed (or SA is blind). Anyone-with-link files are
            # still fetchable at the public uc/download URL. Private files
            # fail closed — the helper never uses the SA token.
            pub, pub_err = gdrive_service.download_public_file_bytes(drive_id)
            if pub:
                _persist_resolved_drive_id(doc, drive_id, "public_download")
                return pub, "drive_public", None
            if pub_err:
                drive_error = f"{drive_error}; {pub_err}"
                logger.info(
                    "public Drive download failed for doc %s: %s",
                    doc.get("id"), pub_err,
                )
        except Exception as exc:  # noqa: BLE001
            drive_error = f"{type(exc).__name__}: {exc}"
            logger.info(
                "Drive fallback skipped for doc %s: %s", doc.get("id"), drive_error,
            )

    if not drive_id:
        extra = f"; {drive_error}" if drive_error else ""
        return None, None, "no Google Drive file id on this document" + extra
    return None, None, drive_error or "Drive download failed"


def materialize_document_file(doc: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Resolve bytes for a document row onto a readable local path.

    The platform keeps no copy of originals: P1B ingest deletes its transient
    copy after indexing, so ``file_path`` is often a stale path and Google
    Drive is the source of truth. Order:

    1. Existing local ``file_path`` with plaintext length > 0 (user uploads)
    2. On-demand preview cache from a prior hydrate
    3. Drive ``drive_file_id`` / ``driveFileId`` on the ledger row
    4. Public Drive uc/download when the id is anyone-with-link and SA media fails

    Returns ``(path, "ok")`` or ``(None, "empty"|reason)``. Reason is a
    logging-safe phrase the router appends to the 404 — never a 500.
    """
    from app.core import file_crypto

    fp = doc.get("file_path") or ""
    if _local_plaintext_size(fp) > 0:
        return fp, "ok"

    cache = _preview_cache_path(doc)
    if _local_plaintext_size(cache) > 0:
        return cache, "ok"

    raw, source, fetch_error = _fetch_remote_document_bytes(doc)
    if raw is None:
        if _local_plaintext_size(fp) == 0 and fp and os.path.exists(fp):
            return None, "empty"
        return None, fetch_error or "missing"
    if len(raw) == 0:
        return None, "empty"

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    file_crypto.write_document(cache, raw)
    logger.info(
        "hydrated document %s from %s (%s bytes) for preview",
        doc.get("id"), source, len(raw),
    )
    if int(doc.get("size") or 0) <= 0:
        set_document_size_if_zero(str(doc.get("id") or ""), len(raw))
    return cache, "ok"


def set_document_drive_md5(doc_id: str, token: Optional[str]) -> Optional[Dict[str, Any]]:
    """Write Drive's content token. Never used as a skip key by itself."""
    if not doc_id:
        return None
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            document.drive_md5 = (token or "").strip() or None
            session.commit()
    return get_document(doc_id)


def set_document_file_path(doc_id: str, file_path: str) -> Optional[Dict[str, Any]]:
    """Point a row at the copy its next index run reads (a re-ingest's fresh download)."""
    if not doc_id:
        return None
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            document.file_path = file_path
            session.commit()
    return get_document(doc_id)


def tombstone_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Mark a Drive-deleted row hidden and drop its chunks. Keeps the row.

    Idempotent: an already-TOMBSTONED row is returned unchanged.
    """
    if not doc_id:
        return None
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            if getattr(document, "ingest_status", None) != TOMBSTONED:
                document.ingest_status = TOMBSTONED
                document.ingest_status_reason = "drive_deleted"
                document.retrieval_visible = False
                delete_document_chunks(session, doc_id)
                document.chunk_count = 0
                session.commit()
    return get_document(doc_id)


def update_document_metadata(doc_id: str, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Merge ``metadata`` into the document's existing metadata_ column.

    Returns the updated document dict, or None if the document was not found.
    """
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            # Copy — SQLAlchemy JSON does not flag in-place mutation of the
            # same dict object, so a stub row would keep drive_file_id=null.
            current = dict(coerce_document_metadata(document.metadata_))
            current.update(metadata)
            document.metadata_ = current
            session.commit()
    with SessionLocal() as session:
        document = session.get(Document, doc_id)
    assert document is not None
    return _document_as_dict(document)


def record_ingest_attempt(doc_id: str, key: str) -> int:
    """Count one indexing attempt on the ledger row; return the new count.

    Stored in ``metadata.ingest_attempts`` as ``{"n": int, "key": str}``. A
    different ``key`` (new source bytes or a new extractor) restarts the
    count at 1. Written BEFORE the work, so an attempt that kills the process
    is still counted.
    """
    if not doc_id:
        return 0
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return 0
            current = dict(coerce_document_metadata(document.metadata_))
            prior = current.get("ingest_attempts")
            n = 0
            if isinstance(prior, dict) and prior.get("key") == key:
                from app.core.ingest_status import attempt_count

                n = attempt_count(prior)
            n += 1
            current["ingest_attempts"] = {"n": n, "key": key}
            document.metadata_ = current
            session.commit()
    return n


def ingest_attempts(doc: Optional[Dict[str, Any]], key: str) -> int:
    """Attempts recorded on ``doc`` for ``key`` (0 when none or a stale key)."""
    from app.core.ingest_status import attempt_count

    meta = (doc or {}).get("metadata") or {}
    prior = meta.get("ingest_attempts") if isinstance(meta, dict) else None
    if not isinstance(prior, dict) or prior.get("key") != key:
        return 0
    return attempt_count(prior)


def record_source_alias(doc_id: str, source_file_id: str, token: Optional[str]) -> None:
    """Remember that ``source_file_id`` holds the same bytes as ``doc_id``.

    A duplicate-content source file never gets a row of its own, so without
    this its id is unknown to resume and the file is downloaded, hashed and
    dropped again on every pass (live 2026-10-03: 50 per pass). Stored in
    ``metadata.source_aliases`` as ``{source_file_id: token}``; resume treats
    an alias as done while the source's content token is unchanged.
    """
    if not doc_id or not source_file_id:
        return
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return
            current = dict(coerce_document_metadata(document.metadata_))
            aliases = dict(current.get("source_aliases") or {})
            if aliases.get(source_file_id) == (token or ""):
                return
            aliases[source_file_id] = token or ""
            current["source_aliases"] = aliases
            document.metadata_ = current
            session.commit()


def find_document_by_sha(
    project_id: str, content_sha256: str,
) -> Optional[Dict[str, Any]]:
    """Return the FIRST existing document in this project with this content
    hash, or None. Used by the Drive walker to skip unchanged files on
    re-walk. Returns None for empty/None hashes so a missing sha cannot
    accidentally match other null-sha rows."""
    if not content_sha256:
        return None
    _ensure_db()
    project_id = storage_project_id(project_id)
    with SessionLocal() as session:
        document = session.scalars(
            select(Document)
            .where(
                Document.project_id == project_id,
                Document.content_sha256 == content_sha256,
            )
            .order_by(Document.uploaded_at)
            .limit(1)
        ).first()
    return _document_as_dict(document) if document else None


def scoped_reingest_of(
    reingest_of: Optional[str],
    content_sha256: Optional[str],
    existing_by_sha: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return ``reingest_of`` only when this file replaces that row.

    A folder walk that passes ``--reingest OLD_ID`` must not hide OLD_ID
    behind every other file in the batch. Match is the named id or the
    same content sha.
    """
    if not reingest_of:
        return None
    old = get_document(reingest_of)
    if old is None:
        return None
    if existing_by_sha and existing_by_sha.get("id") == reingest_of:
        return reingest_of
    if content_sha256 and old.get("content_sha256") == content_sha256:
        return reingest_of
    return None


def supersede_document(old_id: str, new_id: str) -> Optional[Dict[str, Any]]:
    """Hide ``old_id`` and point it at ``new_id``. Never deletes a row.

    Reversible: set ``retrieval_visible=true`` and clear ``superseded_by``.
    """
    if not old_id or not new_id or old_id == new_id:
        return None
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            old = session.get(Document, old_id)
            new = session.get(Document, new_id)
            if old is None or new is None:
                return None
            old.superseded_by = new_id
            old.retrieval_visible = False
            session.commit()
    return get_document(old_id)


def stamp_document_index(
    doc_id: str,
    *,
    chunk_count: int,
    ingest_status: str = INDEXED,
    ingest_status_reason: Optional[str] = None,
    extractor_version: Optional[str] = None,
    stamp_extractor_version: bool = True,
) -> Optional[Dict[str, Any]]:
    """Write ledger columns after a successful (re)index. Never deletes.

    ``stamp_extractor_version=False`` leaves the existing extractor_version
    untouched. Text-extracted-but-not-embedded, and plain TEXT_SPARSE
    after a 1-chunk embed, must not advance the stamp that
    ``docx_stale_extractor_open`` treats as settled. A successful thin
    embed whose prior stamp is the reopen sentinel is the exception
    (terminal-close).
    """
    if not doc_id:
        return None
    _ensure_db()
    version = extractor_version or EXTRACTOR_VERSION
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document is None:
                return None
            document.chunk_count = int(chunk_count)
            document.ingest_status = ingest_status
            document.ingest_status_reason = ingest_status_reason
            if ingest_status in NO_CHUNK_STATUSES:
                delete_document_chunks(session, doc_id)
                document.chunk_count = 0
            if stamp_extractor_version:
                document.extractor_version = version
            document.last_verified_at = _now()
            session.commit()
    return get_document(doc_id)


def delete_document_chunks(session, doc_id: str) -> int:
    """Delete one document's chunks inside the caller's session/transaction.

    Covers the active namespace table and the retired legacy table when they
    exist. A document that leaves the indexed statuses, or is deleted, loses
    its chunks in the same commit as the status change -- never "stale chunks
    of a failed or tombstoned document" still retrievable.
    """
    from sqlalchemy import inspect as _inspect

    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    removed = 0
    existing = set(_inspect(session.get_bind()).get_table_names())
    for table in {rag_chunk_table_name(_rag_vector_namespace()), rag_chunk_table_name("")}:
        if table in existing:
            removed += int(session.execute(
                sqla_text(f"DELETE FROM {table} WHERE doc_id = :doc_id"), {"doc_id": doc_id},
            ).rowcount or 0)
    return removed


def backfill_chunk_counts_from_table(
    *,
    project_id: Optional[str] = None,
    apply: bool = False,
) -> Dict[str, Any]:
    """Set ``documents.chunk_count`` from a COUNT of the active chunk table.

    Targets rows where the ledger column is 0 (or stale) but chunks exist.
    Never deletes. Dry-run unless ``apply``.
    """
    _ensure_db()
    from app.core import ingest_status as ist
    from app.core.rag.vector_store import get_store

    store = get_store()
    stmt = select(Document)
    if project_id:
        stmt = stmt.where(Document.project_id == storage_project_id(project_id))
    with SessionLocal() as session:
        docs = list(session.scalars(stmt).all())

    scanned = 0
    stale = 0
    updated = 0
    by_project: Dict[str, List[Document]] = {}
    for doc in docs:
        scanned += 1
        by_project.setdefault(doc.project_id, []).append(doc)

    for pid, group in by_project.items():
        counts = store.count_by_doc(pid)
        for doc in group:
            real = int(counts.get(doc.id, 0) or 0)
            if real > 0 and int(doc.chunk_count or 0) != real:
                stale += 1
                if apply:
                    ext = os.path.splitext(doc.original_name or "")[1]
                    classified = ist.classify(
                        chunk_count=real, extension=ext,
                    )
                    stamp_document_index(
                        doc.id,
                        chunk_count=real,
                        ingest_status=classified.status,
                        ingest_status_reason=classified.reason,
                    )
                    updated += 1

    return {
        "scanned": scanned,
        "stale": stale,
        "updated": updated,
        "apply": apply,
        "table": store._table_name,
    }


# A copy marker an operator's re-upload adds to the same file name:
# "Letter (1).docx", "Letter - Copy.docx", "Letter copy 2.docx".
_COPY_MARKER_RE = re.compile(r"(?:\s*[-_]?\s*\bcopy\b(?:\s*\d+)?|\s*\(\d+\))+$", re.IGNORECASE)


def _duplicate_name_key(original_name: str) -> str:
    """Name identity for "the same document uploaded again".

    Extension, case, runs of whitespace/underscores and trailing copy markers
    are ignored; nothing else is.
    """
    stem = os.path.splitext((original_name or "").strip())[0]
    stem = _COPY_MARKER_RE.sub("", stem)
    return re.sub(r"[\s_]+", " ", stem).strip().lower()


def supersede_older_duplicates(
    project_id: Optional[str] = None, *, apply: bool = False,
) -> Dict[str, Any]:
    """Hide older uploads behind the newest copy of the same document.

    The structural rule a stale extract needs: within one project, visible
    documents whose names agree under ``_duplicate_name_key`` are copies of one
    document, and the most recently uploaded copy is the live one. Each older
    copy is pointed at it with ``supersede_document`` (reversible, never a
    delete). Dry-run unless ``apply``: returns the planned ``pairs`` either way.
    ``project_id=None`` scans every project.
    """
    _ensure_db()
    with SessionLocal() as session:
        stmt = select(Document).where(Document.superseded_by.is_(None))
        if project_id:
            stmt = stmt.where(Document.project_id == project_id)
        rows = [
            d for d in session.scalars(stmt).all()
            if d.retrieval_visible is None or d.retrieval_visible
        ]
    groups: Dict[Tuple[str, str], List[Any]] = {}
    for d in rows:
        key = _duplicate_name_key(d.original_name or "")
        if key:
            groups.setdefault((d.project_id, key), []).append(d)
    pairs: List[Dict[str, str]] = []
    for docs in groups.values():
        if len(docs) < 2:
            continue
        docs.sort(key=lambda d: (d.uploaded_at is not None, d.uploaded_at, d.id))
        live = docs[-1]
        pairs.extend({"stale_id": d.id, "live_id": live.id} for d in docs[:-1])
    pairs.sort(key=lambda pr: (pr["live_id"], pr["stale_id"]))
    if apply:
        for pr in pairs:
            supersede_document(pr["stale_id"], pr["live_id"])
    return {"applied": bool(apply), "project_id": project_id, "pairs": pairs}


def documents_matching_filename_terms(
    project_id: str,
    terms: List[str],
    *,
    min_terms: int = 2,
    require_letter: bool = False,
    limit: int = 8,
    require_all: bool = False,
) -> List[Dict[str, str]]:
    """Documents whose ``original_name`` or ``file_path`` overlap ``terms``.

    ``require_all`` makes every term mandatory IN THE QUERY. The default
    OR-match is cut to 80 rows before it is ranked, and on a corpus where
    hundreds of names contain "bill" that cut can drop the one file that
    contains all of "bill", "quantities" and "priced".

    Used by letter/signatory retrieval so a named-site letter (the XBCC /
    Wadi Example completion letter lives in ``Misc/`` with the site and
    party in the filename) can be found without listing the whole corpus.
    ``terms`` are already sanitised by the retriever (alphanumeric, ≥4
    chars); LIKE wildcards are therefore not attacker-controlled.

    Returns ``[{id, original_name, file_path}, ...]`` ranked by overlap
    descending, capped at ``limit``. Empty when nothing qualifies.
    """
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in terms or []:
        tok = (raw or "").strip().lower()
        if len(tok) < 3 or tok in seen:
            continue
        if any(ch in tok for ch in "%_"):
            continue
        seen.add(tok)
        cleaned.append(tok)
        if len(cleaned) >= 8:
            break
    if not cleaned or not source_id:
        return []

    name_l = func.lower(Document.original_name)
    # No COALESCE: a NULL path can never contain a non-empty term, and the
    # bare expression is the one the trigram index (Alembic 0022) is on.
    path_l = func.lower(Document.file_path)
    term_clauses = [
        or_(name_l.like(f"%{tok}%"), path_l.like(f"%{tok}%"))
        for tok in cleaned
    ]
    stmt = (
        select(Document.id, Document.original_name, Document.file_path)
        .where(Document.project_id == source_id)
        .where(and_(*term_clauses) if require_all else or_(*term_clauses))
        .where(Document.retrieval_visible.is_(True))
    )
    if require_all:
        min_terms = len(cleaned)
    if require_letter:
        stmt = stmt.where(or_(name_l.like("%letter%"), path_l.like("%letter%")))
    stmt = stmt.limit(80)

    try:
        with SessionLocal() as session:
            rows = session.execute(stmt).all()
    except Exception:
        logger.warning(
            "documents_matching_filename_terms failed for %s",
            project_id,
            exc_info=True,
        )
        return []

    scored: List[Tuple[int, Dict[str, str]]] = []
    for row in rows:
        blob = f"{row.original_name or ''} {row.file_path or ''}".lower()
        hits = sum(1 for tok in cleaned if tok in blob)
        if hits < min_terms:
            continue
        scored.append((
            hits,
            {
                "id": row.id,
                "original_name": row.original_name or "",
                "file_path": row.file_path or "",
            },
        ))
    scored.sort(key=lambda item: -item[0])
    return [doc for _hits, doc in scored[:limit]]


def documents_matching_title_phrase(
    project_id: str,
    phrase: str,
    *,
    limit: int = 8,
) -> List[Dict[str, str]]:
    """Documents whose ``original_name`` or ``file_path`` contains ``phrase``.

    Used by specification-title retrieval (live pack C2). The Variation
    Procedure spec is named in its filename
    (``EXDAX-EXD-PMO-SPE-099650-1.0 Variation Procedure``); cosine and
    term-rescue cannot tell that file from a demolition-spec volume that
    merely mentions variations. Exact-phrase LIKE on the upload name is
    the discriminator those volumes cannot fake.

    ``phrase`` is sanitised here (lowercased, collapsed whitespace, no
    LIKE wildcards). Empty when nothing qualifies.
    """
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    cleaned = " ".join((phrase or "").lower().split())
    if (
        not cleaned
        or len(cleaned) < 8
        or " " not in cleaned
        or any(ch in cleaned for ch in "%_")
        or not source_id
    ):
        return []

    name_l = func.lower(Document.original_name)
    # No COALESCE: a NULL path can never contain the phrase, and the bare
    # expression is the one the trigram index (Alembic 0022) is on.
    path_l = func.lower(Document.file_path)
    needle = f"%{cleaned}%"
    stmt = (
        select(Document.id, Document.original_name, Document.file_path)
        .where(Document.project_id == source_id)
        .where(or_(name_l.like(needle), path_l.like(needle)))
        # Its sibling documents_matching_filename_terms has always skipped
        # retired files. This one did not, and at limit=8 a retired copy can
        # also crowd the live file out of the list.
        .where(Document.retrieval_visible.is_(True))
        .limit(max(1, int(limit or 8)))
    )
    try:
        with SessionLocal() as session:
            rows = session.execute(stmt).all()
    except Exception:
        logger.warning(
            "documents_matching_title_phrase failed for %s",
            project_id,
            exc_info=True,
        )
        return []
    return [
        {
            "id": row.id,
            "original_name": row.original_name or "",
            "file_path": row.file_path or "",
        }
        for row in rows
    ]


def list_documents(
    project_id: str,
    *,
    limit: Optional[int] = None,
    offset: int = 0,
    newest_first: bool = False,
) -> List[Dict[str, Any]]:
    """List a project's documents.

    ``limit``/``offset`` paginate (used by the workspace + the documents
    endpoint so a 2700-doc corpus doesn't serialize every row on every load).
    ``newest_first`` orders by upload time descending — the natural order for a
    paginated sidebar. Defaults preserve the legacy "all docs, oldest-first"
    behaviour so existing callers are untouched.
    """
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    order = Document.uploaded_at.desc() if newest_first else Document.uploaded_at
    with SessionLocal() as session:
        stmt = (
            select(Document)
            .where(Document.project_id == source_id)
            .order_by(order)
        )
        if limit is not None:
            stmt = stmt.limit(limit).offset(offset)
        rows = session.scalars(stmt).all()
    return [_document_as_dict(document) for document in rows]


def list_document_names(project_id: str) -> List[Dict[str, Any]]:
    """``id``, ``original_name``, ``doc_type`` and ``file_path`` of a project's documents.

    For callers that match names (which file did the user mean?): one narrow
    query, no per-document file check -- unlike ``list_documents``, which
    stats every stored file.
    """
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    with SessionLocal() as session:
        rows = session.execute(
            select(Document.id, Document.original_name, Document.doc_type, Document.file_path)
            .where(Document.project_id == source_id)
            .order_by(Document.uploaded_at)
        ).all()
    return [{"id": r.id, "original_name": r.original_name, "doc_type": r.doc_type,
             "file_path": r.file_path} for r in rows]


def count_documents(project_id: str) -> int:
    """Total document count for a project — a cheap indexed COUNT, so a
    paginated listing can report the total without serializing every row."""
    _ensure_db()
    source_id = _master_corpus_source(project_id) or project_id
    with SessionLocal() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(Document)
                .where(Document.project_id == source_id)
            )
            or 0
        )


def audit_document_sizes(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Report documents whose recorded size does not describe reality.

    Read-only counterpart to :func:`repair_document_sizes` — an operator (or a
    test) can assert the corpus is honest without mutating it. Buckets:

    * ``zero_with_file``   — size 0 but a readable file exists (REPAIRABLE; this
      is the "0 B" corpus defect).
    * ``zero_no_file``     — size 0 and no file on disk (metadata-only row; only
      repairable by re-ingesting the source).
    * ``mismatched``       — recorded size differs from the bytes on disk.
    """
    _ensure_db()
    from app.core import file_crypto

    source_id = _master_corpus_source(project_id) or project_id
    with SessionLocal() as session:
        stmt = select(Document)
        if source_id:
            stmt = stmt.where(Document.project_id == source_id)
        rows = session.scalars(stmt).all()
        docs = [_document_as_dict(d) for d in rows]

    report: Dict[str, Any] = {
        "total": len(docs), "ok": 0,
        "zero_with_file": [], "zero_no_file": [], "mismatched": [],
    }
    for doc in docs:
        recorded = int(doc.get("size") or 0)
        path = doc.get("file_path") or ""
        actual: Optional[int] = None
        if path:
            try:
                actual = file_crypto.plaintext_size(path)
            except (OSError, ValueError):
                actual = None
        entry = {"id": doc.get("id"), "original_name": doc.get("original_name"),
                 "recorded_size": recorded, "actual_size": actual}
        if recorded <= 0:
            (report["zero_with_file"] if actual and actual > 0
             else report["zero_no_file"]).append(entry)
        elif actual is not None and actual != recorded:
            report["mismatched"].append(entry)
        else:
            report["ok"] += 1
    report["repairable"] = len(report["zero_with_file"]) + len(report["mismatched"])
    return report


def repair_document_sizes(
    project_id: Optional[str] = None, *, dry_run: bool = False,
) -> Dict[str, Any]:
    """Recompute recorded sizes from the bytes on disk.

    Repairs rows already written by the pre-fix code paths (the 0 B corpus).
    ``dry_run`` reports what WOULD change without writing — the default is to
    write, because a size that disagrees with disk is never the correct value.
    Rows whose file is missing are reported, never guessed at.
    """
    audit = audit_document_sizes(project_id)
    targets = audit["zero_with_file"] + audit["mismatched"]
    repaired = 0
    if not dry_run and targets:
        with _lock:
            with SessionLocal() as session:
                for entry in targets:
                    document = session.get(Document, entry["id"])
                    if document is None or not entry.get("actual_size"):
                        continue
                    document.size = int(entry["actual_size"])
                    repaired += 1
                session.commit()
    return {
        "dry_run": dry_run,
        "scanned": audit["total"],
        "repaired": repaired,
        "repairable": audit["repairable"],
        "unrepairable_no_file": len(audit["zero_no_file"]),
        "details": {k: audit[k] for k in
                    ("zero_with_file", "zero_no_file", "mismatched")},
    }


def get_document(doc_id: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        document = session.get(Document, doc_id)
    return _document_as_dict(document) if document else None


def document_names(doc_ids: List[str]) -> Dict[str, str]:
    """``{id: original_name}`` for the ids that have a documents row.

    One primary-key read for a batch, where ``get_document`` per id costs a
    round trip each. Absent ids are simply missing from the result.
    """
    ids = [d for d in dict.fromkeys(doc_ids or []) if d]
    if not ids:
        return {}
    _ensure_db()
    with SessionLocal() as session:
        rows = session.execute(
            select(Document.id, Document.original_name).where(Document.id.in_(ids))
        ).all()
    return {str(r.id): r.original_name or "" for r in rows}


def delete_document(doc_id: str) -> Optional[Dict[str, Any]]:
    """Delete a document row. Returns the deleted row (so the caller can
    purge the file from disk), or None if it did not exist.

    Also deletes the document's chunks from the RagChunk table. Postgres
    has an ON DELETE CASCADE FK so the cascade is implicit there; SQLite
    (used by dev / tests) doesn't enforce FKs by default, so we delete
    explicitly to keep search results consistent across backends. Without
    this, a search after deletion can still surface chunks from the
    removed document because the hybrid retriever queries the chunks
    table directly.

    If the chunks table has never been created (e.g. a fresh dev/test DB
    with the embedding stack not yet initialized), the explicit delete is
    a no-op — there are no chunks to remove.
    """
    doc = get_document(doc_id)
    if not doc:
        return None
    with _lock:
        with SessionLocal() as session:
            document = session.get(Document, doc_id)
            if document:
                # Row and chunks (active namespace and the retired legacy
                # table) go in ONE transaction. No second connection: a vector
                # store opened here would wait on this transaction's locks.
                delete_document_chunks(session, doc_id)
                session.delete(document)
                session.commit()
    _identity_changed((doc or {}).get("project_id"))
    return doc


def purge_document(doc_id: str) -> Dict[str, Any]:
    """Remove a document completely: its stored file, its row and chunks, and
    its index entry. Returns ``{deleted, file_removed, index_pruned}``.

    ``delete_document`` removes only the row and chunks; every caller that
    stopped there left the file on the shared data volume and the entry in
    the index. One function, so the delete route and the admin purge cannot
    drift apart.
    """
    doc = get_document(doc_id)
    if not doc:
        return {"deleted": False, "file_removed": False, "index_pruned": False}
    fp = doc.get("file_path")
    file_removed = False
    if fp and os.path.exists(fp):
        try:
            os.remove(fp)
            file_removed = True
        except OSError:
            logger.warning("could not remove stored file for %s", doc_id, exc_info=True)
    delete_document(doc_id)
    index_pruned = False
    project_id = doc.get("project_id")
    if project_id:
        try:
            from app.core import doc_index as _doc_index

            index_pruned = _doc_index.drop_index_entry(project_id, doc_id)
        except Exception:  # noqa: BLE001 - never block a delete on index cleanup
            logger.warning("index entry cleanup failed for %s", doc_id, exc_info=True)
    return {"deleted": True, "file_removed": file_removed, "index_pruned": index_pruned}


def purge_documents_older_than(days: int) -> List[Dict[str, Any]]:
    """Delete document rows older than `days`. Returns the purged rows
    (Roadmap V2 · Epic 6 — data retention)."""
    _ensure_db()
    from datetime import timedelta

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with SessionLocal() as session:
        rows = session.scalars(
            select(Document).where(Document.uploaded_at < cutoff)
        ).all()
        purged = [_document_as_dict(document) for document in rows]
    with _lock:
        with SessionLocal() as session:
            session.execute(delete(Document).where(Document.uploaded_at < cutoff))
            session.commit()
    return purged


# ── readiness gate (Roadmap V2 · 0.2) ───────────────────────────────────────

def compute_readiness(project_id: str) -> Dict[str, Any]:
    """A project is 'ready' for progress tracking only once it has a baseline
    schedule, at least one daily and one weekly report, and Aconex connected."""
    source_id = _master_corpus_source(project_id) or project_id
    # Readiness only needs each doc's role — NOT the full serialized rows.
    # Loading all documents here (via list_documents) meant get_project paid the
    # full doc-load TWICE (~8s on the 2713-doc master corpus). Select just the
    # doc_role column instead.
    with SessionLocal() as session:
        roles = list(
            session.scalars(
                select(Document.doc_role).where(Document.project_id == source_id)
            ).all()
        )
        project = session.get(Project, source_id)
    aconex = bool(project.aconex_connected) if project else False

    baseline = ROLE_BASELINE in roles
    daily = roles.count(ROLE_DAILY)
    weekly = roles.count(ROLE_WEEKLY)

    missing: List[str] = []
    if not baseline:
        missing.append("baseline_schedule")
    if daily < 1:
        missing.append("daily_reports")
    if weekly < 1:
        missing.append("weekly_reports")
    if not aconex:
        missing.append("aconex")

    return {
        "baseline_schedule": baseline,
        "daily_reports": daily,
        "weekly_reports": weekly,
        "aconex_connected": aconex,
        "ready": not missing,
        "missing": missing,
    }


# ── project memory / durable facts (Roadmap V2 · Epic 3) ────────────────────

def set_fact(
    project_id: str,
    key: str,
    value: str,
    source_document: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Upsert a durable fact for a project (one row per project+key)."""
    _ensure_db()
    now = _now()
    with _lock:
        with SessionLocal() as session:
            existing = session.scalars(
                select(ProjectFact).where(
                    ProjectFact.project_id == project_id,
                    ProjectFact.key == key,
                )
            ).one_or_none()
            if existing:
                existing.value = str(value)
                existing.source_document = source_document
                existing.confidence = confidence
                existing.updated_at = now
            else:
                session.add(
                    ProjectFact(
                        id=str(uuid.uuid4())[:8],
                        project_id=project_id,
                        key=key,
                        value=str(value),
                        source_document=source_document,
                        confidence=confidence,
                        updated_at=now,
                    )
                )
            session.commit()
    return get_fact(project_id, key)


def get_fact(project_id: str, key: str) -> Optional[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        fact = session.scalars(
            select(ProjectFact).where(
                ProjectFact.project_id == project_id,
                ProjectFact.key == key,
            )
        ).one_or_none()
    return _fact_as_dict(fact) if fact else None


def list_facts(project_id: str) -> List[Dict[str, Any]]:
    _ensure_db()
    with SessionLocal() as session:
        rows = session.scalars(
            select(ProjectFact)
            .where(ProjectFact.project_id == project_id)
            .order_by(ProjectFact.key)
        ).all()
    return [_fact_as_dict(fact) for fact in rows]


def search_facts(project_id: str, query: str) -> List[Dict[str, Any]]:
    """Keyword search over fact keys + values (case-insensitive, any-term)."""
    facts = list_facts(project_id)
    terms = (query or "").lower().split()
    if not terms:
        return facts
    return [
        f for f in facts
        if any(t in f"{f['key']} {f['value']}".lower() for t in terms)
    ]


def delete_fact(project_id: str, key: str) -> bool:
    _ensure_db()
    with _lock:
        with SessionLocal() as session:
            fact = session.scalars(
                select(ProjectFact).where(
                    ProjectFact.project_id == project_id,
                    ProjectFact.key == key,
                )
            ).one_or_none()
            if not fact:
                return False
            session.delete(fact)
            session.commit()
            return True
