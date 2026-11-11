"""S10 ingest reconcile — source-token resume, tombstones, coverage truth.

The ledger already has the columns (``drive_md5``, ``ingest_status``,
``TOMBSTONED``). This module is the missing *use* of those columns:

* resume compares Drive ``md5Checksum`` (then version / modifiedTime) to stored ``drive_md5``
  (id-only skip hid edited files);
* a complete Drive walk can tombstone rows whose ``drive_file_id`` is gone
  (hide from retrieval; never delete chunks);
* coverage is a queryable breakdown, not a file-count identity;
* S11 office census slices that breakdown for ``.pdf`` / ``.xlsx`` /
  ``.pptx`` (status, thin/single-chunk, missing source, extractor_version).

Null ``drive_md5`` is *not* treated as changed. Historical rows were never
stamped; treating them as stale would re-index the whole corpus. After this
module writes the token on ingest, later edits compare md5-to-md5.

A partial Drive walk must not tombstone. Sharded p1b must not tombstone.
Orphan chunks are reported only — no purge.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Collection, Iterable, Mapping, Sequence

from app.core.ingest_status import (
    ALL_STATUSES,
    INDEXED,
    OCR_DEGRADED,
    SPREADSHEET_EXTS,
    TEXT_SPARSE,
    TOMBSTONED,
    UNVERIFIED,
    document_extension,
    resume_is_already_indexed,
)

#: S11 — pptx / xlsx / pdf only. .doc / .xls / .ppt stay out of this tally.
OFFICE_CENSUS_EXTS = (".pdf", ".xlsx", ".pptx")
OFFICE_KIND_MIME = {
    ".pdf": "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
MIME_TO_OFFICE_KIND = {mime: ext for ext, mime in OFFICE_KIND_MIME.items()}


def office_kind(doc: Mapping[str, Any]) -> str | None:
    """``.pdf`` / ``.xlsx`` / ``.pptx`` from name, path, or Drive mime."""
    ext = document_extension(doc)
    if ext in OFFICE_KIND_MIME:
        return ext
    meta = doc.get("metadata") or {}
    if not isinstance(meta, Mapping):
        meta = {}
    mime = str(meta.get("mimeType") or meta.get("mime") or "").strip().lower()
    return MIME_TO_OFFICE_KIND.get(mime)


def _office_source_present(doc: Mapping[str, Any]) -> bool:
    """True when local bytes or a Drive pointer exist. Does not fetch."""
    if doc.get("has_file") is True:
        return True
    if doc.get("has_file") is False:
        return False
    fp = str(doc.get("file_path") or "")
    if fp and os.path.isfile(fp) and os.path.getsize(fp) > 0:
        return True
    from app.core.projects import extract_document_source_pointers

    pointers = extract_document_source_pointers(dict(doc))
    return bool(pointers["drive_file_id"])


def _empty_kind_bucket(ext: str) -> dict[str, Any]:
    return {
        "extension": ext,
        "mime": OFFICE_KIND_MIME[ext],
        "documents": 0,
        "by_status": {},
        "single_chunk": 0,
        "thin": 0,
        "thin_rate": 0.0,
        "single_chunk_rate": 0.0,
        "missing_source": 0,
        "indexed_zero_chunk": 0,
        "by_extractor_version": {},
    }


def tally_office_docs(docs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Ledger census for pptx / xlsx / pdf. Pure. No I/O except path stat.

    ``thin`` is TEXT_SPARSE **or** ``chunk_count == 1`` so a
    ``stamp_as_indexed`` INDEXED single-window row still counts as thin.
    ``indexed_zero_chunk`` is the silent-INDEXED-on-empty lie.
    ``missing_source`` is no local bytes and no Drive pointer — this
    function never hydrates remote objects.
    """
    by_kind = {ext: _empty_kind_bucket(ext) for ext in OFFICE_CENSUS_EXTS}
    for doc in docs:
        kind = office_kind(doc)
        if not kind:
            continue
        bucket = by_kind[kind]
        bucket["documents"] += 1
        status = str(doc.get("ingest_status") or UNVERIFIED)
        if status not in ALL_STATUSES:
            status = UNVERIFIED
        bucket["by_status"][status] = bucket["by_status"].get(status, 0) + 1
        chunks = int(doc.get("chunk_count") or 0)
        if chunks == 1:
            bucket["single_chunk"] += 1
        if status == TEXT_SPARSE or chunks == 1:
            bucket["thin"] += 1
        if status == INDEXED and chunks <= 0:
            bucket["indexed_zero_chunk"] += 1
        if not _office_source_present(doc):
            bucket["missing_source"] += 1
        ver = str(doc.get("extractor_version") or "(none)")
        versions = bucket["by_extractor_version"]
        versions[ver] = versions.get(ver, 0) + 1
    for bucket in by_kind.values():
        n = bucket["documents"]
        if n:
            bucket["thin_rate"] = round(bucket["thin"] / n, 4)
            bucket["single_chunk_rate"] = round(bucket["single_chunk"] / n, 4)
    return {
        "kinds": list(OFFICE_CENSUS_EXTS),
        "by_kind": by_kind,
        "documents_total": sum(b["documents"] for b in by_kind.values()),
        "indexed_zero_chunk": sum(b["indexed_zero_chunk"] for b in by_kind.values()),
        "missing_source": sum(b["missing_source"] for b in by_kind.values()),
    }


def render_office_census_markdown(report: Mapping[str, Any]) -> str:
    """Opaque ledger tally. No client filenames."""
    lines = [
        "# Office extraction census (pptx / xlsx / pdf)",
        "",
        "Queryable via `GET /v1/admin/corpus/coverage` (`office`) and",
        "`scripts/office_extraction_census.py`. Counts come from the",
        "documents ledger. This file is a template until a live or fixture",
        "run overwrites it. No client names.",
        "",
        "| kind | mime | n | INDEXED | TEXT_SPARSE | UNVERIFIED | ZERO_CHUNK | EXTRACT_FAILED | single_chunk | thin | thin_rate | missing_source | indexed_0chunk | extractor_versions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    by_kind = report.get("by_kind") or {}
    for ext in report.get("kinds") or OFFICE_CENSUS_EXTS:
        b = by_kind.get(ext) or _empty_kind_bucket(ext)
        st = b.get("by_status") or {}
        versions = b.get("by_extractor_version") or {}
        ver_s = ", ".join(f"{k}={v}" for k, v in sorted(versions.items())) or "—"
        lines.append(
            f"| {ext} | {b.get('mime')} | {b.get('documents', 0)} "
            f"| {st.get(INDEXED, 0)} | {st.get(TEXT_SPARSE, 0)} "
            f"| {st.get(UNVERIFIED, 0)} | {st.get('ZERO_CHUNK', 0)} "
            f"| {st.get('EXTRACT_FAILED', 0)} "
            f"| {b.get('single_chunk', 0)} | {b.get('thin', 0)} "
            f"| {b.get('thin_rate', 0)} | {b.get('missing_source', 0)} "
            f"| {b.get('indexed_zero_chunk', 0)} | {ver_s} |"
        )
    lines.extend([
        "",
        f"office_docs={report.get('documents_total', 0)} "
        f"missing_source={report.get('missing_source', 0)} "
        f"indexed_zero_chunk={report.get('indexed_zero_chunk', 0)}",
        "",
        "Read-only. Does not re-extract, fetch Drive, or run p1b.",
        "F-3 159 docx source bytes remain owner-gated and are not this tally.",
        "",
    ])
    return "\n".join(lines) + "\n"


def office_extraction_census(*, project_id: str | None = None) -> dict[str, Any]:
    """Load ledger rows and tally pptx / xlsx / pdf. Read-only."""
    from sqlalchemy import select
    from app.core import projects as projects_mod
    from app.core.db import SessionLocal
    from app.core.models import Document

    projects_mod.init_db()
    with SessionLocal() as session:
        stmt = select(Document)
        if project_id:
            stmt = stmt.where(Document.project_id == project_id)
        docs = [projects_mod._document_as_dict(d) for d in session.scalars(stmt).all()]
    return tally_office_docs(docs)


def source_content_token(file_meta: Mapping[str, Any] | None) -> str | None:
    """Drive content identity. ``md5Checksum`` first.

    Drive v3 does not return the v2 identity field, so a file that omits
    the checksum still resolves through ``version`` then ``modifiedTime``.
    A caller that already holds the legacy key can still use it. Never
    SHA-256 — Drive does not publish it, and comparing sha to md5 can
    only ever report "changed".
    """
    if not file_meta:
        return None
    for key in ("md5Checksum", "md5", "version", "modifiedTime", "etag"):
        raw = file_meta.get(key)
        if raw is None:
            continue
        token = str(raw).strip()
        if token:
            return token
    return None


def stored_source_token(doc: Mapping[str, Any] | None) -> str | None:
    """Token previously written to ``documents.drive_md5``."""
    if not doc:
        return None
    raw = doc.get("drive_md5")
    if raw is None:
        meta = doc.get("metadata") or {}
        if isinstance(meta, Mapping):
            raw = meta.get("drive_md5")
    if raw is None:
        return None
    token = str(raw).strip()
    return token or None


def resume_source_changed(
    doc: Mapping[str, Any] | None,
    file_meta: Mapping[str, Any] | None,
) -> bool:
    """True only when *both* tokens exist and differ.

    Missing stored token: historical row, do not thrash.
    Missing source token: Drive omitted md5, version, and modifiedTime, cannot tell.
    """
    stored = stored_source_token(doc)
    source = source_content_token(file_meta)
    if not stored or not source:
        return False
    return stored != source


def should_skip_resume(
    doc: Mapping[str, Any],
    chunk_count: int,
    file_meta: Mapping[str, Any] | None = None,
    *,
    parseable_exts: Collection[str] | None = None,
) -> bool:
    """Combine chunk/extractor resume with source-token change detection."""
    if not resume_is_already_indexed(doc, chunk_count, parseable_exts=parseable_exts):
        return False
    if resume_source_changed(doc, file_meta):
        return False
    return True


def drive_file_id_of(doc: Mapping[str, Any]) -> str | None:
    meta = doc.get("metadata") or {}
    if not isinstance(meta, Mapping):
        return None
    fid = meta.get("drive_file_id")
    if not fid:
        return None
    return str(fid).strip() or None


@dataclass
class ReconcileAction:
    doc_id: str
    reason: str
    drive_file_id: str | None = None


@dataclass
class ReconcilePlan:
    """Work a reconcile pass would do. Apply is opt-in and never purges."""

    to_index: list[ReconcileAction] = field(default_factory=list)
    to_tombstone: list[ReconcileAction] = field(default_factory=list)
    orphans: list[dict[str, Any]] = field(default_factory=list)
    to_reembed: list[ReconcileAction] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    walk_complete: bool = True
    skipped_tombstones_reason: str | None = None

    @property
    def index_count(self) -> int:
        return len(self.to_index)

    @property
    def work_count(self) -> int:
        return len(self.to_index) + len(self.to_tombstone) + len(self.to_reembed)


def plan_tombstones(
    local_docs: Sequence[Mapping[str, Any]],
    seen_drive_ids: Iterable[str],
    *,
    walk_complete: bool,
) -> tuple[list[ReconcileAction], str | None]:
    """Local Drive-backed rows whose file id is gone from a *complete* walk."""
    if not walk_complete:
        return [], "walk_incomplete"
    seen = {str(x).strip() for x in seen_drive_ids if str(x).strip()}
    out: list[ReconcileAction] = []
    for doc in local_docs:
        fid = drive_file_id_of(doc)
        if not fid or fid in seen:
            continue
        if (doc.get("ingest_status") or "") == TOMBSTONED:
            continue
        out.append(
            ReconcileAction(
                doc_id=str(doc.get("id") or ""),
                drive_file_id=fid,
                reason="drive_deleted",
            )
        )
    return [a for a in out if a.doc_id], None


def plan_reconcile(
    *,
    local_docs: Sequence[Mapping[str, Any]],
    drive_files: Sequence[Mapping[str, Any]],
    chunk_counts: Mapping[str, int] | None = None,
    walk_complete: bool = True,
    mismatch_doc_ids: Sequence[str] | None = None,
    orphans: Sequence[Mapping[str, Any]] | None = None,
    coverage: Mapping[str, Any] | None = None,
) -> ReconcilePlan:
    """Idempotent plan: a healthy second pass has ``work_count == 0``.

    ``to_index`` is existing local rows that must be re-indexed (source
    token changed, or still open). New Drive files with no local row are
    *imports* — left to ``reconcile_drive_delta`` / p1b, not this planner.
    """
    counts = chunk_counts or {}
    by_fid: dict[str, Mapping[str, Any]] = {}
    for fm in drive_files:
        fid = str(fm.get("id") or "").strip()
        if fid:
            by_fid[fid] = fm

    to_index: list[ReconcileAction] = []
    for doc in local_docs:
        did = str(doc.get("id") or "")
        if not did:
            continue
        if (doc.get("ingest_status") or "") == TOMBSTONED:
            continue
        fid = drive_file_id_of(doc)
        fm = by_fid.get(fid) if fid else None
        chunks = int(counts.get(did, doc.get("chunk_count") or 0) or 0)
        if should_skip_resume(doc, chunks, fm):
            continue
        if resume_source_changed(doc, fm):
            reason = "source_changed"
        elif chunks <= 0:
            reason = "zero_chunk"
        else:
            reason = "open_or_stale"
        to_index.append(
            ReconcileAction(doc_id=did, drive_file_id=fid, reason=reason)
        )

    tombstones, skip_reason = plan_tombstones(
        local_docs, by_fid.keys(), walk_complete=walk_complete,
    )
    reembed = [
        ReconcileAction(doc_id=str(i), reason="embedding_mismatch")
        for i in (mismatch_doc_ids or [])
        if str(i).strip()
    ]
    return ReconcilePlan(
        to_index=to_index,
        to_tombstone=tombstones,
        orphans=[dict(o) for o in (orphans or [])],
        to_reembed=reembed,
        coverage=dict(coverage or {}),
        walk_complete=walk_complete,
        skipped_tombstones_reason=skip_reason,
    )


def apply_tombstones(
    actions: Sequence[ReconcileAction],
    *,
    execute: bool,
) -> int:
    """Hide gone-from-Drive rows. Never deletes chunks or documents."""
    if not execute:
        return 0
    from app.core import projects as projects_mod

    applied = 0
    for action in actions:
        if projects_mod.tombstone_document(action.doc_id):
            applied += 1
    return applied


def coverage_truth(
    *,
    project_id: str | None = None,
    expected_embedding_model: str | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Queryable ingest coverage. Read-only. Never deletes."""
    from sqlalchemy import text
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    expected = (
        expected_embedding_model
        or os.getenv("RAG_EMBEDDING_MODEL")
        or ""
    ).strip()
    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )
    where_docs = "WHERE project_id = :pid" if project_id else ""
    params: dict[str, Any] = {}
    if project_id:
        params["pid"] = project_id

    by_status: dict[str, int] = {}
    retrieval_visible = 0
    tombstoned = 0
    ocr_degraded = 0
    drive_md5_missing = 0
    drive_backed = 0
    documents_total = 0
    orphans: list[dict[str, Any]] = []
    orphan_total = 0
    mismatch_chunks = 0
    mismatch_docs: list[str] = []
    office: dict[str, Any] = tally_office_docs([])

    with SessionLocal() as session:
        dialect = session.bind.dialect.name if session.bind is not None else "sqlite"
        if dialect == "postgresql":
            drive_pred = "metadata ? 'drive_file_id'"
        else:
            drive_pred = "CAST(metadata AS TEXT) LIKE '%drive_file_id%'"
        rows = session.execute(
            text(
                f"""
                SELECT ingest_status,
                       COUNT(*) AS n,
                       SUM(CASE WHEN retrieval_visible THEN 1 ELSE 0 END)
                         AS visible,
                       SUM(CASE WHEN ingest_status_reason LIKE :ocr THEN 1
                                ELSE 0 END) AS ocr_n
                FROM documents
                {where_docs}
                GROUP BY ingest_status
                """
            ),
            {**params, "ocr": f"%{OCR_DEGRADED}%"},
        ).all()
        for row in rows:
            status = row[0] or "UNVERIFIED"
            n = int(row[1] or 0)
            by_status[status] = n
            documents_total += n
            retrieval_visible += int(row[2] or 0)
            ocr_degraded += int(row[3] or 0)
            if status == TOMBSTONED:
                tombstoned += n

        md5_row = session.execute(
            text(
                f"""
                SELECT
                  SUM(CASE WHEN {drive_pred} THEN 1 ELSE 0 END),
                  SUM(CASE WHEN {drive_pred}
                            AND (drive_md5 IS NULL OR drive_md5 = '')
                           THEN 1 ELSE 0 END)
                FROM documents
                {where_docs}
                """
            ),
            params,
        ).one()
        drive_backed = int(md5_row[0] or 0)
        drive_md5_missing = int(md5_row[1] or 0)

        try:
            orphan_where = "AND c.project_id = :pid" if project_id else ""
            dangling = session.execute(
                text(
                    f"""
                    SELECT c.chunk_id, c.project_id, c.doc_id
                    FROM {chunks_table} c
                    LEFT JOIN documents d ON c.doc_id = d.id
                    WHERE d.id IS NULL
                    {orphan_where}
                    ORDER BY c.project_id, c.doc_id
                    """
                ),
                params,
            ).all()
            orphan_total = len(dangling)
            for row in dangling[: max(0, sample_limit)]:
                orphans.append({
                    "chunk_id": row[0],
                    "project_id": row[1],
                    "doc_id": row[2],
                    "quarantine": "report_only",
                })
        except Exception:
            session.rollback()
            orphan_total = 0
            orphans = []

        if expected:
            try:
                mismatch_where = "AND project_id = :pid" if project_id else ""
                mismatch_chunks = int(
                    session.execute(
                        text(
                            f"""
                            SELECT COUNT(*) FROM {chunks_table}
                            WHERE embedding_model <> :m
                            {mismatch_where}
                            """
                        ),
                        {**params, "m": expected},
                    ).scalar()
                    or 0
                )
                mismatch_docs = [
                    str(r[0])
                    for r in session.execute(
                        text(
                            f"""
                            SELECT DISTINCT doc_id FROM {chunks_table}
                            WHERE embedding_model <> :m
                            {mismatch_where}
                            """
                        ),
                        {**params, "m": expected},
                    ).all()
                    if r[0]
                ]
            except Exception:
                session.rollback()
                mismatch_chunks = 0
                mismatch_docs = []

        from app.core import projects as projects_mod
        from app.core.models import Document
        from sqlalchemy import select as sa_select

        office_stmt = sa_select(Document)
        if project_id:
            office_stmt = office_stmt.where(Document.project_id == project_id)
        office = tally_office_docs(
            [projects_mod._document_as_dict(d) for d in session.scalars(office_stmt).all()]
        )

    return {
        "documents_total": documents_total,
        "by_status": by_status,
        "retrieval_visible": retrieval_visible,
        "tombstoned": tombstoned,
        "ocr_degraded": ocr_degraded,
        "drive_backed": drive_backed,
        "drive_md5_missing": drive_md5_missing,
        "orphans_total": orphan_total,
        "orphans_sample": orphans,
        "orphans_action": "report_only",
        "embedding_model_expected": expected or None,
        "embedding_mismatch_chunks": mismatch_chunks,
        "embedding_mismatch_doc_ids": mismatch_docs,
        "chunks_table": chunks_table,
        "project_id": project_id,
        "office": office,
    }


def is_ghost(doc: Mapping[str, Any], chunks: int) -> bool:
    """Rule (c): a row is a ghost only when it has no recoverable bytes and
    no chunks. Shared by the dry-run tally and the tombstone write path so the
    two can never drift apart."""
    return doc.get("has_file") is not True and chunks <= 0


def is_chunked_spreadsheet(doc: Mapping[str, Any], chunks: int) -> bool:
    """Rule (a): a spreadsheet-extension row still holding chunks in the store.

    Spreadsheets are never chunked (owner ruling 2026-11-06); any chunk rows a
    spreadsheet-extension document still carries are legacy and removed. Shared
    by the dry-run tally and the spreadsheet purge write path so the two can
    never drift apart.
    """
    return document_extension(doc) in SPREADSHEET_EXTS and chunks > 0


def _tally_p3_counts(
    docs: Sequence[Mapping[str, Any]],
    chunk_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Read-only P3 reconcile dry-run tally. Pure: no I/O.

    Three owner/curator rules (SPEC_CORPUS_DECISIONS_v1):
      (a) spreadsheet-extension rows are removed from the chunk store;
      (b) superseded revisions per document-code family + exact duplicates;
      (c) ghosts: rows with no local bytes, no Drive pointer and no chunks.

    ``chunk_counts`` is the live chunk-row count per doc id (the cached
    ``chunk_count`` column can drift); callers may fall back to it.
    """
    ss_by_ext: dict[str, dict[str, int]] = {}
    ss_by_status: dict[str, int] = {}
    ss_documents = 0
    ss_with_chunks = 0
    ss_chunks = 0

    superseded = 0
    retrieval_invisible = 0
    sha_groups: dict[str, int] = {}
    md5_groups: dict[str, int] = {}

    ghosts = 0
    ghost_by_status: dict[str, int] = {}
    ghost_by_bytes_state: dict[str, int] = {}

    for doc in docs:
        did = str(doc.get("id") or "")
        chunks = int(
            chunk_counts.get(did, doc.get("chunk_count") or 0) or 0
        )
        ext = document_extension(doc)
        status = str(doc.get("ingest_status") or UNVERIFIED)

        # (a) spreadsheets in the chunk store
        if ext in SPREADSHEET_EXTS:
            ss_documents += 1
            bucket = ss_by_ext.setdefault(
                ext, {"documents": 0, "with_chunks": 0, "chunks": 0}
            )
            bucket["documents"] += 1
            bucket["chunks"] += chunks
            if is_chunked_spreadsheet(doc, chunks):
                bucket["with_chunks"] += 1
                ss_with_chunks += 1
            ss_chunks += chunks
            ss_by_status[status] = ss_by_status.get(status, 0) + 1

        # (b) superseded + exact duplicates
        if doc.get("superseded_by"):
            superseded += 1
        if doc.get("retrieval_visible") is False:
            retrieval_invisible += 1
        sha = doc.get("content_sha256")
        if sha:
            sha_groups[sha] = sha_groups.get(sha, 0) + 1
        md5 = doc.get("drive_md5")
        if md5:
            md5_groups[md5] = md5_groups.get(md5, 0) + 1

        # (c) ghosts
        if is_ghost(doc, chunks):
            ghosts += 1
            ghost_by_status[status] = ghost_by_status.get(status, 0) + 1
            bs = str(doc.get("bytes_state") or "")
            ghost_by_bytes_state[bs] = ghost_by_bytes_state.get(bs, 0) + 1

    def _dup_stats(groups: Mapping[str, int]) -> dict[str, int]:
        duplicate_docs = 0
        group_count = 0
        for n in groups.values():
            if n > 1:
                group_count += 1
                duplicate_docs += n - 1
        return {"groups": group_count, "duplicate_docs": duplicate_docs}

    return {
        "spreadsheets": {
            "extensions": sorted(SPREADSHEET_EXTS),
            "documents": ss_documents,
            "with_chunks": ss_with_chunks,
            "chunks": ss_chunks,
            "by_extension": ss_by_ext,
            "by_status": ss_by_status,
        },
        "superseded": {
            "superseded_by_set": superseded,
            "retrieval_invisible": retrieval_invisible,
        },
        "duplicates": {
            "content_sha256": _dup_stats(sha_groups),
            "drive_md5": _dup_stats(md5_groups),
        },
        "ghosts": {
            "total": ghosts,
            "by_status": ghost_by_status,
            "by_bytes_state": ghost_by_bytes_state,
        },
    }


def _load_p3_docs_and_chunk_counts(
    session,
    project_id: str | None,
    chunks_table: str,
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """Load the documents ledger and live chunk-row counts for a P3 pass.

    Read-only. ``chunk_counts`` is the live chunk-row count per doc id (the
    cached ``chunk_count`` column can drift); callers fall back to it.
    """
    from sqlalchemy import select, text
    from app.core import projects as projects_mod
    from app.core.models import Document

    stmt = select(Document)
    if project_id:
        stmt = stmt.where(Document.project_id == project_id)
    docs = [
        projects_mod._document_as_dict(d)
        for d in session.scalars(stmt).all()
    ]

    params: dict[str, Any] = {"pid": project_id} if project_id else {}
    chunk_counts: dict[str, int] = {}
    try:
        if project_id:
            rows = session.execute(
                text(
                    f"""
                    SELECT c.doc_id, COUNT(*)
                    FROM {chunks_table} c
                    JOIN documents d ON c.doc_id = d.id
                    WHERE d.project_id = :pid
                    GROUP BY c.doc_id
                    """
                ),
                params,
            ).all()
        else:
            rows = session.execute(
                text(
                    f"""
                    SELECT doc_id, COUNT(*)
                    FROM {chunks_table}
                    GROUP BY doc_id
                    """
                )
            ).all()
        chunk_counts = {str(r[0]): int(r[1] or 0) for r in rows if r[0]}
    except Exception:
        session.rollback()
        chunk_counts = {}
    return docs, chunk_counts


def plan_ghost_tombstones(
    docs: Sequence[Mapping[str, Any]],
    chunk_counts: Mapping[str, int],
) -> list[ReconcileAction]:
    """Rows the ghost rule selects, excluding already-TOMBSTONED rows.

    Pure planner: computes the same set as dry-run rule (c), so the operator
    can confirm the count before the write path runs.
    """
    out: list[ReconcileAction] = []
    for doc in docs:
        did = str(doc.get("id") or "")
        if not did:
            continue
        if (doc.get("ingest_status") or "") == TOMBSTONED:
            continue
        chunks = int(
            chunk_counts.get(did, doc.get("chunk_count") or 0) or 0
        )
        if is_ghost(doc, chunks):
            out.append(ReconcileAction(doc_id=did, reason="ghost"))
    return out


def apply_ghost_tombstones(
    actions: Sequence[ReconcileAction],
    *,
    execute: bool,
) -> int:
    """Hide ghost rows. Never deletes chunks or documents."""
    if not execute:
        return 0
    from app.core import projects as projects_mod

    applied = 0
    for action in actions:
        if projects_mod.tombstone_document_ghost(action.doc_id):
            applied += 1
    return applied


def p3_reconcile_dry_run(*, project_id: str | None = None) -> dict[str, Any]:
    """Read-only P3 reconcile dry-run counts. Never deletes, never re-indexes.

    Loads the documents ledger plus live chunk-row counts and hands them to
    :func:`_tally_p3_counts`. No writes to documents or chunks.
    """
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )
    with SessionLocal() as session:
        docs, chunk_counts = _load_p3_docs_and_chunk_counts(
            session, project_id, chunks_table,
        )

    counts = _tally_p3_counts(docs, chunk_counts)
    counts["project_id"] = project_id
    counts["chunks_table"] = chunks_table
    return counts


def p3_ghost_tombstone(
    *, project_id: str | None = None, execute: bool = False,
) -> dict[str, Any]:
    """Plan and (opt-in) apply ghost tombstones for one project.

    ``ghost_count`` is the dry-run rule-(c) count an operator confirms before
    passing ``execute=true``. Applying hides the row and sets ``TOMBSTONED``;
    it never deletes chunks or documents. Idempotent: a second execute hides
    nothing new.
    """
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )
    with SessionLocal() as session:
        docs, chunk_counts = _load_p3_docs_and_chunk_counts(
            session, project_id, chunks_table,
        )
    actions = plan_ghost_tombstones(docs, chunk_counts)
    applied = apply_ghost_tombstones(actions, execute=execute)
    return {
        "project_id": project_id,
        "chunks_table": chunks_table,
        "ghost_count": len(actions),
        "applied": applied,
        "dry_run": not execute,
    }


def plan_spreadsheet_chunk_purge(
    docs: Sequence[Mapping[str, Any]],
    chunk_counts: Mapping[str, int],
) -> list[ReconcileAction]:
    """Rows the spreadsheet rule selects: spreadsheet-extension documents
    still holding chunks in the chunk store.

    Pure planner: computes the same set as dry-run rule (a) ``with_chunks``,
    so the operator can confirm the count before the write path runs. A second
    pass selects nothing because a purged row falls back to zero chunks.
    """
    out: list[ReconcileAction] = []
    for doc in docs:
        did = str(doc.get("id") or "")
        if not did:
            continue
        chunks = int(
            chunk_counts.get(did, doc.get("chunk_count") or 0) or 0
        )
        if is_chunked_spreadsheet(doc, chunks):
            out.append(ReconcileAction(doc_id=did, reason="spreadsheet_chunks"))
    return out


def apply_spreadsheet_chunk_purge(
    actions: Sequence[ReconcileAction],
    *,
    execute: bool,
) -> int:
    """Remove spreadsheet chunks from the chunk store. Never deletes a document
    and never hides it: a spreadsheet stays a working document, it just leaves
    the RAG."""
    if not execute:
        return 0
    from app.core import projects as projects_mod

    applied = 0
    for action in actions:
        if projects_mod.strip_spreadsheet_chunks(action.doc_id):
            applied += 1
    return applied


def p3_spreadsheet_purge(
    *, project_id: str | None = None, execute: bool = False,
) -> dict[str, Any]:
    """Plan and (opt-in) apply the spreadsheet chunk purge for one project.

    ``spreadsheet_chunk_count`` is the dry-run rule-(a) ``with_chunks`` count an
    operator confirms before passing ``execute=true``. Applying deletes the
    spreadsheet's chunk rows and zeroes its ``chunk_count``; it never deletes
    the document row and never flips ``retrieval_visible``. Idempotent: a
    second execute removes nothing new.
    """
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    chunks_table = rag_chunk_table_name(
        os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip()
    )
    with SessionLocal() as session:
        docs, chunk_counts = _load_p3_docs_and_chunk_counts(
            session, project_id, chunks_table,
        )
    actions = plan_spreadsheet_chunk_purge(docs, chunk_counts)
    applied = apply_spreadsheet_chunk_purge(actions, execute=execute)
    return {
        "project_id": project_id,
        "chunks_table": chunks_table,
        "spreadsheet_chunk_count": len(actions),
        "applied": applied,
        "dry_run": not execute,
    }


def plan_as_dict(plan: ReconcilePlan) -> dict[str, Any]:
    def _act(a: ReconcileAction) -> dict[str, Any]:
        return {
            "doc_id": a.doc_id,
            "drive_file_id": a.drive_file_id,
            "reason": a.reason,
        }

    return {
        "to_index": [_act(a) for a in plan.to_index],
        "to_tombstone": [_act(a) for a in plan.to_tombstone],
        "to_reembed": [_act(a) for a in plan.to_reembed],
        "orphans": list(plan.orphans),
        "orphans_action": "report_only",
        "index_count": plan.index_count,
        "work_count": plan.work_count,
        "walk_complete": plan.walk_complete,
        "skipped_tombstones_reason": plan.skipped_tombstones_reason,
        "coverage": plan.coverage,
    }
