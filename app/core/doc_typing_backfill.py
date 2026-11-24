"""Plan and (opt-in) apply doc_type / doc_role to existing document rows (RQ-3).

New rows are typed at ingest (``app.core.projects.add_document``). This
applies the same two decisions to rows written before the tables existed:

* ``doc_type`` -- ``projects.classify_doc_type`` over the stored name (the
  type table), as ingest calls it, plus the opening text of the document's
  own indexed chunks. Ingest types a row before its text exists, so a name
  that carries no type word left the row ``document`` even when the text
  says plainly what it is; the backfill is where the content can be read;
* ``doc_role`` -- ``projects.resolve_doc_role`` over the stored name and
  metadata source path (readiness role from the name, then the folder-name
  table), exactly as ingest calls it.

Only a column that still holds its default is filled: a row whose type is not
``document`` or whose role is not ``other`` was set by someone (a caller, a
user confirmation) and is left as it is. The plan is read-only: counts per
new type and per new role, every change as ``(doc_id, column, old, new)``,
and a ``plan_hash`` over the changes. ``execute`` writes only when the caller
passes the live plan's hash (the plan is recomputed just before the write)
and each UPDATE is compare-and-set on the old value, so a row that moved
since the plan is skipped and reported, never overwritten. A second execute
plans nothing.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

#: Characters of a document's opening chunks read as its content sample.
CONTENT_SAMPLE_CHARS = int(os.getenv("DOC_TYPING_CONTENT_CHARS", "4000") or 4000)

#: Opening chunks per document read for the sample (bounds the query).
CONTENT_SAMPLE_CHUNKS = int(os.getenv("DOC_TYPING_CONTENT_CHUNKS", "4") or 4)

#: Columns this backfill may fill, with the default value that marks a row
#: as never decided.
COLUMNS = ("doc_type", "doc_role")

logger = logging.getLogger(__name__)


def _metadata(raw: Any) -> Mapping[str, Any] | None:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        try:
            parsed = json.loads(raw)
        except ValueError:
            logger.warning("doc typing backfill: stored metadata is not JSON; no source path read")
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return None


def decide_changes(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """The column changes for these document rows, in doc_id order.

    Each row carries ``id``, ``original_name``, ``doc_type``, ``doc_role`` and
    ``metadata``, and may carry ``content_sample`` (the document's own text). A column is changed only when it holds its default and the
    ingest rule decides something else.
    """
    from app.core.doc_typing import DEFAULT_DOC_TYPE
    from app.core.projects import ROLE_OTHER, classify_doc_type, resolve_doc_role

    defaults = {"doc_type": DEFAULT_DOC_TYPE, "doc_role": ROLE_OTHER}
    out: list[dict[str, str]] = []
    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        did = str(row.get("id"))
        name = str(row.get("original_name") or "")
        decided = {
            "doc_type": classify_doc_type(name, str(row.get("content_sample") or "")),
            "doc_role": resolve_doc_role(name, _metadata(row.get("metadata"))),
        }
        for col in COLUMNS:
            old = str(row.get(col) or defaults[col])
            new = decided[col]
            if old == defaults[col] and new != old:
                out.append({"doc_id": did, "column": col, "old": old, "new": new})
    return out


def _plan_from_changes(project_id: str, scanned: int, changes: list[dict[str, str]]) -> dict[str, Any]:
    counts = {
        col: dict(sorted(Counter(c["new"] for c in changes if c["column"] == col).items()))
        for col in COLUMNS
    }
    digest = hashlib.sha256(json.dumps(
        {"project_id": project_id,
         "changes": [f"{c['doc_id']}:{c['column']}:{c['old']}>{c['new']}" for c in changes]},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {
        "project_id": project_id,
        "documents_scanned": scanned,
        "counts": {"doc_type": counts["doc_type"], "doc_role": counts["doc_role"]},
        "total": len(changes),
        "documents": sorted({c["doc_id"] for c in changes}),
        "changes": changes,
        "plan_hash": digest,
    }


def content_samples(session, project_id: str, limit: int | None = None) -> dict[str, str]:
    """doc_id -> the opening text of its indexed chunks, in chunk order, cut
    at ``limit`` characters. Read-only; a document with no chunks is absent."""
    from sqlalchemy import inspect, select

    from app.core.models import RagChunk

    if not inspect(session.get_bind()).has_table(RagChunk.__tablename__):
        # A database that has never indexed anything (a fresh SQLite) has no
        # chunks table: no document has text yet, so the name alone decides.
        return {}
    return _opening_texts(session.execute(
        select(RagChunk.doc_id, RagChunk.text)
        .where(RagChunk.project_id == project_id,
               RagChunk.chunk_index < CONTENT_SAMPLE_CHUNKS)
        .order_by(RagChunk.doc_id, RagChunk.chunk_index)
    ), CONTENT_SAMPLE_CHARS if limit is None else limit)


def opening_samples(
    session, project_ids: Iterable[str], doc_ids: Iterable[str], limit: int | None = None,
) -> dict[str, str]:
    """doc_id -> the opening text of its indexed chunks, for these documents
    in these projects only. The same sample ``content_samples`` reads, so a
    document's own opening says the same thing to every reader. The project
    filter leads, so the read uses the chunks table's project index.
    Read-only; one query; a document with no chunks is absent."""
    from sqlalchemy import inspect, select

    from app.core.models import RagChunk

    ids = [str(d) for d in dict.fromkeys(doc_ids or []) if d]
    pids = [str(p) for p in dict.fromkeys(project_ids or []) if p]
    if not ids or not pids or not inspect(session.get_bind()).has_table(RagChunk.__tablename__):
        return {}
    return _opening_texts(session.execute(
        select(RagChunk.doc_id, RagChunk.text)
        .where(RagChunk.project_id.in_(pids),
               RagChunk.doc_id.in_(ids),
               RagChunk.chunk_index < CONTENT_SAMPLE_CHUNKS)
        .order_by(RagChunk.doc_id, RagChunk.chunk_index)
    ), CONTENT_SAMPLE_CHARS if limit is None else limit)


def _opening_texts(rows: Iterable[Any], cap: int) -> dict[str, str]:
    out: dict[str, str] = {}
    for doc_id, text in rows:
        have = out.get(doc_id, "")
        if len(have) >= cap:
            continue
        out[doc_id] = (have + "\n" + (text or "") if have else (text or ""))[:cap]
    return out


def plan_doc_typing(session, project_id: str) -> dict[str, Any]:
    """Read-only: every doc_type / doc_role fill for ``project_id``, hashed."""
    from sqlalchemy import select

    from app.core.models import Document

    rows = [
        {"id": d.id, "original_name": d.original_name, "doc_type": d.doc_type,
         "doc_role": d.doc_role, "metadata": d.metadata_}
        for d in session.execute(
            select(Document).where(Document.project_id == project_id)
        ).scalars()
    ]
    samples = content_samples(session, project_id)
    for r in rows:
        r["content_sample"] = samples.get(str(r["id"]), "")
    return _plan_from_changes(project_id, len(rows), decide_changes(rows))


def doc_typing_backfill(
    *, project_id: str, execute: bool = False, plan_hash: str | None = None,
) -> dict[str, Any]:
    """Plan, and with ``execute`` plus the live ``plan_hash``, apply."""
    from sqlalchemy import update

    from app.core.db import SessionLocal
    from app.core.models import Document

    if not project_id:
        raise ValueError("project_id is required")
    with SessionLocal() as session:
        plan = plan_doc_typing(session, project_id)
    result: dict[str, Any] = {
        "plan": plan, "executed": False, "updated": 0, "skipped": [],
        "refused": None,
    }
    if not execute:
        return result
    if not plan_hash or plan_hash != plan["plan_hash"]:
        result["refused"] = "plan_hash does not match the live plan; nothing written"
        return result
    updated = 0
    skipped: list[dict[str, str]] = []
    with SessionLocal() as session:
        for c in plan["changes"]:
            col = getattr(Document, c["column"])
            n = int(session.execute(
                update(Document)
                .where(Document.id == c["doc_id"], Document.project_id == project_id, col == c["old"])
                .values({c["column"]: c["new"]})
            ).rowcount or 0)
            if n:
                updated += n
            else:
                skipped.append(c)
        session.commit()
    result["executed"] = True
    result["updated"] = updated
    result["skipped"] = skipped
    return result
