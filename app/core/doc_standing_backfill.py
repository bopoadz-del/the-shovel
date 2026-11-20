"""Plan and (opt-in) store the kind of standing on existing document rows.

New rows get ``metadata["standing"]`` at ingest (``projects.add_document``).
This applies the same rule (``app.core.doc_standing.standing_for`` over the
row's stored ``doc_type`` and ``doc_role``) to rows written before it existed.

ONE field is written: ``metadata["standing"]`` of the document row. No other
metadata key, no column, and no chunk is touched. The kind is derived, never
set by a user, so a row whose stored kind differs from the rule's (its type
or role was filled later by the doc-typing backfill) is planned too.

The plan is read-only: documents scanned, counts per kind as they would stand
after the write (``after``, including how many stay ``other``), every change
as ``(doc_id, old, new)`` and a ``plan_hash``. ``execute`` writes only when the
caller passes the live plan's hash (the plan is recomputed just before the
write), and each write re-reads the row and skips it, reported, when its
stored kind moved since the plan. A second execute plans nothing.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from app.core.doc_standing import METADATA_KEY, OTHER, standing_for, standing_table, stored_standing
from app.core.doc_typing_backfill import _metadata

#: What a never-decided row's kind reads as in a change.
UNSET = ""


def decide_changes(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, str]], Counter]:
    """The standing changes for these rows in doc_id order, and the count per
    kind every row would have after them. Each row carries ``id``,
    ``doc_type``, ``doc_role`` and ``metadata``."""
    table = standing_table()
    changes: list[dict[str, str]] = []
    after: Counter = Counter()
    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        new = standing_for(row.get("doc_type"), row.get("doc_role"), table)
        after[new] += 1
        old = stored_standing(_metadata(row.get("metadata"))) or UNSET
        if old != new:
            changes.append({"doc_id": str(row.get("id")), "old": old, "new": new})
    return changes, after


def _plan(project_id: str, scanned: int, changes: list[dict[str, str]], after: Counter) -> dict[str, Any]:
    digest = hashlib.sha256(json.dumps(
        {"project_id": project_id,
         "changes": [f"{c['doc_id']}:{c['old']}>{c['new']}" for c in changes]},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {
        "project_id": project_id,
        "field": f"metadata.{METADATA_KEY}",
        "documents_scanned": scanned,
        "after": dict(sorted(after.items())),
        "left_other": int(after.get(OTHER, 0)),
        "changes_by_kind": dict(sorted(Counter(c["new"] for c in changes).items())),
        "total": len(changes),
        "changes": changes,
        "plan_hash": digest,
    }


def plan_doc_standing(session, project_id: str) -> dict[str, Any]:
    """Read-only: every standing write for ``project_id``, hashed."""
    from sqlalchemy import select

    from app.core.models import Document

    rows = [
        {"id": d.id, "doc_type": d.doc_type, "doc_role": d.doc_role, "metadata": d.metadata_}
        for d in session.execute(
            select(Document).where(Document.project_id == project_id)
        ).scalars()
    ]
    changes, after = decide_changes(rows)
    return _plan(project_id, len(rows), changes, after)


def doc_standing_backfill(
    *, project_id: str, execute: bool = False, plan_hash: str | None = None,
) -> dict[str, Any]:
    """Plan, and with ``execute`` plus the live ``plan_hash``, store."""
    from app.core.db import SessionLocal
    from app.core.models import Document

    if not project_id:
        raise ValueError("project_id is required")
    with SessionLocal() as session:
        plan = plan_doc_standing(session, project_id)
    result: dict[str, Any] = {
        "plan": plan, "executed": False, "updated": 0, "skipped": [], "refused": None,
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
            doc = session.get(Document, c["doc_id"])
            meta = _metadata(doc.metadata_) if doc is not None else None
            if (doc is None or doc.project_id != project_id
                    or (stored_standing(meta) or UNSET) != c["old"]):
                skipped.append(c)
                continue
            fresh = dict(meta or {})
            fresh[METADATA_KEY] = c["new"]
            # A new dict, so the JSON column sees the change.
            doc.metadata_ = fresh
            updated += 1
        session.commit()
    result["executed"] = True
    result["updated"] = updated
    result["skipped"] = skipped
    return result
