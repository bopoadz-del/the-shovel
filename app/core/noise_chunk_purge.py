"""Plan and (opt-in) delete the noise chunks already in the index (RQ-1).

The chunkers now refuse noise at write time (``app.core.chunk_noise``). This
removes what the older writers left behind, classified by the same rules:

* ``guard_sentence`` -- starts with a platform guard sentence's lead-in;
* ``empty_row``      -- every line is a row of blank or zero cells;
* ``header_repeat``  -- every line is a label-only row (no figure) that an
  earlier chunk of the SAME document already stated.

The plan is read-only and lists every chunk id per class plus a
``plan_hash`` over them. ``execute`` deletes only when the caller passes the
live plan's hash -- the plan is recomputed from the store just before the
write, so a chunk that changed since the operator's plan changes the hash
and refuses -- and deletes only the planned ids. Documents are never deleted; each touched document's
``chunk_count`` and ingest status are re-derived afterwards (the same
re-stamp the legacy guard purge uses). A second execute plans nothing.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

from app.core import chunk_noise

CLASSES = ("guard_sentence", "empty_row", "header_repeat")


def classify_document_chunks(texts: Iterable[str]) -> list[str | None]:
    """Class of each chunk of one document, in ``chunk_index`` order.

    ``header_repeat`` needs the document's earlier chunks, so it is decided
    here and not per chunk: a chunk qualifies when it is only label rows and
    every one of those rows appeared earlier in the same document.
    """
    seen_rows: set[str] = set()
    out: list[str | None] = []
    for text in texts:
        text = text or ""
        reason = chunk_noise.refusal_reason(text)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        keys = [chunk_noise.row_key(ln) for ln in lines]
        if (
            reason is None
            and lines
            and all(chunk_noise.is_label_row(ln) for ln in lines)
            and all(k in seen_rows for k in keys)
        ):
            reason = "header_repeat"
        seen_rows.update(keys)
        out.append(reason)
    return out


def plan_noise_chunks(session, project_id: str) -> dict[str, Any]:
    """Read-only: every noise chunk of ``project_id`` by class, with a hash."""
    from sqlalchemy import text

    from app.core.ingest_reconcile import _guard_chunk_tables

    rows: list[dict[str, Any]] = []
    for table in _guard_chunk_tables(session):
        by_doc: dict[str, list[tuple[int, str, str]]] = {}
        for cid, did, idx, txt in session.execute(
            text(
                f"SELECT chunk_id, doc_id, chunk_index, text FROM {table} "
                "WHERE project_id = :pid"
            ),
            {"pid": project_id},
        ):
            by_doc.setdefault(str(did), []).append((int(idx), str(cid), str(txt or "")))
        for did, chunks in by_doc.items():
            chunks.sort()
            classes = classify_document_chunks(t for _, _, t in chunks)
            for (idx, cid, _), cls in zip(chunks, classes):
                if cls is not None:
                    rows.append({
                        "table": table, "chunk_id": cid, "doc_id": did,
                        "chunk_index": idx, "class": cls,
                    })
    rows.sort(key=lambda r: (r["table"], r["doc_id"], r["chunk_index"]))
    by_class = {c: [r["chunk_id"] for r in rows if r["class"] == c] for c in CLASSES}
    digest = hashlib.sha256(json.dumps(
        {"project_id": project_id,
         "rows": [f"{r['table']}:{r['chunk_id']}:{r['class']}" for r in rows]},
        sort_keys=True,
    ).encode("utf-8")).hexdigest()
    return {
        "project_id": project_id,
        "counts": {c: len(ids) for c, ids in by_class.items()},
        "total": len(rows),
        "chunk_ids": by_class,
        "documents": sorted({r["doc_id"] for r in rows}),
        "plan_hash": digest,
        "_rows": rows,
    }


def noise_chunk_purge(
    *, project_id: str, execute: bool = False, plan_hash: str | None = None,
) -> dict[str, Any]:
    """Plan, and with ``execute`` plus the live ``plan_hash``, delete."""
    from sqlalchemy import text

    from app.core.db import SessionLocal
    from app.core.ingest_reconcile import _restamp_after_purge

    if not project_id:
        raise ValueError("project_id is required")
    with SessionLocal() as session:
        plan = plan_noise_chunks(session, project_id)
    rows = plan.pop("_rows")
    result: dict[str, Any] = {
        "plan": plan, "executed": False, "deleted": 0, "restamped": [],
        "refused": None,
    }
    if not execute:
        return result
    if not plan_hash or plan_hash != plan["plan_hash"]:
        result["refused"] = "plan_hash does not match the live plan; nothing deleted"
        return result
    deleted = 0
    with SessionLocal() as session:
        for r in rows:
            deleted += int(session.execute(
                text(f"DELETE FROM {r['table']} WHERE chunk_id = :cid AND doc_id = :did"),
                {"cid": r["chunk_id"], "did": r["doc_id"]},
            ).rowcount or 0)
        session.commit()
    result["executed"] = True
    result["deleted"] = deleted
    result["restamped"] = _restamp_after_purge(r["doc_id"] for r in rows)
    return result
