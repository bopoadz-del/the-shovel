"""Re-split stored chunks that are longer than the embedder reads (RQ-4).

The embedding model reads at most its own ``max_seq_length`` tokens and drops
the rest of a passage without a warning, so the tail of every stored chunk
past that window has no vector. This re-splits such a chunk IN PLACE from its
own stored text (the source bytes of Drive-ingested documents are not kept)
and re-embeds the pieces.

* The bound is the loaded embedder's window, measured with its tokenizer at
  run time (``Embedder.token_window``). A character count is used only as a
  pre-screen: a token is at least one character, so a text of at most
  ``window`` characters cannot exceed ``window`` tokens.
* Pure partition: the pieces of a parent, concatenated in order, equal the
  parent text exactly. A split falls at the coarsest boundary that fits --
  paragraph, line, sentence, word -- and inside a word only when one
  whitespace-free run is itself longer than the window.
* Nothing is prepended to a piece. Each piece keeps every column of its
  parent (doc_id, page, knowledge layer, authority, ocr, embedding identity)
  except its id, index, text and embedding. The first piece keeps the parent
  id.
* Scope is an explicit list of document ids; there is no whole-project run.
* The PLAN is read-only and reports, per chunk table and per doc_type,
  parents over the window, pieces to be written and under-floor chunks
  (counted only), an embedding-time estimate measured on a sample, and a
  ``plan_hash``.
* The EXECUTE takes ``expected_plan_hash`` and refuses on a mismatch. It
  writes the active chunk table only (the legacy table is retired and is
  only counted). One document = one transaction: the pieces are embedded
  first, then the document's rows are re-read under lock and their text hash
  compared with the plan's (compare-and-set); a changed document is skipped
  and any failure rolls the document back untouched.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

# Coarsest first. Each pattern matches a SEPARATOR; a unit ends right after
# its separator, so the units of a text concatenate back to the text.
_BOUNDARIES = (
    re.compile(r"\n[ \t]*\n\s*"),      # paragraph
    re.compile(r"\n"),                  # line
    re.compile(r"(?<=[.!?;:])\s+"),     # sentence end (never "3.5": no space)
    re.compile(r"\s+"),                 # word
)


def _units(text: str, boundary: re.Pattern) -> list[str]:
    cuts = [m.end() for m in boundary.finditer(text) if 0 < m.end() < len(text)]
    out, start = [], 0
    for cut in cuts:
        if cut > start:
            out.append(text[start:cut])
            start = cut
    out.append(text[start:])
    merged: list[str] = []
    for unit in out:
        if merged and not unit.strip():
            merged[-1] += unit  # whitespace stays with the unit it follows
        elif unit:
            merged.append(unit)
    return merged


def _hard_cut(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Longest fitting prefixes of one run that has no boundary at all."""
    out: list[str] = []
    rest = text
    while rest and not fits(rest):
        lo, hi = 1, len(rest)
        while lo < hi:  # largest n with fits(rest[:n]); n >= 1 always
            mid = (lo + hi + 1) // 2
            if fits(rest[:mid]):
                lo = mid
            else:
                hi = mid - 1
        out.append(rest[:lo])
        rest = rest[lo:]
    if rest:
        out.append(rest)
    return out


def _split(text: str, fits: Callable[[str], bool], level: int) -> list[str]:
    if level >= len(_BOUNDARIES):
        return _hard_cut(text, fits)  # returns [text] when it fits
    if fits(text):
        return [text]
    units = _units(text, _BOUNDARIES[level])
    if len(units) == 1:
        return _split(text, fits, level + 1)
    pieces: list[str] = []
    cur = ""
    for unit in units:
        if cur and fits(cur + unit):
            cur += unit
            continue
        if cur:
            pieces.append(cur)
            cur = ""
        if fits(unit):
            cur = unit
        else:
            pieces.extend(_split(unit, fits, level + 1))
    if cur:
        pieces.append(cur)
    return pieces


def partition(text: str, count_tokens: Callable[[str], int], window: int) -> list[str]:
    """Pieces of ``text`` that each count at most ``window`` tokens.

    ``"".join(partition(t, ...)) == t`` for every ``t``.
    """
    if window <= 0:
        raise ValueError("window must be positive")

    def fits(s: str) -> bool:
        return len(s) <= window or count_tokens(s) <= window

    return _split(text or "", fits, 0)


def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def doc_rows_hash(rows: Iterable[tuple[str, int, str]]) -> str:
    """Compare-and-set token over one document's ``(chunk_id, index, text)``."""
    body = [f"{cid}:{idx}:{_sha(txt)}" for cid, idx, txt in sorted(rows, key=lambda r: r[1])]
    return _sha(json.dumps(body))


def _active_table() -> str:
    from app.core.models import rag_chunk_table_name
    from app.core.rag.vector_store import _rag_vector_namespace

    return rag_chunk_table_name(_rag_vector_namespace())


def embedder_window() -> tuple[Callable[[str], int], int, str] | None:
    """``(count_tokens, window, model)`` of the configured embedder, or None."""
    from app.core.rag.embeddings import get_embedder

    emb = get_embedder()
    tw = emb.token_window()
    if tw is None:
        return None
    return tw[0], tw[1], emb.model_name


def _clean_ids(doc_ids: Iterable[str]) -> list[str]:
    return sorted({str(d).strip() for d in doc_ids if str(d).strip()})


def plan_resplit(
    session, project_id: str, doc_ids: Iterable[str],
    count_tokens: Callable[[str], int], window: int, model: str,
) -> dict[str, Any]:
    """Read-only plan over ``doc_ids`` of ``project_id``.

    Returns the public plan plus ``_work``: per active-table document, its
    rows hash and the pieces of each over-window parent.
    """
    from sqlalchemy import bindparam, text

    from app.core.chunk_bounds import chunk_min_chars
    from app.core.ingest_reconcile import _guard_chunk_tables

    ids = _clean_ids(doc_ids)
    if not ids:
        raise ValueError("doc_ids is required: the run takes an explicit document list")
    doc_types = {
        str(did): str(dt or "document") for did, dt in session.execute(
            text(
                "SELECT id, doc_type FROM documents "
                "WHERE project_id = :pid AND id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"pid": project_id, "ids": ids},
        )
    }
    floor = chunk_min_chars()
    active = _active_table()
    tables: dict[str, Any] = {}
    work: dict[str, dict[str, Any]] = {}
    hash_rows: list[str] = []
    for table in _guard_chunk_tables(session):
        per_type: dict[str, dict[str, int]] = {}
        by_doc: dict[str, list[tuple[str, int, str]]] = {}
        for cid, did, idx, txt in session.execute(
            text(
                f"SELECT chunk_id, doc_id, chunk_index, text FROM {table} "
                "WHERE project_id = :pid AND doc_id IN :ids"
            ).bindparams(bindparam("ids", expanding=True)),
            {"pid": project_id, "ids": ids},
        ):
            by_doc.setdefault(str(did), []).append((str(cid), int(idx), str(txt or "")))
        for did in sorted(by_doc):
            rows = sorted(by_doc[did], key=lambda r: r[1])
            dtype = doc_types.get(did, "document")
            c = per_type.setdefault(dtype, {
                "documents": 0, "chunks": 0, "parents_over_window": 0,
                "pieces_to_write": 0, "under_floor": 0,
            })
            c["documents"] += 1
            parents: dict[str, list[str]] = {}
            for cid, idx, txt in rows:
                c["chunks"] += 1
                if floor and len(txt) < floor:
                    c["under_floor"] += 1
                pieces = partition(txt, count_tokens, window)
                if len(pieces) > 1:
                    c["parents_over_window"] += 1
                    c["pieces_to_write"] += len(pieces)
                    parents[cid] = pieces
                    hash_rows.append(f"{table}:{did}:{cid}:{_sha(txt)}:{len(pieces)}")
            if table == active and parents:
                work[did] = {
                    "doc_type": dtype,
                    "rows_hash": doc_rows_hash(rows),
                    "parents": parents,
                }
        tables[table] = {
            "executes": table == active,
            "by_doc_type": per_type,
            "parents_over_window": sum(v["parents_over_window"] for v in per_type.values()),
            "pieces_to_write": sum(v["pieces_to_write"] for v in per_type.values()),
            "under_floor": sum(v["under_floor"] for v in per_type.values()),
        }
    digest = _sha(json.dumps({
        "project_id": project_id, "doc_ids": ids, "model": model,
        "window": window, "rows": sorted(hash_rows),
    }, sort_keys=True))
    return {
        "project_id": project_id,
        "doc_ids_requested": len(ids),
        "doc_ids_found": len(doc_types),
        "doc_ids_missing": [d for d in ids if d not in doc_types],
        "model": model,
        "window_tokens": window,
        "active_table": active,
        "tables": tables,
        "plan_hash": digest,
        "executed": False,
        "_work": work,
    }


def _estimate_seconds(work: dict[str, Any], sample: int, encode) -> dict[str, Any]:
    pieces = [p for w in work.values() for ps in w["parents"].values() for p in ps]
    if not pieces or sample <= 0:
        return {"sample": 0, "seconds_per_piece": None, "estimated_seconds": 0.0 if not pieces else None}
    batch = pieces[:sample]
    t0 = time.perf_counter()
    encode(batch)
    per = (time.perf_counter() - t0) / len(batch)
    return {
        "sample": len(batch),
        "seconds_per_piece": round(per, 6),
        "estimated_seconds": round(per * len(pieces), 1),
    }


def _replace_document(session, cls, project_id: str, did: str, item: dict, vectors: dict) -> str:
    """One document under lock: compare-and-set, then rewrite its rows."""
    from sqlalchemy import select

    rows = session.execute(
        select(cls).where(cls.project_id == project_id, cls.doc_id == did)
        .order_by(cls.chunk_index).with_for_update()
    ).scalars().all()
    if doc_rows_hash((r.chunk_id, r.chunk_index, r.text) for r in rows) != item["rows_hash"]:
        return "changed"
    cols = [c.key for c in cls.__mapper__.column_attrs]
    new_rows: list[dict[str, Any]] = []
    for r in rows:
        base = {k: getattr(r, k) for k in cols}
        pieces = item["parents"].get(r.chunk_id)
        if not pieces:
            new_rows.append(base)
            continue
        for k, (piece, vec) in enumerate(zip(pieces, vectors[r.chunk_id])):
            row = dict(base)
            row["chunk_id"] = r.chunk_id if k == 0 else f"{r.chunk_id}#{k}"
            row["text"] = piece
            row["embedding"] = vec
            new_rows.append(row)
    for r in rows:
        session.delete(r)
    session.flush()
    for i, row in enumerate(new_rows):
        row["chunk_index"] = i
        session.add(cls(**row))
    session.flush()
    return "replaced"


def chunk_resplit(
    *, project_id: str, doc_ids: Iterable[str], execute: bool = False,
    expected_plan_hash: str | None = None, doc_type: str | None = None,
    sample: int = 50,
) -> dict[str, Any]:
    """Plan, and with ``execute`` plus the live plan hash, re-split in place."""
    from sqlalchemy.exc import SQLAlchemyError

    from app.core.db import SessionLocal
    from app.core.rag.embeddings import get_embedder

    if not project_id:
        raise ValueError("project_id is required")
    tw = embedder_window()
    if tw is None:
        return {"refused": "the embedder exposes no tokenizer window; nothing planned",
                "executed": False}
    count_tokens, window, model = tw
    with SessionLocal() as session:
        plan = plan_resplit(session, project_id, doc_ids, count_tokens, window, model)
    work = plan.pop("_work")
    embedder = get_embedder()
    plan["embedding_time"] = _estimate_seconds(work, sample, embedder.encode)
    result: dict[str, Any] = {"plan": plan, "executed": False, "refused": None}
    if not execute:
        return result
    if not expected_plan_hash or expected_plan_hash != plan["plan_hash"]:
        result["refused"] = "expected_plan_hash does not match the live plan; nothing written"
        return result
    from app.core.ingest_reconcile import _restamp_after_purge
    from app.core.models import rag_chunk_class_for
    from app.core.rag.vector_store import _rag_vector_namespace

    cls = rag_chunk_class_for(_rag_vector_namespace())
    if cls is None:
        result["refused"] = "active chunk table is not mapped in this process; nothing written"
        return result
    outcome: dict[str, str] = {}
    pieces_written = 0
    for did, item in sorted(work.items()):
        if doc_type and item["doc_type"] != doc_type:
            continue
        flat = [(cid, p) for cid, ps in item["parents"].items() for p in ps]
        vecs = embedder.encode([p for _, p in flat])
        vectors: dict[str, list] = {}
        for (cid, _), v in zip(flat, vecs):
            vectors.setdefault(cid, []).append(v)
        with SessionLocal() as session:
            try:
                state = _replace_document(session, cls, project_id, did, item, vectors)
                if state == "replaced":
                    session.commit()
                    pieces_written += len(flat)
                else:
                    session.rollback()
            except SQLAlchemyError as exc:  # the document stays as it was
                session.rollback()
                state = f"failed: {type(exc).__name__}"
        outcome[did] = state
    result["executed"] = True
    result["documents"] = outcome
    result["pieces_written"] = pieces_written
    result["restamped"] = _restamp_after_purge(
        d for d, s in outcome.items() if s == "replaced"
    )
    return result
