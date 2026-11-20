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
  only counted). Memory and time per call are flat (RQ-4b): pieces are
  embedded a bounded batch at a time into a staging table, at most
  ``max_parents`` parents per call, and the call returns a continuation. Once
  the whole slice is staged, one document = one transaction: the document's
  rows are re-read under lock and their text hash compared with the plan's
  (compare-and-set); a changed document is skipped and any failure rolls the
  document back untouched.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
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


# ── EXECUTE: flat in memory and time (RQ-4b) ───────────────────────────────
#
# Pieces are embedded a bounded batch at a time and every batch's vectors are
# written to a staging table and committed before the next batch is encoded,
# so peak memory is one batch whatever the document size. A call stages at
# most ``max_parents`` parents and returns a continuation (what remains, per
# document). Staging never touches the chunk table, so the plan hash stays
# the same across the calls of one slice. Only when nothing in the slice is
# left unstaged does a call replace documents: one transaction per document,
# compare-and-set on its rows, the vectors read back from staging one parent
# at a time.
#
# Staging is in the database, not in process memory: a continuation must
# survive between HTTP calls, and the service can run more than one task or
# be replaced between two calls.


def _env_positive(name: str, default: int) -> int:
    import os

    try:
        n = int(os.getenv(name, str(default)))
    except ValueError:
        n = default
    return max(1, n)


def resplit_embed_batch() -> int:
    """Pieces per ``encode`` call (env ``RESPLIT_EMBED_BATCH``, default 16)."""
    return _env_positive("RESPLIT_EMBED_BATCH", 16)


def resplit_max_parents() -> int:
    """Parents staged per EXECUTE call (env ``RESPLIT_MAX_PARENTS``, default 100)."""
    return _env_positive("RESPLIT_MAX_PARENTS", 100)


_STAGE_META = None


def _stage_table():
    """``chunk_resplit_stage``: one row per piece vector, keyed to the rows hash."""
    global _STAGE_META
    from sqlalchemy import Column, Integer, LargeBinary, MetaData, String, Table

    from app.core.db import engine
    from app.core.rag.vector_store import ensure_table

    if _STAGE_META is None:
        _STAGE_META = MetaData()
        Table(
            "chunk_resplit_stage", _STAGE_META,
            Column("project_id", String, primary_key=True),
            Column("doc_id", String, primary_key=True),
            Column("parent_id", String, primary_key=True),
            Column("piece_no", Integer, primary_key=True),
            Column("rows_hash", String, nullable=False),
            Column("vector", LargeBinary, nullable=False),
        )
    stage = _STAGE_META.tables["chunk_resplit_stage"]
    ensure_table(stage, bind=engine)
    return stage


def _staged_keys(session, stage, project_id: str, did: str, rows_hash: str) -> set:
    """``(parent_id, piece_no)`` staged for this document state; stale rows dropped."""
    from sqlalchemy import delete, select

    session.execute(delete(stage).where(
        stage.c.project_id == project_id, stage.c.doc_id == did,
        stage.c.rows_hash != rows_hash,
    ))
    keys = {(str(p), int(n)) for p, n in session.execute(
        select(stage.c.parent_id, stage.c.piece_no).where(
            stage.c.project_id == project_id, stage.c.doc_id == did,
        )
    )}
    session.commit()
    return keys


def staged_vectors(session, stage, project_id: str, did: str):
    """``get(parent_id)`` -> that parent's piece vectors, read from staging."""
    import numpy as np
    from sqlalchemy import select

    def get(parent_id: str) -> list:
        return [np.frombuffer(v, dtype=np.float32).copy() for (v,) in session.execute(
            select(stage.c.vector).where(
                stage.c.project_id == project_id, stage.c.doc_id == did,
                stage.c.parent_id == parent_id,
            ).order_by(stage.c.piece_no)
        )]

    return get


def _stage_pieces(stage, project_id: str, todo: list, encode, batch: int) -> int:
    """Encode ``todo`` = ``[(did, rows_hash, parent_id, piece_no, text)]`` in
    batches of ``batch``; each batch is written and committed before the next
    is encoded. Returns the number of ``encode`` calls."""
    import numpy as np

    from app.core.db import SessionLocal

    calls = 0
    for start in range(0, len(todo), batch):
        part = todo[start:start + batch]
        vecs = encode([t[4] for t in part])
        calls += 1
        with SessionLocal() as session:
            session.execute(stage.insert(), [
                {"project_id": project_id, "doc_id": did, "parent_id": cid,
                 "piece_no": k, "rows_hash": rh,
                 "vector": np.asarray(v, dtype=np.float32).tobytes()}
                for (did, rh, cid, k, _), v in zip(part, vecs)
            ])
            session.commit()
    return calls


def _replace_document(session, cls, project_id: str, did: str, item: dict, vectors) -> str:
    """One document under lock: compare-and-set, then rewrite its rows.

    ``vectors(parent_id)`` gives a parent's piece vectors in order. Rows are
    rewritten one parent at a time (flushed, then the session is cleared), so
    the transaction never holds the whole document's rows or vectors.
    """
    from sqlalchemy import select, update

    rows = session.execute(
        select(cls.chunk_id, cls.chunk_index, cls.text)
        .where(cls.project_id == project_id, cls.doc_id == did)
        .order_by(cls.chunk_index).with_for_update()
    ).all()
    if doc_rows_hash((r[0], r[1], r[2]) for r in rows) != item["rows_hash"]:
        return "changed"
    parents = item["parents"]
    total = sum(len(parents.get(r[0]) or [r[0]]) for r in rows)
    shift = max((r[1] for r in rows), default=0) + total + 1
    # Move every row clear of the final index range first, so renumbering
    # never collides with the (project_id, doc_id, chunk_index) unique key.
    session.execute(
        update(cls).where(cls.project_id == project_id, cls.doc_id == did)
        .values(chunk_index=cls.chunk_index + shift)
        .execution_options(synchronize_session=False)
    )
    cols = [c.key for c in cls.__mapper__.column_attrs]
    nxt = 0
    for cid, _, _ in rows:
        pieces = parents.get(cid)
        if not pieces:
            session.execute(
                update(cls).where(cls.chunk_id == cid).values(chunk_index=nxt)
                .execution_options(synchronize_session=False)
            )
            nxt += 1
            continue
        vecs = vectors(cid)
        if len(vecs) != len(pieces):
            raise ValueError(f"staged vectors for a parent: {len(vecs)} != {len(pieces)} pieces")
        obj = session.get(cls, cid)
        base = {k: getattr(obj, k) for k in cols}
        obj.text, obj.embedding, obj.chunk_index = pieces[0], vecs[0], nxt
        for k in range(1, len(pieces)):
            row = dict(base)
            row.update(chunk_id=f"{cid}#{k}", text=pieces[k], embedding=vecs[k],
                       chunk_index=nxt + k)
            session.add(cls(**row))
        nxt += len(pieces)
        session.flush()
        session.expunge_all()
    return "replaced"


# One EXECUTE at a time in this process: a second call is refused, not queued.
_EXECUTE_LOCK = threading.Lock()

_NO_WINDOW = "the embedder exposes no tokenizer window; nothing planned"


def chunk_resplit(
    *, project_id: str, doc_ids: Iterable[str], execute: bool = False,
    expected_plan_hash: str | None = None, doc_type: str | None = None,
    sample: int = 50, max_parents: int | None = None,
) -> dict[str, Any]:
    """Plan, and with ``execute`` plus the live plan hash, re-split in place.

    An EXECUTE stages at most ``max_parents`` parents per call and returns
    ``continuation``; call again with the same plan hash until
    ``continuation.parents_remaining`` is 0. A call made while another
    EXECUTE is running in this process is refused.
    """
    if not project_id:
        raise ValueError("project_id is required")
    if not execute:
        plan, _, _ = _plan_and_work(project_id, doc_ids, sample)
        if plan is None:
            return {"refused": _NO_WINDOW, "executed": False}
        return {"plan": plan, "executed": False, "refused": None}
    if not _EXECUTE_LOCK.acquire(blocking=False):
        return {"plan": None, "executed": False,
                "refused": "another chunk-resplit EXECUTE is in flight; nothing written"}
    try:
        cap = resplit_max_parents() if max_parents is None else int(max_parents)
        if cap <= 0:
            raise ValueError("max_parents must be positive")
        return _execute(project_id, doc_ids, expected_plan_hash, doc_type, sample, cap)
    finally:
        _EXECUTE_LOCK.release()


def _plan_and_work(project_id: str, doc_ids: Iterable[str], sample: int):
    from app.core.db import SessionLocal
    from app.core.rag.embeddings import get_embedder

    tw = embedder_window()
    if tw is None:
        return None, None, None
    count_tokens, window, model = tw
    with SessionLocal() as session:
        plan = plan_resplit(session, project_id, doc_ids, count_tokens, window, model)
    work = plan.pop("_work")
    embedder = get_embedder()
    plan["embedding_time"] = _estimate_seconds(work, sample, embedder.encode)
    return plan, work, embedder


def _execute(project_id, doc_ids, expected_plan_hash, doc_type, sample, max_parents):
    from sqlalchemy import delete
    from sqlalchemy.exc import SQLAlchemyError

    from app.core.db import SessionLocal

    plan, work, embedder = _plan_and_work(project_id, doc_ids, sample)
    if plan is None:
        return {"refused": _NO_WINDOW, "executed": False}
    result: dict[str, Any] = {"plan": plan, "executed": False, "refused": None}
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
    stage = _stage_table()
    in_slice = {d: w for d, w in sorted(work.items())
                if not doc_type or w["doc_type"] == doc_type}

    # 1. Stage the first ``max_parents`` parents that are not fully staged yet.
    todo: list[tuple] = []
    remaining: dict[str, int] = {}
    budget = max_parents
    for did, item in in_slice.items():
        with SessionLocal() as session:
            have = _staged_keys(session, stage, project_id, did, item["rows_hash"])
        left = 0
        for cid, pieces in sorted(item["parents"].items()):
            missing = [k for k in range(len(pieces)) if (cid, k) not in have]
            if not missing:
                continue
            if budget <= 0:
                left += 1
                continue
            budget -= 1
            todo.extend((did, item["rows_hash"], cid, k, pieces[k]) for k in missing)
        remaining[did] = left
    encode_calls = _stage_pieces(stage, project_id, todo, embedder.encode,
                                 resplit_embed_batch())
    parents_remaining = sum(remaining.values())
    result.update({
        "executed": True,
        "staged": {"parents": max_parents - budget, "pieces": len(todo),
                   "encode_calls": encode_calls},
        "continuation": {
            "parents_remaining": parents_remaining,
            "documents_remaining": {d: n for d, n in remaining.items() if n},
            "plan_hash": plan["plan_hash"],
        },
        "documents": {}, "pieces_written": 0, "restamped": [],
    })
    if parents_remaining:
        return result

    # 2. Every parent of the slice is staged: one transaction per document.
    outcome: dict[str, str] = {}
    pieces_written = 0
    for did, item in in_slice.items():
        with SessionLocal() as session:
            try:
                state = _replace_document(session, cls, project_id, did, item,
                                          staged_vectors(session, stage, project_id, did))
                if state == "replaced":
                    session.execute(delete(stage).where(
                        stage.c.project_id == project_id, stage.c.doc_id == did))
                    session.commit()
                    pieces_written += sum(len(p) for p in item["parents"].values())
                else:
                    session.rollback()
            except (SQLAlchemyError, ValueError) as exc:  # the document stays as it was
                session.rollback()
                state = f"failed: {type(exc).__name__}"
        outcome[did] = state
    result["documents"] = outcome
    result["pieces_written"] = pieces_written
    result["restamped"] = _restamp_after_purge(
        d for d, s in outcome.items() if s == "replaced"
    )
    return result
