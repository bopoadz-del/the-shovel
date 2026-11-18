#!/usr/bin/env python3
"""Q-3: count the documents of one project that are structured like an RFI,
and how the index splits them today. Read-only, counts and ids only.

A document is RFI-like when its own text carries the three blocks of a
request-for-information form, each found as a HEADING (a label at the start
of a line, ending the line or followed by ``:``), in this order:

* a question block (the request),
* an answer block AFTER the first question heading (the reply),
* a references block anywhere (what the request points at).

The heading words are a table (``DEFAULT_HEADINGS``) that an operator can
replace with ``RFI_HEADINGS_JSON`` (``{"question": [...], "answer": [...],
"references": [...]}``); no document id, name or expected count lives here.

Per RFI-like document the index's own chunks (in ``chunk_index`` order) are
read and the run reports: chunks per document, and whether the first answer
heading lands in a different chunk from the question heading before it (the
question and its answer are SPLIT). Nothing is chunked, re-split, embedded
or written: the SQL is parameterised SELECTs. The project is the anchor
document's own (read from its row, never taken from an input or printed).

The output is a public workflow log: document ids, counts and chunk indexes
only; never text, a heading, a name, a path or a project id.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterable, Iterator
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

# Generic headings of an RFI form (the operator may replace the table).
DEFAULT_HEADINGS: dict[str, tuple[str, ...]] = {
    "question": ("question", "query", "request", "information requested",
                 "description of request", "description of query", "enquiry"),
    "answer": ("answer", "response", "reply", "engineer's response",
               "consultant's response", "designer's response"),
    "references": ("reference", "references", "ref", "refs",
                   "drawing reference", "specification reference", "attachments"),
}
ROLES = ("question", "answer", "references")
_MAX_IDS = 200
_ID_RE = re.compile(r"^[0-9A-Za-z_-]+$")
MARK = "RFI| "

_DOC_SQL = "SELECT id, project_id FROM documents WHERE id LIKE :p ORDER BY id"


def headings(raw: str | None = None) -> dict[str, tuple[str, ...]]:
    """The heading table: ``RFI_HEADINGS_JSON`` when set (every role must be
    a non-empty list of strings), else the default. Raises ValueError on a
    malformed table, so a typo never silently counts zero."""
    raw = os.environ.get("RFI_HEADINGS_JSON", "") if raw is None else raw
    if not raw.strip():
        return DEFAULT_HEADINGS
    data = json.loads(raw)
    out: dict[str, tuple[str, ...]] = {}
    for role in ROLES:
        words = data.get(role) if isinstance(data, dict) else None
        if not isinstance(words, list) or not words or not all(
                isinstance(w, str) and w.strip() for w in words):
            raise ValueError(f"RFI_HEADINGS_JSON: role {role} needs a non-empty list of words")
        out[role] = tuple(w.strip().lower() for w in words)
    return out


def _heading_re(words: Iterable[str]) -> re.Pattern[str]:
    # A label at the start of a line (after optional numbering or bullets),
    # then ':' / '-' or the end of the line. Longest first so a two-word
    # heading wins over its first word.
    alt = "|".join(re.escape(w).replace(r"\ ", r"\s+")
                   for w in sorted(set(words), key=len, reverse=True))
    return re.compile(rf"(?im)^[ \t>*\-\d.()]*(?:{alt})\b[ \t]*(?:\(s\))?[ \t]*(?::|-|$)")


def compile_headings(table: dict[str, tuple[str, ...]]) -> dict[str, re.Pattern[str]]:
    return {role: _heading_re(table[role]) for role in ROLES}


def heading_hits(chunk_text: str, pats: dict[str, re.Pattern[str]]) -> dict[str, list[int]]:
    """Offsets of each role's headings inside one chunk."""
    return {role: [m.start() for m in pats[role].finditer(chunk_text or "")] for role in ROLES}


def classify(chunks: list[tuple[int, str]], pats: dict[str, re.Pattern[str]]) -> dict[str, Any]:
    """Pure: one document's chunks ``(chunk_index, text)`` -> RFI-like or not,
    and where its first question heading and the first answer heading after
    it land. Order is (chunk_index, offset), so overlapping chunks read in
    document order."""
    ordered = sorted(chunks, key=lambda c: c[0])
    found: dict[str, list[tuple[int, int]]] = {r: [] for r in ROLES}
    for idx, body in ordered:
        for role, offs in heading_hits(body, pats).items():
            found[role].extend((idx, o) for o in offs)
    q = found["question"][0] if found["question"] else None
    a = next((h for h in found["answer"] if q is not None and h > q), None)
    rfi = q is not None and a is not None and bool(found["references"])
    return {
        "rfi_like": rfi,
        "chunks": len(ordered),
        "question_chunk": q[0] if q else None,
        "answer_chunk": a[0] if a else None,
        "split": bool(rfi and q[0] != a[0]),
    }


def summarise(per_doc: Iterable[tuple[str, dict[str, Any]]], cap: int = _MAX_IDS) -> dict[str, Any]:
    """Pure: the run's counts over every document of the project."""
    docs = list(per_doc)
    rfi = [(d, r) for d, r in docs if r["rfi_like"]]
    sizes = sorted(r["chunks"] for _, r in rfi)
    split = [d for d, r in rfi if r["split"]]
    return {
        "documents": len(docs),
        "rfi_like": len(rfi),
        "split": len(split),
        "same_chunk": len(rfi) - len(split),
        "chunks_min": sizes[0] if sizes else 0,
        "chunks_median": sizes[len(sizes) // 2] if sizes else 0,
        "chunks_max": sizes[-1] if sizes else 0,
        "chunks_total": sum(sizes),
        "rows": sorted(((d, r["chunks"], r["question_chunk"], r["answer_chunk"], r["split"])
                        for d, r in rfi))[:cap],
    }


def render(anchor: str, s: dict[str, Any]) -> list[str]:
    lines = [
        (f"== RFI-like in the project of id {anchor}: documents={s['documents']} "
         f"rfi_like={s['rfi_like']} split={s['split']} same_chunk={s['same_chunk']}"),
        (f"   chunks per RFI-like document: min={s['chunks_min']} median={s['chunks_median']} "
         f"max={s['chunks_max']} total={s['chunks_total']}"),
        "   doc_id | chunks | question_chunk | answer_chunk | split",
    ]
    lines += [f"   {d} | {n} | {q} | {a} | {'yes' if sp else 'no'}" for d, n, q, a, sp in s["rows"]]
    if s["rfi_like"] > len(s["rows"]):
        lines.append(f"   (+{s['rfi_like'] - len(s['rows'])} more RFI-like documents)")
    return lines


def group_by_doc(rows: Iterable[tuple]) -> Iterator[tuple[str, list[tuple[int, str]]]]:
    """``(doc_id, chunk_index, text)`` rows ordered by doc_id -> one list per doc."""
    cur: str | None = None
    buf: list[tuple[int, str]] = []
    for doc_id, idx, body in rows:
        if cur is not None and doc_id != cur:
            yield cur, buf
            buf = []
        cur = str(doc_id)
        buf.append((int(idx), body or ""))
    if cur is not None:
        yield cur, buf


def _chunks_sql(table: str) -> str:
    return (f"SELECT doc_id, chunk_index, text FROM {table} "
            "WHERE project_id = :pr ORDER BY doc_id, chunk_index")


def emit(line: str, *, err: bool = False) -> None:
    stream = sys.stderr if err else sys.stdout
    stream.write(f"{MARK}{line}\n")
    stream.flush()


def main() -> int:
    anchor = os.environ.get("RFI_ANCHOR_ID", "").strip()
    if not anchor or not _ID_RE.match(anchor):
        emit("RFI_ANCHOR_ID must be one document id or unambiguous prefix", err=True)
        return 2
    try:
        pats = compile_headings(headings())
    except (ValueError, json.JSONDecodeError):
        emit("RFI_HEADINGS_JSON is malformed", err=True)
        return 2
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    table = rag_chunk_table_name(os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip())
    try:
        with SessionLocal() as s:
            docs = s.execute(text(_DOC_SQL), {"p": anchor + "%"}).all()
            if len(docs) != 1:
                emit(f"== id {anchor}: {len(docs)} matching documents, need exactly 1")
                return 1
            rows = s.execute(text(_chunks_sql(table)).execution_options(stream_results=True),
                             {"pr": docs[0][1]})
            summ = summarise((d, classify(ch, pats)) for d, ch in group_by_doc(rows))
    except Exception as exc:  # by class only: a message can quote text
        emit(f"== id {anchor}: READ FAILED: {type(exc).__name__}")
        return 1
    for line in render(anchor, summ):
        emit(line)
    emit("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
