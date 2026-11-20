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

The heading rule only sees headings written in the letters of the heading
table. So the run also counts, over EVERY document of the project, the
dominant Unicode letter script of its text (the first word of each letter's
Unicode name: LATIN, ARABIC, ...), and how many documents are written mainly,
or in part (at least ``_MIXED_SHARE`` of their letters), in a script the
heading table does not use: the documents the rule cannot see. The table's
own scripts are read from the table, never assumed.

A second, independent selector answers "does the project hold RFI forms at
all": the documents whose own NAME or PATH names the RFI document class, read
with the product's document-class vocabulary
(``app.core.rag.handed_set.named_source_kinds``; the class is
``RFI_SAMPLE_KIND``, default ``rfi``). The run counts them, how many the
heading rule calls RFI-like, and prints, for the first ``RFI_SAMPLES`` of them
(default 3), a heading SKELETON: one line per heading-like cell (a short label
ending in ``:``, an all-capitals label, or a cell opening with a word of the
heading table) with its chunk index, line, cell position, word count, letter
case, terminator, table role and whether the line-start rule sees it. The
skeleton carries shapes only, never the words.

The output is a public workflow log: document ids, counts, chunk indexes and
Unicode script names only; never text, a heading, a name, a path or a
project id.
"""
from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from collections import Counter
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
# A document is "in part" in another script when at least this share of its
# letters are in scripts the heading table does not use.
_MIXED_SHARE = 0.05

_DOC_SQL = "SELECT id, project_id FROM documents WHERE id LIKE :p ORDER BY id"
_NAMES_SQL = "SELECT id, original_name, file_path FROM documents WHERE project_id = :pr"
_DEFAULT_SAMPLES = 3
_MAX_SAMPLES = 10
# Per sample, the heading-like cells printed (in document order).
_MAX_SKELETON = 60
# A heading-like cell has at most this many words.
_HEADING_MAX_WORDS = 6
# How extraction lays out a form's table on one line: tabs, pipes, or runs
# of three or more spaces between cells.
_CELL_SPLIT = re.compile(r"\t+|\s*\|\s*|\s{3,}")
_LEAD = re.compile(r"^[ \t>*\-\d.()\[\]#]*")
_NAME_SEPS = re.compile(r"[_\-./\\]+")


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


def letter_script(ch: str) -> str | None:
    """The Unicode script of one letter (first word of its Unicode name), or
    None for a non-letter."""
    if not ch.isalpha():
        return None
    return unicodedata.name(ch, "?").split(" ", 1)[0]


def script_counts(texts: Iterable[str]) -> Counter:
    """Pure: letters per Unicode script over the given texts."""
    out: Counter = Counter()
    for t in texts:
        for ch in t or "":
            sc = letter_script(ch)
            if sc:
                out[sc] += 1
    return out


def table_scripts(table: dict[str, tuple[str, ...]]) -> frozenset[str]:
    """The scripts the heading table itself is written in."""
    return frozenset(script_counts(w for words in table.values() for w in words))


def script_profile(counts: Counter, seen: frozenset[str]) -> dict[str, Any]:
    """Pure: a document's dominant script (most letters; ties by name) and
    whether it is mainly, or in part, in a script outside ``seen``."""
    total = sum(counts.values())
    if not total:
        return {"script": None, "unseen": False, "mixed": False}
    dom = min(counts, key=lambda k: (-counts[k], k))
    outside = sum(n for k, n in counts.items() if k not in seen)
    return {"script": dom, "unseen": dom not in seen,
            "mixed": dom in seen and outside / total >= _MIXED_SHARE}


def classify(chunks: list[tuple[int, str]], pats: dict[str, re.Pattern[str]],
             seen: frozenset[str] = frozenset()) -> dict[str, Any]:
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
        **script_profile(script_counts(b for _, b in ordered), seen),
        "rfi_like": rfi,
        "chunks": len(ordered),
        "question_chunk": q[0] if q else None,
        "answer_chunk": a[0] if a else None,
        "split": bool(rfi and q[0] != a[0]),
        # Which roles' headings the document carries at all, and whether an
        # answer heading follows a question heading (references aside): a
        # zero rfi_like count is then read as "no such headings" or "one
        # block missing", not as "no RFIs".
        "roles": tuple(r for r in ROLES if found[r]),
        "q_then_a": a is not None,
    }


def summarise(per_doc: Iterable[tuple[str, dict[str, Any]]], cap: int = _MAX_IDS) -> dict[str, Any]:
    """Pure: the run's counts over every document of the project."""
    docs = list(per_doc)
    rfi = [(d, r) for d, r in docs if r["rfi_like"]]
    sizes = sorted(r["chunks"] for _, r in rfi)
    split = [d for d, r in rfi if r["split"]]
    by_script = Counter(r.get("script") or "(no letters)" for _, r in docs)
    return {
        "documents": len(docs),
        "by_script": sorted(by_script.items(), key=lambda kv: (-kv[1], kv[0])),
        "unseen_script": sum(1 for _, r in docs if r.get("unseen")),
        "mixed_script": sum(1 for _, r in docs if r.get("mixed")),
        "rfi_like": len(rfi),
        "split": len(split),
        "same_chunk": len(rfi) - len(split),
        "chunks_min": sizes[0] if sizes else 0,
        "chunks_median": sizes[len(sizes) // 2] if sizes else 0,
        "chunks_max": sizes[-1] if sizes else 0,
        "chunks_total": sum(sizes),
        "with_role": {r: sum(1 for _, x in docs if r in x["roles"]) for r in ROLES},
        "q_then_a": sum(1 for _, x in docs if x["q_then_a"]),
        "rows": sorted(((d, r["chunks"], r["question_chunk"], r["answer_chunk"], r["split"])
                        for d, r in rfi))[:cap],
    }


def render(anchor: str, s: dict[str, Any]) -> list[str]:
    lines = [
        (f"== RFI-like in the project of id {anchor}: documents={s['documents']} "
         f"rfi_like={s['rfi_like']} split={s['split']} same_chunk={s['same_chunk']}"),
        (f"   chunks per RFI-like document: min={s['chunks_min']} median={s['chunks_median']} "
         f"max={s['chunks_max']} total={s['chunks_total']}"),
        ("   documents with a heading of: "
         + " ".join(f"{r}={s['with_role'][r]}" for r in ROLES)
         + f" question_then_answer={s['q_then_a']}"),
        ("   documents by dominant letter script: "
         + " ".join(f"{k}={n}" for k, n in s["by_script"])),
        (f"   outside the heading table's scripts: mainly={s['unseen_script']} "
         f"in_part={s['mixed_script']} (the rule cannot see these headings)"),
        "   doc_id | chunks | question_chunk | answer_chunk | split",
    ]
    lines += [f"   {d} | {n} | {q} | {a} | {'yes' if sp else 'no'}" for d, n, q, a, sp in s["rows"]]
    if s["rfi_like"] > len(s["rows"]):
        lines.append(f"   (+{s['rfi_like'] - len(s['rows'])} more RFI-like documents)")
    return lines


def name_kinds(name: str | None, path: str | None) -> tuple[str, ...]:
    """The document classes a document's own name or path names, read with
    the product's class vocabulary. Separators (``_ - . /`` and a backslash)
    count as spaces, so ``RFI_012`` names its class like ``RFI 012`` does."""
    from app.core.rag.handed_set import named_source_kinds

    joined = " ".join(_NAME_SEPS.sub(" ", x or "") for x in (name, path))
    return named_source_kinds(joined)


def sample_count(raw: str | None = None) -> int:
    """``RFI_SAMPLES``: how many named documents get a skeleton (default 3,
    at most ``_MAX_SAMPLES``). Raises ValueError on a non-integer."""
    raw = os.environ.get("RFI_SAMPLES", "") if raw is None else raw
    if not raw.strip():
        return _DEFAULT_SAMPLES
    return max(0, min(_MAX_SAMPLES, int(raw)))


def letter_case(label: str) -> str:
    """Pure: ``UPPER``, ``lower``, ``Title``, ``mixed`` or ``caseless``."""
    letters = [c for c in label if c.isalpha()]
    if not any(c.isupper() or c.islower() for c in letters):
        return "caseless"
    if not any(c.islower() for c in letters):
        return "UPPER"
    if not any(c.isupper() for c in letters):
        return "lower"
    words = [w for w in label.split() if any(c.isalpha() for c in w)]
    if all(next(c for c in w if c.isalpha()).isupper() for w in words):
        return "Title"
    return "mixed"


def _role_at_start(cell: str, pats: dict[str, re.Pattern[str]]) -> str:
    for role in ROLES:
        if pats[role].match(cell):
            return role
    return "-"


def heading_skeleton(chunks: list[tuple[int, str]], pats: dict[str, re.Pattern[str]],
                     cap: int = _MAX_SKELETON) -> list[dict[str, Any]]:
    """Pure: the heading-like cells of one document, in document order, as
    SHAPES (no words). A line is split into cells on tabs, ``|`` and runs of
    3+ spaces. A cell is heading-like when its label (what precedes its first
    ``:``, after numbering and bullets) has 1..``_HEADING_MAX_WORDS`` words
    with letters and the cell has a ``:``, or the label is in capitals, or
    the cell opens with a word of the heading table. ``rule`` says whether
    the line-start rule (``heading_hits``) sees a heading of that role on
    that line; ``value`` whether text follows the label in the same cell."""
    out: list[dict[str, Any]] = []
    for idx, body in sorted(chunks, key=lambda c: c[0]):
        for ln, line in enumerate((body or "").splitlines()):
            rule_roles = {r for r, offs in heading_hits(line, pats).items() if offs}
            cells = [c for c in _CELL_SPLIT.split(line) if c.strip()]
            for pos, raw in enumerate(cells):
                cell = _LEAD.sub("", raw).strip()
                label, colon, rest = cell.partition(":")
                label = label.strip()
                words = [w for w in label.split() if any(ch.isalpha() for ch in w)]
                if not words or len(words) > _HEADING_MAX_WORDS:
                    continue
                role = _role_at_start(cell, pats)
                case = letter_case(label)
                if not (colon or case == "UPPER" or role != "-"):
                    continue
                end = "colon" if colon else ("eol" if len(cells) == 1 else "none")
                out.append({
                    "chunk": idx, "line": ln, "cell": pos, "words": len(words),
                    "case": case, "end": end, "role": role,
                    "rule": role != "-" and role in rule_roles,
                    "value": bool(rest.strip()),
                })
                if len(out) >= cap:
                    return out
    return out


def render_samples(kind: str, named_rows: list[dict[str, Any]],
                   samples: list[tuple[str, int, list[dict[str, Any]]]]) -> list[str]:
    """The named-class count and each sample's skeleton (shapes, no words)."""
    with_role = {r: sum(1 for x in named_rows if r in x["roles"]) for r in ROLES}
    lines = [
        (f"== documents whose own name or path names the class {kind}: {len(named_rows)} "
         f"(rfi_like={sum(1 for x in named_rows if x['rfi_like'])}; with a heading of: "
         + " ".join(f"{r}={with_role[r]}" for r in ROLES) + ")"),
    ]
    for doc, nchunks, cells in samples:
        lines.append(f"   sample {doc}: chunks={nchunks} heading_like={len(cells)} "
                     f"table_role={sum(1 for c in cells if c['role'] != '-')} "
                     f"seen_by_rule={sum(1 for c in cells if c['rule'])}")
        lines.append("     chunk | line | cell | words | case | end | role | rule | value")
        lines += [
            (f"     {c['chunk']} | {c['line']} | {c['cell']} | {c['words']} | {c['case']} | "
             f"{c['end']} | {c['role']} | {'yes' if c['rule'] else 'no'} | "
             f"{'yes' if c['value'] else 'no'}")
            for c in cells
        ]
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


def drop_hidden(rows: Iterable[tuple], hidden: set[str]) -> Iterator[tuple]:
    """Pure: the ``(doc_id, chunk_index, text)`` rows whose document is not in
    ``hidden``, in the same order."""
    for row in rows:
        if str(row[0]) not in hidden:
            yield row


def project_chunk_rows(s: Any, table: str, project_id: str) -> Iterator[tuple]:
    """The project's own chunks, ``(doc_id, chunk_index, text)`` ordered by
    document. A chat session's file is that session's, not the project's
    (``projects.document_in_scope``), so the documents the product's own
    ``session_hidden_document_ids`` names are dropped: the count reads what
    the project's retrieval may read, never a session upload."""
    from app.core.projects import session_hidden_document_ids

    hidden = {str(d) for d in session_hidden_document_ids(project_id, s)}
    sql = (f"SELECT doc_id, chunk_index, text FROM {table} "
           "WHERE project_id = :pr ORDER BY doc_id, chunk_index")
    rows = s.execute(text(sql).execution_options(stream_results=True), {"pr": project_id})
    return drop_hidden(rows, hidden)


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
        table = headings()
        pats = compile_headings(table)
        seen = table_scripts(table)
    except (ValueError, json.JSONDecodeError):
        emit("RFI_HEADINGS_JSON is malformed", err=True)
        return 2
    try:
        n_samples = sample_count()
    except ValueError:
        emit("RFI_SAMPLES must be an integer", err=True)
        return 2
    kind = os.environ.get("RFI_SAMPLE_KIND", "").strip() or "rfi"
    from app.core.db import SessionLocal
    from app.core.models import rag_chunk_table_name

    table = rag_chunk_table_name(os.getenv("RAG_VECTOR_NAMESPACE", "v2").strip())
    try:
        with SessionLocal() as s:
            docs = s.execute(text(_DOC_SQL), {"p": anchor + "%"}).all()
            if len(docs) != 1:
                emit(f"== id {anchor}: {len(docs)} matching documents, need exactly 1")
                return 1
            named = {str(d) for d, name, path in s.execute(text(_NAMES_SQL), {"pr": docs[0][1]})
                     if kind in name_kinds(name, path)}
            rows = project_chunk_rows(s, table, docs[0][1])
            per_doc: list[tuple[str, dict[str, Any]]] = []
            samples: list[tuple[str, int, list[dict[str, Any]]]] = []
            for d, ch in group_by_doc(rows):
                r = classify(ch, pats, seen)
                per_doc.append((d, r))
                if d in named and len(samples) < n_samples:
                    samples.append((d, r["chunks"], heading_skeleton(ch, pats)))
            summ = summarise(per_doc)
    except Exception as exc:  # by class only: a message can quote text
        emit(f"== id {anchor}: READ FAILED: {type(exc).__name__}")
        return 1
    for line in render(anchor, summ):
        emit(line)
    for line in render_samples(kind, [r for d, r in per_doc if d in named], samples):
        emit(line)
    emit("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
