#!/usr/bin/env python3
"""One static page per test run from its ``rag_retrieval_trace`` lines (Q-2).

Measurement only. No server, no product import, no live call: the input is
a log export taken by whoever holds read access (a CloudWatch
``filter-log-events`` JSON, or one event per line), plus the run's row files.

Per row the page shows the row's id and repeat, the build before and after,
the answer's length, and for every trace line of that row: each leg's top
ranks, the fused order, what was handed, what was cut and the rule that cut
it (``dropped_by``), and the scores.

What it never writes: document names, question text, expected answers,
answer text, source lists. Every field reaches the page through an explicit
allowlist (``_SCALAR_KEYS``, ``_CHUNK_KEYS``, ``_LEG_KEYS``), so a field the
trace gains later is left out until someone adds it here on purpose. A
string value is written only if it has the shape of an id or a code
(``_ID_RE``); anything else becomes ``(withheld)``.

Usage::

    python scripts/trace_page.py --events events.json \\
        --rows "TESTER_EVIDENCE/*_v2_*.json" --out TESTER_EVIDENCE/trace_page_<run>.html \\
        [--scores q1_scores.jsonl] [--slack-s 15]
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

TRACE_EVENT = "rag_retrieval_trace"

# Labels the product writes for the query (``inject._emit_retrieval_trace``
# and ``emit_search_trace``). Anything else is shown as ``other``.
_QUERY_LABELS = frozenset({"user_message", "followup_context", "tool_query"})

# Top-level fields shown in a trace's summary. Numbers and booleans only,
# plus the ``path``/``query`` codes and the request id.
_SCALAR_KEYS = (
    "request_id", "path", "query", "query_chars", "retrieved_k", "handed_k",
    "requested_k", "effective_k", "pool_k", "pool_depth", "char_budget",
    "chars_handed", "budget_spent", "threshold_fired", "identifier_miss",
    "degraded", "top_score",
)
_CHUNK_KEYS = (
    "rank", "chunk_id", "doc_id", "score", "vector_score", "lexical_score",
    "fused_rank", "chars", "layer", "doc_type", "doc_role", "dropped_by",
)
_LEG_KEYS = (
    "leg_rank", "other_leg_rank", "fused_rank", "chars", "pool_position",
    "handed_position", "dropped_by",
)
_FUSED_COUNT_KEYS = (
    "n_candidates", "n_both_legs", "n_lexical_only", "n_vector_only", "fused_k",
)

# An id or a code: no spaces, no path separators, bounded length. A document
# name ("Bill of Quantities.pdf") fails this and is withheld.
_ID_RE = re.compile(r"^[A-Za-z0-9_\-:.]{1,80}$")
_NAME_SUFFIX_RE = re.compile(r"\.(pdf|docx?|xlsx?|xls|txt|md|dwg|xer|csv|pptx?)$", re.I)
WITHHELD = "(withheld)"

#: Inputs the page could not use, counted by reason and printed by ``main``,
#: so "no trace for this row" is never confused with "its line was unreadable".
SKIPPED: Counter = Counter()


# ── input ────────────────────────────────────────────────────────────────


def _trace_payload(message: str) -> Optional[Dict[str, Any]]:
    """The trace JSON inside one log message, or None.

    The message is either ``rag_retrieval_trace {json}`` or a JSON log
    record whose ``message`` field holds that text."""
    if not isinstance(message, str) or TRACE_EVENT not in message:
        return None
    text = message
    stripped = message.strip()
    if stripped.startswith("{"):
        try:
            outer = json.loads(stripped)
        except ValueError:
            outer = None
        if isinstance(outer, dict) and isinstance(outer.get("message"), str):
            text = outer["message"]
    at = text.find(TRACE_EVENT)
    if at < 0:
        return None
    brace = text.find("{", at)
    if brace < 0:
        return None
    try:
        payload, _ = json.JSONDecoder().raw_decode(text[brace:])
    except ValueError:
        SKIPPED["undecodable_trace_line"] += 1
        return None
    return payload if isinstance(payload, dict) else None


def load_events(path: str) -> List[Tuple[float, Dict[str, Any]]]:
    """(epoch seconds, trace payload) for every trace line in the export.

    Accepts the ``filter-log-events`` JSON (``{"events": [...]}``), a JSON
    list of events, or one event object per line. Lines that are not trace
    lines are skipped."""
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    events: List[Any]
    try:
        doc = json.loads(raw)
        events = doc.get("events", []) if isinstance(doc, dict) else list(doc)
    except ValueError:
        events = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    out: List[Tuple[float, Dict[str, Any]]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        payload = _trace_payload(ev.get("message", ""))
        ts = ev.get("timestamp")
        if payload is None or not isinstance(ts, (int, float)):
            continue
        out.append((float(ts) / 1000.0, payload))
    out.sort(key=lambda e: e[0])
    return out


def _epoch(stamp: str) -> Optional[float]:
    try:
        dt = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        SKIPPED["row_without_start_time"] += 1
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def row_from_record(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The fields of one tester row the page may use. Never the question,
    the expected answer, the answer text or the sources."""
    start = _epoch(rec.get("time_utc", ""))
    if start is None:
        return None
    timing = rec.get("timing") if isinstance(rec.get("timing"), dict) else {}
    t_end = timing.get("t_end")
    answer = rec.get("answer")
    return {
        "id": _safe_str(rec.get("id")),
        "rep": rec.get("rep") if isinstance(rec.get("rep"), int) else None,
        "start": start,
        "t_end": float(t_end) if isinstance(t_end, (int, float)) else None,
        "sha_before": _safe_str(rec.get("sha_before")),
        "sha_after": _safe_str(rec.get("sha_after")),
        "answer_chars": len(answer) if isinstance(answer, str) else None,
    }


def load_rows(pattern: str) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except (OSError, ValueError):
            continue
        row = row_from_record(rec) if isinstance(rec, dict) else None
        if row is not None:
            rows.append(row)
    rows.sort(key=lambda r: r["start"])
    return rows


def load_scores(path: Optional[str]) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """Q-1 scorer output, one JSON object per line keyed by ``row`` and
    ``rep``. Only numeric and boolean values are kept."""
    if not path:
        return {}
    out: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not isinstance(rec, dict):
                continue
            key = (_safe_str(rec.get("row")), rec.get("rep"))
            out[key] = {
                k: v for k, v in rec.items()
                if k not in ("row", "rep") and _ID_RE.match(str(k))
                and isinstance(v, (int, float, bool))
            }
    return out


# ── join ─────────────────────────────────────────────────────────────────


def assign(
    events: List[Tuple[float, Dict[str, Any]]],
    rows: List[Dict[str, Any]],
    slack_s: float = 15.0,
    default_window_s: float = 400.0,
) -> Tuple[Dict[int, List[Tuple[float, Dict[str, Any]]]], int]:
    """Each trace line goes to the latest row that started at or before it
    and whose window (its end time plus ``slack_s``) still covers it. A line
    no row covers is counted as unmatched, never guessed onto a row."""
    by_row: Dict[int, List[Tuple[float, Dict[str, Any]]]] = {i: [] for i in range(len(rows))}
    unmatched = 0
    for ts, payload in events:
        owner = None
        for i, row in enumerate(rows):
            if row["start"] > ts:
                break
            window = row["t_end"] if row["t_end"] is not None else default_window_s
            if ts <= row["start"] + window + slack_s:
                owner = i
        if owner is None:
            unmatched += 1
        else:
            by_row[owner].append((ts, payload))
    return by_row, unmatched


# ── projection (the allowlist) ───────────────────────────────────────────


def _safe_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value)
    if not _ID_RE.match(s) or _NAME_SUFFIX_RE.search(s):
        return WITHHELD
    return s


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_str(value)


def project_trace(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The trace line reduced to the allowlisted fields."""
    summary = {}
    for key in _SCALAR_KEYS:
        if key not in payload:
            continue
        value = payload[key]
        if key == "query":
            value = value if value in _QUERY_LABELS else "other"
        summary[key] = _safe(value)
    fused = payload.get("fused_candidates") if isinstance(payload.get("fused_candidates"), dict) else {}
    legs: Dict[str, List[Dict[str, Any]]] = {}
    for leg, entries in (fused.get("leg_top") or {}).items():
        name = _safe_str(leg) or WITHHELD
        legs[name] = [
            {k: _safe(e.get(k)) for k in _LEG_KEYS if k in e}
            for e in (entries or []) if isinstance(e, dict)
        ]
    counts = {k: _safe(fused.get(k)) for k in _FUSED_COUNT_KEYS if k in fused}

    def chunks(key: str) -> List[Dict[str, Any]]:
        return [
            {k: _safe(c.get(k)) for k in _CHUNK_KEYS if k in c}
            for c in (payload.get(key) or []) if isinstance(c, dict)
        ]

    handed = chunks("chunks")
    cut = chunks("not_handed")
    return {
        "summary": summary,
        "fused_counts": counts,
        "leg_top": legs,
        "handed": handed,
        "cut": cut,
        "cut_by": dict(Counter(c.get("dropped_by") or "unexplained" for c in cut)),
    }


# ── page ─────────────────────────────────────────────────────────────────


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return html.escape(f"{v:.6g}")
    return html.escape(str(v))


def _table(rows: List[Dict[str, Any]], keys: Iterable[str]) -> str:
    keys = [k for k in keys if any(k in r for r in rows)]
    if not rows or not keys:
        return "<p class=empty>none</p>"
    head = "".join(f"<th>{html.escape(k)}</th>" for k in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{_cell(r.get(k))}</td>" for k in keys) + "</tr>"
        for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def _kv(d: Dict[str, Any]) -> str:
    if not d:
        return "<p class=empty>none</p>"
    return "<table>" + "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_cell(v)}</td></tr>" for k, v in d.items()
    ) + "</table>"


def _fused_order(view: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Handed and cut chunks that carry a fused rank, in fused order."""
    rows = [dict(c, handed="yes") for c in view["handed"]] + [
        dict(c, handed="no") for c in view["cut"]
    ]
    ranked = [r for r in rows if isinstance(r.get("fused_rank"), int)]
    return sorted(ranked, key=lambda r: r["fused_rank"])


def render(
    run_label: str,
    rows: List[Dict[str, Any]],
    by_row: Dict[int, List[Tuple[float, Dict[str, Any]]]],
    unmatched: int,
    scores: Optional[Dict[Tuple[str, Any], Dict[str, Any]]] = None,
) -> str:
    scores = scores or {}
    parts = [
        "<!doctype html><meta charset=utf-8>",
        f"<title>trace page {html.escape(run_label)}</title>",
        "<style>body{font:13px sans-serif;margin:1em}table{border-collapse:collapse;"
        "margin:.3em 0}td,th{border:1px solid #bbb;padding:2px 5px;text-align:right}"
        "th{background:#eee}.empty{color:#888}h2{margin-top:1.5em}</style>",
        f"<h1>Retrieval trace page: {html.escape(run_label)}</h1>",
        "<p>Ids, ranks, scores and counts only. Document names and question text are "
        "not written; look a document id up in the app.</p>",
        f"<p>rows: {len(rows)}; trace lines matched: "
        f"{sum(len(v) for v in by_row.values())}; unmatched: {unmatched}</p>",
    ]
    for i, row in enumerate(rows):
        lines = by_row.get(i, [])
        head = {
            "row": row["id"], "rep": row["rep"],
            "build": f"{row['sha_before']}>{row['sha_after']}",
            "answer_chars": row["answer_chars"], "t_end_s": row["t_end"],
            "trace_lines": len(lines),
        }
        head.update(scores.get((row["id"], row["rep"]), {}))
        parts.append(f"<h2>row {_cell(row['id'])} rep {_cell(row['rep'])}</h2>")
        parts.append(_kv(head))
        for n, (ts, payload) in enumerate(lines, 1):
            view = project_trace(payload)
            parts.append(f"<h3>trace {n} (+{ts - row['start']:.1f} s)</h3>")
            parts.append(_kv(view["summary"]))
            parts.append("<h4>fused candidates</h4>" + _kv(view["fused_counts"]))
            for leg, entries in view["leg_top"].items():
                parts.append(f"<h4>leg {html.escape(leg)}: top ranks</h4>")
                parts.append(_table(entries, _LEG_KEYS))
            parts.append("<h4>fused order (pool)</h4>")
            parts.append(_table(_fused_order(view), ("fused_rank", "handed") + _CHUNK_KEYS))
            parts.append("<h4>handed</h4>" + _table(view["handed"], _CHUNK_KEYS))
            parts.append("<h4>cut, and why</h4>" + _kv(view["cut_by"]))
            parts.append(_table(view["cut"], _CHUNK_KEYS))
    return "\n".join(parts) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--events", required=True, help="log export with the trace lines")
    ap.add_argument("--rows", required=True, help="glob of the run's row JSON files")
    ap.add_argument("--out", required=True, help="the page to write")
    ap.add_argument("--scores", help="Q-1 scorer output (JSON lines)")
    ap.add_argument("--slack-s", type=float, default=15.0)
    ap.add_argument("--label", default="", help="run label shown on the page")
    args = ap.parse_args(argv)
    events = load_events(args.events)
    rows = load_rows(args.rows)
    if not rows:
        print("no rows matched --rows", file=sys.stderr)
        return 2
    by_row, unmatched = assign(events, rows, args.slack_s)
    page = render(args.label or args.out, rows, by_row, unmatched, load_scores(args.scores))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"trace_page rows={len(rows)} lines={len(events)} unmatched={unmatched} "
          f"skipped={dict(SKIPPED)} out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
