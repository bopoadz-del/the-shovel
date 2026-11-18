#!/usr/bin/env python3
"""Public view of an owner-document ops response: a WHITELIST, never a deny-list.

This repository is public: a workflow's run log, step summary and artifacts
are public pages. The document exclude / include routes answer with the
document's original file name, free-text reasons and refusal prose. This
filter prints only the fields named in ``VIEWS`` for the response kind, and
renders every value by its type:

* null, booleans and numbers print as they are;
* a string prints only when it is a short code (letters, digits, ``_``,
  ``-``, ``:``; no dot, no slash, no space), so an id, a status, a hash or a
  ``YYYY-MM-DDTHH:MM:SSZ`` time prints, and a file name, a path or a sentence
  prints as ``<text len=N>``;
* a list prints as its length;
* a dict prints only the sub-fields its view names (``VIEWS`` nests), any
  other dict prints as ``<object keys=N>``.

A key that is not in the view is never read. A field named in ``TEXT_FIELDS``
prints its length even when it looks like a code (stored reasons, refusal
prose and ``not_eligible``).

Usage: ``ops_public_view.py <kind> <response.json>`` prints Markdown lines;
``ops_public_view.py reason-gate <value>`` exits 1 unless the value is empty
or a short code (the dispatch reason input).
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

CODE_RE = re.compile(r"[A-Za-z0-9_:-]{1,80}")
# The operator's reason input: lower-case code, as stored on the row.
REASON_RE = re.compile(r"[a-z0-9_-]{1,64}")

_STAMP = ("by", "at", "reason")

# kind -> ordered whitelist. A tuple value names the sub-fields of a dict.
VIEWS: dict[str, dict[str, Any]] = {
    "exclude-plan": {"plan": {
        "document_id": None, "doc_type": None, "ingest_status": None,
        "chunk_count": None, "chunk_rows": None, "rows_to_delete": None,
        "project_chunk_total": None, "already_excluded": None,
        "plan_hash": None,
    }, "executed": None, "refused": None},
    "exclude-applied": {
        "executed": None, "refused": None, "chunks_deleted": None,
        "index_pruned": None, "chunk_rows_after": None,
        "chunk_count_after": None, "project_chunk_total_after": None,
        "excluded": _STAMP, "ingest_status": None, "retrieval_visible": None,
    },
    "include-plan": {"plan": {
        "document_id": None, "doc_type": None, "ingest_status": None,
        "ingest_status_reason": None, "retrieval_visible": None,
        "chunk_count": None, "chunk_rows": None, "excluded": _STAMP,
        "eligible": None, "not_eligible": None, "status_after": None,
        "plan_hash": None,
    }, "executed": None, "refused": None},
    "include-applied": {
        "executed": None, "refused": None, "excluded_after": _STAMP,
        "included_back": _STAMP, "ingest_status": None,
        "retrieval_visible": None, "chunk_count": None,
    },
}
TEXT_FIELDS = frozenset({"reason", "refused", "not_eligible", "ingest_status_reason"})


def render(value: Any, key: str = "") -> str:
    """One value, by type; text that is not a short code prints as its length."""
    if value is None or isinstance(value, bool):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if key not in TEXT_FIELDS and CODE_RE.fullmatch(value):
            return value
        return f"<text len={len(value)}>"
    if isinstance(value, list):
        return f"<list len={len(value)}>"
    if isinstance(value, dict):
        return f"<object keys={len(value)}>"
    return f"<{type(value).__name__}>"


def view_lines(obj: Any, view: dict[str, Any], prefix: str = "") -> list[str]:
    """``- path: value`` for each whitelisted field, in view order."""
    src = obj if isinstance(obj, dict) else {}
    out: list[str] = []
    for key, sub in view.items():
        path = f"{prefix}{key}"
        if key not in src:
            continue
        val = src[key]
        if isinstance(sub, dict) and isinstance(val, dict):
            out.extend(view_lines(val, sub, prefix=f"{path}."))
        elif isinstance(sub, tuple) and isinstance(val, dict):
            parts = [f"{k}={render(val[k], k)}" for k in sub if k in val]
            out.append(f"- {path}: {{{', '.join(parts)}}}")
        else:
            out.append(f"- {path}: {render(val, key)}")
    return out


def reason_ok(value: str) -> bool:
    return value == "" or bool(REASON_RE.fullmatch(value))


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "reason-gate":
        if reason_ok(argv[1]):
            return 0
        # The value itself is never echoed.
        print(f"reason refused: not a short code (a-z 0-9 _ -, at most 64); len={len(argv[1])}")
        return 1
    if len(argv) != 2 or argv[0] not in VIEWS:
        print(f"usage: ops_public_view.py <{'|'.join(VIEWS)}> <response.json> | reason-gate <value>")
        return 2
    try:
        with open(argv[1], encoding="utf-8") as fh:
            obj = json.load(fh)
    except FileNotFoundError:
        print("- response: none")
        return 0
    except ValueError as exc:
        print(f"- response: unreadable ({type(exc).__name__})")
        return 1
    print("\n".join(view_lines(obj, VIEWS[argv[0]])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
