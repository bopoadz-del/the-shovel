"""Document-type registry — Roadmap V2 · Epic 2 (custom document types).

Document types are data, not hardcoded `if` branches. Built-ins live in
`config/document_types.yaml`; custom types are added at runtime (no redeploy)
and persisted to `DATA_DIR/custom_document_types.json`. Classification is
content-aware, not filename-only, and an unmatched document is reported as
`unrecognised` so the caller can ASK rather than silently guess.
"""

import json
import logging
import re
import os
import threading
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _builtin_file() -> str:
    here = os.path.dirname(os.path.abspath(__file__))  # app/core
    return os.path.normpath(os.path.join(here, "..", "..", "config",
                                         "document_types.yaml"))


def _custom_file() -> str:
    data_dir = os.getenv("DATA_DIR", "./data")
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError:
        import tempfile
        data_dir = tempfile.gettempdir()
    return os.path.join(data_dir, "custom_document_types.json")


def _load_builtins() -> List[Dict[str, Any]]:
    path = _builtin_file()
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("document_types", []) or []


def _load_custom() -> List[Dict[str, Any]]:
    path = _custom_file()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        logger.warning("failed to load custom document types from %s", path, exc_info=True)
        return []


def list_types() -> List[Dict[str, Any]]:
    """All document types — custom entries override built-ins by name."""
    by_name: Dict[str, Dict[str, Any]] = {}
    for t in _load_builtins():
        by_name[t["name"]] = {**t, "source": "builtin"}
    for t in _load_custom():
        by_name[t["name"]] = {**t, "source": "custom"}
    return list(by_name.values())


def add_type(definition: Dict[str, Any]) -> Dict[str, Any]:
    """Register (or replace) a custom document type. No code change needed."""
    name = (definition.get("name") or "").strip()
    if not name:
        raise ValueError("Document type requires a 'name'")
    entry = {
        "name": name,
        "description": definition.get("description", ""),
        "match": definition.get("match", {}) or {},
        "expected_fields": definition.get("expected_fields", []) or [],
    }
    with _lock:
        custom = [t for t in _load_custom() if t.get("name") != name]
        custom.append(entry)
        with open(_custom_file(), "w", encoding="utf-8") as f:
            json.dump(custom, f, indent=2)
    return {**entry, "source": "custom"}


def remove_type(name: str) -> bool:
    """Remove a custom document type (built-ins cannot be removed)."""
    with _lock:
        custom = _load_custom()
        kept = [t for t in custom if t.get("name") != name]
        if len(kept) == len(custom):
            return False
        with open(_custom_file(), "w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2)
    return True


#: Separators a file name uses between words. A multi-word keyword
#: ("bill of quantities") must match "Bill_of_Quantities.xlsx" as well as
#: "Bill of Quantities.xlsx", so both sides are folded to single spaces.
_NAME_SEPARATORS = re.compile(r"[\s_\-.]+")

#: Distinct content keywords one type needs before content ALONE (a name and
#: extension that match no type) may decide the type. One shared phrase is
#: too weak: generic words recur across kinds of document.
CONTENT_ONLY_MIN_HITS = int(os.getenv("DOC_TYPE_CONTENT_MIN_HITS", "2") or 2)


def _fold_name(text: str) -> str:
    return _NAME_SEPARATORS.sub(" ", (text or "").lower()).strip()


def _unrecognised() -> Dict[str, Any]:
    return {
        "name": "unrecognised",
        "score": 0,
        "matched_on": [],
        "expected_fields": [],
        "needs_user_confirmation": True,
    }


def classify(filename: str = "", content_sample: str = "") -> Dict[str, Any]:
    """Classify a document by filename + content keywords.

    Scoring: extension +3, filename keyword +2, content keyword +1. A document
    that matches nothing is `unrecognised` with `needs_user_confirmation: True`
    — never a silent default.

    When neither the name nor the extension matches any type, content alone
    decides only if one type has at least ``CONTENT_ONLY_MIN_HITS`` distinct
    content keywords and no other type scores as high. A tie between kinds is
    left `unrecognised`: the table's order is not a precedence among kinds.
    """
    fn = (filename or "").lower()
    ext = os.path.splitext(fn)[1]
    stem = _fold_name(os.path.splitext(fn)[0] if ext else fn)
    content = (content_sample or "").lower()

    best_type: Dict[str, Any] = {}
    best_score = 0
    best_matched: List[str] = []
    name_matched_any = False
    content_scores: List[int] = []

    for t in list_types():
        m = t.get("match", {}) or {}
        score = 0
        matched: List[str] = []
        for kw in m.get("filename", []) or []:
            # Word-boundary match, not substring: "spec" must not fire inside
            # "Inspection" (a live Work Inspection Request classified as a
            # specification). Controlled-document names are full of hyphens,
            # underscores and parentheses, so those count as separators, and
            # a multi-word keyword matches across any of them.
            kw_n = _fold_name(kw)
            if (re.search(rf"(?<![a-z0-9]){re.escape(kw.lower())}(?![a-z0-9])", fn)
                    or re.search(rf"(?<![a-z0-9]){re.escape(kw_n)}(?![a-z0-9])", stem)):
                score += 2
                matched.append(f"filename:{kw}")
        exts = [e.lower() for e in (m.get("extensions", []) or [])]
        if ext and ext in exts:
            score += 3
            matched.append(f"ext:{ext}")
        if score:
            name_matched_any = True
        content_hits = 0
        for kw in m.get("content", []) or []:
            if kw.lower() in content:
                score += 1
                content_hits += 1
                matched.append(f"content:{kw}")
        content_scores.append(content_hits)
        if score > best_score:
            best_score, best_type, best_matched = score, t, matched

    if best_score == 0:
        return _unrecognised()
    # Content alone (no name or extension matched): enough distinct evidence
    # for one kind, and no tie between kinds.
    if not name_matched_any and (
        best_score < CONTENT_ONLY_MIN_HITS or content_scores.count(best_score) > 1
    ):
        return _unrecognised()
    return {
        "name": best_type["name"],
        "score": best_score,
        "matched_on": best_matched,
        "expected_fields": best_type.get("expected_fields", []),
        "needs_user_confirmation": False,
    }
