"""Document-type registry — Roadmap V2 · Epic 2 (custom document types).

Document types are data, not hardcoded `if` branches. Built-ins live in
`config/document_types.yaml`; custom types are added at runtime (no redeploy)
and persisted to `DATA_DIR/custom_document_types.json`. Classification is
content-aware, not filename-only, and an unmatched document is reported as
`unrecognised` so the caller can ASK rather than silently guess.
"""

import functools
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


def _table_key(path: str):
    """What identifies one version of the types file on disk: its path, its
    modification time and its size. ``None`` when there is no file. An edit
    to the file changes the key, so a cached read never outlives the file."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (path, st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=4)
def _read_table(key) -> Dict[str, Any]:
    """The parsed types file for one ``_table_key``: read once per version of
    the file, not once per classify call (the backfill classifies every
    document of a project in one pass)."""
    with open(key[0], "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _builtin_table() -> Dict[str, Any]:
    key = _table_key(_builtin_file())
    return _read_table(key) if key is not None else {}


def _load_builtins() -> List[Dict[str, Any]]:
    return list(_builtin_table().get("document_types", []) or [])


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
#: too weak: generic words recur across kinds of document. Never below 1, so
#: a document that matches nothing at all always fails the bar.
CONTENT_ONLY_MIN_HITS = max(1, int(os.getenv("DOC_TYPE_CONTENT_MIN_HITS", "2") or 2))


#: Arabic spelling variants folded to one form on BOTH sides (keyword and
#: document): short vowels / tanween / shadda and tatweel dropped, hamza-
#: carrying alefs to bare alef, alef maqsura to ya, ta marbuta to ha. A
#: keyword then matches however the writer spelt the word.
_AR_MARKS = re.compile("[ً-ْٰـ]")
_AR_LETTERS = str.maketrans({"أ": "ا", "إ": "ا",
                             "آ": "ا", "ٱ": "ا",
                             "ى": "ي", "ة": "ه"})


#: Letters and digits that make up one word, in Latin and Arabic script. A
#: keyword matches only a WHOLE word of the folded text: "صور" (photos) must
#: not fire inside "تصور" (concept), nor "عقد" (contract) inside "معقد"
#: (complex). Arabic punctuation (comma, semicolon, question mark) is not a
#: letter and so separates words.
_WORD_CHARS = "a-z0-9\u0621-\u063a\u0641-\u064a\u0660-\u0669\u066e-\u06d3"


def _ar_prefixes() -> List[str]:
    """One-word Arabic clitics (article, conjunction, preposition) that may
    stand glued before a keyword: "والعقد" is "and the contract". Read from
    the types table (``arabic_prefixes``), longest first, so the regex tries
    "لل" before "ل". Cached per version of the file, like the type list."""
    key = _table_key(_builtin_file())
    return list(_prefixes_for(key)) if key is not None else []


@functools.lru_cache(maxsize=4)
def _prefixes_for(key) -> tuple:
    raw = _read_table(key).get("arabic_prefixes") or []
    out = {_norm(str(p)) for p in raw if str(p).strip()}
    return tuple(sorted(out, key=lambda p: (-len(p), p)))


def _has_word(keyword: str, text: str, prefixes: List[str]) -> bool:
    """``keyword`` (already folded) occurs in ``text`` (already folded) as
    whole words: nothing word-like before or after it. A keyword that starts
    with an Arabic letter may carry ONE leading prefix from ``prefixes``."""
    if not keyword:
        return False
    lead = ""
    if prefixes and re.match(r"[\u0621-\u064a]", keyword):
        lead = "(?:" + "|".join(re.escape(p) for p in prefixes) + ")?"
    return re.search(rf"(?<![{_WORD_CHARS}]){lead}{re.escape(keyword)}(?![{_WORD_CHARS}])",
                     text) is not None


def _norm(text: str) -> str:
    return _AR_MARKS.sub("", (text or "").lower()).translate(_AR_LETTERS)


def _is_arabic(text: str) -> bool:
    return re.search(r"[\u0621-\u064a]", text) is not None


def _fold_name(text: str) -> str:
    return _NAME_SEPARATORS.sub(" ", _norm(text)).strip()


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

    The same holds on every path: when two kinds share the top score (by
    name, extension, content or any mix), the document is `unrecognised`
    rather than given to whichever kind the table lists first.
    """
    fn = _norm(filename)
    ext = os.path.splitext(fn)[1]
    stem = _fold_name(os.path.splitext(fn)[0] if ext else fn)
    content = _norm(content_sample)

    best_type: Dict[str, Any] = {}
    best_score = 0
    best_matched: List[str] = []
    name_matched_any = False
    scores: List[int] = []
    prefixes = _ar_prefixes()

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
            # Arabic words are matched the same way (see ``_has_word``).
            if (_has_word(_norm(kw), fn, prefixes)
                    or _has_word(_fold_name(kw), stem, prefixes)):
                score += 2
                matched.append(f"filename:{kw}")
        exts = [e.lower() for e in (m.get("extensions", []) or [])]
        if ext and ext in exts:
            score += 3
            matched.append(f"ext:{ext}")
        if score:
            name_matched_any = True
        for kw in m.get("content", []) or []:
            # Arabic content words are whole words too; a Latin content
            # phrase keeps its substring match (unchanged by this rule).
            kw_c = _norm(kw)
            if (_has_word(kw_c, content, prefixes) if _is_arabic(kw_c)
                    else kw_c in content):
                score += 1
                matched.append(f"content:{kw}")
        scores.append(score)
        if score > best_score:
            best_score, best_type, best_matched = score, t, matched

    # A tie at the top between kinds is unrecognised on every path: no kind
    # wins by its place in the table.
    if scores.count(best_score) > 1:
        return _unrecognised()
    # No name or extension matched, so the score is content hits alone: it
    # decides only with enough distinct evidence for one kind. A document
    # that matched nothing (score 0) fails the bar here too, since the bar
    # is at least 1; any name match scores >= 2.
    if not name_matched_any and best_score < CONTENT_ONLY_MIN_HITS:
        return _unrecognised()
    return {
        "name": best_type["name"],
        "score": best_score,
        "matched_on": best_matched,
        "expected_fields": best_type.get("expected_fields", []),
        "needs_user_confirmation": False,
    }
