"""Which of a project's files does a message name? Pure string matching.

Kept free of heavy imports so the retrieval worker process
(``app.core.rag.retrieval_worker``) can run it: matching a message against
thousands of document names is CPU work that, run on a thread, starved the
web process's event loop of the GIL (live 2026-10-07, 15 users).
"""
from __future__ import annotations

import os
import re
from typing import Iterable, List

FILE_NAME_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]{11,}")
_FILE_LIKE_RE = re.compile(r"[_.\-0-9]")


def user_names_project_file(user_low: str, original_name: str) -> bool:
    """True when the user message names this project file.

    Full ``original_name`` match first. A distinctive stem (≥12 chars) also
    matches so a user can say ``site_waterproofing_spec`` without the
    upload timestamp suffix. The user token is the *shorter* string when
    the stored name has a timestamp; match both directions.
    """
    name = (original_name or "").strip().lower()
    if not name or not user_low:
        return False
    if name in user_low:
        return True
    stem = os.path.splitext(name)[0]
    if len(stem) >= 12 and stem in user_low:
        return True
    if len(stem) < 12:
        return False
    for m in FILE_NAME_TOKEN_RE.finditer(user_low):
        token_stem = os.path.splitext(m.group(0).rstrip(".,;:)"))[0]
        # A plain word ("specification") is language, not a file name; a
        # stem the user typed joins words or carries a digit.
        if not _FILE_LIKE_RE.search(token_stem):
            continue
        if len(token_stem) >= 12 and token_stem in stem:
            return True
    return False


def names_mentioned(user_low: str, names: Iterable[str]) -> List[str]:
    """The names (stripped) that ``user_low`` mentions, in the given order."""
    out = []
    for raw in names:
        name = (raw or "").strip()
        if name and user_names_project_file(user_low, name):
            out.append(name)
    return out
