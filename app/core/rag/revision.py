"""Revision currency (audit §5.2) — ALWAYS ON, flag-independent.

A stale-revision answer on a drawing (a dimension, detail, or spec that changed
in a later revision) is a real construction-safety and liability hazard, and it
is exactly the failure this product exists to prevent. This module supplies the
pure, filename-derived signals the retriever uses to:

  1. firmly DOWN-RANK any document whose own filename marks it superseded
     (``superseded|obsolete|archive|old|previous|deprecated``), so a current
     document of comparable relevance always outranks it; and
  2. among retrieved chunks that share a DRAWING NUMBER, prefer the highest
     revision — suppressing a lower revision when a strictly-higher, comparable
     one is also retrieved.

Everything keys off the ORIGINAL FILENAME the uploader gave — the one signal the
retriever already resolves per chunk (``_doc_name_for_id``). So nothing here
depends on ``RAG_LAYERED``, on the layered-authority backfill, or on any
persisted metadata column: revision currency holds even with layered RAG off.

Conservatism is the rule. An unparseable name yields no signal and the chunk is
left exactly where cosine put it; two revisions are only ordered when they are
the same KIND (both numeric or both alphabetic) — guessing that ``Rev 2``
supersedes ``Rev B`` could suppress a current document, the one outcome worse
than showing both.

Multi-character revisions ARE handled (the 2026-07-30 follow-up): ``Rev 10``
outranks ``Rev 9`` and ``Rev AA`` outranks ``Rev Z``. A revision marker is only
read when it follows a boundary, so a drawing-number digit run (the ``2100`` in
``STR2100``) is never mistaken for a revision; anything still ambiguous (a >3
digit run, a lowercase token) yields no signal and both chunks are kept.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

# Keyword set shared verbatim with the layered-RAG 'historical' authority
# classifier (app/core/rag/layers.py::_AUTHORITY_PATTERNS). Kept here as the
# single source of truth so the always-on penalty below and the (flag-gated)
# layered down-weight can never disagree on what "superseded" means.
SUPERSEDED_PATTERN = r"superseded|obsolete|archive|\bold\b|previous|deprecated"
_SUPERSEDED_RE = re.compile(SUPERSEDED_PATTERN, re.IGNORECASE)

# Drawing sheet code, e.g. 'A-101', 'STR2100', 'M_401'. Same shape the
# construction container extracts (_extract_drawing_number): 1-3 letters, an
# optional -/_ separator, then 3-6 digits.
_DRAWING_NUMBER_RE = re.compile(r"[A-Z]{1,3}[-_]?\d{3,6}", re.IGNORECASE)

# Revision token, e.g. 'Rev C', 'R2', 'rev A', 'Rev 10', 'Rev AA'. Handles
# MULTI-CHARACTER revisions (the 2026-07-30 follow-up to §5.2): 'Rev 10' now
# parses to '10', not '1' — the old single-char capture ranked Rev 10 BELOW
# Rev 9, so the retriever could suppress the CURRENT sheet in favour of a stale
# one (the exact hazard this module prevents).
#
# Conservatism preserved two ways: (1) the marker MUST follow a non-alphanumeric
# boundary (start / space / _ / - / .), so a drawing-number digit run like the
# '2100' in 'STR2100' is never mistaken for a revision; (2) a digit run longer
# than 3 (or a lowercase token) yields no match and the chunk keeps its place —
# missing a revision is always safe, mis-ordering the current one is not.
_REVISION_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])[Rr][Ee]?[Vv]?\s*[-_]?\s*([0-9]{1,3}|[A-Z]{1,2})\b"
)

# Firm demotion applied to a superseded chunk's fused score. Large relative to
# the cosine spread among top-K neighbours (typically ~0.0-0.2) so a current
# document of comparable relevance reliably outranks a superseded one — yet a
# subtraction, NOT a hard drop, so a superseded chunk still survives as a LAST
# resort when nothing current matches (a flagged stale answer beats no answer).
SUPERSEDED_PENALTY = 0.5


def is_superseded(name: Optional[str]) -> bool:
    """True when a filename marks the document superseded/obsolete/archived."""
    return bool(name and _SUPERSEDED_RE.search(name))


def drawing_number(name: Optional[str]) -> str:
    """Uppercased drawing sheet code from a filename, or '' when none matches."""
    if not name:
        return ""
    m = _DRAWING_NUMBER_RE.search(name)
    return m.group(0).upper() if m else ""


def revision_token(name: Optional[str]) -> str:
    """Uppercased revision token from a filename, or '' when none parses.

    Handles single- AND multi-character revisions: 'A-101_RevC' -> 'C',
    'A-101-Rev10' -> '10', 'A-101-RevAA' -> 'AA'. A marker not preceded by a
    boundary (e.g. the 'R2100' inside 'STR2100') is ignored, and a digit run
    longer than 3 yields '' — both keep the chunk in place rather than risk a
    mis-ordering.
    """
    if not name:
        return ""
    m = _REVISION_RE.search(name)
    return m.group(1).upper() if m else ""


def revision_rank(token: str) -> Optional[Tuple[int, int]]:
    """Sortable key for a revision token, or ``None`` when it cannot be ordered.

    Returns ``(kind, value)`` where ``kind`` is ``0`` for a numeric revision
    (``9 < 10 < 11`` — multi-digit ordered by integer value, the §5.2 follow-up
    fix) and ``1`` for an alphabetic one, ordered as the standard drawing
    sequence ``A < B < ... < Z < AA < AB`` via base-26 (so 'AA' correctly
    supersedes 'Z'). Callers must compare ranks ONLY within the same ``kind``:
    a numeric and a letter revision of the same sheet are deliberately left
    un-ordered so a mixed-scheme sheet keeps both rather than risking
    suppression of the current one. An empty/unparseable token returns
    ``None`` (no ordering, keep as-is).
    """
    if not token:
        return None
    token = token.upper()
    if token.isdigit():
        return (0, int(token))
    if token.isalpha() and 1 <= len(token) <= 2 and all("A" <= c <= "Z" for c in token):
        # Base-26 spreadsheet-style: A=1 .. Z=26, AA=27, AB=28.
        value = 0
        for c in token:
            value = value * 26 + (ord(c) - ord("A") + 1)
        return (1, value)
    return None
