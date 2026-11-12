"""Edition-aware candidate demotion in the retrieval ranking.

An ask that names a standard AND its edition/version/year — "FIDIC 2017",
"ISO 9001:2015", "NFPA 51B (2019)" — must not be answered from a source that
carries a DIFFERENT edition of the same standard (the project's 1999
conditions when 2017 was asked). The retriever demotes such a candidate below
candidates that match the asked edition or carry no edition; a candidate with
no edition token is unaffected.

Nothing here names a standard, a year or a clause number: the standard and the
edition tokens are read from the ask at run time. The only hardcoded shapes are
generic — a 4-digit year in the 1900–2099 range, and an all-caps acronym with an
optional numeric code as the standard name — not any specific standard or
edition. Demotion (never promotion) keeps the failure mode safe: a false
standard pick only lowers a candidate that also carries a conflicting edition
token, and a candidate with no edition token never moves.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.core.rag.vector_store import Chunk

# A 4-digit year token. The RANGE is the general mechanism; no specific year is
# written here — the ask supplies the year at run time.
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# Explicit edition/version marker with its value AFTER the word: "edition 2017",
# "version 4", "ed. 3". A value BEFORE the word ("2017 edition") is already
# covered by the year token above.
_EDITION_VALUE_RE = re.compile(
    r"\b(?:edition|version|ed\.)\b\s*[:\-]?\s*([A-Za-z0-9]+)",
    re.IGNORECASE,
)

# A standard name: an all-caps acronym, optionally with a numeric code suffix —
# "FIDIC", "NFPA 51B", "ISO 9001". The letters/numbers come from the ask.
_ACRONYM_CODE_RE = re.compile(r"[A-Z]{2,6}(?:[\s_-]?\d+[A-Z]?)?")

# How far (chars) either side of an edition token to look for its standard name.
_STANDARD_WINDOW = 40

# Firm demotion for a candidate that carries a conflicting edition. Large
# relative to the top-K cosine spread (~0.0–0.2), like the superseded-document
# penalty, so a matching / no-edition candidate reliably outranks it — a
# subtraction, not a drop, so it still survives as a last resort.
EDITION_CONFLICT_PENALTY = 0.5


@dataclass(frozen=True)
class EditionStandard:
    """One (standard, edition) pair the ask names."""

    standard: str
    edition: str


def _nearest_standard(text: str, pos: int) -> Optional[str]:
    """The acronym/code nearest to the edition token at ``pos``.

    Prefers the closest acronym BEFORE the token (the dominant "FIDIC 2017"
    form), then the first acronym AFTER it ("2017 FIDIC"). The window keeps an
    unrelated acronym elsewhere in the sentence from being mistaken for the
    standard being edition-qualified.
    """
    prefix = text[max(0, pos - _STANDARD_WINDOW):pos]
    last: Optional[re.Match] = None
    for match in _ACRONYM_CODE_RE.finditer(prefix):
        last = match
    if last is not None:
        return last.group(0)
    suffix = text[pos:pos + _STANDARD_WINDOW]
    match = _ACRONYM_CODE_RE.search(suffix)
    return match.group(0) if match else None


def extract_edition_standards(query: str) -> List[EditionStandard]:
    """(standard, edition) pairs the ask names. Empty when it names none."""
    text = query or ""
    found: List[EditionStandard] = []
    seen: set = set()

    for match in _YEAR_RE.finditer(text):
        standard = _nearest_standard(text, match.start())
        if standard:
            key = (standard.upper(), match.group(0))
            if key not in seen:
                seen.add(key)
                found.append(EditionStandard(standard, match.group(0)))

    for match in _EDITION_VALUE_RE.finditer(text):
        standard = _nearest_standard(text, match.start())
        if standard:
            key = (standard.upper(), match.group(1).upper())
            if key not in seen:
                seen.add(key)
                found.append(EditionStandard(standard, match.group(1)))

    return found


def _standard_pattern(standard: str) -> re.Pattern:
    """A word-boundary pattern for the standard name, tolerant of separators in
    the code suffix ("ISO 9001" matches "ISO-9001" / "ISO9001")."""
    match = re.match(r"([A-Z]{2,6})(?:[\s_-]?(\d+[A-Z]?))?$", standard, re.IGNORECASE)
    if not match:
        return re.compile(r"\b" + re.escape(standard) + r"\b", re.IGNORECASE)
    acronym, code = match.group(1), match.group(2)
    if code:
        return re.compile(
            r"\b" + re.escape(acronym) + r"[\s_-]*" + re.escape(code) + r"\b",
            re.IGNORECASE,
        )
    # A pure acronym may be glued to its edition/code by a separator in a
    # filename ("ACME_1999", "ACME-2017"), so a trailing word boundary is too
    # strict; only a trailing LETTER rules the match out ("ACMEB" is a
    # different standard, "ACME2017" is ACME's 2017 edition).
    return re.compile(r"\b" + re.escape(acronym) + r"(?![A-Za-z])", re.IGNORECASE)


def _edition_tokens(blob: str) -> List[str]:
    """Edition tokens (years + explicit edition/version values) in a blob."""
    out: List[str] = []
    for match in _YEAR_RE.finditer(blob):
        out.append(match.group(0))
    for match in _EDITION_VALUE_RE.finditer(blob):
        out.append(match.group(1))
    return out


def _adjacent_edition_tokens(blob: str, start: int, end: int) -> List[str]:
    """Edition tokens within _STANDARD_WINDOW of a standard occurrence's span."""
    window = blob[max(0, start - _STANDARD_WINDOW):end + _STANDARD_WINDOW]
    return _edition_tokens(window)


def candidate_conflicts(
    text: str,
    source_name: str,
    standard: str,
    edition: str,
) -> bool:
    """True when the candidate carries the SAME standard with a DIFFERENT
    edition ADJACENT to an occurrence of the standard name, and no occurrence
    of the standard has the asked edition adjacent.

    A year token far from the standard (a signing date, a project code in a
    filename) does not make the candidate conflict; only an edition token that
    actually qualifies an occurrence of the standard does. A candidate with no
    edition token, or one where a matching occurrence exists, is unaffected
    (False).
    """
    joined = "\n".join(part for part in (text, source_name) if part)
    pattern = _standard_pattern(standard)
    edition_lower = (edition or "").lower()
    matched = False
    conflicting = False
    for occurrence in pattern.finditer(joined):
        adjacent = _adjacent_edition_tokens(
            joined, occurrence.start(), occurrence.end())
        if not adjacent:
            continue
        if any(token.lower() == edition_lower for token in adjacent):
            # This occurrence names the asked edition; a matching occurrence
            # means the candidate is not in conflict regardless of any other
            # stray year token elsewhere.
            matched = True
        else:
            conflicting = True
    return conflicting and not matched


def apply_edition_conflict_demotion(
    query: str,
    scored: List[Tuple[float, Chunk]],
    name_by_id: Dict[str, str],
) -> None:
    """In-place: demote candidates that carry a conflicting edition of a
    standard the ask named, below matching / no-edition candidates. A query
    with no edition pair is a no-op: the inner loop never fires."""
    pairs = extract_edition_standards(query)
    for i, (score, chunk) in enumerate(scored):
        name = (
            name_by_id.get(chunk.doc_id, "")
            or getattr(chunk, "source_name", "")
            or ""
        )
        for pair in pairs:
            if candidate_conflicts(
                chunk.text or "", name, pair.standard, pair.edition,
            ):
                demoted = score - EDITION_CONFLICT_PENALTY
                chunk.score = round(demoted, 6)
                scored[i] = (demoted, chunk)
                break
