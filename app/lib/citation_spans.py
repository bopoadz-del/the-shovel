"""Citation spans in an answer: the one rule for what is a citation, not a claim.

Used by the offline answer scorer (``scripts/answer_source_score.py``) and by
the conflict check (``app.agents.evidence_conflicts``): a number inside a
citation, a retrieval key or a numbered source marker is never a figure the
sentence claims. Blanking keeps every position (a span becomes spaces of the
same length), so offsets into the blanked text line up with the original.

Stdlib only; no document, project or question is named here.
"""

from __future__ import annotations

import re

# ``[source: NAME]`` / ``(source: NAME)`` written by the answer. A document
# name may itself hold parentheses ("Bill (Priced).pdf"): the bracket form
# ends only at ``]``, the paren form allows one level of inner parentheses.
NAMED_CITE_RE = re.compile(
    r"\[\s*source\s*:\s*([^\]]+)\]"
    r"|\(\s*source\s*:\s*((?:[^()]|\([^()]*\))+)\)",
    re.IGNORECASE,
)
# Raw retrieval keys a model may leave in an answer before the exit check.
KEYED_CITE_RE = re.compile(r"\b(?:chunk_id|chunk|doc_id)\s*[=:]\s*([A-Za-z0-9_\-:.]+)", re.IGNORECASE)
# A bare numbered marker: a small integer standing alone right before the end
# of a sentence or line ("... 30 minutes 2." / "... mixes 1. The").
MARKER_RE = re.compile(r"(?<=[^\d\s,.])\s(\d{1,2})(?=[.;]?(?:\n|$)|[.;]\s)")
# A bracketed numbered marker the answer contract writes (``[3]``, ``[2][4]``).
BRACKET_MARKER_RE = re.compile(r"\[\d{1,3}\]")

#: A matched span as spaces of the same length (positions stay aligned).
_SPACES = lambda m: " " * len(m.group(0))


def blank_named(text: str) -> str:
    """Blank the named and keyed citation spans."""
    return KEYED_CITE_RE.sub(_SPACES, NAMED_CITE_RE.sub(_SPACES, text or ""))


def blank_markers(text: str, listed: int) -> str:
    """Blank bare numbered markers that a Sources panel of ``listed`` entries
    numbers (a marker exists only when the panel numbers it)."""
    return MARKER_RE.sub(
        lambda m: " " * len(m.group(0)) if 1 <= int(m.group(1)) <= listed else m.group(0), text)


def blank_citations(text: str, listed: int = 0) -> str:
    """Every citation span blanked: named and keyed citations, bracketed
    markers, and bare markers a panel of ``listed`` entries numbers."""
    return blank_markers(BRACKET_MARKER_RE.sub(_SPACES, blank_named(text)), listed)
