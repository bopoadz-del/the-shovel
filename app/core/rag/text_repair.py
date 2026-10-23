"""Repair text-layer artefacts that make an indexed chunk unreadable.

One artefact so far: FAKE BOLD. Some PDF producers draw bold by printing each
glyph twice (occasionally three times) at a tiny offset, and the text layer
comes out as ``PPaarrttyy aanndd EEnnggiinneeeerr ddeettaaiillss``. Live, that
turned a Contract Data row into key ``PPLLEECCOO((EE: XX22MM SSaauuddii ...``
value ``Clause (as`` -- and "Clause (as" was then returned as the Engineer.

Applied where chunk text is READ, so documents already in the index are
repaired without a re-index and every downstream rule sees the same text.

What matters most is what is NOT touched. ``1100``, ``2233``, ``AABB-0011``
are real quantities and drawing codes, and a repair that rewrote them would
be a worse defect than the one it fixes. So:

* a CHUNK is treated as fake-bold only when it carries at least three tokens
  that are unmistakably doubled (three or more repeated pairs, with letters);
  one such token in clean text is a coincidence and is left alone. The chunk,
  not the line: a table breaks lines at cells, and a doubled cell is often a
  single word (``|: | EEnnggiinneeeerr``);
* within such a chunk, only LINES that themselves carry a doubled word are
  touched;
* inside such a line only tokens that are doubled THROUGHOUT are collapsed,
  so a clean word keeps its own double letters (``address``, ``committee``);
* a digits-only token is collapsed only when the tokens on both sides of it
  were, because a number is the one thing that cannot vouch for itself.
"""
from __future__ import annotations

import re
from typing import List, Optional

_TOKEN_RE = re.compile(r"\S+")
_EDGE_PUNCT = ".,;:!?|)]}([{\"'"
_MIN_STRONG_GROUPS = 3
_MIN_STRONG_TOKENS_PER_CHUNK = 3


def _collapse_exact(tok: str, k: int) -> Optional[str]:
    """``tok`` with every glyph repeated exactly ``k`` times, undone; else None."""
    if len(tok) < k or len(tok) % k:
        return None
    out = []
    for i in range(0, len(tok), k):
        group = tok[i:i + k]
        if group != group[0] * k:
            return None
        out.append(group[0])
    return "".join(out)


def _collapse(tok: str) -> Optional[str]:
    """Collapse a doubled/tripled token, tolerating one un-doubled edge mark.

    ``ddeettaaiillss:`` and ``11..33..11((bb)):`` end in a single colon that
    was printed once; the run before it is still doubled throughout.
    """
    for k in (2, 3):
        whole = _collapse_exact(tok, k)
        if whole is not None:
            return whole
        if len(tok) > k and tok[-1] in _EDGE_PUNCT:
            body = _collapse_exact(tok[:-1], k)
            if body is not None:
                return body + tok[-1]
        if len(tok) > k and tok[0] in _EDGE_PUNCT:
            body = _collapse_exact(tok[1:], k)
            if body is not None:
                return tok[0] + body
    return None


def _is_strong(tok: str, collapsed: Optional[str]) -> bool:
    """Unmistakably doubled: enough groups, and letters among them."""
    if collapsed is None:
        return False
    letters = sum(1 for ch in collapsed if ch.isalpha())
    return len(collapsed) >= _MIN_STRONG_GROUPS and letters >= 2


def _repair_line(line: str) -> str:
    matches = list(_TOKEN_RE.finditer(line))
    collapsed: List[Optional[str]] = [_collapse(m.group(0)) for m in matches]
    if not any(_is_strong(m.group(0), c) for m, c in zip(matches, collapsed)):
        return line
    take: List[bool] = []
    for i, (m, c) in enumerate(zip(matches, collapsed)):
        if c is None:
            take.append(False)
        elif m.group(0).strip(_EDGE_PUNCT).isdigit():
            # A number cannot vouch for itself: both neighbours must have been
            # doubled (a missing neighbour at the line edge does not object).
            left = collapsed[i - 1] is not None if i else True
            right = collapsed[i + 1] is not None if i + 1 < len(matches) else True
            take.append(left and right)
        else:
            take.append(True)
    out, pos = [], 0
    for m, c, use in zip(matches, collapsed, take):
        out.append(line[pos:m.start()])
        out.append(c if use and c is not None else m.group(0))
        pos = m.end()
    out.append(line[pos:])
    return "".join(out)


def repair_fake_bold(text: str) -> str:
    """Undo glyph doubling on the lines that show it; everything else verbatim."""
    if not text:
        return text
    # Cheap exit: no line can qualify without a run of three doubled letters.
    if not re.search(r"([A-Za-z])\1([A-Za-z])\2([A-Za-z])\3", text):
        return text
    strong = sum(
        1 for m in _TOKEN_RE.finditer(text)
        if _is_strong(m.group(0), _collapse(m.group(0)))
    )
    if strong < _MIN_STRONG_TOKENS_PER_CHUNK:
        return text
    return "\n".join(_repair_line(line) for line in text.split("\n"))
