"""Discrepancy sweep: the pure detection half (owner 06:01Z, 19 Nov UTC).

The owner's rule: when the drafted answer states a value for what was asked,
the exit takes the DISTINCTIVE IDENTIFIERS of that statement -- a standard or
code reference, an item or drawing reference, read by the typed readers the
product already has -- and runs ONE narrow lexical search over the project for
passages that carry the same identifier together with a typed value of the
same kind whose value DIFFERS. Every such passage is added to the turn's
evidence, and the existing conflict statement then does the rest (every
value, every origin with its standing, "the documents differ", the user
decides).

This module is the DETECTION half and is deliberately pure: no retrieval, no
model, no I/O. It decides, from the draft's cited sentence and the candidate
passages the caller already retrieved, which candidates state a *different*
figure for the *same* distinctive identifier. The caller owns the single
lexical search (``bm25_search``) and the evidence merge; the trace records
identifiers as digests (also the caller, so no passage text or identifier
leaves the turn).

Readers are reused, not re-implemented, from ``citation_provenance`` (the
same typed readers the conflict statement uses):

- ``_DESIGNATION_RE`` reads a clause / code / standard / table designation
  ("ACI 318", "Vol 6", "Table 7.3.1.1") -- the distinctive identifier.
- ``_figure_mentions`` / ``_figure_unit`` read a typed value (a number with a
  unit, currency or percent; a bare number is never a figure).
- ``_competing_figure_sentences`` already answers "does this text state
  another figure in the same unit" with the product's own close-number /
  rounding tolerance, so a repeated figure or a rounding is never a
  difference.

No case, document, value, row, standard or item is named anywhere here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from app.agents.citation_provenance import (
    _DESIGNATION_RE,
    _close,
    _competing_figure_sentences,
    _figure_mentions,
    _figure_unit,
)
from app.agents.clarify_cases import _CLAUSE_VALUE_RE, _REFERENCE_RE
from app.agents.evidence_conflicts import extract as _typed_values
from app.agents.evidence_conflicts import same as _same_value


@dataclass(frozen=True)
class Stated:
    """One typed value the draft states, as the figure reader read it."""

    value: str
    number: float
    unit: str


@dataclass(frozen=True)
class SweepHit:
    """A candidate passage that states a different figure for the same
    distinctive identifier. ``passage`` is the candidate's own text; the
    caller adds it to the evidence (and records the trace as digests)."""

    identifier: str
    stated_value: str
    differing_value: str
    source: str
    passage: str


def _designations(text: str) -> list[str]:
    """The distinctive identifiers in ``text``, first-seen, normalised.

    A clause / code / standard / table designation, read by shape -- never a
    list of known identifiers.
    """
    t = text or ""
    out: list[str] = []
    have: set[str] = set()
    taken: list[tuple[int, int]] = []
    # A designation read over a figure ("Depth 5" in "Depth 5 m") is the
    # value itself, not what the value is stated for: never an identifier.
    figures = [m.span() for m, _n in _figure_mentions(t)]
    for m in _DESIGNATION_RE.finditer(t):
        if any(m.start() < end and start < m.end() for start, end in figures):
            continue
        # A sentence-ending period after "ACI 318." is punctuation, not part
        # of the designation; internal dots ("Table 7.3.1.1") are kept.
        token = re.sub(r"[.,;:]+$", "", m.group(0).strip()).lower().split()
        token = " ".join(token)
        if not token or token in have:
            continue
        have.add(token)
        out.append(token)
        taken.append(m.span())
    # A bare clause number ("4.2.1 Minimum cover ...", "per 4.2.1") is how a
    # document heads and cites its own clauses without a label word. It is
    # read by the clarifying-question exit's reference reader, kept only in
    # the clause shape (three or more numbers joined by dots, so a decimal
    # "4.5" is never one), and only where it is not already part of a
    # labelled designation ("Clause 4.2.1") or of a figure.
    for m in _REFERENCE_RE.finditer(t):
        token = m.group(0)
        if not _CLAUSE_VALUE_RE.match(token) or _is_date_shape(token):
            continue
        if any(m.start() < end and start < m.end()
               for start, end in figures + taken):
            continue
        if token in have:
            continue
        have.add(token)
        out.append(token)
    return out


def _is_date_shape(token: str) -> bool:
    """A dotted date ("12.03.2026", "2026.03.12") carries a four-digit year
    part; a clause number does not. Shape only."""
    return any(len(part) >= 4 for part in token.split("."))


def _clause_number(token: str) -> str:
    """The trailing number of a designation token ("clause 4.2.1" ->
    "4.2.1", "cl.4.2.1" -> "4.2.1"), or "" when it has none."""
    m = re.search(r"\d+(?:\.\d+)*$", token)
    return m.group(0) if m else ""


def _same_identifier(a: str, b: str) -> bool:
    """Two designation tokens name the same thing.

    Equal tokens do. A bare clause number names the same thing as a labelled
    designation that ends in that number ("4.2.1" and "clause 4.2.1"): the
    bare form carries no label to disagree with. Two DIFFERENT labels on the
    same number ("table 7.3.1.1" and "clause 7.3.1.1") never do -- different
    structures are never one identifier.
    """
    if a == b:
        return True
    if _CLAUSE_VALUE_RE.match(a):
        return _clause_number(b) == a
    if _CLAUSE_VALUE_RE.match(b):
        return _clause_number(a) == b
    return False


def _stated_figures(text: str) -> list[Stated]:
    """Every typed value the draft states (a figure with its unit)."""
    out: list[Stated] = []
    for m, number in _figure_mentions(text or ""):
        out.append(Stated(
            value=m.group(0).strip(),
            number=number,
            unit=_figure_unit(m),
        ))
    return out


def _measured(text: str) -> list:
    """The measured values in ``text`` as the conflict statement reads them
    (a quantity in a unit, or money in one currency), each normalised to its
    kind's base unit. A date, a grade or a labelled phrase is not measured."""
    return [v for v in _typed_values(text or "") if isinstance(v.norm, tuple)]


def _converted_difference(stated_text: str, text: str) -> tuple[str, str] | None:
    """The first stated value and a candidate value of the SAME kind that
    differs after unit conversion ("40 mm" against "0.05 m"), else None.

    The reader and the equality are the conflict statement's own
    (``evidence_conflicts.extract`` / ``same``), so this recall and the
    judgement it feeds agree on what one value is: a figure written in
    another unit of the same kind that converts to the stated value ("50 mm",
    "0.05 m") is the same value and never recalled, nor is one that rounds to
    it (the product's close-number tolerance on the base values). A value of
    another kind (a length against a mass) is never compared.
    """
    theirs = _measured(text)
    for mine in _measured(stated_text):
        for other in theirs:
            if other.kind != mine.kind or _same_value(mine, other):
                continue
            if _close(*_in_stated_unit(mine, other)):
                continue
            return mine.raw.strip(), other.raw.strip()
    return None


def _in_stated_unit(mine, other) -> tuple[float, float]:
    """The stated value as written and the other value expressed in the
    stated value's unit, so the close-number tolerance applies at the scale
    the draft wrote it (in base metres, 40 mm and 41 mm would sit inside the
    tolerance's absolute floor). A value written as zero has no scale; the
    base values are compared then."""
    written = [float(n.replace(",", "")) for n in _WRITTEN_NUMBER_RE.findall(mine.raw)]
    number = sum(written) / len(written) if written else 0.0
    base = sum(mine.norm) / 2
    if not number or not base:
        return base, sum(other.norm) / 2
    return number, sum(other.norm) / 2 / (base / number)


_WRITTEN_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def sweep_differences(
    stated_text: str,
    candidates: Iterable[tuple[str, str]],
) -> list[SweepHit]:
    """The candidate passages that state a different figure for the same
    distinctive identifier.

    ``stated_text`` is the draft's cited sentence. ``candidates`` is
    ``(source, passage_text)`` -- the passages the caller's one narrow lexical
    search returned. A candidate hits only when it shares at least one
    designation with the stated sentence AND states, for that designation, a
    figure that differs from a stated figure: in the same unit (the product's
    own close-number tolerance applies), or in another unit of the same kind
    once both are converted (the conflict statement's own reader and
    equality). A figure stated for that designation is one inside its scope
    (``_identifier_scope``); a figure elsewhere in the passage is stated for
    something else. Pure; no retrieval, no model, no I/O.
    """
    stated_ids = _designations(stated_text)
    stated = _stated_figures(stated_text)

    # An empty ``stated_ids`` or ``stated`` is handled by the loop itself:
    # no candidate can share a designation from an empty id set, and no
    # candidate can hit when there is no stated figure to compare against.
    hits: list[SweepHit] = []
    for source, text in candidates:
        theirs = _designations(text)
        shared = {a for a in stated_ids
                  if any(_same_identifier(a, b) for b in theirs)}
        if not shared:
            continue
        scoped = _identifier_scope(text, shared)
        hit = None
        for s in stated:
            differing = _competing_figure_sentences(
                scoped, s.number, s.unit, frozenset(),
            )
            if differing:
                hit = (s.value, differing[0][0])
                break
        if hit is None:
            hit = _converted_difference(stated_text, scoped)
        if hit is None:
            continue
        hits.append(SweepHit(
            identifier=",".join(sorted(shared)),
            stated_value=hit[0],
            differing_value=hit[1],
            source=source,
            passage=text,
        ))
    return hits


def _identifier_scope(text: str, shared: set[str]) -> str:
    """The part of a candidate passage that states something FOR one of the
    ``shared`` designations; a figure outside it is stated for something
    else ("... comply with <code>. The slab is 200 mm" states no value for
    the code).

    Two places are in scope, both read by shape:

    - a sentence that names a shared designation;
    - the body under a heading line led by a shared designation (the line
      opens with it, states no figure and does not end as a sentence),
      down to the next line led by any designation -- how a document heads
      a clause and states its values in the lines beneath.

    Designations are compared with ``_same_identifier``, the relation that
    decided they are shared.
    """
    kept: list[str] = []
    under = False
    for line in (text or "").splitlines():
        line = line.strip()
        lead = _leading_designation(line)
        if lead is not None:
            heading = not _stated_figures(line) and not line.endswith((".", "!", "?"))
            under = heading and _names_shared([lead], shared)
        for sentence in _SENTENCE_END_RE.split(line):
            if sentence.strip() and (under or _names_shared(_designations(sentence), shared)):
                kept.append(sentence.strip())
    return "\n".join(kept)


def _leading_designation(line: str) -> str | None:
    """The designation ``line`` opens with, else None."""
    opening = " ".join(line.lower().split())
    return next((d for d in _designations(line) if opening.startswith(d)), None)


def _names_shared(found: Iterable[str], shared: set[str]) -> bool:
    """Does any designation in ``found`` name one of the ``shared`` ones?"""
    return any(_same_identifier(a, b) for a in shared for b in found)


# A sentence ends at . ! ? followed by space and a capital, or at a line
# break. A dot inside a designation ("Table 7.3.1.1", "4.2 m") is followed by
# a digit, never by a space and a capital, so it does not end the sentence.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[(])|\n+")


def sweep_targets(draft: str) -> tuple[list[str], str]:
    """What a drafted answer gives the sweep to look for.

    Returns ``(stated, query)``: ``stated`` is every sentence of the draft
    that states a typed value (a figure with its unit) AND carries a
    distinctive identifier -- the only sentences whose value another document
    could state differently for the same identified thing; ``query`` is the
    ONE lexical query the caller runs, the identifiers of those sentences,
    first-seen. A draft with no such sentence gives ``([], "")`` and the
    caller runs no search. Pure.
    """
    stated: list[str] = []
    ids: list[str] = []
    for sentence in _SENTENCE_END_RE.split(draft or ""):
        sentence = sentence.strip()
        if not sentence or not _stated_figures(sentence):
            continue
        found = _designations(sentence)
        if not found:
            continue
        stated.append(sentence)
        ids.extend(i for i in found if i not in ids)
    return stated, " ".join(ids)
