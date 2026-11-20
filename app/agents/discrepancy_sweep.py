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
    _competing_figure_sentences,
    _figure_mentions,
    _figure_unit,
)


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
    out: list[str] = []
    have: set[str] = set()
    for m in _DESIGNATION_RE.finditer(text or ""):
        # A sentence-ending period after "ACI 318." is punctuation, not part
        # of the designation; internal dots ("Table 7.3.1.1") are kept.
        token = re.sub(r"[.,;:]+$", "", m.group(0).strip()).lower().split()
        token = " ".join(token)
        if not token or token in have:
            continue
        have.add(token)
        out.append(token)
    return out


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


def sweep_differences(
    stated_text: str,
    candidates: Iterable[tuple[str, str]],
) -> list[SweepHit]:
    """The candidate passages that state a different figure for the same
    distinctive identifier.

    ``stated_text`` is the draft's cited sentence. ``candidates`` is
    ``(source, passage_text)`` -- the passages the caller's one narrow lexical
    search returned. A candidate hits only when it shares at least one
    designation with the stated sentence AND carries a figure that differs
    from a stated figure (same unit; the product's own close-number tolerance
    applies). Pure; no retrieval, no model, no I/O.
    """
    stated_ids = set(_designations(stated_text))
    stated = _stated_figures(stated_text)
    if not stated_ids or not stated:
        return []

    hits: list[SweepHit] = []
    for source, text in candidates:
        shared = stated_ids & set(_designations(text))
        if not shared:
            continue
        for s in stated:
            differing = _competing_figure_sentences(
                text, s.number, s.unit, frozenset(),
            )
            if not differing:
                continue
            hits.append(SweepHit(
                identifier=",".join(sorted(shared)),
                stated_value=s.value,
                differing_value=differing[0][0],
                source=source,
                passage=text,
            ))
            break  # one hit per candidate passage
    return hits
