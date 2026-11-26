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

``sweep_differences`` is the DETECTION half and is deliberately pure: no retrieval, no
model, no I/O. It decides, from the draft's cited sentence and the candidate
passages the caller already retrieved, which candidates state a *different*
figure for the *same* distinctive identifier.

``sweep_rows`` is the CALLER half the answer exit runs: it takes the draft's
sentences that state a typed value beside an identifier, makes the ONE
lexical search (injected: the exit passes the project's ``bm25_search``) with
those identifiers, and returns each differing passage as an evidence row of
the shape the turn already records. Its log line carries counts only, so no
passage text or identifier leaves the turn.

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

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from app.agents.citation_provenance import (
    _DESIGNATION_RE,
    _competing_figure_sentences,
    _figure_mentions,
    _figure_unit,
)

_LOG = logging.getLogger(__name__)


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

    # An empty ``stated_ids`` or ``stated`` is handled by the loop itself:
    # no candidate can share a designation from an empty id set, and no
    # candidate can hit when there is no stated figure to compare against.
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


#: How many passages the one narrow search reads.
SWEEP_K = 20


def stated_sentences(draft: str) -> list[str]:
    """The draft's sentences that state a typed value next to a distinctive
    identifier -- the only statements the sweep checks."""
    from app.agents.evidence_conflicts import _sentences

    return [s for s in _sentences(draft or "") if _designations(s) and _stated_figures(s)]


def sweep_rows(
    draft: str,
    handed: Iterable[dict],
    search: Callable[[str, int], Sequence[Any]],
    names: Callable[[list[str]], dict[str, str]],
) -> list[dict[str, Any]]:
    """The caller half: the evidence rows the sweep adds to the turn.

    ``draft`` is the answer as drafted; ``handed`` the evidence rows the turn
    already has (a passage already handed is never searched for again).
    ``search(query, k)`` is the ONE lexical search over the project, run once
    with the stated sentences' identifiers; it returns chunks (``doc_id``,
    ``chunk_index``, ``text``, ``page``). ``names(doc_ids)`` names the
    documents of the passages found. Each passage that states a different
    figure for the same identifier comes back as an evidence row of the same
    shape the turn records for handed passages. A passage whose document has
    no name is not returned: it could not be cited.

    A designation that is part of the name of a handed document (a drawing or
    sheet number the draft cites as its source) names where a value is
    written, not what the value is for: it is neither searched for nor
    matched on.
    """
    handed = [r for r in handed if isinstance(r, dict)]
    labels = {d for r in handed for d in _designations(str(r.get("doc_name") or ""))}
    stated = [s for s in stated_sentences(draft) if set(_designations(s)) - labels]
    if not stated:
        return []
    identifiers = list(dict.fromkeys(i for s in stated for i in _designations(s) if i not in labels))
    have = {(str(r.get("doc_id") or ""), r.get("chunk_index")) for r in handed}
    fresh = [c for c in search(" ".join(identifiers), SWEEP_K)
             if str(getattr(c, "text", "") or "").strip()
             and (str(c.doc_id or ""), c.chunk_index) not in have]
    candidates = [(str(i), str(c.text)) for i, c in enumerate(fresh)]
    hit: set[int] = set()
    for sentence in stated:
        hit.update(int(h.source) for h in sweep_differences(sentence, candidates)
                   if set(h.identifier.split(",")) - labels)
    found = [fresh[i] for i in sorted(hit)]
    _LOG.info("discrepancy_sweep: identifiers=%d searched=%d differing=%d",
              len(identifiers), len(fresh), len(found))
    named = names(list(dict.fromkeys(str(c.doc_id or "") for c in found))) if found else {}
    rows: list[dict[str, Any]] = []
    for c in found:
        name = named.get(str(c.doc_id or ""), "")
        if not name:
            continue
        page = getattr(c, "page", None)
        page = page if isinstance(page, int) and not isinstance(page, bool) else None
        rows.append({"doc_id": str(c.doc_id), "doc_name": name, "page": page,
                     "page_or_section": f"p. {page}" if page else "",
                     "chunk_index": c.chunk_index, "snippet": str(c.text).strip()})
    return rows
