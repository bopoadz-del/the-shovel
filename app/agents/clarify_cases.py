"""Clarify-before-choosing: the exit mechanism's decision (PR 2 part b).

The owner's 15:03Z rule (19 Nov UTC): when what the product holds supports
more than one materially different answer to the question as asked, and the
question does not say which case is meant, the turn ends as ONE short
clarifying question that names the cases it actually found, each with its
source. The next turn answers the case the user picks.

This module is the DETECTION half and is deliberately pure: no retrieval, no
model, no I/O. ``clarifying_question`` takes the question as asked, the draft
answer, and the cases already read from the handed evidence, and returns the
clarifying question (or ``None`` when the turn should stand as written). It
is tested against recorded strings, the way ``missing_input`` is.

A ``Case`` is one value claim in the handed passages:

- ``value``: a number with its unit, or a clause / document-id / rate
  reference (read by the typed readers already in the product);
- ``subject``: the noun phrase the passage's OWN sentence, heading, or detail
  title names for that value -- read from the passage's own words, never from
  a list of known structures, elements, locations, depths, sizes, materials,
  or stages;
- ``source``: document name (and page/chunk) from the evidence records.

The value+subject+source EXTRACTION from evidence is the shared reader's job
(``citation_provenance``'s figure/sentence primitives) and is wired in the
caller. This module never lists a known subject; it only compares the subjects
it is handed. Passing cases in keeps the decision testable.

Bounds (each one a test, below):

1. One clarifying question per turn, never a chain.
2. Only when the cases lead to DIFFERENT answers (same value in every case:
   answer).
3. Never when the question already names the case.
4. Never to ask for something the documents state plainly (no invented
   question).
5. Options listed are only cases actually found, each with its source.
6. "all" / no choice is the caller's next turn (every case side by side), not
   this function's output -- the question is still one question.
7. A question that is merely broad is answered, not bounced back (no value
   conflict -> this detector never fires).

Case 2 vs case 3: different values for DIFFERENT subjects, question names
none -> ask (this mechanism). Different values for the SAME subject -> the
both-values statement (#453, on main). When the passages do not let the
product tell them apart, it behaves as case 2 (it asks).

Kill-switch: ``CLARIFY_CASES=0``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass

FLAG_ENV = "CLARIFY_CASES"

#: A number token: plain, comma-grouped, or with one decimal part. Clause ids
#: such as "8.8.1" and ids such as "AB-2001-101" parse as MORE than one number
#: and are therefore compared as strings, never as floats.
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w)")

#: Words that do not identify a subject.
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or",
    "with", "without", "by", "from", "per", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having", "do", "does",
    "did", "will", "would", "shall", "should", "can", "could", "may",
    "might", "must", "this", "that", "these", "those", "it", "its",
    "their", "our", "your", "his", "her", "which", "what", "when",
    "where", "who", "whom", "how", "not", "no", "yes", "as", "than",
    "then", "there", "here", "each", "any", "some", "all", "every", "one",
}

#: Month names and abbreviations. A calendar date ("Jan-25", "30-Jan-25")
#: matches the letter-number reference SHAPE but is not a clause or document
#: reference; these mark the letter-number codes that are dates so the reader
#: skips them. A month is a language category, not a project or test case.
_MONTH_WORDS = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
})


def _is_date_token(value: str) -> bool:
    """True when a letter-number token is a calendar date, not a reference."""
    lead = re.match(r"[A-Za-z]+", (value or "").strip())
    return bool(lead) and lead.group(0).lower() in _MONTH_WORDS


def enabled() -> bool:
    """ON by default; ``CLARIFY_CASES=0`` is the kill switch."""
    return str(os.getenv(FLAG_ENV, "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )


@dataclass(frozen=True)
class Case:
    """One value claim in the handed passages."""

    subject: str
    value: str
    source: str


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _numbers(value: str) -> list[float]:
    out: list[float] = []
    for m in _NUMBER_RE.finditer(value or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            continue
    return out


def _number_of(value: str) -> float | None:
    """The single float a value carries, else None when it has none, or more
    than one (a clause / document id compared as a string)."""
    nums = _numbers(value)
    return nums[0] if len(nums) == 1 else None


def _unit_of(value: str) -> str:
    """The non-numeric remainder of a value ("mm" in "200 mm"), normalised."""
    for m in _NUMBER_RE.finditer(value or ""):
        value = value[: m.start()] + " " + value[m.end() :]
    return _norm(value)


def _subject_tokens(subject: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]{3,}", (subject or "").lower())
        if w not in _STOPWORDS
    }


def _same_subject(a: str, b: str) -> bool:
    """True only when two subjects are confidently the same phrase.

    Conservative: the default is "different" (which asks), because collapsing
    two different structures into one case is the exact per-case miss the
    standing order forbids. Same only when one token set is a subset of the
    other and they share a content token of length >= 4 (a near-identical
    phrase, not two phrases that merely share one word).
    """
    ta, tb = _subject_tokens(a), _subject_tokens(b)
    if not ta or not tb:
        return _norm(a) == _norm(b)
    smaller, larger = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    if not smaller <= larger:
        return False
    return any(len(t) >= 4 for t in smaller)


def _question_names_case(question: str, case: Case) -> bool:
    """Bound 3: the question already names its case. Conservative -- only
    treated as named when clearly so, so a vague ask still asks."""
    q = _norm(question)
    value = _norm(case.value)
    if value and value in q:
        return True
    tokens = {t for t in _subject_tokens(case.subject) if len(t) >= 4}
    return bool(tokens) and all(t in q for t in tokens)


def _text_commits(text: str, value: str) -> bool:
    """True when the draft states this case's value."""
    n = _number_of(value)
    if n is not None:
        for found in _numbers(text):
            if abs(found - n) <= 1e-3 * max(1.0, abs(found), abs(n)):
                return True
        return False
    return _norm(value) in _norm(text)


def clarifying_question(question: str, draft: str, cases: list[Case]) -> str | None:
    """The ONE clarifying question for this turn, or None to stand as written.

    ``cases`` are the value claims read from the handed evidence. The question
    fires only when the draft silently commits to exactly one of several
    materially different cases, and the question names none of them.
    """
    if not enabled():
        return None
    if not (question or "").strip() or not (draft or "").strip():
        return None
    if len(cases) < 2:
        return None

    # Dedup by (subject, value, source), preserving first appearance.
    unique: list[Case] = []
    seen: set[tuple[str, str, str]] = set()
    for case in cases:
        key = (_norm(case.subject), _norm(case.value), _norm(case.source))
        if key in seen:
            continue
        seen.add(key)
        unique.append(case)

    # Bound 2: same value in every case -> answer, never ask.
    distinct_values = {
        (round(n, 6), _unit_of(c.value))
        if (n := _number_of(c.value)) is not None
        else _norm(c.value)
        for c in unique
    }
    if len(distinct_values) < 2:
        return None

    # Bound 3: the question already names one of the cases.
    if any(_question_names_case(question, c) for c in unique):
        return None

    # Same-subject conflict (different values, one subject) is the both-values
    # statement (#453). Only a subject that DIFFERS makes this mechanism ask.
    if all(_same_subject(unique[0].subject, c.subject) for c in unique[1:]):
        return None

    # Bound 7 / the draft: it must have silently picked exactly one case.
    committed = [c for c in unique if _text_commits(draft, c.value)]
    if len(committed) != 1:
        return None

    # Bound 5: list only the cases actually found, each with its source.
    options = []
    for c in unique:
        if c.subject and _norm(c.subject) != _norm(c.value):
            options.append(f"{c.value} in {c.source} ({c.subject})")
        else:
            options.append(f"{c.value} in {c.source}")
    return (
        "Retrieved context supports more than one answer: "
        + "; ".join(options)
        + ". Which one is meant?"
    )


# --- the reader: value claims read from the handed passages (pure) --------

_SENTENCE_BREAK_RE = re.compile(r"[.!?]|\n")

#: A clause / document reference the passage states, read by SHAPE only:
#: numbers joined by dots (8.8.1) or a letter-number code (C35/45,
#: AB-2001-101). Never an enumerated id -- the shape decides, so a new code
#: is read the same way as an old one.
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]{1,6}[-/]?\d[A-Za-z0-9./-]*\d|\d+(?:\.\d+){2,})"
    r"(?![A-Za-z0-9])",
)


def _sentence_span(text: str, start: int) -> tuple[int, int]:
    """The sentence boundaries around ``start`` (split on . ! ? and newlines)."""
    t = text or ""
    lo = 0
    for m in _SENTENCE_BREAK_RE.finditer(t):
        if m.start() < start:
            lo = m.end()
            continue
        hi = m.end() if m.group() == "\n" else m.start() + 1
        return lo, hi
    return lo, len(t)


def cases_from_passages(passages: Iterable[tuple[str, str]]) -> list[Case]:
    """The value claims read from ``(source, text)`` passages.

    One ``Case`` per value the passage states. A measure (a number with a
    unit) reuses the product's figure reader; a reference is read by shape
    (dotted clause ids and letter-number codes). The subject is the sentence
    (or heading line) that states the value, so two passages about different
    things are told apart by the passage's own words -- never by a known
    list of subjects. Pure: no I/O, no model.
    """
    from app.agents.citation_provenance import _figure_mentions

    cases: list[Case] = []
    for source, text in passages:
        t = text or ""
        for m in _REFERENCE_RE.finditer(t):
            value = m.group(0).strip()
            if _is_date_token(value):
                continue
            lo, hi = _sentence_span(t, m.start())
            cases.append(Case(
                subject=t[lo:hi].strip(), value=value, source=source or "",
            ))
        for m, _figure in _figure_mentions(t):
            value = m.group(0).strip()
            lo, hi = _sentence_span(t, m.start())
            cases.append(Case(
                subject=t[lo:hi].strip(), value=value, source=source or "",
            ))
    return cases
