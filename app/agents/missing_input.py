"""AN ANSWER THAT NAMES ITS MISSING INPUT MUST GO AND GET IT (item 4).

From a recorded gate battery (``FLEET_OPS/artifacts/gate_battery_13b2bf7_2026-08-31.md``):
a delay-damages-per-calendar-day ask FAILED with "do not state the specific
monetary rate per calendar day". A neighbouring question in the same
session and corpus retrieved the Accepted Contract Amount to complete its
arithmetic; the per-day ask said the figure it needed was not available.
It is not a retrieval limit. The answer identified its missing input and
did not go and get it.

The owner's rule: **one bounded retrieval attempt for any input the answer
names as missing.** One, not a loop -- an agent that keeps searching until
it finds something is how a wrong answer gets manufactured. If the single
attempt comes back with nothing that carries the named terms, the refusal
stands exactly as written.

This module is the DETECTION half and is deliberately pure: no retrieval, no
model, no I/O. It answers two questions about a draft answer --

    does this answer name something it lacks?   -> names_missing_input
    what should be searched for?                -> fetch_query

-- so both can be tested against the recorded string rather than against a
mock of the retriever.

WHAT IT DOES NOT DO. It does not decide whether the thing exists. An answer
that refuses a drawing genuinely not in the corpus will have that drawing
named as a missing input here; the bounded fetch then returns nothing and
the refusal is unchanged. Distinguishing "absent" from "not retrieved" is
the fetch's job, not the detector's -- and the platform cannot know which it
is without looking, which is the whole reason the rule is "attempt once".
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

#: Cue phrases an answer uses when it names something it does not have. The
#: capture group is the THING; everything after a clause boundary is dropped.
_CUES = (
    r"do(?:es)?\s+not\s+(?:state|contain|include|provide|specify|give|mention)",
    r"(?:is|are|was|were)\s+not\s+(?:stated|contained|included|provided|"
    r"specified|given|available|present)",
    r"(?:could|can|do)\s*n[o']?t\s+(?:find|confirm|locate|identify|determine)",
    r"(?:I|we)\s+(?:do|did)\s+not\s+have",
    r"no\s+(?:record|figure|value|rate|amount)\s+(?:of|for)",
    r"without",
)
_CUE_RE = re.compile(
    # "." ends the phrase EXCEPT inside a clause number: 8.8.1, 1.1.75.
    r"(?:%s)\s+(?P<thing>(?:[^.;:\n]|\.(?=\d)){4,120})" % "|".join(_CUES),
    re.IGNORECASE,
)

#: Words that end the named thing early -- past them the sentence has moved
#: on from WHAT is missing to what follows from it.
_STOP_RE = re.compile(
    r"\b(?:in the|within the|from the|among the|so |therefore|because|"
    r"which |that would|I (?:cannot|can't|am unable)|and (?:I|we) )",
    re.IGNORECASE,
)

#: Leading noise on the captured phrase.
_LEAD_RE = re.compile(
    r"^(?:the|a|an|any|its|their|this|that|specific|exact|particular|"
    r"actual|explicit)\s+", re.IGNORECASE
)

#: A named thing has to be a noun phrase with content. These alone are not.
_EMPTY_PHRASES = frozenset({
    "it", "them", "this", "that", "these", "those", "information",
    "details", "data", "context", "documents", "document", "excerpts",
    "sources", "the corpus", "anything", "enough information",
})

_QUESTION_STOPWORDS = frozenset("""
what which when where whom whose does did doing done have has had having
this that these those with from into your our their please tell show give
about there here will shall would could should must calculate compute work
the and for are was were can may might how many much
""".split())

FLAG_ENV = "MISSING_INPUT_FETCH"


def enabled() -> bool:
    """ON by default; ``MISSING_INPUT_FETCH=0`` is the kill switch."""
    return str(os.getenv(FLAG_ENV, "1")).strip().lower() not in (
        "0", "false", "no", "off",
    )


def _clean(phrase: str) -> str:
    text = _STOP_RE.split(phrase, maxsplit=1)[0]
    text = text.strip(" ,—-–\"'`*")
    while True:
        stripped = _LEAD_RE.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    return " ".join(text.split())


#: The passive form puts the named thing BEFORE the cue: "the delay damages
#: rate is not stated". Without this the capture starts after "stated" and
#: finds only the place it was looked for, never the thing.
_PASSIVE_RE = re.compile(
    r"(?P<thing>[A-Za-z0-9](?:[^.;:\n]|\.(?=\d)){3,90}?)\s+"
    r"(?:is|are|was|were)\s+not\s+"
    r"(?:stated|contained|included|provided|specified|given|available|present)",
    re.IGNORECASE,
)


def _is_nameable(thing: str) -> bool:
    """Enough of a name to search on.

    One generic word is not. "I could not confirm this reference" names
    ``reference``, and a search for that returns the whole corpus -- then
    the support guard, which only needs one word to match, waves it through.
    Two content words, or one carrying a number ("Schedule 10", "8.8.1"),
    is the floor.
    """
    if not thing or thing.lower() in _EMPTY_PHRASES:
        return False
    words = [
        w for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*", thing)
        if w.lower() not in _QUESTION_STOPWORDS
    ]
    if not words:
        return False
    if any(re.search(r"\d", w) for w in words):
        return True
    return len([w for w in words if len(w) >= 3]) >= 2


def names_missing_input(text: str) -> Optional[str]:
    """The first thing this answer says it does not have, or None.

    First rather than all, because the rule is ONE bounded attempt: an
    answer listing three gaps still gets one search, on the gap it named
    first, which is the one it led with.
    """
    body = text or ""
    candidates = []
    for match in _CUE_RE.finditer(body):
        candidates.append((match.start(), _clean(match.group("thing"))))
    for match in _PASSIVE_RE.finditer(body):
        candidates.append((match.start(), _clean(match.group("thing"))))
    for _pos, thing in sorted(candidates):
        if _is_nameable(thing):
            return thing
    return None


def fetch_query(missing: str, question: str = "") -> str:
    """The single query the bounded attempt runs.

    The NAMED THING leads, because that is what the answer said it lacked
    and searching the original question again is what already failed. Terms
    from the question follow so the search stays inside this project's
    subject -- "rate per calendar day" alone is a phrase half the corpus
    could match.
    """
    q_terms: List[str] = []
    seen = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-/]+", missing)}
    for word in re.findall(r"[A-Za-z][A-Za-z0-9\-/]+", question or ""):
        low = word.lower()
        if len(word) < 4 or low in seen or low in _QUESTION_STOPWORDS:
            continue
        seen.add(low)
        q_terms.append(word)
        if len(q_terms) >= 6:
            break
    return " ".join([missing] + q_terms).strip()




def fetched_context_supports(chunks_text: str, missing: str) -> bool:
    """Did the one attempt actually come back with the named thing?

    The guard that keeps a bounded fetch from making an answer worse. A
    search always returns SOMETHING; injecting whatever came back and asking
    again is how a refusal becomes a confident wrong answer. Content words
    from the named thing have to appear in what was retrieved.
    """
    body = (chunks_text or "").lower()
    if not body.strip():
        return False
    # A digit is content: "Schedule 10" must not be satisfied by "Schedule 9",
    # which is what happens when only letter-initial tokens are counted.
    words = [
        w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/\.]*", missing or "")
        if w.lower() not in _QUESTION_STOPWORDS and len(w) >= 2
    ]
    if not words:
        return False
    hits = sum(1 for w in words if w in body)
    # Every content word for a one- or two-word name; a clear majority for a
    # longer phrase, so "monetary rate per calendar day" is not satisfied by
    # the word "rate" alone.
    need = len(words) if len(words) <= 2 else (len(words) + 1) // 2 + 1
    return hits >= min(need, len(words))
