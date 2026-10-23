"""User-supplied milestone durations are arithmetic, not a Contract Data veto.

Live theshovel.ai ~27d6940: a tester asked, with Milestone 1/3/5 durations and a
common start, which milestone drives completion and by how much over Milestone 1.
The assistant rejected the premise from Contract Data access-date rows.

Owner rule: the numbers in the question are the operands. Access-date
particulars must not override them. Kill-switch
``HYPOTHETICAL_MILESTONE_ARITHMETIC=0`` restores the rejection.
"""
from __future__ import annotations

import os
import re
from typing import Any

HEADING = "HYPOTHETICAL MILESTONE ARITHMETIC"

# M1=397d / M1 = 397 days / Milestone 5: 731d
_SUPPLIED_DURATION_RE = re.compile(
    r"(?i)\b(?:milestone\s+)?m(?P<n>\d+)\s*[=:]\s*(?P<days>\d+)\s*(?:days?|d)\b"
)
_MILESTONE_IS_RE = re.compile(
    r"(?i)\bmilestone\s+(?P<n>\d+)\s+(?:is|of)\s+(?P<days>\d+)\s*(?:days?|d)\b"
)
_OVER_BASELINE_RE = re.compile(
    r"(?i)\bover\s+(?:milestone\s+)?m(?P<n>\d+)\b"
)
_COMPARE_ASK_RE = re.compile(
    r"(?i)(?:drives?\s+completion|by\s+how\s+much|how\s+much\s+over|"
    r"which\s+milestone|longest|later\s+than)"
)
_PREMISE_REJECTION_RE = re.compile(
    r"(?i)^\s*(?:i\s+can'?t\s+answer\s+that\s+as\s+posed|"
    r"the\s+premise\s+(?:does\s+not|doesn't)\s+hold|"
    r"i\s+cannot\s+answer\s+that\s+as\s+posed)"
)


def hypothetical_milestone_arithmetic_enabled() -> bool:
    """ON by default. ``HYPOTHETICAL_MILESTONE_ARITHMETIC=0`` restores the FAIL."""
    raw = (os.getenv("HYPOTHETICAL_MILESTONE_ARITHMETIC", "1") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def parse_supplied_milestone_durations(query: str) -> dict[int, int]:
    """Milestone number → days, taken only from assignments in ``query``."""
    out: dict[int, int] = {}
    blob = query or ""
    for cre in (_SUPPLIED_DURATION_RE, _MILESTONE_IS_RE):
        for match in cre.finditer(blob):
            out[int(match.group("n"))] = int(match.group("days"))
    return out


def parse_baseline_milestone(query: str, supplied: dict[int, int]) -> int | None:
    """The milestone named after 'over', else the shortest supplied duration."""
    if not supplied:
        return None
    match = _OVER_BASELINE_RE.search(query or "")
    if match:
        n = int(match.group("n"))
        if n in supplied:
            return n
    return min(supplied, key=lambda n: (supplied[n], n))


def query_is_hypothetical_milestone_arithmetic(query: str) -> bool:
    """True when the user supplied ≥2 M#=Nd pairs and asked which is longest."""
    if not hypothetical_milestone_arithmetic_enabled():
        return False
    blob = query or ""
    if not _COMPARE_ASK_RE.search(blob):
        return False
    return len(parse_supplied_milestone_durations(blob)) >= 2


def compose_hypothetical_milestone_arithmetic(query: str) -> dict[str, Any] | None:
    """Longest supplied duration, and its day-delta over the named baseline."""
    if not query_is_hypothetical_milestone_arithmetic(query):
        return None
    supplied = parse_supplied_milestone_durations(query)
    baseline = parse_baseline_milestone(query, supplied)
    if baseline is None:
        return None
    longest = max(supplied.values())
    drivers = tuple(sorted(n for n, days in supplied.items() if days == longest))
    if not drivers:
        return None
    delta = longest - supplied[baseline]
    line = format_hypothetical_milestone_line(
        drivers, longest, baseline, supplied[baseline], delta,
    )
    return {
        "drivers": drivers,
        "driver_days": longest,
        "baseline": baseline,
        "baseline_days": supplied[baseline],
        "delta": delta,
        "line": line,
    }


def format_hypothetical_milestone_line(
    drivers: tuple[int, ...],
    driver_days: int,
    baseline: int,
    baseline_days: int,
    delta: int,
) -> str:
    """User-facing lead: longest milestone + day delta over the baseline."""
    names = _join_milestone_names(drivers)
    verb = "drive" if len(drivers) > 1 else "drives"
    return (
        f"{names} {verb} completion, by {delta} days over "
        f"Milestone {baseline} ({driver_days} − {baseline_days})."
    )


def _join_milestone_names(drivers: tuple[int, ...]) -> str:
    labels = [f"Milestone {n}" for n in drivers]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def answer_leads_with_premise_rejection(text: str) -> bool:
    """True when the visible answer opens by refusing the user's numbers."""
    return bool(_PREMISE_REJECTION_RE.search(text or ""))


def answer_states_hypothetical_milestone(text: str, composed: dict[str, Any]) -> bool:
    """True when ``text`` already leads with the driver and the day delta."""
    if not text or not composed:
        return False
    lead = (text or "").lstrip().split("\n", 1)[0]
    if answer_leads_with_premise_rejection(lead):
        return False
    delta = str(composed.get("delta", ""))
    if delta not in lead:
        return False
    drivers = composed.get("drivers") or ()
    return any(
        f"Milestone {n}" in lead or f"M{n}" in lead
        for n in drivers
    )


def instruction_for_query(query: str) -> str:
    """Inject heading. Includes the composed line when the numbers parse."""
    if not query_is_hypothetical_milestone_arithmetic(query):
        return ""
    composed = compose_hypothetical_milestone_arithmetic(query)
    result = composed["line"] if composed else (
        "the longest supplied duration drives completion; the delta over "
        "the named baseline is longest minus baseline"
    )
    return (
        f"{HEADING} — the user supplied milestone durations and a common "
        "start IN THE QUESTION. Use those numbers. Contract Data "
        "access-date particulars must not override them. Do not reject "
        f"the premise. {result} That IS the answer. Lead with that "
        "result.\n"
    )


def graft_hypothetical_milestone_answer(text: str, query: str) -> str:
    """Lead with the arithmetic. A rejection-first body is pushed down."""
    composed = compose_hypothetical_milestone_arithmetic(query)
    if not composed:
        return text
    line = composed["line"]
    raw = text or ""
    if answer_states_hypothetical_milestone(raw, composed):
        return text
    body = raw.strip()
    if not body:
        return line
    return f"{line}\n\n{body}"
