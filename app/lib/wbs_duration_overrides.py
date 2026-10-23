"""Parse chat duration overrides and apply them to a generated WBS.

Guarantees that a turn like ``use 6 days per slab and re-run`` actually
changes template durations and recomputes CPM — the model cannot drop
the override, and ``generate_wbs`` does not need the LLM to pass a param.

The parser is deliberately narrow:

* Explicit ``N days per/for <activity>`` always yields an override.
* Bare ``consider 6 and re-run`` only yields an override when history
  (or the current message) mentions working days next to an activity
  family, so ``use 6 trucks and re-run`` is ignored.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Longest-first so "floor slab" wins over "slab".
ACTIVITY_FAMILIES: Tuple[str, ...] = (
    "ground floor slab",
    "floor slab",
    "slab pour",
    "slab curing",
    "tree removal",
    "site clearance",
    "pile cap",
    "wall panel",
    "composite deck",
    "stair core",
    "cable trench",
    "roof deck",
    "formwork",
    "reinforcement",
    "excavation",
    "backfill",
    "cladding",
    "roofing",
    "glazing",
    "commissioning",
    "mobilisation",
    "mobilization",
    "rebar",
    "curing",
    "column",
    "beam",
    "trench",
    "deck",
    "pour",
    "pile",
    "slab",
    "wall",
)

_STOP = frozenset({
    "the", "a", "an", "each", "every", "all", "this", "that", "it", "its",
    "day", "days", "working", "calendar", "please", "and", "or", "to",
    "for", "per", "on", "of", "with", "from", "use", "consider", "assume",
    "take", "apply", "change", "set", "make", "new", "old", "instead",
    "rather", "than", "again", "now", "then", "just", "schedule", "wbs",
    "programme", "program", "template", "activity", "activities", "time",
    "duration", "cycle", "ones", "one", "them", "those", "these",
})

# Inflection / CESMM aliases so "tree removal" hits "Remove trees …" and
# WBS rows that folded tree work into "Site clearance — Hall A/B"
# (a live WBS duration-override ask). Not a fuzzy score floor — exact stems
# and an explicit family alias.
_TOKEN_STEMS = {
    "removal": "remove",
    "removing": "remove",
    "removed": "remove",
    "trees": "tree",
    "clearance": "clear",
    "clearing": "clear",
    "cleared": "clear",
}
_MATCH_ALIASES: Dict[str, Tuple[str, ...]] = {
    "tree removal": ("site clearance", "remove tree"),
    "remove tree": ("tree removal", "site clearance"),
}

_RERUN_RE = re.compile(
    r"\b(re-?run|re-?generat\w*|re-?calculat\w*|re-?comput\w*|"
    r"run\s+(?:it\s+)?again|do\s+it\s+again)\b",
    re.IGNORECASE,
)

# "use 6 days per slab" / "consider 6 working days for each slab pour"
_USE_DAYS_FOR_RE = re.compile(
    r"\b(?:use|consider|assume|take|apply|override|change|set|make|"
    r"revise|update|switch(?:\s+to)?)"
    r"(?:\s+it)?(?:\s+as)?"
    r".{0,40}?"
    r"(\d{1,3})\s*(?:working\s+)?days?"
    r".{0,24}?"
    r"(?:per|for\s+each|for\s+every|for|on\s+each|on|of|across)\s+"
    r"(?:the\s+|a\s+|each\s+|every\s+)?"
    r"([a-z][a-z0-9\s\-]{1,40}?)"
    r"(?=\s+and\s+re|\s+and\s+run|\s*[,.;]|$)",
    re.IGNORECASE,
)

# "6 days per slab" / "6-day slab pour"
_DAYS_PER_RE = re.compile(
    r"\b(\d{1,3})\s*[- ]?(?:working\s+)?days?\s+"
    r"(?:per|for\s+each|for\s+every|for|on\s+each)\s+"
    r"(?:the\s+|a\s+|each\s+|every\s+)?"
    r"([a-z][a-z0-9\s\-]{1,40}?)"
    r"(?=\s+and\s+re|\s+and\s+run|\s*[,.;]|$)",
    re.IGNORECASE,
)

# "slab duration to 6 days" / "each slab to 6 days"
_ACTIVITY_TO_DAYS_RE = re.compile(
    r"\b(?:each\s+|every\s+|the\s+)?"
    r"([a-z][a-z0-9\s\-]{1,40}?)"
    r"\s+(?:duration|cycle|cycle\s+time)?"
    r"\s*(?:to|at|as|of|=)\s*"
    r"(\d{1,3})\s*(?:working\s+)?days?",
    re.IGNORECASE,
)

# "consider 6" / "use 6 days" — match only, activity inferred later
_BARE_OVERRIDE_RE = re.compile(
    r"\b(?:use|consider|assume|take|apply|override|switch\s+to)"
    r"(?:\s+it)?(?:\s+as)?"
    r"\s+(\d{1,3})(?:\s*(?:working\s+)?days?)?\b",
    re.IGNORECASE,
)

_HAS_DAYS_RE = re.compile(r"\bdays?\b", re.IGNORECASE)

_TARGET_COUNT_RE = re.compile(
    r"\b(\d{2,4})\s*-?\s*activit",
    re.IGNORECASE,
)

_MIN_DAYS = 1
_MAX_DAYS = 365


def _clamp_days(raw: Any) -> Optional[int]:
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None
    if days < _MIN_DAYS or days > _MAX_DAYS:
        return None
    return days


def _normalize_match(raw: str) -> Optional[str]:
    words = [
        w for w in re.split(r"[\s\-]+", (raw or "").lower())
        if w and w not in _STOP
    ]
    if not words:
        return None
    cleaned: List[str] = []
    for w in words:
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        cleaned.append(w)
    if not cleaned:
        return None
    return " ".join(cleaned)


def flatten_history(history: Optional[Sequence[Any]]) -> str:
    """Join recent chat turns into one string for activity-family inference."""
    if not history:
        return ""
    parts: List[str] = []
    for item in history:
        if isinstance(item, dict):
            content = item.get("content") or item.get("text") or ""
            parts.append(str(content))
        elif isinstance(item, str):
            parts.append(item)
    return "\n".join(parts[-12:])


def infer_activity_match(text: str) -> Optional[str]:
    """Return the longest activity-family mention in ``text``, or None."""
    low = text or ""
    if not low.strip():
        return None
    for fam in ACTIVITY_FAMILIES:
        if re.search(rf"\b{re.escape(fam)}s?\b", low, re.IGNORECASE):
            return fam
    return None


def _explicit_overrides(text: str) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    raw = text or ""
    for rx, days_idx, match_idx in (
        (_USE_DAYS_FOR_RE, 1, 2),
        (_DAYS_PER_RE, 1, 2),
        (_ACTIVITY_TO_DAYS_RE, 2, 1),
    ):
        for m in rx.finditer(raw):
            days = _clamp_days(m.group(days_idx))
            match = _normalize_match(m.group(match_idx))
            if days is None or not match:
                continue
            found.append({
                "match": match,
                "days": days,
                "source": "user_message",
            })
    return found


def parse_duration_overrides(
    text: str,
    history: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Extract ``[{match, days, source}]`` from the current message.

    History is only used to infer *which* activity a bare
    ``consider 6 and re-run`` applies to.
    """
    raw = text or ""
    found = _explicit_overrides(raw)
    if found:
        return _dedupe_overrides(found)

    if not _RERUN_RE.search(raw):
        return []

    hist = flatten_history(history)
    if not _HAS_DAYS_RE.search(raw) and not _HAS_DAYS_RE.search(hist):
        return []

    days: Optional[int] = None
    for m in _BARE_OVERRIDE_RE.finditer(raw):
        days = _clamp_days(m.group(1))
    if days is None:
        return []

    match = infer_activity_match(raw) or infer_activity_match(hist)
    if not match:
        return []
    return [{
        "match": match,
        "days": days,
        "source": "user_message+history",
    }]


def message_wants_wbs_duration_rerun(
    text: str,
    history: Optional[Sequence[Any]] = None,
) -> bool:
    """True when this turn must re-call ``generate_wbs`` with overrides."""
    parsed = parse_duration_overrides(text, history)
    if not parsed:
        return False
    if _RERUN_RE.search(text or ""):
        return True
    low = (text or "").lower()
    return any(
        p in low
        for p in (
            "wbs", "schedule", "programme", "program of works",
            "activity list", "activities",
        )
    )


def coerce_duration_overrides(raw: Any) -> List[Dict[str, Any]]:
    """Normalize a tool-arg dict or list into ``[{match, days}]``."""
    pairs: List[Tuple[Any, Any]] = []
    if isinstance(raw, dict):
        # A single {match, days} row vs a {slab: 6} map.
        if "days" in raw or "duration_days" in raw or "duration" in raw:
            pairs.append((
                raw.get("match") or raw.get("activity") or raw.get("name"),
                raw.get("days") or raw.get("duration_days") or raw.get("duration"),
            ))
        else:
            pairs.extend(raw.items())
    elif isinstance(raw, list):
        for row in raw:
            if isinstance(row, dict):
                pairs.append((
                    row.get("match") or row.get("activity") or row.get("name"),
                    row.get("days") or row.get("duration_days") or row.get("duration"),
                ))
    out: List[Dict[str, Any]] = []
    for match, days in pairs:
        norm = _normalize_match(str(match or ""))
        clamped = _clamp_days(days)
        if norm and clamped is not None:
            out.append({"match": norm, "days": clamped, "source": "params"})
    return out


def merge_duration_overrides(
    *groups: Any,
) -> List[Dict[str, Any]]:
    """Later groups win on the same ``match`` key."""
    by_match: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        rows = (
            group
            if isinstance(group, list)
            else coerce_duration_overrides(group)
        )
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            match = _normalize_match(str(row.get("match") or ""))
            days = _clamp_days(row.get("days"))
            if not match or days is None:
                continue
            merged = dict(row)
            merged["match"] = match
            merged["days"] = days
            by_match[match] = merged
    # Longer matches first so "floor slab" applies before "slab".
    return sorted(by_match.values(), key=lambda r: len(r["match"]), reverse=True)


def _dedupe_overrides(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return merge_duration_overrides(rows)


def _stem_token(token: str) -> str:
    t = (token or "").lower()
    return _TOKEN_STEMS.get(t, t)


def _match_phrases(match: str) -> Tuple[str, ...]:
    """Return the match plus any explicit family aliases."""
    m = (match or "").strip().lower()
    if not m:
        return ()
    aliases = _MATCH_ALIASES.get(m) or _MATCH_ALIASES.get(_normalize_match(m) or "")
    if aliases:
        return (m,) + tuple(a for a in aliases if a != m)
    return (m,)


def activity_matches_override(name: str, match: str) -> bool:
    n = (name or "").lower()
    if not n:
        return False
    for phrase in _match_phrases(match):
        if _phrase_hits_activity(n, phrase):
            return True
    return False


def _phrase_hits_activity(name_low: str, match: str) -> bool:
    m = (match or "").strip().lower()
    if not m:
        return False
    if m in name_low:
        return True
    tokens = [
        _stem_token(t)
        for t in re.split(r"[\s\-]+", m)
        if t and t not in _STOP
    ]
    if not tokens:
        return False
    if all(t in name_low for t in tokens):
        return True
    name_stems = {
        _stem_token(t)
        for t in re.split(r"[\s\-—–/]+", name_low)
        if t and t not in _STOP
    }
    return all(t in name_stems for t in tokens)


def apply_duration_overrides(
    activities: Sequence[Dict[str, Any]],
    overrides: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Set ``duration_days`` on matching activities. Returns
    ``(activities, applied, unmatched)``.
    """
    acts = [dict(a) for a in activities]
    applied: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    claimed: set = set()
    for ovr in overrides:
        match = str(ovr.get("match") or "")
        days = _clamp_days(ovr.get("days"))
        if not match or days is None:
            continue
        n = 0
        ids: List[str] = []
        for i, a in enumerate(acts):
            if i in claimed:
                continue
            if not activity_matches_override(str(a.get("name") or ""), match):
                continue
            a["duration_days"] = days
            claimed.add(i)
            n += 1
            if a.get("id"):
                ids.append(str(a["id"]))
        row = {
            "match": match,
            "days": days,
            "activities_updated": n,
            "activity_ids": ids[:20],
            "source": ovr.get("source") or "params",
        }
        if n:
            applied.append(row)
        else:
            unmatched.append(row)
    return acts, applied, unmatched


def collect_overrides(
    *texts: Optional[str],
    explicit: Any = None,
    history: Optional[Sequence[Any]] = None,
) -> List[Dict[str, Any]]:
    """Merge explicit tool params with overrides parsed from any text source."""
    parsed: List[Dict[str, Any]] = []
    for t in texts:
        if t:
            parsed.extend(parse_duration_overrides(str(t), history))
    return merge_duration_overrides(parsed, explicit)


def infer_target_count(text: str, default: int = 200) -> int:
    m = _TARGET_COUNT_RE.search(text or "")
    if not m:
        return default
    try:
        n = int(m.group(1))
    except (TypeError, ValueError):
        return default
    return max(20, min(1000, n))
