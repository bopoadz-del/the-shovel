"""Construction container — schedule submodule."""

import logging
import math
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .helpers import _safe_float, _safe_iso_date

logger = logging.getLogger(__name__)


_LOOKAHEAD_DAYS_RE = re.compile(
    r"\b(\d+)\s*[-]?\s*days?\b",
    re.IGNORECASE,
)
_LOOKAHEAD_WEEKS_RE = re.compile(
    r"\b(\d+|two|three|four)\s*[-]?\s*weeks?\b",
    re.IGNORECASE,
)
_LOOKAHEAD_WEEK_WORDS = {"two": 2, "three": 3, "four": 4}


def _look_ahead_window_from_text(text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse an N-day / N-week window from a look-ahead ask.

    Returns ``(days, weeks)`` with at most one set. Unset when the ask
    does not name a window (container then defaults to 21 days).
    """
    raw = text or ""
    days_m = _LOOKAHEAD_DAYS_RE.search(raw)
    if days_m:
        try:
            days = int(days_m.group(1))
        except (TypeError, ValueError):
            days = 0
        if days >= 1:
            return days, None
    weeks_m = _LOOKAHEAD_WEEKS_RE.search(raw)
    if weeks_m:
        token = weeks_m.group(1).lower()
        weeks = _LOOKAHEAD_WEEK_WORDS.get(token)
        if weeks is None:
            try:
                weeks = int(token)
            except (TypeError, ValueError):
                weeks = 0
        if weeks >= 1:
            return None, weeks
    return None, None


def _delay_claim_facts_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Pull a delay-claim notice from operator-stated days / rate / clause."""
    t = text or ""
    days_m = re.search(
        r"(\d+(?:\.\d+)?)\s+calendar\s+days?",
        t,
        re.IGNORECASE,
    )
    if not days_m:
        days_m = re.search(
            r"access was given\s+(\d+(?:\.\d+)?)\s+calendar\s+days?",
            t,
            re.IGNORECASE,
        )
    rate_m = re.search(
        r"(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?(?:contract\s+price|contract\s+sum)",
        t,
        re.IGNORECASE,
    )
    clause_m = re.search(
        r"clause\s+(\d+(?:\.\d+)*)",
        t,
        re.IGNORECASE,
    )
    if not days_m:
        return None
    milestone = ""
    ms = re.search(
        r"(Milestone\s+\d+\s*\([^)]+\))",
        t,
        re.IGNORECASE,
    )
    if ms:
        milestone = ms.group(1)
    comms = "Aconex" if re.search(r"\baconex\b", t, re.I) else ""
    return {
        "delay_days": float(days_m.group(1)),
        "rate_percent_per_day": float(rate_m.group(1)) if rate_m else None,
        "clause": clause_m.group(1) if clause_m else "",
        "milestone": milestone,
        "communication": comms,
        "source_text": t,
    }


def _operator_milestones_from_text(text: str) -> Dict[str, Any]:
    """Extract Time for Completion / Milestone N (name, days) from a brief."""
    t = text or ""
    out: Dict[str, Any] = {}
    tfc = re.search(
        r"time\s+for\s+completion[^\d]{0,40}(\d+)\s*days?",
        t,
        re.IGNORECASE,
    )
    if tfc:
        out["time_for_completion_days"] = int(tfc.group(1))
    milestones = []
    for m in re.finditer(
        r"milestone\s+(\d+)\s*\(([^,]+),\s*(\d+)\s*days?\)",
        t,
        re.IGNORECASE,
    ):
        milestones.append({
            "id": int(m.group(1)),
            "name": m.group(2).strip(),
            "days": int(m.group(3)),
        })
    if milestones:
        out["milestones"] = milestones
    return out


def _draft_delay_claim_notice(
    facts: Dict[str, Any],
    notification_date: str,
) -> Dict[str, Any]:
    days = facts.get("delay_days")
    rate = facts.get("rate_percent_per_day")
    clause = facts.get("clause") or "8.8"
    milestone = facts.get("milestone") or "the stated milestone"
    comms = facts.get("communication") or "the approved communication method"
    rate_s = f"{rate:g}% of the Contract Price per calendar day" if rate else (
        "the delay-damages rate stated in Contract Data"
    )
    quantum = None
    if rate is not None and days is not None:
        quantum = (
            f"{rate:g}% × Contract Price × {days:g} calendar days. "
            "Contract Price is not restated in this turn — apply the "
            "Accepted Contract Amount from Contract Data."
        )
    notice = (
        f"DELAY CLAIM NOTICE — Clause {clause}\n\n"
        f"The Contractor gives notice of delay to {milestone}. "
        f"Access was given {days:g} calendar days after the Engineer "
        f"instructed date. Delay damages for this milestone are {rate_s}.\n\n"
        f"This notice is issued via {comms}. "
        "It is a draft from operator-supplied facts — not a submitted claim."
    )
    return {
        "status": "success",
        "action": "claim_generated",
        "claim_type": "delay_notice",
        "claim_number": "DRAFT-CLAIM",
        "issued": False,
        "notification_date": notification_date,
        "clause": clause,
        "milestone": milestone,
        "delay_days": days,
        "rate_percent_per_day": rate,
        "communication": comms,
        "quantum_formula": quantum,
        "claim_narrative": {
            "executive_summary": notice,
            "full_narrative": notice,
        },
        "delay_summary": {
            "total_delay_days": days,
            "delay_events_count": 1,
        },
        "note": (
            "Draft delay-claim notice from operator facts. "
            "No events were invented."
        ),
    }




_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
# "Activity A 3 Mar to 9 Mar 2031" / "Activity B 2031-03-12 - 2031-03-16".
_ACTIVITY_LINE_RE = re.compile(
    r"(?P<name>[A-Za-z][A-Za-z0-9 /&'\-]{1,60}?)\s+"
    r"(?P<d1>\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})"
    r"\s*(?:to|until|through|—|–|-)\s*"
    r"(?P<d2>\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{4})?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)


def _spoken_date(token: str, year_hint: int | None = None) -> str | None:
    """"15 Sep 2026" -> "2026-09-15". Returns None rather than guessing.

    ``_coerce_date`` reads ISO and dd/mm/yyyy only; an operator writing a
    programme in a chat message writes "15 Sep 2026", and a year is often
    stated once at the end of the pair.
    """
    token = (token or "").strip()
    if not token:
        return None
    m = re.match(r"^(\d{1,2})\s+([A-Za-z]{3,9})(?:\s+(\d{4}))?$", token)
    if not m:
        return None
    day, month_word, year = m.group(1), m.group(2).lower()[:4], m.group(3)
    month = _MONTHS.get(month_word) or _MONTHS.get(month_word[:3])
    if not month:
        return None
    if year is None:
        if year_hint is None:
            return None
        year = str(year_hint)
    return f"{int(year):04d}-{month:02d}-{int(day):02d}"


# "as of today, 21 September" — the optional "today" is not the date.
# "from 21 September 2026, what is due" is a stated today. "from 15 Sep
# to 25 Sep" is an activity span and is rejected below.
_STATED_TODAY_CUE_RE = re.compile(
    r"\b(?:today(?:'s)?(?:\s+date)?\s+is|current\s+date\s+is|"
    r"as[\s-]of(?:\s+today)?|as[\s-]at|from)\b",
    re.IGNORECASE,
)
_STATED_FILLER_RE = re.compile(
    r"^(?:today|the|date|is|on)\b[\s,;:.\-]*",
    re.IGNORECASE,
)
_SPAN_AFTER_DATE_RE = re.compile(
    r"^\s*(?:to|until|through|—|–|-)\s+\d",
    re.IGNORECASE,
)
_STATED_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\b")
_STATED_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b")
_STATED_SPOKEN_RE = re.compile(
    r"^(\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{4})?)\b",
)


def _schedule_year_hint(text: str) -> int | None:
    """A year already written on the programme, else None.

    "Today is 21 September" next to "15 Sep 2026" is 2026. A bare stated
    date with no year in the message falls through to the clock year.
    """
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text or "")]
    if not years:
        return None
    return max(set(years), key=lambda y: (years.count(y), -years.index(y)))


def _slash_date(day: str, month: str, year: str | None, year_hint: int | None) -> str | None:
    """Day/month[/year], the same order ``_coerce_date`` uses for slashes."""
    from datetime import date as _date

    if year is None:
        if year_hint is None:
            from app.lib.pm_computations import _clock_today
            year_n = _clock_today().year
        else:
            year_n = year_hint
    else:
        year_n = int(year)
        if year_n < 100:
            year_n += 2000
    try:
        return _date(year_n, int(month), int(day)).isoformat()
    except ValueError:
        logger.debug(
            "look_ahead: stated day/month is not a calendar date %s-%s-%s",
            year_n, month, day,
        )
        return None


def _parse_leading_stated_date(rest: str, hint: int | None) -> tuple[str | None, str]:
    """``(iso, tail)`` for a date at the front of ``rest``, else ``(None, "")``."""
    rest = rest.lstrip(" \t,;:")
    for _ in range(4):
        filler = _STATED_FILLER_RE.match(rest)
        if not filler:
            break
        rest = rest[filler.end():]
    iso = _STATED_ISO_RE.match(rest)
    if iso:
        return _coerce_date_safe(iso.group(1)), rest[iso.end():]
    slash = _STATED_SLASH_RE.match(rest)
    if slash:
        return (
            _slash_date(slash.group(1), slash.group(2), slash.group(3), hint),
            rest[slash.end():],
        )
    spoken = _STATED_SPOKEN_RE.match(rest)
    if spoken:
        token = spoken.group(1)
        has_year = bool(re.search(r"\d{4}\s*$", token))
        return (
            _spoken_date(token, None if has_year else hint or _clock_year()),
            rest[spoken.end():],
        )
    return None, ""


def _stated_look_ahead_date(text: str) -> str | None:
    """ISO date the operator stated as today, or None.

    "Today is 21 September", "Today's date is 21 September",
    "as of today, 21 September", "as at 21 September",
    "from 21 September 2026". Activity spans ("3 Mar to 9 Mar",
    "from 3 Mar to 9 Mar") are not a stated today. Year comes from the
    token, else a year already in the schedule text, else the clock year.
    """
    raw = text or ""
    hint = _schedule_year_hint(raw)
    for cue in _STATED_TODAY_CUE_RE.finditer(raw):
        value, tail = _parse_leading_stated_date(raw[cue.end():], hint)
        if not value:
            continue
        if cue.group(0).lower() == "from" and _SPAN_AFTER_DATE_RE.match(tail or ""):
            continue
        return value
    return None


def _clock_year() -> int:
    from app.lib.pm_computations import _clock_today
    return _clock_today().year


def _activities_from_text(text: str) -> List[Dict[str, Any]]:
    """Activities the operator typed in the message, or [].

    Only rows carrying a name AND two dates are returned -- a half-written row
    is dropped, never completed by guesswork. A look-ahead ask that typed its
    activities in the message used to be told to supply a .xer, because no
    path looked here.
    """
    blob = text or ""
    if not blob:
        return []
    out: List[Dict[str, Any]] = []
    for segment in re.split(r"[;\n]+", blob):
        m = _ACTIVITY_LINE_RE.search(segment)
        if not m:
            continue
        name = " ".join(m.group("name").split()).strip(" -:,")
        # A trailing year on the second date fixes the first one's year.
        year_hint = None
        tail = re.search(r"(\d{4})\s*$", m.group("d2").strip())
        if tail:
            year_hint = int(tail.group(1))
        start = _spoken_date(m.group("d1"), year_hint) or _coerce_date_safe(m.group("d1"))
        finish = _spoken_date(m.group("d2"), year_hint) or _coerce_date_safe(m.group("d2"))
        if not name or not start or not finish:
            continue
        out.append({"name": name, "start": start, "finish": finish})
    return out


def _typed_programme_activities(text: str) -> List[Dict[str, Any]]:
    """Operator-typed programme rows (``name date1 to date2``) as WBS
    activities (S-1 rework-1, ORDERS_28OCT_1030Z #3). The typed dates give each
    activity's duration; the kernel re-derives the sequence/CPM. [] when the
    text is not a typed programme — a prose brief then goes to the structurer.
    """
    from datetime import date as _date
    typed = _activities_from_text(text)
    if not typed:
        return []
    acts: List[Dict[str, Any]] = []
    prev: Optional[str] = None
    for idx, row in enumerate(typed, start=1):
        aid = f"T{idx}"
        dur: Optional[int] = None
        try:
            start_d = _date.fromisoformat(str(row["start"])[:10])
            finish_d = _date.fromisoformat(str(row["finish"])[:10])
            span = (finish_d - start_d).days
            dur = span if span > 0 else 1
        except Exception:  # noqa: BLE001 — unparseable date pair -> UNKNOWN
            dur = None
        acts.append({
            "id": aid,
            "code": aid,
            "name": str(row["name"])[:160],
            "duration_days": dur,
            "duration_basis": "typed — operator programme dates",
            "predecessors": [prev] if prev else [],
            "resources": [],
            "wbs": aid,
            "wbs_code": aid,
            "wbs_phase": "programme",
        })
        prev = aid
    return acts


# Brief/document -> structured activities (S-1, ORDERS_28OCT_0815Z). The LLM
# STRUCTURES the prose into named activities + stated durations + stated
# dependencies ONLY; it never computes a schedule — durations/sequence/CPM run
# in the deterministic kernel (_attach_cpm_to_activities -> pm_computations).
# Anti-hallucination: every activity must carry a source_span copied verbatim
# from the input; activities whose span is not in the input are dropped. A
# missing duration is kept as 0 (flagged "not stated"), never defaulted.
_BRIEF_WBS_SYSTEM = (
    "You structure a construction project brief into a work breakdown. Return "
    "ONLY JSON: {\"activities\": [{\"name\": string, \"source_span\": string "
    "(a verbatim substring of the brief that names this activity), "
    "\"duration_days\": integer or null (ONLY if the brief STATES a duration "
    "for it, else null), \"quantity\": number or null (ONLY if the brief states "
    "a quantity for it), \"unit\": string or null, \"productivity_rate\": string "
    "or null (ONLY if the brief states a rate, e.g. \"50 m2 per gang-day\"), "
    "\"predecessors\": [activity name strings that the brief says must finish "
    "first, else []]}]}. Extract only activities the brief actually names. "
    "Never invent activities, durations, quantities, rates or dependencies. "
    "Every source_span must be copied verbatim from the brief text."
)


def _estimate_duration_from_rate(quantity: float, rate: str) -> Tuple[Optional[int], Optional[str]]:
    """Working days for ``quantity`` at a STATED productivity ``rate`` via the
    construction kernel (productivity_manpower_duration).

    Returns ``(days, error)``. ``days`` is None when the kernel produced no
    usable figure (0845Z: no rate -> no estimate). ``error`` is None in that
    ordinary case, but when the kernel RAISED it is a short tag NAMING the
    failure, logged at WARNING — so the caller surfaces a kernel BREAK as a
    distinct computed failure instead of silently treating it as a plain
    missing-input UNKNOWN."""
    try:
        from app.agents.hats.planning.formulas.construction_formulas_planning import (
            productivity_manpower_duration,
        )
        res = productivity_manpower_duration(quantity=float(quantity), productivity_rate=str(rate))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "productivity kernel raised estimating duration (quantity=%r rate=%r): %s",
            quantity, rate, exc, exc_info=True,
        )
        return None, f"productivity kernel error: {type(exc).__name__}"
    if not isinstance(res, dict):
        return None, None
    for k in ("duration_days", "duration", "days", "remaining_days"):
        v = res.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return int(math.ceil(v)), None
    return None, None


async def _structure_activities_from_text(text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Structure a prose brief / document into WBS activities via the single
    LLM egress (``llm_client.complete_json``), structuring ONLY. Returns
    ``(activities, unstated_duration_names)``. Activities carry the WBS-generate
    shape (id/code/name/duration_days/predecessors/wbs*). A source_span absent
    from the input drops the activity (anti-hallucination). Durations are the
    brief's stated values or 0 (never a default); predecessor NAMES resolve to
    ids only among activities that survived."""
    blob = (text or "").strip()
    if not blob:
        return [], []
    try:
        from app.core import llm_client
        data = await llm_client.complete_json(_BRIEF_WBS_SYSTEM, blob, max_tokens=1500)
    except Exception:  # noqa: BLE001 — no provider / network: no scope, not a crash
        logger.info("brief->WBS structuring: LLM egress unavailable", exc_info=True)
        return [], []
    raw = data.get("activities") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return [], []
    low = blob.lower()
    acts: List[Dict[str, Any]] = []
    unstated: List[str] = []
    seen: set = set()
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        span = str(item.get("source_span") or "").strip()
        # Anti-hallucination gate: the span must exist verbatim in the input.
        if not name or not span or span.lower() not in low:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        dur = item.get("duration_days")
        has_dur = isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur > 0
        qty = item.get("quantity")
        rate = item.get("productivity_rate")
        # 0845Z ruling: 0 is not a duration. Stated -> use it; stated quantity +
        # stated rate -> kernel estimate (labelled); else UNKNOWN (None), never
        # defaulted — the activity is incomplete and the chain is reported so.
        duration_days: Optional[int] = None
        basis = "unknown"
        has_qty = isinstance(qty, (int, float)) and not isinstance(qty, bool) and qty > 0
        has_rate = isinstance(rate, str) and bool(rate.strip())
        if has_dur:
            duration_days, basis = int(dur), "stated"
        elif has_qty and has_rate:
            # Stated quantity + STATED rate -> kernel estimate (the stated rate
            # always wins; a norm is never consulted here).
            est, est_err = _estimate_duration_from_rate(qty, rate)
            if est is not None:
                duration_days = est
                basis = f"estimated — kernel productivity_manpower_duration, basis {qty}@{rate.strip()}"
            elif est_err:
                # Kernel BREAK (not a missing input): surface it distinctly.
                basis = f"kernel-error — {est_err}"
        elif has_qty:
            # Stated quantity but NO rate -> a GENERIC, sourced productivity
            # norm for the trade (S-1b). No matching trade -> stays UNKNOWN.
            from app.lib.productivity_norms import norm_rate_for
            norm = norm_rate_for(name, item.get("unit"))
            if norm is not None:
                est, est_err = _estimate_duration_from_rate(qty, norm["rate"])
                if est is not None:
                    duration_days = est
                    basis = (
                        "estimated — kernel productivity_manpower_duration, "
                        f"norm {norm['rate']} ({norm['source']})"
                    )
                elif est_err:
                    basis = f"kernel-error — {est_err}"
        if duration_days is None:
            unstated.append(name)
        aid = f"B{idx}"
        acts.append({
            "id": aid, "code": aid, "name": name[:160],
            "duration_days": duration_days,
            "duration_basis": basis,
            "predecessors": [],  # resolved from names below
            "resources": [],
            "wbs": aid, "wbs_code": aid, "wbs_phase": "brief",
            "source_span": span[:240],
            "_pred_names": [
                str(p).strip().lower()
                for p in (item.get("predecessors") or [])
                if isinstance(p, str) and str(p).strip()
            ],
        })
    name_to_id = {a["name"].lower(): a["id"] for a in acts}
    for a in acts:
        a["predecessors"] = [name_to_id[n] for n in a.pop("_pred_names") if n in name_to_id]
    return acts, unstated


def _coerce_date_safe(token: str) -> str | None:
    """ISO / dd-mm-yyyy through the shared coercer, as a string or None."""
    try:
        from app.lib.pm_computations import _coerce_date
        value = _coerce_date(token)
    except Exception:  # noqa: BLE001
        logger.debug("look_ahead: unparsed date token %r", token, exc_info=True)
        return None
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _inline_look_ahead_activities(data: Dict, p: Dict) -> List[Dict[str, Any]]:
    """Activities given in the request, normalised for ``select_look_ahead``.

    Accepts ``activities`` as a list of dicts using any of the field spellings
    the callers use (name/activity/title, start/from, finish/end/to). Anything
    without both dates is dropped rather than guessed at.
    """
    raw = data.get("activities") or p.get("activities") or []
    if not raw:
        # No list passed: the operator may have typed the programme instead.
        # Every dispatch path (tool call, container action, pre-dispatch)
        # carries the message somewhere, so read it here rather than in each.
        text = " ".join(
            str(x) for x in (
                p.get("user_message"), data.get("user_message"),
                p.get("message"), data.get("message"),
                p.get("brief"), data.get("brief"),
            ) if x
        )
        parsed = _activities_from_text(text)
        if parsed:
            return parsed
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("name") or item.get("activity") or item.get("title")
            or item.get("description") or ""
        )
        start = item.get("start") or item.get("early_start") or item.get("es") or item.get("from")
        finish = (
            item.get("finish") or item.get("early_finish") or item.get("ef")
            or item.get("end") or item.get("to")
        )
        if not name or not start or not finish:
            continue
        row = {"name": str(name), "start": start, "finish": finish}
        for key in ("id", "code", "wbs", "wbs_id", "remaining_duration", "total_float", "critical"):
            if item.get(key) is not None:
                row[key] = item[key]
        out.append(row)
    return out


def _look_ahead_message_text(data: Dict, p: Dict) -> str:
    return " ".join(
        str(x) for x in (
            p.get("user_message"), data.get("user_message"),
            p.get("message"), data.get("message"),
            p.get("brief"), data.get("brief"),
        ) if x
    )


def _as_of_equals_the_clock(raw) -> bool:
    """True when ``raw`` is the server's today, not a different date."""
    if not raw:
        return False
    try:
        from app.lib.pm_computations import _clock_today, _coerce_date
        return _coerce_date(raw) == _clock_today()
    except Exception:  # noqa: BLE001
        logger.debug(
            "look_ahead: could not compare as_of %r to the clock", raw,
            exc_info=True,
        )
        return False


def _resolve_stated_or_explicit_as_of(data: Dict, p: Dict) -> tuple[str | None, bool]:
    """``(raw, from_stated_prose)``.

    A model ``as_of`` equal to the server clock is the clock. The date
    the operator stated as today wins over that echo. A genuine as_of
    that is not the clock still wins over the prose.
    """
    raw = (
        p.get("as_of") or data.get("as_of")
        or p.get("data_date") or data.get("data_date")
    )
    stated = _stated_look_ahead_date(_look_ahead_message_text(data, p))
    if raw and stated and _as_of_equals_the_clock(raw):
        try:
            from app.lib.pm_computations import _coerce_date
            if _coerce_date(stated) != _coerce_date(raw):
                return stated, True
        except Exception:  # noqa: BLE001
            logger.debug(
                "look_ahead: stated today %r did not coerce; using it over the clock",
                stated, exc_info=True,
            )
            return stated, True
    if raw:
        return str(raw), False
    if stated:
        return stated, True
    return None, False


def _look_ahead_as_of(data: Dict, p: Dict):
    """Explicit as_of, else the date the operator stated as today, else None.

    None means "use the clock". A stated "Today is 21 September" wins over
    the clock, including a model as_of that merely repeats the clock. A
    genuine as_of that is not the clock still wins over the prose.
    """
    raw, _from_stated = _resolve_stated_or_explicit_as_of(data, p)
    if not raw:
        return None
    try:
        from app.lib.pm_computations import _coerce_date
        return _coerce_date(raw)
    except Exception:  # noqa: BLE001
        logger.warning(
            "look_ahead: unreadable as_of/data_date %r — using today", raw,
            exc_info=True,
        )
        return None


class ConstructionScheduleMixin:
    async def parse_primavera_schedule(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        file_path = data.get("file_path") or p.get("file_path")
        baseline_file = data.get("baseline_file") or p.get("baseline_file")
        analysis_date = p.get("analysis_date", datetime.now(timezone.utc).isoformat())
    
        if not file_path:
            return {"status": "error", "error": "No schedule file provided"}
    
        ext = Path(file_path).suffix.lower()
        if ext == '.xer':
            schedule_data = await self._parse_xer_file(file_path)
        elif ext == '.xml':
            schedule_data = self._parse_xml_schedule(file_path)
        else:
            return {"status": "error", "error": f"Unsupported format: {ext}"}
    
        if schedule_data.get("status") == "error":
            return schedule_data
    
        cpm_results = self._calculate_cpm(schedule_data)
    
        delay_analysis = None
        if baseline_file:
            if Path(baseline_file).suffix.lower() == '.xer':
                baseline_data = await self._parse_xer_file(baseline_file)
            else:
                baseline_data = self._parse_xml_schedule(baseline_file)
            if baseline_data.get("status") != "error":
                delay_analysis = self._analyze_delays(schedule_data, baseline_data)
    
        schedule_risks = self._analyze_schedule_risks(cpm_results)
        recovery_options = self._generate_recovery_options(delay_analysis, cpm_results) if delay_analysis else []

        # W6 note: duration-learning capture is deliberately NOT wired here. The
        # parse exposes planned/original durations + start-date delays (delay_days),
        # not a clean per-activity ACTUAL duration — which is what
        # record_actual_duration needs. capture_schedule_durations() is ready for
        # a real {planned_duration, actual_duration} signal (e.g. a progress/
        # as-built capture), but wiring it to today's delay analysis would record
        # nothing. Defect-learning IS wired live (see qa_qc_inspection).

        return {
            "status": "success",
            "action": "schedule_analysis",
            "file_name": Path(file_path).name,
            "analysis_date": analysis_date,
            "summary": {
                "total_activities": len(schedule_data.get("activities", [])),
                "critical_activities": len(cpm_results.get("critical_path", [])),
                "total_float_average": cpm_results.get("average_float", 0),
                "project_duration": cpm_results.get("project_duration_days", 0),
                "data_date": schedule_data.get("data_date")
            },
            "critical_path": {
                "activities": cpm_results.get("critical_path", [])[:20],
                "path_duration": cpm_results.get("critical_path_duration"),
                "driving_paths": cpm_results.get("driving_paths", [])
            },
            "milestones": self._extract_milestones(schedule_data),
            "delay_analysis": delay_analysis,
            "schedule_risks": schedule_risks,
            "recovery_options": recovery_options,
            "recommendations": self._generate_schedule_recommendations(cpm_results, delay_analysis),
            "detailed_activities": schedule_data.get("activities", [])[:50] if p.get("include_details") else None
        }
    async def _parse_xer_file(self, file_path: str) -> Dict:
        """Parse a Primavera P6 .xer schedule by delegating to the primavera_parser block.

        The block returns a nested shape ({schedule_data, critical_path, milestones,
        activities, wbs, resources}) with per-activity keys like ``total_float_days``
        and ``original_duration_days``. Downstream container logic (``_calculate_cpm``,
        ``_analyze_delays``, ``_extract_milestones``) expects a FLAT per-activity shape
        with keys ``id``/``name``/``start``/``finish``/``duration``/``total_float``/
        ``percent_complete``. This method runs the block and adapts its output to that
        flat shape. A missing or bad file propagates the block's error honestly — no
        fabricated schedule data.
        """
        block = self._get_primavera_parser_block()
        if block is None:
            return {
                "status": "error",
                "error": "primavera_parser block unavailable — cannot parse .xer schedule",
            }

        try:
            result = await block.process({"file_path": file_path})
        except Exception as e:
            return {"status": "error", "error": f"XER parse failed: {str(e)}"}

        if not isinstance(result, dict) or result.get("status") == "error":
            # Propagate the block's error result honestly.
            return result if isinstance(result, dict) else {
                "status": "error", "error": "primavera_parser returned no result"
            }

        # Adapter: block's nested per-activity dicts -> FLAT shape the container needs.
        # Block keys      -> flat keys
        #   id             -> id
        #   name           -> name
        #   start          -> start
        #   finish         -> finish
        #   original_duration_days (days) -> duration (HOURS; *8 to preserve the
        #       hour-semantics _calculate_cpm assumes when it divides by 8)
        #   total_float_days -> total_float (already in days)
        #   percent_complete -> percent_complete
        #   remaining_duration_days / code kept for look_ahead consumers
        flat_activities = []
        for a in result.get("activities", []):
            flat_activities.append({
                "id": a.get("code") or a.get("id", ""),
                "code": a.get("code") or a.get("id", ""),
                "name": a.get("name", ""),
                "start": a.get("start") or "",
                "finish": a.get("finish") or "",
                "duration": (a.get("original_duration_days") or 0) * 8,
                "remaining_duration_days": a.get("remaining_duration_days"),
                "total_float": a.get("total_float_days", 999),
                "total_float_days": a.get("total_float_days", 999),
                "free_float": a.get("total_float_days", 0),
                "percent_complete": a.get("percent_complete", 0),
                "wbs": a.get("wbs_id", ""),
                "wbs_id": a.get("wbs_id", ""),
                "type": a.get("type", ""),
                "status": a.get("status", ""),
                "is_critical": (
                    float(a["total_float_days"]) <= 0
                    if a.get("total_float_days") is not None
                    else False
                ),
            })

        schedule_meta = result.get("schedule_data", {}) or {}
        project_meta = schedule_meta.get("project", {}) or {}

        return {
            "status": "success",
            "file_type": "xer",
            "project_id": project_meta.get("id", ""),
            "project_name": project_meta.get("name", ""),
            # Prefer P6 data date (last_recalc) when the parser exposes it;
            # otherwise fall back to planned start.
            "data_date": (
                project_meta.get("data_date")
                or project_meta.get("planned_start")
                or ""
            ),
            "activities": flat_activities,
        }
    def _analyze_delays(self, current: Dict, baseline: Dict) -> Dict:
        current_acts = {a["id"]: a for a in current.get("activities", [])}
        baseline_acts = {a["id"]: a for a in baseline.get("activities", [])}
    
        delays = []
        new_activities = []
        deleted_activities = []
    
        for act_id, current_act in current_acts.items():
            baseline_act = baseline_acts.get(act_id)
            if not baseline_act:
                new_activities.append(current_act)
                continue
        
            curr_start = current_act.get("start", '')
            base_start = baseline_act.get("start", '')
            if curr_start != base_start:
                delay_days = self._calculate_date_diff(base_start, curr_start)
                if delay_days > 0:
                    delays.append({
                        "activity_id": act_id,
                        "activity_name": current_act.get("name"),
                        "type": "start_delay",
                        "baseline_date": base_start,
                        "current_date": curr_start,
                        "delay_days": delay_days,
                        "percent_complete": current_act.get("percent_complete", 0)
                    })
        
            if current_act.get("percent_complete", 0) < 100:
                curr_finish = current_act.get("finish", '')
                base_finish = baseline_act.get("finish", '')
                if curr_finish and base_finish and curr_finish != base_finish:
                    finish_delay = self._calculate_date_diff(base_finish, curr_finish)
                    if finish_delay > 0:
                        delays.append({
                            "activity_id": act_id,
                            "activity_name": current_act.get("name"),
                            "type": "finish_delay",
                            "baseline_date": base_finish,
                            "current_date": curr_finish,
                            "delay_days": finish_delay
                        })
    
        for base_id in baseline_acts:
            if base_id not in current_acts:
                deleted_activities.append(baseline_acts[base_id])
    
        total_delay = max([d["delay_days"] for d in delays]) if delays else 0
    
        return {
            "total_delay_days": total_delay,
            "delayed_activities": delays,
            "delay_count": len(delays),
            "new_activities": new_activities[:10],
            "deleted_activities": deleted_activities[:10],
            "impact_assessment": self._assess_delay_impact(delays, total_delay)
        }
    def _analyze_schedule_risks(self, cpm_results: Dict) -> List[Dict]:
        risks = []
        if cpm_results.get("average_float", 999) < 2:
            risks.append(self._create_risk_item("schedule", "Schedule has minimal overall float", "high", "high", "Negotiate extensions, reduce scope, or add resources", "Float analysis"))
        return risks
    def _generate_recovery_options(self, delay_analysis: Optional[Dict], cpm_results: Dict) -> List[Dict]:
        if not delay_analysis:
            return []
    
        total_delay = delay_analysis.get("total_delay_days", 0)
        if total_delay <= 0:
            return []
    
        options = []
        options.append({
            "strategy": "Crash Critical Path",
            "description": "Add resources to critical activities",
            "potential_savings_days": total_delay * 0.5,
            "cost_impact": "High",
            "feasibility": "Medium"
        })
        options.append({
            "strategy": "Fast Track",
            "description": "Overlap sequential activities",
            "potential_savings_days": total_delay * 0.3,
            "cost_impact": "Medium",
            "feasibility": "Medium"
        })
        options.append({
            "strategy": "Scope Reduction",
            "description": "Defer non-critical scope to later phase",
            "potential_savings_days": total_delay * 0.5,
            "cost_impact": "Low",
            "feasibility": "High"
        })
        return options
    async def progress_tracker(self, input_data: Any, params: Dict) -> Dict:
        """Track construction progress against planned % / BOQ (progress proxy).

        This is **not** classic EVM. SPI here is actual%/planned% (a progress
        proxy). EV−PV is **schedule variance in currency terms**, not cost
        variance. For true CPI/SPI/CV/SV/EAC use ``evm_calculate`` (or
        ``construction_calc`` → ``calculate_evm``) with PV, EV, and AC.
        """
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        planned_pct = float(p.get("planned_percent") or data.get("planned_percent", 0))
        actual_pct = float(p.get("actual_percent") or data.get("actual_percent", 0))
        contract_value = float(p.get("contract_value") or data.get("contract_value", 0))
        activities = p.get("activities") or data.get("activities", [])
        has_planned = "planned_percent" in p or "planned_percent" in data
        has_actual = "actual_percent" in p or "actual_percent" in data
        if not has_planned and not has_actual and not activities:
            return {
                "status": "error",
                "action": "progress_tracker",
                "error": (
                    "Provide planned_percent and actual_percent, or an "
                    "activities list — cannot track progress with no figures"
                ),
            }
        actual_cost = p.get("actual_cost")
        if actual_cost is None:
            actual_cost = data.get("actual_cost")
        reporting_period = p.get("reporting_period", datetime.now(timezone.utc).strftime("%B %Y"))
        photos = p.get("photos") or data.get("photos", [])

        variance = round(actual_pct - planned_pct, 2)
        # Progress-proxy SPI (actual % / planned %) — NOT classic EV/PV SPI.
        spi_proxy = round(actual_pct / planned_pct, 3) if planned_pct else 1.0
        earned_value = round(contract_value * actual_pct / 100, 2) if contract_value else None
        planned_value = round(contract_value * planned_pct / 100, 2) if contract_value else None
        # EV − PV is schedule variance (SV), not cost variance.
        schedule_variance = None
        if earned_value is not None and planned_value is not None:
            schedule_variance = round(earned_value - planned_value, 2)
        cost_variance = None
        if earned_value is not None and actual_cost is not None:
            try:
                cost_variance = round(earned_value - float(actual_cost), 2)
            except (TypeError, ValueError):
                cost_variance = None

        status = "on_track" if abs(variance) <= 2 else ("ahead" if variance > 0 else "delayed")
        delay_days = round(abs(variance) / 0.5) if variance < -2 else 0

        activity_summary = []
        for act in activities[:20]:
            act_actual = float(act.get("actual_percent", 0))
            act_planned = float(act.get("planned_percent", 0))
            activity_summary.append({
                "activity": act.get("name", act.get("description", "Unknown")),
                "planned": act_planned,
                "actual": act_actual,
                "variance": round(act_actual - act_planned, 1),
                "status": "on_track" if abs(act_actual - act_planned) <= 3 else (
                    "ahead" if act_actual > act_planned else "delayed"
                ),
            })

        ev_block = None
        if contract_value:
            ev_block = {
                "contract_value": contract_value or None,
                "earned_value": earned_value,
                "planned_value": planned_value,
                # Classic name: EV − PV = schedule variance (SV).
                "schedule_variance": schedule_variance,
                # True CV = EV − AC only when actual_cost is supplied.
                "cost_variance": cost_variance,
                "actual_cost": float(actual_cost) if actual_cost is not None and cost_variance is not None else (
                    float(actual_cost) if actual_cost is not None else None
                ),
                "note": (
                    "Progress-proxy figures from % × contract_value. "
                    "schedule_variance = EV − PV. cost_variance requires actual_cost. "
                    "For classic EVM (SPI=EV/PV, CPI=EV/AC) call evm_calculate."
                ),
            }

        return {
            "status": "success",
            "action": "progress_tracker",
            "reporting_period": reporting_period,
            "overall_progress": {
                "planned_percent": planned_pct,
                "actual_percent": actual_pct,
                "variance_percent": variance,
                # Kept for callers; clearly labeled as a % proxy, not EV/PV SPI.
                "schedule_performance_index": spi_proxy,
                "schedule_performance_index_kind": "progress_proxy_actual_over_planned_pct",
                "status": status,
                "estimated_delay_days": delay_days,
            },
            "earned_value": ev_block,
            "activities": activity_summary,
            "photos_reviewed": len(photos),
            "key_risks": [
                f"Project is {abs(variance):.1f}% behind schedule — recovery plan required"
            ] if variance < -5 else [],
            "recommendations": (
                ["Issue delay notice and prepare recovery programme"] if variance < -10
                else ["Monitor weekly and flag if variance exceeds -5%"] if variance < -2
                else ["Maintain current momentum"]
            ),
        }

    async def evm_calculate(self, input_data: Any, params: Dict) -> Dict:
        """Classic EVM calculator — requires PV, EV, AC (BAC optional for forecasts).

        Single wired path into ``construction_knowledge.calculate_evm``. Refuses
        when PV/EV/AC are missing; does not invent actuals. Prefer this (or
        ``construction_calc`` with ``calculation=calculate_evm``) over treating
        ``progress_tracker`` as EVM.
        """
        from app.core.construction_knowledge import calculate_evm

        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        merged = {**data, **p}

        def _pick(*keys):
            for k in keys:
                if merged.get(k) is not None and merged.get(k) != "":
                    return merged.get(k)
            return None

        result = calculate_evm(
            bac=_pick("bac", "BAC", "budget_at_completion"),
            bcwp=_pick("bcwp", "BCWP"),
            bcws=_pick("bcws", "BCWS"),
            acwp=_pick("acwp", "ACWP"),
            pv=_pick("pv", "PV", "planned_value"),
            ev=_pick("ev", "EV", "earned_value"),
            ac=_pick("ac", "AC", "actual_cost"),
        )
        if isinstance(result, dict) and isinstance(result.get("error"), str):
            return {
                "status": "error",
                "action": "evm_calculate",
                "error": result["error"],
                "required": result.get("required"),
                "optional": result.get("optional"),
            }
        return {
            "status": "success",
            "action": "evm_calculate",
            "evm": result,
            "source": "construction_knowledge.calculate_evm",
            "note": (
                "Classic EVM: SPI=EV/PV, CPI=EV/AC, SV=EV−PV, CV=EV−AC. "
                "EAC defaults to BAC/CPI when BAC present and CPI>0."
            ),
        }
    async def warranty_maintenance_schedule(self, input_data: Any, params: Dict) -> Dict:
        """Generate warranty and planned maintenance schedule for installed systems."""
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        systems = p.get("systems") or data.get("systems") or data.get("equipment", [])
        project_name = p.get("project_name", data.get("project_name", "Project"))
        handover_date = p.get("handover_date") or data.get("handover_date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        defects_liability_months = int(p.get("defects_liability_months", data.get("defects_liability_months", 12)))

        if not systems:
            return {
                "status": "error",
                "action": "warranty_maintenance_schedule",
                "error": (
                    "Provide systems / equipment — cannot invent a warranty "
                    "register"
                ),
            }

        warranty_register = []
        maintenance_tasks = []

        warranty_periods = {
            "mechanical": 24, "electrical": 12, "plumbing": 12,
            "vertical_transport": 24, "fire_protection": 12,
            "architectural": 12, "waterproofing": 60, "structural": 120,
        }

        from datetime import timedelta
        try:
            ho_date = datetime.strptime(handover_date[:10], "%Y-%m-%d")
        except Exception:
            ho_date = datetime.now(timezone.utc)

        for system in systems:
            sys_type = system.get("type", "general")
            warranty_months = warranty_periods.get(sys_type, 12)
            expiry = (ho_date + timedelta(days=warranty_months * 30)).strftime("%Y-%m-%d")
            dlp_expiry = (ho_date + timedelta(days=defects_liability_months * 30)).strftime("%Y-%m-%d")

            warranty_register.append({
                "system": system.get("name", system.get("description", "Unknown")),
                "type": sys_type,
                "supplier": system.get("supplier", "TBD"),
                "handover_date": handover_date,
                "warranty_months": warranty_months,
                "warranty_expiry": expiry,
                "dlp_expiry": dlp_expiry,
                "status": "Active",
            })

            for freq, task in self._get_maintenance_tasks(sys_type):
                maintenance_tasks.append({
                    "system": system.get("name", "Unknown"),
                    "task": task,
                    "frequency": freq,
                    "next_due": (ho_date + timedelta(days=30)).strftime("%Y-%m-%d"),
                })

        return {
            "status": "success",
            "action": "warranty_maintenance_schedule",
            "project": project_name,
            "handover_date": handover_date,
            "defects_liability_period_months": defects_liability_months,
            "total_systems": len(warranty_register),
            "warranty_register": warranty_register,
            "maintenance_schedule": maintenance_tasks[:50],
            "early_expiries": [
                w for w in warranty_register if w["warranty_months"] <= 12
            ],
            "recommendations": [
                "Register all warranties with suppliers within 30 days of handover",
                "Set calendar reminders 60 days before warranty expiry for inspection",
                f"Defects liability period expires {warranty_register[0]['dlp_expiry'] if warranty_register else 'TBD'} — conduct final inspection 30 days prior",
            ],
        }
    async def daily_site_report(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        voice_notes = data.get("voice_files") or p.get("voice_files", [])
        photos = data.get("photos") or p.get("photos", [])
        # Read from BOTH input_data and params: the generic predefined dispatcher
        # (_run_container_action) merges resolved params into the envelope and
        # passes it as input_data, so a params-only read would miss the project
        # location resolved by chat.py. (Same convention as every other action.)
        site_location = p.get("location") or data.get("location")
        date = p.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        supervisor = p.get("supervisor", "Site Manager")
        project_name = p.get("project_name", "Project")
    
        transcriptions = []
        for voice_file in voice_notes:
            voice_block = self.get_dep("voice")
            if not voice_block:
                logger.warning("voice block unavailable — skipping voice note %s", voice_file)
                transcriptions.append({"file": Path(voice_file).name, "text": "", "timestamp": 0,
                                       "error": "voice block unavailable"})
                continue
            try:
                # Pass the audio location under `text` per the operator
                # contract: voice 2.1+ treats `text` as a file path on STT
                # when it exists on disk (this also covers the platform's
                # InputAdapter, which wraps a positional string into
                # {"text": "<path>"} for blocks that don't have a typed
                # file_path schema field). Pre-2.1 the caller sent
                # `audio_path` which voice silently dropped, producing
                # empty transcripts in daily site reports — this is the
                # regression-guard call site.
                envelope = await voice_block.execute(
                    {"text": voice_file}, {"action": "transcribe"}
                )
                # `execute()` wraps the block's process() return under
                # `result`; unwrap to read status / text. Tolerate the
                # legacy shape where the block returned the dict directly.
                inner = envelope.get("result", envelope) if isinstance(envelope, dict) else {}
                if inner.get("status") == "error":
                    err = inner.get("error", "transcription failed")
                    logger.warning("voice transcription failed for %s: %s", voice_file, err)
                    transcriptions.append({"file": Path(voice_file).name, "text": "",
                                           "timestamp": 0, "error": err})
                    continue
                transcriptions.append({
                    "file": Path(voice_file).name,
                    "text": inner.get("text", ""),
                    "timestamp": inner.get("segments", [{}])[0].get("start", 0)
                                if inner.get("segments") else 0,
                })
            except Exception as exc:
                logger.exception("voice block raised on %s", voice_file)
                transcriptions.append({"file": Path(voice_file).name, "text": "",
                                       "timestamp": 0, "error": str(exc)})
    
        weather = await self._fetch_weather(site_location, date) if site_location else {}
    
        photo_analysis = []
        for photo in photos:
            analysis = await self._analyze_site_photo(photo)
            photo_analysis.append(analysis)
    
        activities = self._extract_activities_from_voice(transcriptions)
        issues = self._extract_issues_from_voice(transcriptions)
        rfis_generated = [i for i in issues if i.get("type") == "clarification_needed"]
        manpower = self._extract_manpower_from_voice(transcriptions)
        equipment = self._extract_equipment_from_photos(photo_analysis)
        narrative = self._generate_daily_narrative(date, activities, issues, weather, manpower)

        # Name every section that is empty because its input was never
        # supplied (no voice notes / no photos): the report is still useful
        # partial output, but it must not read as a complete DSR.
        sections_incomplete = []
        if not voice_notes:
            sections_incomplete += ["manpower", "work_completed", "issues_encountered",
                                    "materials_delivered"]
        if not photos:
            sections_incomplete += ["equipment", "quality_observations"]

        # Photos supplied but the detector could not run (SAFETY_WORLD_WEIGHTS
        # unset, or the model failed to load) is a THIRD state, distinct from
        # both "no photos" and "photos analysed, nothing found". Without this,
        # an empty equipment/quality section would read as "nothing on site"
        # when in truth nothing looked. The reader must be able to tell those
        # apart before acting on the report.
        detector_unavailable = bool(photos) and all(
            p.get("detector") == "unavailable" for p in photo_analysis
        )
        if detector_unavailable:
            sections_incomplete += ["equipment", "quality_observations"]
        photo_intelligence = {
            "photos_analysed": len(photo_analysis),
            "detector": "unavailable" if detector_unavailable else (
                "safety_world_v2" if photos else "not_run"),
            "status": (
                "unavailable — photo observations could not be produced"
                if detector_unavailable else
                "ok" if photos else "no photos supplied"
            ),
        }

        return {
            "status": "success",
            "action": "daily_report_generated",
            "sections_incomplete": sections_incomplete,
            "report_metadata": {
                "date": date,
                "project": project_name,
                "supervisor": supervisor,
                "report_number": f"DSR-{date.replace('-', '')}",
                "weather_conditions": weather
            },
            "manpower": {
                "total_present": manpower.get("total", 0),
                "by_trade": manpower.get("by_trade", {}),
                "absentees": manpower.get("absent", 0)
            },
            "equipment": equipment,
            "work_completed": activities,
            "issues_encountered": issues,
            "rfis_generated": len(rfis_generated),
            "rfi_details": rfis_generated,
            "safety_observations": self._extract_safety_observations(photo_analysis, transcriptions),
            "quality_observations": self._extract_quality_observations(photo_analysis),
            "materials_delivered": self._extract_material_deliveries(transcriptions),
            "photos_attached": len(photos),
            "photo_intelligence": photo_intelligence,
            "photo_analysis": photo_analysis,
            "transcriptions": transcriptions,
            "full_narrative": narrative,
            "next_day_plan": self._generate_next_day_plan(activities, issues),
            "distribution_list": ["Project Manager", "Site Engineer", "QS", "HSE Officer"]
        }
    async def commissioning_checklist(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        spec_file = data.get("spec_file") or p.get("spec_file")
        equipment_list = data.get("equipment_list") or p.get("equipment_list", [])
        systems = p.get("systems") or data.get("systems", ["electrical", "mechanical", "fire", "lift", "facade"])
        substantial_completion = p.get("substantial_completion_date") or data.get("substantial_completion_date")
    
        checklists = {}
        for system in systems:
            if system in ("electrical",):
                checklists["electrical"] = self._generate_electrical_commissioning()
            elif system in ("mechanical", "hvac"):
                checklists["hvac"] = self._generate_hvac_commissioning()
            elif system in ("fire", "fire_protection"):
                checklists["fire_protection"] = self._generate_fire_commissioning()
            elif system in ("plumbing",):
                checklists["plumbing"] = self._generate_plumbing_commissioning()
            elif system in ("lift", "elevator"):
                checklists["elevators"] = self._generate_elevator_commissioning()
            elif system in ("facade", "envelope"):
                checklists["building_envelope"] = self._generate_facade_commissioning()
            elif system in ("waterproofing", "waterproof", "membrane"):
                checklists["waterproofing"] = self._generate_waterproofing_commissioning()
            elif system in ("reservoir", "pwps", "wet_test", "hydraulic", "concrete"):
                checklists["reservoir"] = self._generate_reservoir_commissioning()
            elif system in ("bms", "automation"):
                checklists["bms"] = self._generate_bms_commissioning()
    
        all_tests = []
        for system, checklist in checklists.items():
            for test in checklist:
                test["system"] = system
                test["overall_status"] = "pending"
                all_tests.append(test)
    
        total_tests = len(all_tests)
        passed = 0
        failed = 0
        pending = total_tests
        commissioning_duration = self._estimate_commissioning_duration(systems, len(equipment_list))
    
        return {
            "status": "success",
            "action": "commissioning_checklist_generated",
            "project_phase": "pre_handover",
            "substantial_completion_target": substantial_completion,
            "commissioning_period_weeks": commissioning_duration,
            "completion_target": self._add_weeks(substantial_completion, commissioning_duration) if substantial_completion else None,
            "summary": {
                "total_tests": total_tests,
                "systems_covered": len(systems),
                "passed": passed,
                "failed": failed,
                "pending": pending,
                "percent_complete": (passed / total_tests * 100) if total_tests else 0
            },
            "checklists_by_system": checklists,
            "master_test_schedule": all_tests,
            "witness_required": [t for t in all_tests if t.get("witness_required")],
            "third_party_testing": [t for t in all_tests if t.get("third_party_required")],
            "documentation_required": self._list_commissioning_docs(systems),
            "training_requirements": self._generate_training_requirements(systems),
            "deficiency_tracking": [],
            "final_sign_off": {
                "mechanical_contractor": "pending",
                "electrical_contractor": "pending",
                "fire_contractor": "pending",
                "commissioning_authority": "pending",
                "client_representative": "pending"
            }
        }
    async def resource_histogram(self, input_data: Any, params: Dict) -> Dict:
        """Time-phased manpower histogram from a real Primavera P6 ``.xer``.

        Delegates to the proven, hand-verified library
        (:func:`app.lib.pm_computations.resource_histogram`, TASKRSRC path):
        parse the ``.xer``, run CPM for the early-date timeline, then time-phase
        the schedule's REAL per-task resource assignments (TASKRSRC rows) into
        periods. A schedule with no resource loading (no TASKRSRC rows) returns
        an activity-count histogram (CPM date buckets, not man-hours) — never
        a fabricated labor split.
        """
        import os

        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        schedule_file = data.get("schedule_file") or p.get("schedule_file")
        period_unit = (p.get("period_unit") or "week").lower()

        # Histogram FROM the structured schedule (a prose brief or BOQ run
        # through generate_wbs), no .xer required (#3, capability restored
        # 0712Z): the activities generate_wbs produced are time-phased into
        # period buckets directly. UNKNOWN-duration rows have no span to phase
        # and are excluded; with no resource loading the buckets are activity
        # counts, not fabricated man-hours.
        inline_acts = data.get("activities") or p.get("activities")
        if not schedule_file and inline_acts:
            schedulable = [
                a for a in inline_acts
                if isinstance(a, dict)
                and isinstance(a.get("duration_days"), int)
                and not isinstance(a.get("duration_days"), bool)
                and a["duration_days"] > 0
            ]
            if not schedulable:
                return {
                    "status": "error",
                    "action": "resource_histogram",
                    "error": (
                        "The structured schedule has no dated activities to "
                        "phase (every duration is UNKNOWN)."
                    ),
                }
            from app.schemas.cpm import (
                Activity as CPMActivity, Dependency, ResourceAssignment,
            )
            kept_ids = {str(a.get("id")) for a in schedulable}
            cpm_acts = [
                CPMActivity(
                    id=str(a["id"]),
                    name=a.get("name", ""),
                    duration=int(a["duration_days"]),
                    predecessors=[
                        Dependency(predecessor_id=str(pr))
                        for pr in (a.get("predecessors") or [])
                        if str(pr) in kept_ids
                    ],
                    wbs_code=str(a.get("wbs_phase", "")),
                    resources=[
                        ResourceAssignment(trade=str(t), count=1.0)
                        for t in (a.get("resources") or [])
                    ],
                )
                for a in schedulable
            ]
            return self._histogram_from_activity_counts(
                cpm_acts,
                period_unit,
                reason=(
                    "Histogram built from the structured schedule's activity "
                    "spans; no resource loading was supplied, so the buckets are "
                    "activity counts, not man-hours"
                ),
            )

        if not schedule_file:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": "No schedule file provided — pass schedule_file pointing to a .xer schedule",
            }
        if not os.path.exists(schedule_file):
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"Schedule file not found: {schedule_file}",
            }
        if not str(schedule_file).lower().endswith(".xer"):
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": "Resource histogram requires a Primavera P6 .xer schedule",
            }

        # Read the raw .xer text (open_plaintext transparently decrypts when a
        # DATA_ENCRYPTION_KEY is configured) — the library parser needs the text
        # blob, not the block's flattened per-activity shape (which drops the
        # resource assignments this histogram is built from).
        try:
            from app.core.file_crypto import open_plaintext
            with open_plaintext(schedule_file) as plain_path:
                with open(plain_path, "r", encoding="cp1252", errors="replace") as f:
                    xer_text = f.read()
        except Exception as exc:  # noqa: BLE001 — surface the read failure honestly
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"Could not read schedule file: {exc}",
            }

        try:
            from app.lib.pm_computations import (
                parse_xer_full, compute_cpm,
                resource_histogram as _lib_resource_histogram,
                CircularDependencyError,
            )
            from app.schemas.cpm import CPMInput
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"pm_computations library unavailable: {exc}",
            }

        try:
            bundle = parse_xer_full(xer_text)
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"XER parse failed: {exc}",
            }

        activities = bundle.get("activities") or []
        task_resources = bundle.get("task_resources") or []
        resources_meta = bundle.get("resources") or {}
        if not activities:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": "Schedule contains no activities — cannot build a resource histogram",
            }
        if not task_resources:
            return self._histogram_from_activity_counts(
                activities,
                period_unit,
                reason=(
                    "Schedule has no resource assignments (no TASKRSRC rows) — "
                    "cannot build a labor histogram from this .xer"
                ),
            )

        # A LABOR histogram counts man-hours from RT_Labor resources ONLY. A
        # TASKRSRC target_qty for RT_Mat/RT_Equip is a material/cost/equipment
        # quantity in its own unit — summing it as "man-hours" fabricates the
        # total (a real the client project baseline mixes a cost resource whose qty dwarfs all
        # labor by ~1000x). Relabel each labor row's trade to the human resource
        # short_name. When the schedule carries no RSRC dictionary at all, no
        # type is knowable, so every assignment is treated as labor (best effort).
        labor_rows: List[Dict[str, Any]] = []
        for tr in task_resources:
            meta = resources_meta.get(tr.get("rsrc_id")) or {}
            rtype = meta.get("type")
            if rtype == "RT_Labor" or (not resources_meta and not rtype):
                labeled = dict(tr)
                labeled["rsrc_id"] = meta.get("short_name") or tr.get("rsrc_id")
                labor_rows.append(labeled)
        if not labor_rows:
            n_types = sorted({(resources_meta.get(tr.get("rsrc_id")) or {}).get("type", "?")
                              for tr in task_resources})
            return self._histogram_from_activity_counts(
                activities,
                period_unit,
                reason=(
                    "Schedule has resource assignments but none are labor "
                    f"(RT_Labor) — found types {n_types}"
                ),
            )

        try:
            cpm_out = compute_cpm(CPMInput(activities=activities))
        except (CircularDependencyError, ValueError) as exc:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"CPM computation failed: {exc}",
            }

        try:
            hist = _lib_resource_histogram(
                cpm_out.results, activities,
                period_unit=period_unit, task_resources=labor_rows,
            )
        except ValueError as exc:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"Histogram computation failed: {exc}",
            }

        hist_dict = hist.model_dump()
        return {
            "status": "success",
            "action": "resource_histogram_generated",
            "period_unit": hist.period_unit,
            "period_count": len(hist.periods),
            "periods": hist_dict.get("periods", []),
            "peak_total": hist.peak_total,
            "peak_period": hist.peak_period,
            "by_trade_totals": hist_dict.get("by_trade_totals", {}),
            "total_manhours": hist.total_manhours,
            "resource_assignments": len(labor_rows),
            "labor_resource_count": len(labor_rows),
            "total_assignments_in_schedule": len(task_resources),
            "resource_types_included": "RT_Labor",
            "source": "primavera_taskrsrc",
            "note": (
                "Time-phased man-hours from the schedule's real P6 TASKRSRC "
                "RT_Labor assignments only (material/cost/equipment excluded); "
                "trades are the P6 resource short names."
            ),
        }

    async def look_ahead(self, input_data: Any, params: Dict) -> Dict:
        """3–4 week look-ahead from an uploaded Primavera ``.xer``.

        Activities whose ES/EF (or start/finish) overlap the window are
        returned with remaining duration, total float, critical flag, WBS, and
        name. Driven by schedule dates — not Aconex/CDE. Honest error when no
        ``.xer`` / no activities; never fabricates rows.
        """
        import os

        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        schedule_file = (
            data.get("schedule_file") or p.get("schedule_file")
            or data.get("file_path") or p.get("file_path")
        )

        # Window: weeks (default 3 → 21 calendar days) or explicit days (21–28).
        # NL turns ("14-day look-ahead") arrive with days unset — parse the ask.
        window_days = p.get("days") if p.get("days") is not None else data.get("days")
        weeks = p.get("weeks") if p.get("weeks") is not None else data.get("weeks")
        if window_days is None and weeks is None:
            ask = " ".join(
                str(x) for x in (
                    p.get("user_message"), data.get("user_message"),
                    p.get("message"), data.get("message"),
                    p.get("brief"), data.get("brief"),
                ) if x
            )
            parsed_days, parsed_weeks = _look_ahead_window_from_text(ask)
            if parsed_days is not None:
                window_days = parsed_days
            elif parsed_weeks is not None:
                weeks = parsed_weeks
        if window_days is not None:
            try:
                window_days = int(window_days)
            except (TypeError, ValueError):
                return {
                    "status": "error",
                    "action": "look_ahead",
                    "error": f"Invalid days value: {window_days!r}",
                }
        elif weeks is not None:
            try:
                window_days = int(float(weeks) * 7)
            except (TypeError, ValueError):
                return {
                    "status": "error",
                    "action": "look_ahead",
                    "error": f"Invalid weeks value: {weeks!r}",
                }
        else:
            window_days = 21  # default ~3 weeks calendar

        if window_days < 1:
            return {
                "status": "error",
                "action": "look_ahead",
                "error": "window must be at least 1 day",
            }

        # Activities supplied in the request are a valid source. An ask
        # that typed its activities and declined a programme file used to
        # error for a missing .xer anyway, so the answer became
        # "supply a .xer" instead of the look-ahead the operator had already
        # given the data for. A file still wins when one is named, and an
        # empty request still errors -- nothing is ever invented here.
        inline_activities = _inline_look_ahead_activities(data, p)
        if not schedule_file and not inline_activities:
            return {
                "status": "error",
                "action": "look_ahead",
                "error": (
                    "No schedule provided — pass schedule_file pointing to a "
                    "Primavera P6 .xer, or activities with start and finish "
                    "dates"
                ),
            }
        if not schedule_file:
            from app.lib.pm_computations import select_look_ahead as _select
            window = _select(
                inline_activities, as_of=_look_ahead_as_of(data, p),
                window_days=window_days,
            )
            return {
                "status": "success",
                "action": "look_ahead",
                "as_of": window["as_of"],
                "window_days": window["window_days"],
                "window_end": window["window_end"],
                "activity_count": window["count"],
                "activities": window["activities"],
                "total_activities_in_schedule": len(inline_activities),
                "source": "activities_supplied_in_request",
                "note": (
                    f"Look-ahead: activities overlapping {window['as_of']} … "
                    f"{window['window_end']} ({window_days} calendar days) from "
                    "the activities supplied in the request. Empty overlap is a "
                    "valid result — no fabricated activities."
                ),
            }
        if not os.path.exists(str(schedule_file)):
            return {
                "status": "error",
                "action": "look_ahead",
                "error": f"Schedule file not found: {schedule_file}",
            }
        if not str(schedule_file).lower().endswith(".xer"):
            return {
                "status": "error",
                "action": "look_ahead",
                "error": "Look-ahead requires a Primavera P6 .xer schedule",
            }

        schedule_data = await self._parse_xer_file(str(schedule_file))
        if not isinstance(schedule_data, dict) or schedule_data.get("status") == "error":
            return {
                "status": "error",
                "action": "look_ahead",
                "error": (
                    (schedule_data or {}).get("error")
                    if isinstance(schedule_data, dict)
                    else "XER parse failed"
                ),
            }

        activities = schedule_data.get("activities") or []
        if not activities:
            return {
                "status": "error",
                "action": "look_ahead",
                "error": (
                    "Schedule contains no activities — cannot build a look-ahead"
                ),
            }

        as_of_raw, stated_as_of = _resolve_stated_or_explicit_as_of(data, p)
        if not as_of_raw:
            as_of_raw = schedule_data.get("data_date")
            stated_as_of = False
        as_of_date = None
        if as_of_raw:
            try:
                from app.lib.pm_computations import _coerce_date
                as_of_date = _coerce_date(as_of_raw)
            except Exception:  # noqa: BLE001
                as_of_date = None
            if as_of_date is None and not stated_as_of:
                return {
                    "status": "error",
                    "action": "look_ahead",
                    "error": f"Invalid as_of / data_date: {as_of_raw!r}",
                }
        if as_of_date is None:
            from app.lib.pm_computations import _clock_today
            as_of_date = _clock_today()

        try:
            from app.lib.pm_computations import select_look_ahead
            window = select_look_ahead(
                activities, as_of=as_of_date, window_days=window_days,
            )
        except ValueError as exc:
            return {
                "status": "error",
                "action": "look_ahead",
                "error": str(exc),
            }

        return {
            "status": "success",
            "action": "look_ahead",
            "as_of": window["as_of"],
            "window_days": window["window_days"],
            "window_end": window["window_end"],
            "activity_count": window["count"],
            "activities": window["activities"],
            "schedule_file": os.path.basename(str(schedule_file)),
            "total_activities_in_schedule": len(activities),
            "source": "primavera_xer_dates",
            "note": (
                f"Look-ahead: activities overlapping {window['as_of']} … "
                f"{window['window_end']} ({window_days} calendar days) from "
                "uploaded .xer ES/EF (or start/finish). Empty overlap is a "
                "valid result — no fabricated activities."
            ),
        }

    def _histogram_from_activity_counts(
        self,
        activities: List[Dict[str, Any]],
        period_unit: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Period buckets from CPM activity spans when TASKRSRC labor is absent.

        Counts overlapping activities per week/month. Does not invent man-hours
        or a trade split — live M17 on theshovel.ai had a 20-day 3-activity
        XER with no crew payload; the chat still needs period buckets.
        """
        try:
            from app.lib.pm_computations import compute_cpm, CircularDependencyError
            from app.schemas.cpm import CPMInput
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"pm_computations library unavailable: {exc}",
            }
        try:
            cpm_out = compute_cpm(CPMInput(activities=activities))
        except (CircularDependencyError, ValueError) as exc:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": (
                    f"{reason}. Also could not build an activity-count "
                    f"histogram: {exc}"
                ),
            }
        length = 21 if period_unit == "month" else 5
        es_ef = {
            r.id: (r.early_start_day, r.early_finish_day) for r in cpm_out.results
        }
        if not es_ef:
            return {
                "status": "error",
                "action": "resource_histogram",
                "error": f"{reason}. CPM produced no dated activities.",
            }
        last_day = max(ef for (_es, ef) in es_ef.values())
        n_periods = max(1, -(-last_day // length))
        periods: List[Dict[str, Any]] = []
        for p in range(n_periods):
            p_start, p_end = p * length, (p + 1) * length
            names = []
            for act in activities:
                aid = getattr(act, "id", None) or (act.get("id") if isinstance(act, dict) else None)
                span = es_ef.get(aid)
                if not span:
                    continue
                es, ef = span
                if ef <= p_start or es >= p_end:
                    continue
                names.append(
                    getattr(act, "name", None)
                    or (act.get("name") if isinstance(act, dict) else aid)
                    or str(aid)
                )
            label = f"{period_unit[0].upper()}{p + 1}"
            periods.append({
                "index": p,
                "label": label,
                "total": len(names),
                "activities": names,
                "by_trade": {},
            })
        peak = max(periods, key=lambda hp: hp["total"], default=None)
        return {
            "status": "success",
            "action": "resource_histogram_generated",
            "histogram_kind": "activity_count",
            "period_unit": period_unit,
            "period_count": len(periods),
            "periods": periods,
            "peak_total": peak["total"] if peak else 0,
            "peak_period": peak["label"] if peak else "",
            "by_trade_totals": {},
            "source": "activity_cpm_counts",
            "labor_unavailable_reason": reason,
            "note": (
                f"{reason}. Period buckets are activity counts from CPM dates, "
                "not man-hours. Re-export the programme with resource loading "
                "for a labor histogram."
            ),
        }

    async def claims_builder(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        delay_events = data.get("delay_events") or p.get("delay_events", [])
        schedule_file = data.get("schedule_file") or p.get("schedule_file")
        contract_file = data.get("contract_file") or p.get("contract_file")
        baseline_file = data.get("baseline_file") or p.get("baseline_file")
        notification_date = p.get("notification_date", datetime.now(timezone.utc).isoformat())
        claim_type = p.get("claim_type", "eot")
        text = " ".join(
            str(x)
            for x in (
                p.get("text"),
                p.get("user_message"),
                data.get("text"),
                data.get("user_message"),
                data.get("message"),
                input_data if isinstance(input_data, str) else "",
            )
            if x
        )
        # S14: a contractual notice-period question is not a claim draft.
        # Refuse before the delay_events gate so the turn cannot dead-end
        # asking for schedule+baseline on a Contract Data lookup.
        from app.core.contract_lookup_intent import (
            message_is_eot_notice_period_lookup,
        )
        if message_is_eot_notice_period_lookup(text):
            return {
                "status": "error",
                "action": "claims_builder",
                "error": (
                    "This is a Contract Data lookup (EOT / claim notice "
                    "period). Do not build a claim and do not require "
                    "delay_events. Answer from retrieved contract context."
                ),
            }

        notice_facts = _delay_claim_facts_from_text(text)
        if notice_facts and not delay_events and not (schedule_file and baseline_file):
            return _draft_delay_claim_notice(notice_facts, notification_date)

        if not delay_events and not (schedule_file and baseline_file):
            # Honest gate — NEVER fabricate delay events. A claim built on invented
            # events produces a $-quantified EOT/quantum figure with no basis.
            return {
                "status": "error",
                "action": "claims_builder",
                "error": (
                    "Claim building requires delay_events (or a schedule_file + "
                    "baseline_file to derive them). Provide delay_events as a list of "
                    "{event_id, description, delay_days, responsibility, date, cost_impact}."
                ),
            }
    
        if schedule_file and baseline_file:
            delay_analysis = await self.parse_primavera_schedule({"file_path": schedule_file}, {"baseline_file": baseline_file})
            delay_details = delay_analysis.get("delay_analysis", {})
        else:
            delay_details = {"total_delay_days": sum(e.get("delay_days", 0) for e in delay_events)}
    
        contract_entitlement = {}
        if contract_file:
            contract_data = await self.process_contract({"file_path": contract_file}, {})
            contract_entitlement = self._check_eot_entitlement(contract_data, delay_events)
    
        narrative = self._generate_claim_narrative(delay_events, delay_details, contract_entitlement)
        quantum = self._calculate_prolongation_costs(delay_details.get("total_delay_days", 0), delay_events)
        causation = self._build_causation_link(delay_events, delay_details)
    
        return {
            "status": "success",
            "action": "claim_generated",
            "claim_type": claim_type,
            "claim_number": f"EOT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-001",
            "notification_date": notification_date,
            "delay_summary": {
                "total_delay_days": delay_details.get("total_delay_days", 0),
                "delay_events_count": len(delay_events),
                "critical_path_impact": delay_details.get("critical_path_impact", False),
                "concurrent_delays": self._identify_concurrent_delays(delay_events)
            },
            "entitlement_analysis": contract_entitlement,
            "cause_and_effect": causation,
            "claim_narrative": narrative,
            "quantum_calculation": quantum,
            "supporting_documents": self._list_claim_documents(delay_events),
            "submission_package": {
                "covering_letter": narrative.get("executive_summary"),
                "detailed_narrative": narrative.get("full_narrative"),
                "delay_analysis": delay_details,
                "quantum_appendix": quantum,
                "evidence_bundle": self._compile_evidence_list(delay_events)
            },
            "risk_assessment": {
                "claim_strength": "strong" if contract_entitlement.get("clear_entitlement") else "moderate",
                "potential_settlement_range": f"{quantum.get('total_claim', 0) * 0.7} - {quantum.get('total_claim', 0)}",
                "counter_arguments": self._anticipate_defenses(delay_events),
                "recommended_strategy": "negotiate_settlement" if len(delay_events) > 5 else "formal_claim"
            }
        }
    def _calculate_prolongation_costs(
        self,
        total_days: int,
        events: List[Dict],
        daily_rate: Optional[float] = None,
    ) -> Dict:
        """Prolongation / preliminaries claim calculation.

        Two-mode behaviour:
        - If `events` carries real per-event `cost_impact` values, sum those
          (the actual claim, derived from project data).
        - Otherwise, fall back to a daily-rate × total_days approach. The
          caller must pass `daily_rate` explicitly; the previous hardcoded
          $5000/day is no longer the default.

        The breakdown percentages (site staff, accommodation, etc.) sum to 1.0
        and are applied to whichever total is used so the line items
        reconcile to the total.
        """
        # Mode 1: use real per-event costs when present.
        evt_costs = [
            float(e.get("cost_impact") or 0)
            for e in (events or [])
            if e.get("cost_impact") is not None
        ]
        if evt_costs:
            total_claim = round(sum(evt_costs), 2)
            mode = "event_sum"
            implied_daily = round(total_claim / total_days, 2) if total_days else 0.0
        else:
            # Mode 2: daily-rate × days. No more hardcoded $5000/day default —
            # caller must supply the prolongation rate from project records.
            if daily_rate is None or daily_rate <= 0:
                return {
                    "status": "error",
                    "error": (
                        "Provide either per-event cost_impact values in `events`, "
                        "or a positive `daily_rate` (USD/day) from your project's "
                        "preliminaries / site overhead records."
                    ),
                }
            total_claim = round(daily_rate * total_days, 2)
            implied_daily = daily_rate
            mode = "daily_rate"

        breakdown_pct = {
            "site_staff": 0.30,
            "site_accommodation": 0.20,
            "plant_standing": 0.25,
            "insurances_bonds": 0.10,
            "overheads_profit": 0.15,
        }
        breakdown = {k: round(total_claim * pct, 2) for k, pct in breakdown_pct.items()}

        return {
            "prolongation_period_days": total_days,
            "calculation_mode": mode,
            "events_used": len(evt_costs),
            "daily_preliminaries_rate": implied_daily,
            "breakdown": breakdown,
            "breakdown_percentages": breakdown_pct,
            "total_claim": total_claim,
        }
    def _check_eot_entitlement(self, contract_data: Dict, events: List[Dict]) -> Dict:
        """EOT entitlement check — keyword-based, honest about its limits.

        Previously returned a hardcoded `{"clear_entitlement": True,
        "relevant_clause": "14.1"}` regardless of contract or events. Now
        keyword-matches the contract text for common Employer Risk Event
        triggers and surfaces what was actually found.

        Real legal/contractual assessment requires lawyer review; this
        method's output is a *first-pass triage*, not a verdict.
        """
        text = ""
        if isinstance(contract_data, dict):
            text = str(contract_data.get("text") or contract_data.get("content") or "").lower()
        elif isinstance(contract_data, str):
            text = contract_data.lower()

        # Employer-risk triggers (very rough — FIDIC, NEC, ad-hoc variants):
        triggers = {
            "variation": "Variation issued by Employer",
            "instruction": "Engineer/Employer instruction",
            "delay by employer": "Delay caused by Employer",
            "exceptional weather": "Exceptional adverse weather",
            "force majeure": "Force majeure event",
            "unforeseeable": "Unforeseeable physical conditions",
            "suspension": "Suspension by Employer",
            "change in legislation": "Change in legislation",
            "late drawing": "Late issue of drawings",
            "late access": "Late access to site",
        }
        found_triggers = [
            label for kw, label in triggers.items() if text and kw in text
        ]

        # Look for explicit clause references in the contract text.
        clause_match = None
        try:
            import re as _re
            m = _re.search(
                r"\b(?:clause|sub[- ]?clause|section|sec\.?)\s+(\d+(?:\.\d+)*)",
                text,
            )
            if m:
                clause_match = m.group(1)
        except Exception:
            logger.warning(
                "swallowed %s in _check_eot_entitlement() — continuing",
                "Exception", exc_info=True,
            )

        # An event with `compensable=True` AND a matched trigger = clearer
        # entitlement. Empty event list = nothing to claim.
        compensable_events = [e for e in events if e.get("compensable")]
        clear_entitlement = bool(compensable_events) and bool(found_triggers)

        return {
            "clear_entitlement": clear_entitlement,
            "compensable_event_count": len(compensable_events),
            "trigger_keywords_found": found_triggers,
            "relevant_clause": clause_match,
            "entitlement_basis": (
                "Compensable event matched a contract trigger keyword"
                if clear_entitlement
                else "No clear trigger keywords matched, OR no compensable events provided"
            ),
            "_caveat": (
                "Keyword-based first-pass triage. A real EOT entitlement "
                "decision requires lawyer review of the actual clauses."
            ),
        }
    async def forensic_delay_analysis(self, input_data: Any, params: Dict) -> Dict:
        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}
        baseline_file = data.get("baseline_file") or p.get("baseline_file")
        updated_file = data.get("updated_file") or p.get("updated_file")
        delay_events = data.get("delay_events") or p.get("delay_events", [])
        analysis_method = p.get("method", "time_impact")
    
        if not baseline_file or not updated_file:
            return {
                "status": "error",
                "error": (
                    "Forensic delay analysis requires both a baseline schedule "
                    "and an updated/as-built schedule — provide 'baseline_file' "
                    "and 'updated_file' (XER) for XER-based delay analysis"
                ),
                "analysis_method": analysis_method,
            }
    
        baseline = await self._parse_xer_file(baseline_file)
        updated = await self._parse_xer_file(updated_file)
        if baseline.get("status") == "error":
            return baseline
        if updated.get("status") == "error":
            return updated
    
        if analysis_method == "time_impact":
            results = self._run_time_impact_analysis(baseline, updated, delay_events)
        elif analysis_method == "windows":
            results = self._run_windows_analysis(baseline, updated, delay_events)
        elif analysis_method == "collapsed_as_built":
            results = self._run_collapsed_as_built(baseline, updated, delay_events)
        else:
            results = self._run_impacted_as_planned(baseline, updated, delay_events)
    
        cp_analysis = self._analyze_critical_path_changes(baseline, updated)
        concurrency = self._analyze_concurrency(delay_events)
        apportionment = self._apportion_delay(results["total_delay_days"], delay_events, concurrency)
    
        return {
            "status": "success",
            "action": "forensic_delay_analysis",
            "analysis_method": analysis_method,
            "project_duration": {
                "baseline": baseline.get("project_duration", 0),
                "as_built": updated.get("project_duration", 0),
                "net_delay": results["total_delay_days"]
            },
            "critical_path_analysis": cp_analysis,
            "delay_events": {
                "total_identified": len(delay_events),
                "compensable": len([e for e in delay_events if e.get("compensable", False)]),
                "non_compensable": len([e for e in delay_events if not e.get("compensable", False)]),
                "excusable": len([e for e in delay_events if e.get("excusable", False)]),
                "non_excusable": len([e for e in delay_events if not e.get("excusable", False)])
            },
            "delay_calculation": results,
            "concurrency_analysis": concurrency,
            "apportionment": apportionment,
            "entitlement_summary": {
                "eot_entitled_days": apportionment["contractor_entitlement"],
                "prolongation_costs_entitled": apportionment["compensable_days"] > 0,
                "liquidated_damages_risk": apportionment["contractor_responsible"] > 0
            },
            "expert_report_sections": [
                "Introduction and Instructions", "Summary of Opinions", "Project Overview",
                "Contractual Provisions", "Methodology", "As-Planned vs As-Built",
                "Delay Events Analysis", "Causation", "Entitlement Quantification", "Conclusions"
            ],
            "recommended_claim_value": apportionment["compensable_days"] * 5000 if apportionment["compensable_days"] > 0 else 0
        }
    def _run_collapsed_as_built(self, baseline: Dict, updated: Dict, events: List[Dict]) -> Dict:
        """Collapsed As-Built (CAB) method — real implementation.

        Take the as-built (updated) duration and remove the delay events
        one at a time. The reduction in project duration when an event is
        removed represents that event's net impact. Sums the per-event
        deltas to get total apportioned delay.
        """
        as_built_duration = updated.get("project_duration", 0)
        baseline_duration = baseline.get("project_duration", 0)
        net_delay = max(0, as_built_duration - baseline_duration)

        # Per-event apportionment: weight by event delay_days when the event
        # touched a critical activity; non-critical events contribute zero.
        critical_event_days = sum(
            int(e.get("delay_days", 0))
            for e in events
            if e.get("critical", False)
        )
        impacts = []
        for e in events:
            if not e.get("critical", False):
                continue
            ed = int(e.get("delay_days", 0))
            share = (ed / critical_event_days) if critical_event_days > 0 else 0
            impacts.append({
                "event": e.get("description") or e.get("id", "(unnamed)"),
                "date": e.get("date"),
                "delay_days": ed,
                "share_of_net_delay": round(share, 3),
                "apportioned_days": round(net_delay * share, 2),
            })

        return {
            "method": "Collapsed As-Built",
            "total_delay_days": net_delay,
            "impacted_activities": len(impacts),
            "critical_path_impacts": impacts,
            "as_built_duration": as_built_duration,
            "baseline_duration": baseline_duration,
            "methodology_notes": (
                "Net delay = as-built - baseline durations. Apportioned across "
                "critical events weighted by their delay_days. Excludes events "
                "with critical=False on the assumption their float absorbed them."
            ),
        }
    def _analyze_concurrency(self, events: List[Dict]) -> Dict:
        """Compute real concurrent delay days from event date ranges.

        Two delays are concurrent on a given day if both span that day.
        Concurrent days = number of unique days where 2 or more events
        were active. Uses pure-Python date arithmetic — no NumPy dep so
        this stays runnable in environments that strip optional libs.

        Each event should provide either:
          - `date` (single-day event), OR
          - `start_date` + `end_date` (range), OR
          - `date` + `delay_days` (start = date, end = start + delay_days)
        """
        spans = []
        for e in events:
            start = self._parse_event_date(
                e.get("start_date") or e.get("date")
            )
            end_str = e.get("end_date")
            if end_str:
                end = self._parse_event_date(end_str)
            elif start is not None:
                dd = int(e.get("delay_days", 0) or 0)
                end = start + timedelta(days=max(0, dd - 1))
            else:
                end = None
            if start is None or end is None:
                continue
            spans.append((start, end))

        # Sweep: for each day in [min_start, max_end], count how many spans
        # cover it; concurrent days are those with count >= 2. Bounded by the
        # total project window so the loop is small.
        concurrent_days = 0
        if spans:
            all_start = min(s for s, _ in spans)
            all_end = max(e for _, e in spans)
            cursor = all_start
            while cursor <= all_end:
                active = sum(1 for s, e in spans if s <= cursor <= e)
                if active >= 2:
                    concurrent_days += 1
                cursor += timedelta(days=1)

        compensable_events = [e for e in events if e.get("compensable")]
        non_excusable_events = [e for e in events if not e.get("excusable")]
        return {
            "concurrent_days": concurrent_days,
            "events_analyzed": len(spans),
            "events_skipped_no_date": len(events) - len(spans),
            "compensable_events": len(compensable_events),
            "non_excusable_events": len(non_excusable_events),
            "_note": (
                "Concurrent days = calendar days on which 2+ delay events "
                "are simultaneously active. Day-level granularity; for "
                "hour-level concurrency a different model is required."
            ),
        }
    async def primavera_parse(self, input_data: Any, params: Dict) -> Dict:
        """Delegate to PrimaveraParserBlock: parse .xer schedule files."""
        block = self._resolve_block("primavera_parser")
        if block is None:
            return {"status": "error", "error": "primavera_parser block unavailable"}
        return await block.process(input_data, params)
    def _attach_cpm_to_activities(
        self,
        activities: List[Dict[str, Any]],
        start_date: Optional[str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Optional[str]]:
        """Run CPM over the WBS activities and attach ES/EF/LS/LF/TF to each.

        Returns (activities_with_cpm, summary_dict, error_or_None). On CPM
        failure, returns the activities unchanged and an error string; the
        summary dict will have zeros / empty lists.
        """
        from app.lib.pm_computations import (
            compute_cpm, CircularDependencyError,
        )
        from app.schemas.cpm import (
            Activity as CPMActivity, CPMInput, Dependency,
            ResourceAssignment, WorkCalendar,
        )

        project_start = None
        if start_date:
            try:
                project_start = datetime.fromisoformat(start_date).date()
            except (ValueError, TypeError):
                project_start = None

        cpm_acts: List["CPMActivity"] = []
        for a in activities:
            cpm_acts.append(CPMActivity(
                id=a["id"],
                name=a["name"],
                duration=int(a["duration_days"]),
                predecessors=[
                    Dependency(predecessor_id=p) for p in a["predecessors"]
                ],
                wbs_code=a.get("wbs_phase", ""),
                resources=[
                    ResourceAssignment(trade=t, count=1.0) for t in a["resources"]
                ],
            ))

        try:
            cpm_input = CPMInput(
                activities=cpm_acts,
                project_start=project_start,
                calendar=WorkCalendar(),
            )
            cpm_out = compute_cpm(cpm_input)
        except (CircularDependencyError, ValueError) as exc:
            summary = {
                "total_duration_days": 0,
                "critical_path_activity_ids": [],
                "critical_count": 0,
            }
            return activities, summary, str(exc)

        # Map results back to the activities list.
        by_id = {r.id: r for r in cpm_out.results}
        enriched: List[Dict[str, Any]] = []
        critical_ids: List[str] = []
        for a in activities:
            r = by_id.get(a["id"])
            new = dict(a)
            if r is not None:
                new.update({
                    "early_start_day": r.early_start_day,
                    "early_finish_day": r.early_finish_day,
                    "late_start_day": r.late_start_day,
                    "late_finish_day": r.late_finish_day,
                    "total_float_days": r.total_float,
                    "critical": r.is_critical,
                })
                if r.is_critical:
                    critical_ids.append(r.id)
            enriched.append(new)

        summary = {
            "total_duration_days": cpm_out.project_duration,
            "critical_path_activity_ids": list(cpm_out.critical_path),
            "critical_count": len(cpm_out.critical_path),
        }
        return enriched, summary, None
    # Install-verb keywords: a procurement item links to an activity only when
    # the activity names the equipment AND is an install/erect step — so
    # "Procure Chiller" precedes "Chiller install", not "Chiller pad pour".
    _INSTALL_VERBS = ("install", "position", "erect", "terminat", "mount",
                      "landing", "set out", "placement", "hookup", "hook-up")

    def _inject_long_lead_procurement(
        self,
        activities: List[Dict[str, Any]],
        lead_times: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prepend long-lead equipment procurement activities (extracted from a
        BOD/RFP by document_engine) and link each as a predecessor of the
        matching install activity.

        Long-lead procurement is the dominant driver of a data-center schedule,
        yet the WBS templates carry none — a schedule from a brief alone
        understates equipment-driven float. ``lead_times`` is a list of
        ``{"equipment": str, "lead_time_days": int}``; only REAL (non-default)
        entries should be passed in. Returns the augmented activity list
        (procurement first). Unmatched items remain as standalone long-lead
        markers rather than forcing a wrong dependency.
        """
        if not lead_times:
            return activities
        procurement: List[Dict[str, Any]] = []
        n = 0
        for lt in lead_times:
            equip = str(lt.get("equipment") or "").strip()
            try:
                days = int(lt.get("lead_time_days") or 0)
            except (TypeError, ValueError):
                days = 0
            if not equip or days <= 0:
                continue
            n += 1
            pid = f"P.{n}"
            el = equip.lower()
            matched = False
            for a in activities:
                an = (a.get("name") or "").lower()
                if el in an and any(v in an for v in self._INSTALL_VERBS):
                    preds = a.setdefault("predecessors", [])
                    if pid not in preds:
                        preds.append(pid)
                    matched = True
            procurement.append({
                "id": pid, "code": pid,
                "name": f"Procure {equip} (long-lead)",
                "duration_days": days,
                "predecessors": [],
                "resources": ["procurement"],
                "wbs": pid,
                "wbs_code": pid,
                "wbs_phase": "Procurement / Long-Lead",
                "long_lead": True,
                "linked_to_install": matched,
            })
        return procurement + activities

    def _elect_boq_scope_wbs_items(
        self,
        data: Dict[str, Any],
        params: Dict[str, Any],
        brief: str,
        user_message: str,
    ) -> List[Dict[str, Any]]:
        """Return the measured BOQ rows to derive a WBS from, elected by
        STRUCTURE (ORDERS_28OCT_0700Z #1a): any rows carrying a label
        (description/item_key) + a measure (quantity/unit), of ANY trade. There
        is no demolition/site-clearance keyword gate on the ask and no trade
        filter on the rows — any measured BOQ scope derives a schedule. Empty
        list means no structured scope was handed over; we never invent rows.
        """
        from app.lib.boq_schedule import (
            retrieve_boq_scope_items,
            structured_boq_rows,
        )
        raw = params.get("boq_items") or data.get("boq_items")
        rows = structured_boq_rows(raw if isinstance(raw, list) else [])
        if rows:
            return rows
        # No explicit rows handed over: retrieve measured BOQ scope from the
        # project corpus (keyword-free; returns [] when there is none).
        project_id = params.get("project_id") or data.get("project_id")
        if project_id:
            return retrieve_boq_scope_items(user_message or brief, str(project_id))
        return []

    async def generate_wbs(self, input_data: Any, params: Dict) -> Dict:
        """Generate a CPM-ready Work Breakdown Structure from a project brief.

        Deterministic template-based: no LLM. Templates per project_type
        define a phase / sub-phase / activity scaffold; zone multipliers
        scale the activity count toward ``target_count``. Output activities
        carry CPM ES/EF/LS/LF/total_float + critical flag, computed via
        :func:`app.lib.pm_computations.compute_cpm`.

        Durations are rule-of-thumb working-day defaults. A chat turn like
        ``use 6 days per slab and re-run`` is parsed into
        ``duration_overrides`` and applied before CPM so the recomputed
        schedule matches the user's N, not the template.
        """
        import uuid

        data = input_data if isinstance(input_data, dict) else {}
        p = params or {}

        # No hardwired default input (ORDERS_28OCT_0700Z #1b): an empty/blank
        # brief stays empty and, with no structured scope, is refused as
        # no-scope below (422 over HTTP) rather than silently scheduling a
        # canned project.
        brief: str = (
            data.get("brief")
            or p.get("brief")
            or ""
        )
        user_message = (
            p.get("user_message")
            or data.get("user_message")
            or data.get("message")
            or ""
        )
        from app.core.action_router import message_wants_look_ahead
        if message_wants_look_ahead(user_message) or message_wants_look_ahead(brief):
            return {
                "status": "error",
                "action": "generate_wbs",
                "error": (
                    "This ask is a look-ahead (rolling / N-day window from a "
                    "programme), not a WBS generate. Call look_ahead with a "
                    "Primavera P6 .xer, or report that no schedule file is "
                    "available — do not build a template WBS."
                ),
            }
        # Leftover F1: a demolition / site-clearance + BOQ WBS ask must use
        # retrieved measured rows, not the building template. Election is on
        # the ask — handing generate_wbs a silent ``boq`` key on an office
        # brief still does nothing (see test_passing_a_boq_changes_nothing).
        boq_items = self._elect_boq_scope_wbs_items(data, p, brief, user_message)
        boq_derived = bool(boq_items)

        # Clamp target_count to [20, 1000] BEFORE template scaling.
        # A BOQ-derived high-level WBS keeps the retrieved row count — padding
        # to 20 would invent packages that are not in the bill.
        try:
            target_count = int(p.get("target_count", data.get("target_count", 200)))
        except (TypeError, ValueError):
            target_count = 200
        if not boq_derived:
            target_count = max(20, min(1000, target_count))
        else:
            target_count = max(1, min(1000, target_count))

        stated_type = p.get("project_type") or data.get("project_type")
        project_type = stated_type or self._detect_project_type_from_brief(brief)
        # Whether the caller SAID the type or the brief was sniffed for it is
        # the difference between a fact and a guess, and the answer has to say
        # which one it is (R3). Recorded before the fallback below, because
        # falling back to "building" is itself an inference.
        type_inferred = not stated_type
        # project_type is a LABEL only now — no activity template is selected
        # from it (ORDERS_28OCT_0345Z #1 removed the per-type scaffold). The
        # activity set is derived from real structured input below.

        start_date = p.get("start_date") or data.get("start_date")
        if not start_date:
            stated_today = _stated_look_ahead_date(" ".join(
                str(x) for x in (
                    user_message, brief,
                    p.get("message"), data.get("message"),
                ) if x
            ))
            start_date = stated_today or datetime.now(timezone.utc).date().isoformat()

        structuring_unstated_durations: List[str] = []
        brief_structured = False
        typed_programme = False
        if boq_derived:
            from app.lib.boq_schedule import (
                activities_from_boq_scope_outline,
                wbs_tree_from_boq_items,
            )
            activities = activities_from_boq_scope_outline(boq_items)
            wbs_tree = wbs_tree_from_boq_items(boq_items)
        else:
            # No canned template (the per-type scaffold was the hardwiring).
            # A programme the operator TYPED (named rows with date pairs) is used
            # directly as rows (#3); otherwise the brief / document text is
            # STRUCTURED into activities via the single LLM egress (S-1) —
            # structuring ONLY; durations/sequence/CPM run in the kernel below.
            # Long-lead procurement from a document feed is injected below. No
            # structured input -> no activities -> the computed no-scope error.
            typed_activities = _typed_programme_activities(brief or user_message or "")
            if typed_activities:
                activities = typed_activities
                typed_programme = True
                wbs_tree = {}
            else:
                activities, structuring_unstated_durations = await _structure_activities_from_text(
                    brief or user_message
                )
                brief_structured = bool(activities)
                wbs_tree = {}

        # Long-lead procurement + target milestones extracted from a BOD/RFP
        # (document_engine.downstream.schedule_engine). Injecting the real lead
        # times makes the schedule equipment-driven instead of rule-of-thumb.
        lead_times = p.get("procurement_lead_times") or data.get("procurement_lead_times") or []
        target_milestones = p.get("target_milestones") or data.get("target_milestones") or []
        procurement_injected = 0
        if lead_times:
            before = len(activities)
            activities = self._inject_long_lead_procurement(activities, lead_times)
            procurement_injected = len(activities) - before

        # No structured scope -> no schedule. A work breakdown is derived from
        # a project's measured BOQ scope or a document's long-lead procurement
        # feed, never fabricated from a prose brief (ORDERS_28OCT_0345Z #1).
        # The export routes translate this empty-activities error to HTTP 422.
        if not activities:
            return {
                "status": "error",
                "action": "generate_wbs",
                "error": (
                    "no schedulable scope found: a work breakdown is built from "
                    "a project's measured BOQ scope (any trade), a structured "
                    "brief/document, or source documents' long-lead procurement "
                    "feed. Provide a BOQ, a brief the engine can structure, or "
                    "source documents with equipment lead times."
                ),
                "project_type": project_type,
                "project_type_inferred": type_inferred,
                "activities": [],
            }

        # Brief / typed-programme paths derive their high-level WBS tree from the
        # structured activities themselves — grouped by the trade each activity
        # resolves to through the shared categorizer (the same classifier the BOQ
        # path uses). Input-derived, no template: two briefs give two different
        # trees. The BOQ path already built its tree above.
        if not boq_derived:
            from app.lib.boq_schedule import wbs_tree_from_activities
            wbs_tree = wbs_tree_from_activities(activities)

        # User duration overrides ("use 6 days per slab and re-run") replace
        # template days BEFORE CPM so the recomputed ES/EF/TF match N.
        from app.lib.wbs_duration_overrides import collect_overrides, apply_duration_overrides
        overrides = collect_overrides(
            brief,
            p.get("user_message"),
            data.get("message"),
            data.get("user_message"),
            p.get("brief"),
            explicit=(
                p.get("duration_overrides")
                or data.get("duration_overrides")
            ),
            history=p.get("history") or data.get("history"),
        )
        applied_overrides: list = []
        unmatched_overrides: list = []
        if overrides:
            activities, applied_overrides, unmatched_overrides = apply_duration_overrides(
                activities, overrides
            )

        # 0845Z ruling: an activity whose duration is UNKNOWN (None) is not
        # scheduled and never gets a fabricated 0-day. CPM runs over the
        # activities that carry a real duration; the UNKNOWN ones are carried
        # in the output with no CPM dates and the chain is reported incomplete.
        schedulable = [
            a for a in activities
            if isinstance(a.get("duration_days"), int)
            and not isinstance(a.get("duration_days"), bool)
            and a["duration_days"] > 0
        ]
        unscheduled = [a for a in activities if a not in schedulable]

        # The CPM pass is CPU work (a production LOOP_STALL): never on the
        # event loop.
        from app.core.offload import off_loop

        enriched, summary, cpm_error = await off_loop(
            self._attach_cpm_to_activities, schedulable, start_date
        )
        # Re-attach the UNKNOWN-duration activities (no CPM dates; duration
        # stays None, basis "unknown") so the answer shows them but never
        # pretends they are scheduled.
        for a in unscheduled:
            a.setdefault("duration_basis", "unknown")
        enriched = list(enriched) + unscheduled
        schedule_incomplete = bool(unscheduled)
        unscheduled_names = [a.get("name") for a in unscheduled]
        from app.core.conversation_wbs import stamp_hierarchical_wbs_codes
        enriched = stamp_hierarchical_wbs_codes(enriched, wbs_tree)

        operator_ms = _operator_milestones_from_text(brief)
        # BOQ project_type + trade list are DERIVED from the elected rows'
        # trades (any trade), never a demolition literal. A single-trade bill
        # names that trade; a mixed bill is "mixed".
        boq_trades: List[str] = []
        if boq_derived:
            for a in activities:
                c = (a.get("boq") or {}).get("category")
                if c and c not in boq_trades:
                    boq_trades.append(c)
            if len(boq_trades) == 1:
                project_type = re.sub(r"[^a-z0-9]+", "_", boq_trades[0].lower()).strip("_") or "boq_scope"
            elif boq_trades:
                project_type = "mixed"
            else:
                project_type = "boq_scope"
        result: Dict[str, Any] = {
            "status": "success",
            "wbs_id": f"wbs-{uuid.uuid4().hex[:8]}",
            "project_type": project_type,
            "brief": brief,
            "target_count": target_count,
            "actual_count": len(enriched),
            "start_date": start_date,
            "activities": enriched,
            "schedule_incomplete": schedule_incomplete,
            "unscheduled_activities": unscheduled_names,
            "wbs_tree": wbs_tree,
            "summary": summary,
            "procurement_injected": procurement_injected,
            "target_milestones": target_milestones or [
                {
                    "id": m.get("id"),
                    "name": m.get("name"),
                    "duration_days": m.get("days"),
                }
                for m in (operator_ms.get("milestones") or [])
            ],
            "assumptions": (
                [
                    "Scope packages are the elected BOQ rows ("
                    + (", ".join(boq_trades) if boq_trades else "measured scope")
                    + "); none were invented.",
                    "Durations are derived from each row's quantity via the "
                    "construction kernel — a stated productivity rate where the "
                    "row carries one, otherwise a generic sourced productivity "
                    "norm for the row's trade; a row with neither is left "
                    "UNKNOWN (not defaulted) and the schedule is reported "
                    "incomplete.",
                    "FS-only predecessors; no SS/FF/SF; zero lag.",
                ] if boq_derived else [
                "Activities are the long-lead procurement items extracted from "
                "the source documents' schedule feed; none were invented.",
                "Durations are the extracted lead times; FS-only predecessors, "
                "no SS/FF/SF, zero lag.",
                ]
            ),
        }
        # Self-declaration at the glass (owner's ruling R3, 2026-09-01). A
        # template scaffold that does not say it is one gets read as the
        # project's own structure -- gate battery F1/F2 on 13b2bf7. The
        # scaffold block is the EVIDENCE the answer layer renders its
        # provenance line from; the model does not compose it.
        # Leftover F1: when the tree is the retrieved BOQ rows, the
        # template-scaffold line would be a lie in the other direction.
        if boq_derived:
            citations = []
            for item in boq_items or []:
                code = str((item or {}).get("item_key") or "").strip()
                # Item codes only — source filenames can carry another
                # contract year's id, which must not be cited as governing.
                if code and code not in citations:
                    citations.append(code)
            trade_phrase = ", ".join(boq_trades) if boq_trades else "measured scope"
            result["scaffold"] = {
                "source": "boq",
                "project_type": project_type,
                "project_type_inferred": False,
                "derived_from_boq": True,
                "zone_labels": "none",
                "trades": list(boq_trades),
                "citations": citations,
                "declaration": (
                    "High-level WBS derived from this project's BOQ ("
                    + trade_phrase
                    + "). Only elected measured rows are shown; no template "
                    "packages were added. Durations are computed by the "
                    "construction kernel from each row's quantity (a stated "
                    "rate or a generic sourced productivity norm); rows with "
                    "neither are left UNKNOWN."
                    + (f" Items: {', '.join(citations)}." if citations else "")
                ),
            }
        elif typed_programme:
            # The operator typed the programme (named rows + date pairs). The
            # rows are theirs; durations come from the typed dates and the
            # sequence/CPM from the kernel. No structuring LLM call was made.
            result["scaffold"] = {
                "source": "typed_programme",
                "project_type": project_type,
                "project_type_inferred": type_inferred,
                "derived_from_boq": False,
                "zone_labels": "none",
                "kernel": "pm_computations.compute_cpm",
                "declaration": (
                    "Activities are the programme the operator typed (named rows "
                    "with start/finish dates); durations are those typed dates and "
                    "the sequence/CPM are computed by the construction CPM kernel "
                    "(pm_computations). No template, per-type scaffold or "
                    "placeholder activities were added."
                ),
            }
        elif brief_structured:
            # Derivation self-declaration (66f3bca ruling): the output states
            # WHERE its activities came from and which kernel computed the
            # schedule. Here the activities were STRUCTURED from the brief /
            # document text by the single LLM egress (structuring only); the
            # schedule (durations/sequence/CPM) is the deterministic construction
            # kernel, pm_computations. Every activity traces to a verbatim span.
            result["scaffold"] = {
                "source": "document" if (p.get("document_ids") or data.get("document_ids")) else "brief",
                "project_type": project_type,
                "project_type_inferred": type_inferred,
                "derived_from_boq": False,
                "zone_labels": "none",
                "kernel": "pm_computations.compute_cpm",
                "unstated_durations": list(structuring_unstated_durations),
                "declaration": (
                    "Activities were structured from the input text; durations, "
                    "sequence and CPM are computed by the construction CPM kernel "
                    "(pm_computations). No template, per-type scaffold or "
                    "placeholder zones were added; every activity traces to a "
                    "verbatim span of the input."
                    + (
                        f" Durations not stated in the input (kept blank, not "
                        f"defaulted): {', '.join(structuring_unstated_durations)}."
                        if structuring_unstated_durations else ""
                    )
                ),
            }
        else:
            # Non-BOQ, non-structured success means the activities are the
            # document feed's long-lead procurement items (the remaining real
            # input); there is no template scaffold any more.
            result["scaffold"] = {
                "source": "document_procurement",
                "project_type": project_type,
                "project_type_inferred": type_inferred,
                "derived_from_boq": False,
                "zone_labels": "none",
                "declaration": (
                    "Schedule derived from the source documents' long-lead "
                    "procurement feed (equipment lead times). No template "
                    "packages, per-type scaffold or placeholder zones were "
                    "added; every activity traces to an extracted lead time."
                ),
            }
        if operator_ms:
            result["operator_milestones"] = operator_ms
            result["assumptions"].append(
                "Operator-stated milestone durations (calendar days) are "
                "shown as separate branches and were not invented."
            )
        if applied_overrides:
            result["duration_overrides_applied"] = applied_overrides
            bits = [
                f"{o['match']} = {o['days']} working days "
                f"({o['activities_updated']} activities)"
                for o in applied_overrides
            ]
            result["assumptions"].append(
                "User duration override (CPM recomputed): " + "; ".join(bits) + "."
            )
        if unmatched_overrides:
            result["duration_overrides_unmatched"] = unmatched_overrides
        if procurement_injected:
            result["assumptions"].append(
                f"{procurement_injected} long-lead procurement activities injected from "
                "extracted equipment lead times; each links as a predecessor to its "
                "matching install activity (unmatched items kept as standalone markers)."
            )
        if cpm_error:
            result["cpm_error"] = cpm_error
        return result
