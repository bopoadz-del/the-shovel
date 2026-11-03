"""Contract Data Q&A must stay on RAG — not generate_wbs / schedule.

Live: "What is the Time for Completion for the whole of the Works?"
and "Milestone 5 Time for Completion" were stolen to the predefined
schedule workflow and answered "Schedule built: N activities…". Those
are Contract Data lookups. Clash stays off unless the user types clash;
this is the same class of steal-guard for TfC / milestones / delay
damages / DNP / performance bond / Engineer / Aconex.
"""
from __future__ import annotations

import re

# Explicit "build me a programme" — never a document lookup.
_WBS_GENERATE_RE = re.compile(
    r"\b(?:generate|create|build|produce|draft|prepare|make|develop|"
    r"give\s+me|export)\b.{0,60}\b"
    r"(?:wbs|work\s+breakdown|l[1-4]\s+schedule|level\s+[1-4]\s+schedule|"
    r"construction\s+schedule|project\s+schedule|activity\s+list|gantt|"
    r"programme|program\s+of\s+works)\b",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_WBS_RE = re.compile(
    r"\b(?:l[1-4]\s+schedule|level\s+[1-4]\s+schedule|\d{2,4}\s+activit|"
    r"work\s+breakdown)\b",
    re.IGNORECASE,
)
# P6 / XER parse is a real schedule tool, not Contract Data.
_P6_FILE_RE = re.compile(
    r"\b(?:xer|\.xer|\bp6\b|primavera|baseline\s+programme|"
    r"baseline\s+program)\b",
    re.IGNORECASE,
)

# Drafting a claim / EOT / delay-analysis is work, not a Contract Data lookup.
# Must sit after the P6-file check and before lookup cues so "delay damages"
# inside a claim-prep ask does not steal the turn onto RAG.
_CLAIM_WORK_RE = re.compile(
    r"\b(?:draft|prepare|build|write|raise|submit|assemble|compile|create|"
    r"run|perform|do|start)\b.{0,60}\b"
    r"(?:claims?|eot|extension\s+of\s+time|delay\s+analysis|"
    r"window\s+analysis|time\s+impact\s+analysis|notice\s+of\s+claim)\b",
    re.IGNORECASE | re.DOTALL,
)

# FIDIC / Contract Data fact lookups. Bare "milestone" is too broad
# (P6 "extract the key milestones from the programme").
_LOOKUP_CUES = (
    re.compile(r"\btime\s+for\s+completion\b", re.IGNORECASE),
    re.compile(r"\b(?:delay|liquidated)\s+damages\b", re.IGNORECASE),
    re.compile(r"\bdefects\s+notification\s+period\b", re.IGNORECASE),
    re.compile(r"\bdnp\b", re.IGNORECASE),
    re.compile(r"\bperformance\s+bond\b", re.IGNORECASE),
    re.compile(r"\baccepted\s+contract\s+amount\b", re.IGNORECASE),
    re.compile(
        r"\b(?:who\s+is\s+(?:the\s+)?engineer|engineer\s+under\s+"
        r"(?:this\s+|the\s+)?contract)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\baconex\b", re.IGNORECASE),
    re.compile(
        r"\b(?:approved\s+)?method\s+of\s+(?:electronic\s+)?communication\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhow\s+many\s+milestones\b", re.IGNORECASE),
    re.compile(
        r"\bmilestone\s+\d+\b.{0,40}\btime\s+for\s+completion\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\btime\s+for\s+completion\b.{0,40}\bmilestone\s+\d+\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"\bparent\s+company\s+guarantee\b", re.IGNORECASE),
    re.compile(
        r"\b(?:commencement\s+date|"
        r"when\s+does\s+(?:the\s+|this\s+)?contract\s+commence)\b",
        re.IGNORECASE,
    ),
)

# Numbered contract-volume Schedules ("Schedule 10: Not Used"), not a P6
# programme. Live: "What does Schedule 10 of the
# contract contain?" missed every cue below, so the keyword router stole
# the turn to parse_primavera_schedule / generate_wbs at conf 0.2.
_CONTRACT_SCHEDULE_N_RE = re.compile(
    r"\b(?:what\s+(?:does|is)|which)\s+schedule\s+\d+\b"
    r"|\bschedule\s+\d+\s+of\s+the\s+(?:contract|agreement|tender|volume|conditions?)"
    r"|\bschedule\s+\d+\b.{0,80}\b(?:contain|cover|include|used)\b",
    re.IGNORECASE | re.DOTALL,
)

# S14 / F-BAT-B fresh_eot_notice_period: "Within how many days must the
# Contractor give notice of an EOT claim on this project?" is Contract
# Data (the project's notice period), not claims_builder. The words
# "EOT" / "claim" / "notice" used to shape-match the claim-build path,
# which then dead-ended on missing delay_events. Claim-build verbs
# (draft / prepare / build / …) stay excluded via _CLAIM_WORK_RE.
_EOT_NOTICE_PERIOD_RE = re.compile(
    r"(?:"
    r"(?:how\s+many\s+days|within\s+(?:how\s+many\s+)?days?).{0,120}"
    r"(?:give\s+)?notice"
    r"|"
    r"(?:eot|extension\s+of\s+time|claim).{0,48}notice\s+period"
    r"|"
    r"notice\s+period.{0,48}(?:eot|extension\s+of\s+time|claim|clause\s+20)"
    r"|"
    r"(?:time\s*[- ]?bar|timebar).{0,48}(?:eot|extension\s+of\s+time|claim|notice)"
    r"|"
    r"(?:eot|extension\s+of\s+time|claim).{0,48}(?:time\s*[- ]?bar|timebar)"
    r"|"
    r"(?:give\s+)?notice\s+of\s+(?:an?\s+)?(?:eot|extension\s+of\s+time)\s+claim"
    r")",
    re.IGNORECASE | re.DOTALL,
)

# Schedule/WBS generative actions the keyword router must not dispatch
# for a Contract Data lookup.
CONTRACT_LOOKUP_BLOCKED_ACTIONS = frozenset({
    "generate_wbs",
    "parse_primavera_schedule",
    "resource_histogram",
    "claims_builder",
    "forensic_delay_analysis",
    "progress_tracker",
    "drawing_qto",
})


def message_is_eot_notice_period_lookup(text: str) -> bool:
    """True for contractual EOT / claim notice-period Q&A (S14).

    Positive: days to give notice of an EOT claim; EOT notice period;
    claim time-bar. Negative: draft / prepare / build an EOT claim
    (those stay on claims_builder and still need delay events).
    """
    raw = text or ""
    if not raw.strip():
        return False
    if _CLAIM_WORK_RE.search(raw):
        return False
    return bool(_EOT_NOTICE_PERIOD_RE.search(raw))


def message_is_contract_data_lookup(text: str) -> bool:
    """True when the turn is a Contract Data fact lookup, not a WBS generate.

    Positive: Time for Completion / Milestone N TfC / delay damages / DNP /
    performance bond / Engineer / Aconex / Accepted Contract Amount /
    numbered contract Schedule N contents / Parent Company Guarantee /
    contract Commencement Date / EOT claim notice period.

    Negative: "create an L2 schedule", "generate a WBS", "extract milestones
    from the XER / programme", "prepare an EOT claim" (claim-build).
    """
    raw = text or ""
    if not raw.strip():
        return False
    if _WBS_GENERATE_RE.search(raw) or _EXPLICIT_WBS_RE.search(raw):
        return False
    if _P6_FILE_RE.search(raw):
        return False
    if _CLAIM_WORK_RE.search(raw):
        return False
    if _CONTRACT_SCHEDULE_N_RE.search(raw):
        return True
    if message_is_eot_notice_period_lookup(raw):
        return True
    return any(cue.search(raw) for cue in _LOOKUP_CUES)
