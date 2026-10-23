"""Clash stays off unless the user actually asks for it.

A leftover project-assistant turn that said \"Do not run clash detection\"
was routed to ``bim_clash_detection`` because the matcher only looked for
the word ``clash``. Negated phrasing is not a request.

Also treats ``interference`` / ``mep conflict`` as clash asks so those
ACTION_PATTERNS keywords are not silently stripped by the orchestrator
filter (Fable W3).
"""
from __future__ import annotations

import re

# Positive ask: clash, interference, or MEP conflict.
# Bare "conflict" is too broad (schedule conflict) — use "mep conflict".
_CLASH_ASK = re.compile(
    r"\b(?:clash(?:es|ing)?|interference|mep\s+conflict)\b",
    re.IGNORECASE,
)

# Opt-out of *running* clash — not "do not include clashes below 5mm".
# Bare no/without/never with a wide gap over-suppressed real asks (Fable W1).
_CLASH_OPT_OUT = re.compile(
    r"(?:"
    r"\b(?:do\s+not|don't|dont)\s+(?:run|perform|enable|turn\s+on)\s+clash"
    r"|\b(?:skip|avoid|disable|exclude|omit)\s+(?:running\s+)?clash"
    r"|\bturn\s+off\s+clash"
    r"|\bclash(?:\s+\w+){0,6}\s+(?:is\s+)?not\s+(?:needed|required|necessary)"
    r")",
    re.IGNORECASE,
)


def message_wants_clash(text: str) -> bool:
    """True only when the message asks to run clash / interference detection."""
    raw = text or ""
    if not _CLASH_ASK.search(raw):
        return False
    if _CLASH_OPT_OUT.search(raw):
        return False
    return True
