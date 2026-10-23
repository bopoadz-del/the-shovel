"""Split schedule-delay advice so a slip does not become a claim.

R004 and keyword routing used to fire EOT + prolongation + CREATE_CLAIM on
any schedule delay ("the chiller delivery has slipped four weeks"). That is
commercially dangerous: a reported slip is not entitlement. Classify the
message first. Stay silent or ask when ownership of the risk is unclear.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import FrozenSet, List, Optional

from app.core.dependency_graph import SuggestedAction


class DelayKind(str, Enum):
    """What a delay turn is allowed to recommend."""

    NONE = "none"
    ASK = "ask"
    EOT_ONLY = "eot_only"
    COST_ONLY = "cost_only"
    CONCURRENT = "concurrent"
    CULPABLE = "culpable"
    CLAIM = "claim"


_DELAY_SIGNAL = re.compile(
    r"\b(?:"
    r"delay(?:s|ed|ing)?"
    r"|slip(?:s|ped|ping)?"
    r"|slippage"
    r"|behind\s+(?:programme|program|schedule)"
    r"|pushed\s+out"
    r"|late(?:r)?"
    r"|overdue"
    r"|eot"
    r"|extension\s+of\s+time"
    r"|prolongation"
    r"|claim(?:s|ing|ed)?"
    r")\b",
    re.IGNORECASE,
)

_CONCURRENT = re.compile(
    r"\b(?:concurrent\s+delays?|delay\s+concurrency|concurrency\s+of\s+delay)\b",
    re.IGNORECASE,
)

_CULPABLE = re.compile(
    r"\b(?:"
    r"culpable\s+delay"
    r"|contractor(?:['’]s)?\s+(?:delay|risk|fault|default)"
    r"|our\s+(?:own\s+)?delay"
    r"|we\s+(?:are|were)\s+late"
    r"|contractor-caused"
    r"|contractor\s+caused"
    r")\b",
    re.IGNORECASE,
)

_EXPLICIT_CLAIM = re.compile(
    r"\b(?:"
    r"delay\s+claim"
    r"|eot\s+claim"
    r"|claim\s+notice"
    r"|(?:build|file|prepare|draft|raise|submit)\s+(?:a\s+|an\s+)?(?:delay\s+)?claim"
    r"|claim\s+for\s+delay"
    r"|schedule\s+delay\s+claim"
    r"|disruption\s+claim"
    r"|claiming\s+\d+\s+days"
    r"|claiming\s+days"
    r"|loss\s+and\s+expense"
    r")\b",
    re.IGNORECASE,
)

_EOT_ONLY = re.compile(
    r"\b(?:"
    r"extension\s+of\s+time"
    r"|eot\s+entitlement"
    r"|time\s+(?:entitlement|only|extension)"
    r"|eot\b(?!\s+claim)"
    r")\b",
    re.IGNORECASE,
)

_COST_ONLY = re.compile(
    r"\b(?:"
    r"prolongation(?:\s+cost(?:s)?)?"
    r"|preliminar(?:y|ies)\s+cost"
    r"|disruption\s+cost"
    r"|standing\s+time"
    r"|loss\s+and\s+expense"
    r")\b",
    re.IGNORECASE,
)

# A reported slip with no entitlement language. Must not become a claim.
_SLIP_REPORT = re.compile(
    r"\b(?:"
    r"(?:has\s+)?slipped"
    r"|delivery\s+(?:has\s+)?slipped"
    r"|behind\s+(?:programme|program|schedule)"
    r"|pushed\s+out"
    r"|schedule\s+slip(?:page)?"
    r")\b",
    re.IGNORECASE,
)

_CLAIM_ACTIONS: FrozenSet[SuggestedAction] = frozenset(
    {
        SuggestedAction.CREATE_CLAIM,
        SuggestedAction.CALCULATE_PROLONGATION_COST,
        SuggestedAction.CHECK_EOT_ENTITLEMENT,
    }
)


def classify_delay(text: str) -> DelayKind:
    """Return the only delay advice this turn is allowed to give."""
    raw = text or ""
    if not _DELAY_SIGNAL.search(raw):
        return DelayKind.NONE
    # S14: "notice of an EOT claim" as a Contract Data days/period
    # question is not a claim-build turn. claims_builder_permitted
    # must stay False so the keyword "eot claim" cannot steal RAG.
    from app.core.contract_lookup_intent import message_is_contract_data_lookup
    if message_is_contract_data_lookup(raw):
        return DelayKind.NONE
    if _CONCURRENT.search(raw):
        return DelayKind.CONCURRENT
    if _CULPABLE.search(raw):
        return DelayKind.CULPABLE
    if _EXPLICIT_CLAIM.search(raw):
        return DelayKind.CLAIM
    has_eot = bool(_EOT_ONLY.search(raw))
    has_cost = bool(_COST_ONLY.search(raw))
    if has_eot and has_cost:
        # Time + cost named, but not an explicit claim request — ask.
        return DelayKind.ASK
    if has_eot:
        return DelayKind.EOT_ONLY
    if has_cost:
        return DelayKind.COST_ONLY
    return DelayKind.ASK


def claims_builder_permitted(text: str) -> bool:
    """True only when the user explicitly asked to build a claim."""
    return classify_delay(text) == DelayKind.CLAIM


def delay_advice_text(kind: DelayKind) -> str:
    """Assertive inject text. Empty means omit claim language entirely."""
    if kind == DelayKind.CLAIM:
        return (
            "User asked for a delay claim — run entitlement analysis, then "
            "draft only from supplied delay events. Do not invent a claim number."
        )
    if kind == DelayKind.EOT_ONLY:
        return (
            "Time-risk delay — assess EOT entitlement only. Do not package "
            "cost or recommend a claim unless the user asked."
        )
    if kind == DelayKind.COST_ONLY:
        return (
            "Cost-risk delay — assess prolongation / preliminaries only. "
            "Do not advise an EOT claim unless the user asked."
        )
    if kind == DelayKind.CONCURRENT:
        return (
            "Concurrent delay language — do not recommend a claim. Ask for "
            "the concurrent-event analysis first."
        )
    if kind == DelayKind.CULPABLE:
        return (
            "Culpable / contractor-risk delay — do not recommend a claim."
        )
    if kind == DelayKind.ASK:
        return (
            "Schedule delay noted — do not assume EOT or cost entitlement. "
            "Ask whether the event is employer-risk, contractor-risk, or "
            "concurrent before advising a claim."
        )
    return ""


def suggested_actions_for_delay(kind: DelayKind) -> List[SuggestedAction]:
    """Actions allowed for this delay kind. Never CREATE_CLAIM on ASK."""
    if kind == DelayKind.CLAIM:
        return [
            SuggestedAction.CHECK_EOT_ENTITLEMENT,
            SuggestedAction.CALCULATE_PROLONGATION_COST,
            SuggestedAction.CREATE_CLAIM,
        ]
    if kind == DelayKind.EOT_ONLY:
        return [SuggestedAction.CHECK_EOT_ENTITLEMENT, SuggestedAction.UPDATE_SCHEDULE]
    if kind == DelayKind.COST_ONLY:
        return [
            SuggestedAction.CALCULATE_PROLONGATION_COST,
            SuggestedAction.UPDATE_BUDGET,
        ]
    if kind in {DelayKind.CONCURRENT, DelayKind.CULPABLE, DelayKind.ASK}:
        return [SuggestedAction.UPDATE_SCHEDULE, SuggestedAction.UPDATE_RISK_REGISTER]
    return []


def filter_claim_actions(
    actions: List[SuggestedAction],
    text: Optional[str] = None,
    kind: Optional[DelayKind] = None,
) -> List[SuggestedAction]:
    """Strip claim/EOT/prolongation actions unless the kind allows them."""
    resolved = kind if kind is not None else classify_delay(text or "")
    allowed = set(suggested_actions_for_delay(resolved))
    if resolved == DelayKind.CLAIM:
        return list(actions)
    if resolved == DelayKind.EOT_ONLY:
        banned = {SuggestedAction.CREATE_CLAIM, SuggestedAction.CALCULATE_PROLONGATION_COST}
        return [a for a in actions if a not in banned]
    if resolved == DelayKind.COST_ONLY:
        banned = {SuggestedAction.CREATE_CLAIM, SuggestedAction.CHECK_EOT_ENTITLEMENT}
        return [a for a in actions if a not in banned]
    return [a for a in actions if a not in _CLAIM_ACTIONS]


def rewrite_r004_description(text: str, original: str) -> str:
    """Replace R004's default claim-check sentence with kind-specific advice."""
    kind = classify_delay(text)
    if kind == DelayKind.NONE:
        return original
    advised = delay_advice_text(kind)
    return advised or original
