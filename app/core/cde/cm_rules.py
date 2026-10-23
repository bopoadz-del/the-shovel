"""CM overlay on CDE events — not on chat utterances.

When a clash / RFI / delay-related CDE row arrives, run the existing CM
overlay (``classify_delay``, ``CrossDomainReasoner.analyze_turn``,
``message_wants_clash``). Do not duplicate those classifiers.

``inject()`` cites live CDE ids (and the CDE-allocated reference when the
row has one) or stays silent. It never invents an RFI or claim number.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from app.core.cde.events import CdeEvent
from app.core.clash_intent import message_wants_clash
from app.core.delay_advice import DelayKind, classify_delay, delay_advice_text


class CdeEventKind(str, Enum):
    NONE = "none"
    CLASH = "clash"
    RFI = "rfi"
    DELAY = "delay"


def _blob(event: CdeEvent) -> str:
    return (
        f"{event.mail_type} {event.correspondence_type} "
        f"{event.doc_type} {event.reference}"
    ).lower()


def _is_rfi_row(event: CdeEvent) -> bool:
    blob = _blob(event)
    return "rfi" in blob or "request for information" in blob


def classify_cde_event(event: CdeEvent) -> CdeEventKind:
    """Clash / RFI / delay — otherwise the overlay stays silent.

    RFI is a CDE mail type, not a chat keyword. Clash and delay reuse the
    chat-path helpers on the event text so the two paths cannot drift.
    """
    text = event.text()
    if _is_rfi_row(event):
        return CdeEventKind.RFI
    if message_wants_clash(text) or "clash" in (event.doc_type or "").lower():
        return CdeEventKind.CLASH
    if classify_delay(text) != DelayKind.NONE:
        return CdeEventKind.DELAY
    return CdeEventKind.NONE


def relevant_events(events: Iterable[CdeEvent]) -> list[CdeEvent]:
    return [e for e in events if e.id and classify_cde_event(e) != CdeEventKind.NONE]


def run_cm_overlay(event: CdeEvent) -> Optional[dict[str, Any]]:
    """Run the existing CM overlay on one live CDE row. None when silent."""
    if not event.id:
        return None
    kind = classify_cde_event(event)
    if kind == CdeEventKind.NONE:
        return None
    text = event.text()
    from app.core.cross_domain_reasoner import CrossDomainReasoner

    analysis = CrossDomainReasoner().analyze_turn(text)
    delay_kind = classify_delay(text)
    cited: dict[str, str] = {"cde_id": event.id}
    if event.reference.strip():
        cited["reference"] = event.reference.strip()
    return {
        "kind": kind.value,
        "source": event.source,
        "cde_id": event.id,
        "reference": event.reference.strip(),
        "subject": (event.subject or event.title).strip(),
        "delay_kind": delay_kind.value,
        "delay_advice": delay_advice_text(delay_kind),
        "matched_template": analysis.get("matched_template"),
        "cross_domain_context": analysis.get("cross_domain_context") or "",
        "suggested_tools": analysis.get("suggested_tools") or [],
        "cited": cited,
        "source_of_truth": "cde",
    }


def inject(events: Sequence[CdeEvent]) -> str:
    """CM prompt fragment grounded on live CDE rows, or empty.

    Silent when there is nothing to cite. Never fabricates an RFI / claim
    number — a row without a CDE id is skipped; a row without a CDE
    reference is cited by ``cde_id`` only.
    """
    overlays: list[dict[str, Any]] = []
    for event in events:
        overlay = run_cm_overlay(event)
        if overlay:
            overlays.append(overlay)
    if not overlays:
        return ""

    lines = [
        "[CDE-grounded CM]",
        "Cite only these live CDE rows. Do not invent RFI or claim numbers. "
        "Aconex remains the system of record.",
    ]
    for overlay in overlays:
        cited = overlay["cited"]
        label = f"{overlay['kind']} cde_id={cited['cde_id']}"
        if cited.get("reference"):
            label += f" reference={cited['reference']}"
        subject = overlay.get("subject") or ""
        lines.append(f"- {label}" + (f": {subject}" if subject else ""))
        advice = (overlay.get("delay_advice") or "").strip()
        if advice:
            lines.append(f"  {advice}")
        ctx = (overlay.get("cross_domain_context") or "").strip()
        if ctx:
            # Cap so a long linkage list cannot drown the system prompt.
            lines.append(f"  {ctx[:400]}")
    return "\n".join(lines)
