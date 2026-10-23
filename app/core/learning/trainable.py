"""A routing pattern is trainable only when an independent source labeled it.

Own dispatch (the orchestrator recording what it just did) is not a label.
Battery-grade marks and ``/v1/feedback/route`` corrections are.
"""

from __future__ import annotations

from typing import Any, Mapping

#: Sources that may mark a routing_decisions row trainable.
INDEPENDENT_LABEL_SOURCES: frozenset[str] = frozenset({
    "battery_grade",
    "feedback_route",
    "user_correction",
    "feedback",
})

#: Sources that are the router's own writeback. Never trainable.
OWN_DISPATCH_SOURCES: frozenset[str] = frozenset({
    "smart_orchestrator",
    "learned",
    "keyword_fallback",
    "keyword",
    "auto",
    "runtime",
})


def is_independently_labeled(
    pattern_source: str | None = None,
    observation: Mapping[str, Any] | None = None,
) -> bool:
    """True only when an independent source supplied the label.

    ``corrected=True`` on an own-dispatch row is not enough — that flag
    can be set by the same process that routed. The source token must be
    one of ``INDEPENDENT_LABEL_SOURCES``.
    """
    sources = {(pattern_source or "").strip().lower()}
    if observation:
        sources.add(str(observation.get("source") or "").strip().lower())
    sources.discard("")
    if sources & OWN_DISPATCH_SOURCES and not (sources & INDEPENDENT_LABEL_SOURCES):
        return False
    return bool(sources & INDEPENDENT_LABEL_SOURCES)
