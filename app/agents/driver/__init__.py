"""Driver mode (F-DRIVER Phase B), behind ``DRIVER_MODE`` -- off by default.

In driver mode the model drives the turn. It is given the project's profile,
the conversation and the hats' descriptions; it chooses a hat itself
(``select_hat``); it calls tools with ``tool_choice`` auto. No code chooses a
route or forces a tool, and nothing is injected automatically -- knowledge
is reached through tools. Steps and model are configuration.
"""
from __future__ import annotations

import contextvars
import os
from typing import Any, Mapping

_TRUTHY = ("1", "true", "yes", "on")
#: The header a request carries to ask for driver mode when DRIVER_MODE=request.
REQUEST_HEADER = "x-fork-driver"
_REQUESTED: "contextvars.ContextVar[bool]" = contextvars.ContextVar("driver_requested", default=False)


def mode() -> str:
    """``off`` (default), ``on`` (every turn) or ``request`` (only turns that
    ask with the ``X-Fork-Driver: on`` header -- driver mode measured on live
    without changing anyone else's turn)."""
    raw = (os.getenv("DRIVER_MODE") or "").strip().lower()
    if raw in _TRUTHY:
        return "on"
    return "request" if raw == "request" else "off"


def mark_request(headers: Mapping[str, Any], user_switched_on: bool = False) -> None:
    """Call at a turn's arrival: remember whether this turn takes driver mode
    -- the request asked for it, or an admin switched it on for the user.
    Honoured only when DRIVER_MODE=request."""
    asked = str(headers.get(REQUEST_HEADER) or "").strip().lower() in _TRUTHY
    _REQUESTED.set(mode() == "request" and (asked or bool(user_switched_on)))


def enabled() -> bool:
    m = mode()
    return m == "on" or (m == "request" and _REQUESTED.get())


def max_steps() -> int:
    try:
        return max(1, int(os.getenv("DRIVER_MAX_STEPS") or "6"))
    except ValueError:
        return 6


def model() -> str:
    """Model for driver turns; empty means the provider's default."""
    return (os.getenv("DRIVER_MODEL") or "").strip()
