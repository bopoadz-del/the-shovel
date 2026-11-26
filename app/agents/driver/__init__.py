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
from typing import Any, Mapping, Optional

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
    """Driver mode for this turn: on for every turn, asked for by the
    request, or the turn's project is on the catalogue path."""
    m = mode()
    return m == "on" or (m == "request" and _REQUESTED.get()) or _ON_SHEET.get() is True


def max_steps() -> int:
    try:
        return max(1, int(os.getenv("DRIVER_MAX_STEPS") or "6"))
    except ValueError:
        return 6


def model() -> str:
    """Model for driver turns; empty means the provider's default."""
    return (os.getenv("DRIVER_MODEL") or "").strip()


#: The project-profile setting that puts a project's turns on the catalogue
#: path: the model leads, with the working sheet, and every turn writes a
#: trace record. Read from the project's own profile; no project is named.
ANSWER_PATH_KEY = "answer_path"
CATALOGUE = "catalogue"


def project_setting(project_id: Any, user_id: Any, key: str) -> str:
    """One setting of the project's own profile (lower case), or ``""`` when
    the project has none, is not found or cannot be read."""
    if not project_id:
        return ""
    try:
        from app.core import projects

        project = projects.get_project_accessible(str(project_id), user_id)
    except Exception:  # noqa: BLE001 -- an unreadable profile keeps the default
        return ""
    profile = (project or {}).get("profile")
    if not isinstance(profile, dict):
        return ""
    return str(profile.get(key) or "").strip().lower()


def answer_path(project_id: Any, user_id: Any = None) -> str:
    """The project's ``answer_path`` setting, or ``""`` (the default path)."""
    return project_setting(project_id, user_id, ANSWER_PATH_KEY)


def catalogue_path(project_id: Any, user_id: Any = None) -> bool:
    """True when the project's profile sets ``answer_path: "catalogue"``."""
    return answer_path(project_id, user_id) == CATALOGUE


#: Whether this turn's project is on the catalogue path: None until a door
#: marks the turn (``mark_project``), then True or False.
_ON_SHEET: "contextvars.ContextVar[Optional[bool]]" = contextvars.ContextVar("driver_on_sheet", default=None)


def mark_project(on: Any) -> bool:
    """Call at a turn's arrival with ``catalogue_path(project_id, user_id)``
    (read in a thread: it reads the database). Set here, in the turn's own
    context -- a mark set inside the thread would not reach the turn. Every
    later check of this turn (``enabled``, ``on_sheet``) reads the mark."""
    _ON_SHEET.set(bool(on))
    return bool(on)


def on_sheet() -> Optional[bool]:
    """The turn's mark: True / False once marked, None when no door marked it."""
    return _ON_SHEET.get()


def sheet_max_rounds() -> int:
    """Rounds a catalogue-path turn may take (``DRIVER_SHEET_MAX_ROUNDS``,
    default 12): the ceiling that ends a turn whose steps are still open."""
    try:
        return max(1, int(os.getenv("DRIVER_SHEET_MAX_ROUNDS") or "12"))
    except ValueError:
        return 12


def sheet_max_seconds() -> float:
    """Seconds a catalogue-path turn may spend before its ceiling
    (``DRIVER_SHEET_MAX_SECONDS``, default 150)."""
    try:
        return max(1.0, float(os.getenv("DRIVER_SHEET_MAX_SECONDS") or "150"))
    except ValueError:
        return 150.0
