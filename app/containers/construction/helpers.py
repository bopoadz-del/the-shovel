"""Shared helpers for the construction container package."""

import re
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert anything to a float without crashing on bad input.

    The audit flagged several _safe_float(p.get(...)) sites that crash with
    ValueError when the user passes a string like "10%" or "1,200" or None.
    This swallows those cases and returns `default` instead.
    """
    if value is None or value == "":
        return default
    try:
        if isinstance(value, str):
            value = value.replace(",", "").replace("%", "").strip()
            if not value:
                return default
        return float(value)
    except (ValueError, TypeError):
        return default


def _parse_money_str(value: Any) -> Optional[float]:
    """Parse a money string handling US (1,234,567.89) and EU (1.234.567,89)
    thousands/decimal conventions. Returns None if unparseable.

    Heuristics for ambiguous cases (single separator, no other):
    - followed by exactly 3 digits → thousands separator ("500.000" → 500000)
    - followed by 1-2 or 4+ digits → decimal separator ("1.5", "1.2345")
    Multiple occurrences of one separator → all thousands ("1.234.567" → 1234567).
    """
    if value is None:
        return None
    s = re.sub(r"[^0-9,.\-]", "", str(value))
    if not s or s in ("-", ".", ","):
        return None

    has_comma = "," in s
    has_period = "." in s

    if has_comma and has_period:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif has_comma:
        if s.count(",") > 1:
            s = s.replace(",", "")
        else:
            after = s.split(",")[-1]
            s = s.replace(",", "") if len(after) == 3 else s.replace(",", ".")
    elif has_period:
        if s.count(".") > 1:
            s = s.replace(".", "")
        else:
            after = s.split(".")[-1]
            if len(after) == 3:
                s = s.replace(".", "")

    try:
        return float(s)
    except ValueError:
        return None


def _safe_iso_date(value: Any) -> Optional[datetime]:
    """Parse a date string tolerating common AEC formats.

    Handles: ISO 8601 (with/without Z), `YYYY-MM-DD`, `DD/MM/YYYY`,
    `MM/DD/YYYY`, Primavera's date strings. Returns None on failure
    instead of crashing — the audit flagged datetime.fromisoformat()
    blowing up on legitimate non-ISO inputs.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    # ISO 8601 with optional trailing Z
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        logger.warning(
            "swallowed %s in _safe_iso_date() — continuing",
            "ValueError", exc_info=True,
        )
    # Common alternates
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
