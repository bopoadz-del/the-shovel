"""Planning / CPM + PE assist calculators (additive library, gap-fill).

Critical-path float plus the Planning Engineer formula-sheet arithmetic:
progress/quantity, productivity → manhours → manpower → duration, unit
conversions, and material/concrete-mix helpers that refuse invented rates.
"""
from __future__ import annotations

from app.lib.formula_registry import SIGNED_FIGURE, formula

import logging

import re

from typing import Any, Dict, Optional, Tuple

# Formulas (and the helpers only they use) live in their owner's package;
# imported back so existing imports of this module keep working.
from app.agents.base.formulas.construction_formulas_planning import (  # noqa: F401 -- moved
    pe_unit_convert,
)
from app.agents.hats.planning.formulas.construction_formulas_planning import (  # noqa: F401 -- moved
    _crew_cost_from_aliases,
    _daily_production_from_rate_alias,
    critical_path_float,
    logger,
    productivity_manpower_duration,
    progress_quantity,
)
from app.agents.hats.quantities.formulas.construction_formulas_planning import (  # noqa: F401 -- moved
    concrete_mix_proportions,
    material_consumption,
)

# Live A2-1: "remaining 3,400 m2 of plastering cost if productivity stays
# at 42 m2 per gang-day and a gang costs SAR 1,950 per day"
_PROD_COST_ASK_RE = re.compile(
    r"(?i)\b(cost|price|sar|aed|usd|gbp|eur).{0,80}\b(gang|crew)|"
    r"\b(gang|crew).{0,40}\b(cost|price|sar|aed|usd)"
)

_QTY_M2_RE = re.compile(rf"(?i)({SIGNED_FIGURE})\s*m2\b")

_PER_GANG_DAY_RE = re.compile(
    rf"(?i)({SIGNED_FIGURE})\s*m2\s+per\s+(?:gang|crew)[- ]?day"
)

_GANG_DAY_COST_RE = re.compile(
    r"(?i)(?:(?:gang|crew).{0,32}(?:costs?|at)\s*)?(?:SAR|AED|USD|GBP|EUR)\s*"
    rf"({SIGNED_FIGURE})\s*per\s+day"
    r"|(?:gang|crew).{0,24}(?:costs?|at)\s*(\d[\d,]*(?:\.\d+)?)"
)

_CURRENCY_RE = re.compile(r"\b(SAR|AED|USD|GBP|EUR)\b", re.IGNORECASE)

def looks_like_productivity_cost_ask(text: str) -> bool:
    """True when the operator asked for cost from gang-day productivity."""
    raw = text or ""
    if not _PROD_COST_ASK_RE.search(raw):
        return False
    return bool(_PER_GANG_DAY_RE.search(raw) or _QTY_M2_RE.search(raw))

def parse_productivity_cost_ask(
    text: str,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """Return (quantity, daily_production, crew_cost_per_day, currency)."""
    raw = text or ""
    qty = daily = cost = None
    m_qty = _QTY_M2_RE.search(raw)
    if m_qty:
        qty = float(m_qty.group(1).replace(",", ""))
    m_daily = _PER_GANG_DAY_RE.search(raw)
    if m_daily:
        daily = float(m_daily.group(1).replace(",", ""))
    m_cost = _GANG_DAY_COST_RE.search(raw)
    if m_cost:
        raw_cost = next((g for g in m_cost.groups() if g), None)
        if raw_cost:
            cost = float(raw_cost.replace(",", ""))
    curr_m = _CURRENCY_RE.search(raw)
    currency = (curr_m.group(1) if curr_m else "SAR").upper()
    return qty, daily, cost, currency

def format_productivity_cost_line(inner: dict, currency: str = "SAR") -> str:
    """User-facing duration + cost line from a productivity result dict."""
    if not isinstance(inner, dict):
        return ""
    duration = inner.get("duration")
    cost = inner.get("total_cost")
    if duration is None:
        return ""
    try:
        dur_f = float(duration)
    except (TypeError, ValueError):
        logger.debug("duration is not numeric: %r", duration)
        return ""
    qty = inner.get("quantity")
    daily = inner.get("daily_production")
    crew = inner.get("crew_cost_per_day")
    bits = [f"Duration = {dur_f:.2f} gang-days"]
    if qty not in (None, "") and daily not in (None, ""):
        bits[0] = (
            f"Duration = {float(qty):g} / {float(daily):g} = {dur_f:.2f} gang-days"
        )
    if cost is not None:
        try:
            cost_f = float(cost)
        except (TypeError, ValueError):
            cost_f = None
        if cost_f is not None:
            crew_bit = f" × {currency} {float(crew):,.2f}" if crew not in (None, "") else ""
            bits.append(
                f"Cost = {dur_f:.2f}{crew_bit} = {currency} {cost_f:,.2f}"
            )
    return ". ".join(bits) + "."

def compose_productivity_cost_from_ask(text: str) -> dict | None:
    """Run duration × gang-day rate from the operator ask (live A2-1)."""
    if not looks_like_productivity_cost_ask(text):
        return None
    qty, daily, cost, currency = parse_productivity_cost_ask(text)
    if qty is None or daily is None or cost is None:
        return None
    from app.lib import construction_formulas as _cf
    env = _cf.run_calculation(
        "productivity_manpower_duration",
        {
            "quantity": qty,
            "daily_production": daily,
            "crew_cost_per_day": cost,
        },
    )
    if not isinstance(env, dict) or env.get("status") != "success":
        return None
    inner = env.get("result") if isinstance(env.get("result"), dict) else {}
    if inner.get("total_cost") is None or inner.get("duration") is None:
        return None
    # Echo inputs so the formatter can show qty / daily × rate.
    inner = dict(inner)
    inner.setdefault("quantity", qty)
    inner.setdefault("daily_production", daily)
    inner.setdefault("crew_cost_per_day", cost)
    line = format_productivity_cost_line(inner, currency)
    if not line or currency not in line:
        return None
    return {
        "duration": inner["duration"],
        "total_cost": inner["total_cost"],
        "currency": currency,
        "line": line,
        "envelope": env,
        "result": inner,
    }

ADDITIONAL_CALCULATORS = {
    "critical_path_float": critical_path_float,
    "progress_quantity": progress_quantity,
    "productivity_manpower_duration": productivity_manpower_duration,
    "pe_unit_convert": pe_unit_convert,
    "material_consumption": material_consumption,
    "concrete_mix_proportions": concrete_mix_proportions,
}
