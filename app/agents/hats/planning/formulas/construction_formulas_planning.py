"""Formulas from app.lib.construction_formulas_planning owned by the planning hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import logging
import re
from typing import Any, Dict, Optional, Tuple


logger = logging.getLogger(__name__)

@formula(
    owner='planning',
    display_name='Total float and criticality',
    description='Total float of an activity from its early and late dates, and whether it is critical.',
    inputs={
        'early_start': Param('days', 0, 100000, label='early start'),
        'early_finish': Param('days', 0, 100000, label='early finish'),
        'late_start': Param('days', 0, 100000, label='late start'),
        'late_finish': Param('days', 0, 100000, label='late finish'),
    },
    outputs={'total_float': '-', 'is_critical': '-', 'consistency_check_lf_minus_ef': '-'},
)
def critical_path_float(
    early_start: float,
    early_finish: float,
    late_start: float,
    late_finish: float,
) -> dict:
    """Total float TF = LS - ES = LF - EF; the activity is on the critical path
    when TF <= 0. (Free float needs the successor's ES; not computed here.)

    NEGATIVE float is critical too -- MORE critical, not less:     a real 2013 P6
    baseline surfaced an activity at TF = -8.6 days (behind its constraint
    dates) that the old `TF == 0` test reported as NOT critical. On a live
    schedule, negative-float activities are exactly the ones driving the
    forecast delay; a planner asking "is this critical?" must never be told
    no about one of them."""
    if float(early_finish) < float(early_start) or float(late_finish) < float(late_start):
        return {"error": "finish dates must be >= the matching start dates."}
    es, ef = float(early_start), float(early_finish)
    ls, lf = float(late_start), float(late_finish)
    tf = ls - es
    tf_check = lf - ef
    on_critical = tf < 1e-9
    behind = tf < -1e-9
    return {
        "total_float": round(tf, 3),
        "is_critical": on_critical,
        "consistency_check_lf_minus_ef": round(tf_check, 3),
        "standard": "CPM (PMBOK / planning)",
        "note": (f"TF = LS - ES = {ls} - {es} = {tf:.1f} (= LF - EF = {lf} - {ef} "
                 f"= {tf_check:.1f}); "
                 + ("NEGATIVE float - critical and behind schedule."
                    if behind else
                    ("critical" if on_critical else "has float") + ".")),
    }

@formula(
    owner='planning',
    display_name='Progress and remaining quantity',
    description='Planned and actual percent complete, remaining quantity and progress variance.',
    inputs={
        'total_qty': Param('qty', 0, 1e12, label='total quantity'),
        'planned_qty': Param('qty', 0, 1e12, label='planned quantity'),
        'actual_qty': Param('qty', 0, 1e12, label='actual quantity'),
    },
    outputs={'planned_percent': '%', 'actual_percent': '%', 'planned_qty': '-', 'actual_qty': '-', 'remaining_qty': '-', 'progress_variance_percent': '%'},
)
def progress_quantity(
    total_qty: float,
    planned_qty: Optional[float] = None,
    actual_qty: Optional[float] = None,
) -> dict:
    """Progress & quantity sheet: Planned %, Actual %, Remaining Qty, Progress Variance.

    Requires ``total_qty`` > 0. Planned and/or actual quantities are optional —
    only the outputs whose inputs are present are returned. Never invents a qty.
    """
    total = float(total_qty)
    if total <= 0:
        return {
            "error": "progress_quantity requires total_qty > 0 — refuse empty total.",
            "required": ["total_qty"],
            "optional": ["planned_qty", "actual_qty"],
        }
    out: Dict[str, Any] = {
        "total_qty": total,
        "units": "same as input quantities",
        "formulas_used": [],
        "standard": "PE formula sheet (Progress & Quantity)",
    }
    if planned_qty is not None:
        pq = float(planned_qty)
        planned_pct = (pq / total) * 100.0
        out["planned_qty"] = pq
        out["planned_percent"] = round(planned_pct, 3)
        out["formulas_used"].append("Planned % = Planned Qty / Total Qty × 100")
    if actual_qty is not None:
        aq = float(actual_qty)
        actual_pct = (aq / total) * 100.0
        remaining = total - aq
        out["actual_qty"] = aq
        out["actual_percent"] = round(actual_pct, 3)
        out["remaining_qty"] = round(remaining, 6)
        out["formulas_used"].append("Actual % = Actual Qty / Total Qty × 100")
        out["formulas_used"].append("Remaining Qty = Total Qty − Actual Qty")
    if planned_qty is not None and actual_qty is not None:
        out["progress_variance_percent"] = round(
            out["actual_percent"] - out["planned_percent"], 3
        )
        out["formulas_used"].append("Progress Variance = Actual % − Planned %")
    if planned_qty is None and actual_qty is None:
        return {
            "error": (
                "progress_quantity needs planned_qty and/or actual_qty in "
                "addition to total_qty — no invented quantities."
            ),
            "required": ["total_qty", "planned_qty|actual_qty"],
        }
    out["note"] = "; ".join(out["formulas_used"])
    return out

def _crew_cost_from_aliases(
    crew_cost_per_day: Optional[float],
    day_rate: Optional[float],
    gang_cost_per_day: Optional[float],
) -> Optional[float]:
    for raw in (crew_cost_per_day, day_rate, gang_cost_per_day):
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None

# A rate quoted against one of these is output per day, not per man-hour.
_PER_DAY_RATE = ("gang-day", "gang day", "crew-day", "crew day", "per day", "/day")


def _unit_is_per_hour(rate_unit: Optional[str]) -> bool:
    unit = str(rate_unit or "").lower()
    return "hour" in unit or "/h" in unit


def _unit_is_per_day(rate_unit: Optional[str]) -> bool:
    """True when the quoted unit is a day rate, not an hour rate."""
    unit = str(rate_unit or "").lower()
    if not unit or _unit_is_per_hour(unit):
        return False
    return any(token in unit for token in _PER_DAY_RATE)


# The number, then the unit words written on the same value. The words stay
# whole: a leading area token is not a reason to drop "per gang-day".
_RATE_VALUE = re.compile(r"^([+-]?\d[\d,]*(?:\.\d+)?)(?:\s+(.*))?$")


def _quoted_rate(raw: Any) -> Tuple[Optional[float], str]:
    """A rate's number and the unit words written beside it.

    A bare number has no unit. The unit is whatever was written, not the
    longest prefix a converter happens to know.
    """
    if isinstance(raw, bool) or raw is None or raw == "":
        return None, ""
    if isinstance(raw, (int, float)):
        return float(raw), ""
    match = _RATE_VALUE.match(str(raw).strip())
    if match is None:
        return None, ""
    return float(match.group(1).replace(",", "")), (match.group(2) or "").strip()


def _daily_production_from_rate_alias(
    *,
    productivity_rate: Optional[float],
    rate_unit: Optional[str],
    crew_cost: Optional[float],
) -> Optional[float]:
    """``productivity_rate`` plus a day unit (or a crew day-rate).

    ``productivity`` on this calculator is qty / man-hour. A rate quoted
    per day is ``daily_production``, not that field.
    """
    if productivity_rate is None or productivity_rate == "":
        return None
    try:
        rate = float(productivity_rate)
    except (TypeError, ValueError):
        logger.debug("productivity_rate is not numeric: %r", productivity_rate)
        return None
    if rate <= 0:
        return None
    unit = str(rate_unit or "").lower()
    if _unit_is_per_day(unit) or crew_cost is not None or not unit:
        # No unit + crew cost (or a bare rate next to a day-cost) is a
        # day rate. A unit that names hours stays on the man-hour path.
        if "hour" in unit or "/h" in unit:
            return None
        return rate
    return None


def _must_be_positive(field: str, value: float, required: list) -> dict:
    """A zero or negative input is refused in the input's own words and asked
    for again -- never normalised (a negative rate is not read as positive)."""
    from app.lib import formula_registry

    spec = formula_registry.get("productivity_manpower_duration")
    params = formula_registry.parameters(spec) if spec is not None else {}
    label = (params[field].label if field in params and params[field].label else field.replace("_", " "))
    return {
        "error": f"The {label} must be greater than zero; {value:g} was given, so it cannot be used.",
        "needs_input": [field],
        "required": required,
    }

@formula(
    owner='planning',
    display_name='Quantity, productivity, manpower and duration',
    description='Links quantity, productivity, manpower and duration: any one from the others.',
    inputs={
        'quantity_executed': Param('qty', 0, 1e12, label='quantity executed'),
        'man_hours': Param('man-hour', 0, 1e8, label='man-hours'),
        'quantity': Param('qty', 0, 1e12, label='work quantity'),
        'productivity': Param(label='labour productivity'),
        'manpower': Param('persons', 0, 1e6, label='crew size'),
        'working_hours': Param('h', 0, 1e8, label='working hours'),
        'remaining_manhours': Param('man-hour', 0, 1e10, label='remaining man-hours'),
        'available_hours': Param('h', 0, 1e8, label='available hours'),
        'daily_production': Param('qty/day', 0, 1e9, label='daily production'),
        'remaining_qty': Param('qty', 0, 1e12, label='remaining quantity'),
        'remaining_days': Param('days', 0, 36500, label='remaining duration'),
        'productivity_rate': Param(label='production rate'),
        'rate_unit': Param(label='rate unit'),
        'crew_cost_per_day': Param('currency/day', 0, 1e9, label='crew cost per day'),
        'day_rate': Param('currency/day', 0, 1e9, label='day rate'),
        'gang_cost_per_day': Param('currency/day', 0, 1e9, label='gang cost per day'),
    },
    outputs={
        'productivity': '1/man-hour',
        'required_manpower': '-',
        'manhours_required': 'h',
        'daily_required_production': '-',
        'duration': 'days',
        'total_cost': 'currency',
    },
)
def productivity_manpower_duration(
    *,
    quantity_executed: Optional[float] = None,
    man_hours: Optional[float] = None,
    quantity: Optional[float] = None,
    productivity: str = None,
    manpower: Optional[float] = None,
    working_hours: Optional[float] = None,
    remaining_manhours: Optional[float] = None,
    available_hours: Optional[float] = None,
    daily_production: Optional[float] = None,
    remaining_qty: Optional[float] = None,
    remaining_days: Optional[float] = None,
    productivity_rate: str = None,
    rate_unit: Optional[str] = None,
    crew_cost_per_day: Optional[float] = None,
    day_rate: Optional[float] = None,
    gang_cost_per_day: Optional[float] = None,
) -> dict:
    """Qty → productivity → manpower → duration (PE formula sheet).

    Computes every output whose required inputs are present. Refuses when
    nothing can be computed — never invents a productivity rate.

    ``productivity`` and ``productivity_rate`` stay text. A unit written on
    the value ("m2 per gang-day", "per hour") has to reach this function
    whole; a numeric declaration would keep only the leading number.

    ``productivity_rate`` (m2 per gang-day) pairs with ``quantity`` as
    daily production, and ``crew_cost_per_day`` prices the resulting
    duration. These aliases must reach the function, or a call that uses
    them is refused for missing paired inputs.
    """
    # The unit written on the rate decides the output field. A per-day rate
    # is duration in days; an hour rate is man-hours. A bare productivity
    # number stays man-hours, and a bare production rate stays a day rate.
    prod_value, prod_unit = _quoted_rate(productivity)
    rate_value, rate_written = _quoted_rate(productivity_rate)
    if prod_unit and not rate_unit:
        rate_unit = prod_unit
    if rate_written and not rate_unit:
        rate_unit = rate_written
    if prod_value is not None and _unit_is_per_day(prod_unit):
        if daily_production is None:
            daily_production = prod_value
        productivity = None
    else:
        productivity = prod_value
    if rate_value is not None and _unit_is_per_hour(rate_written):
        if productivity is None:
            productivity = rate_value
        productivity_rate = None
    else:
        productivity_rate = rate_value
    crew_cost = _crew_cost_from_aliases(
        crew_cost_per_day, day_rate, gang_cost_per_day,
    )
    # A zero or negative rate given under its alias is refused in its own
    # words like every other input here -- never dropped as "not given",
    # which turned -120 into a generic "needs paired inputs" (LANE2).
    if productivity_rate not in (None, ""):
        try:
            alias_rate = float(productivity_rate)
        except (TypeError, ValueError):
            alias_rate = None
        if alias_rate is not None and alias_rate <= 0:
            return _must_be_positive("productivity_rate", alias_rate, ["quantity", "productivity_rate"])
    if daily_production is None:
        aliased_daily = _daily_production_from_rate_alias(
            productivity_rate=productivity_rate,
            rate_unit=rate_unit,
            crew_cost=crew_cost,
        )
        if aliased_daily is not None:
            daily_production = aliased_daily
    if productivity is None and productivity_rate is not None and daily_production is None:
        try:
            prod_alias = float(productivity_rate)
        except (TypeError, ValueError):
            prod_alias = None
        if prod_alias is not None and prod_alias > 0:
            productivity = prod_alias
    # A per-day unit describes the rate next to it. That figure is duration
    # in days, including when the same number was also passed as productivity.
    if productivity is not None and _unit_is_per_day(rate_unit):
        quoted = productivity_rate if productivity_rate not in (None, "") else productivity
        try:
            same_rate = float(productivity) == float(quoted)
        except (TypeError, ValueError):
            same_rate = False
        if same_rate:
            if daily_production is None:
                daily_production = float(productivity)
            productivity = None

    results: Dict[str, Any] = {
        "formulas_used": [],
        "standard": "PE formula sheet (Productivity / Manpower / Duration)",
    }
    computed = False

    # Productivity = Quantity Executed / Man-hours
    if quantity_executed is not None and man_hours is not None:
        mh = float(man_hours)
        if mh <= 0:
            return _must_be_positive("man_hours", mh, ["quantity_executed", "man_hours"])
        prod = float(quantity_executed) / mh
        results["productivity"] = round(prod, 6)
        results["formulas_used"].append(
            "Productivity = Quantity Executed / Man-hours"
        )
        computed = True
        if productivity is None:
            productivity = prod

    # Manhours Required = Quantity / Productivity
    if quantity is not None and productivity is not None:
        prod = float(productivity)
        if prod <= 0:
            return _must_be_positive("productivity", prod, ["quantity", "productivity"])
        mh_req = float(quantity) / prod
        results["manhours_required"] = round(mh_req, 4)
        results["formulas_used"].append(
            "Manhours Required = Quantity / Productivity"
        )
        computed = True
        if remaining_manhours is None:
            remaining_manhours = mh_req

    # Manhours = Manpower × Working Hours
    if manpower is not None and working_hours is not None:
        mh = float(manpower) * float(working_hours)
        results["manhours"] = round(mh, 4)
        results["formulas_used"].append(
            "Manhours = Manpower × Working Hours"
        )
        computed = True

    # Required Manpower = Remaining Manhours / Available Hours
    if remaining_manhours is not None and available_hours is not None:
        avail = float(available_hours)
        if avail <= 0:
            return _must_be_positive("available_hours", avail, ["remaining_manhours", "available_hours"])
        req_mp = float(remaining_manhours) / avail
        results["required_manpower"] = round(req_mp, 4)
        results["formulas_used"].append(
            "Required Manpower = Remaining Manhours / Available Hours"
        )
        computed = True

    # Duration = Quantity / Daily Production
    if quantity is not None and daily_production is not None:
        daily = float(daily_production)
        if daily <= 0:
            return _must_be_positive("daily_production", daily, ["quantity", "daily_production"])
        dur = float(quantity) / daily
        results["duration"] = round(dur, 4)
        results["formulas_used"].append(
            "Duration = Quantity / Daily Production"
        )
        computed = True
        if crew_cost is not None:
            cost = dur * crew_cost
            results["total_cost"] = round(cost, 2)
            results["crew_cost_per_day"] = round(crew_cost, 2)
            results["formulas_used"].append(
                "Cost = Duration × Crew Cost per Day"
            )

    # Daily Required Production = Remaining Qty / Remaining Days
    if remaining_qty is not None and remaining_days is not None:
        days = float(remaining_days)
        if days <= 0:
            return _must_be_positive("remaining_days", days, ["remaining_qty", "remaining_days"])
        daily_req = float(remaining_qty) / days
        results["daily_required_production"] = round(daily_req, 6)
        results["formulas_used"].append(
            "Daily Required Production = Remaining Qty / Remaining Days"
        )
        computed = True

    if not computed:
        return {
            "error": (
                "productivity_manpower_duration needs paired inputs (no invented "
                "rates): quantity_executed (qty) + man_hours (h); "
                "quantity (qty) + productivity (qty/h); "
                "quantity (qty) + daily_production (qty/day); "
                "quantity (qty) + productivity_rate (qty/gang-day) + "
                "crew_cost_per_day (currency/day); "
                "manpower (persons) + working_hours (h); "
                "remaining_manhours (h) + available_hours (h); "
                "or remaining_qty (qty) + remaining_days (days)."
            ),
            "required_pairs": [
                ["quantity_executed", "man_hours"],
                ["quantity", "productivity"],
                ["manpower", "working_hours"],
                ["remaining_manhours", "available_hours"],
                ["quantity", "daily_production"],
                ["remaining_qty", "remaining_days"],
                ["quantity", "productivity_rate", "crew_cost_per_day"],
            ],
        }

    note_parts = list(results["formulas_used"])
    if results.get("duration") is not None and quantity is not None and daily_production is not None:
        note_parts.append(
            f"{float(quantity):g} / {float(daily_production):g} = "
            f"{results['duration']:g} days"
        )
    if results.get("total_cost") is not None and results.get("duration") is not None:
        note_parts.append(
            f"{results['duration']:g} × {results['crew_cost_per_day']:g} = "
            f"{results['total_cost']:g}"
        )
    results["note"] = "; ".join(note_parts)
    return results
