"""Formulas from app.lib.construction_formulas owned by the procurement hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math
from typing import Any, Dict


@formula(
    owner='procurement',
    display_name='Number of cranes needed',
    description='Number of cranes needed for a lifting demand and crane cycle capacity.',
    inputs={
        'total_lift_demand_tons': Param('t', 0, 1e8, label='total lift demand'),
        'crane_capacity_tons': Param('t', 0, 10000, label='crane capacity'),
        'utilization_pct': Param('%', 0, 100, label='crane utilisation'),
        'shifts_per_day': Param('-', 1, 24, label='shifts per day'),
        'hours_per_shift': Param('h', 1, 24, label='hours per shift'),
        'cycle_time_minutes': Param('min', 1, 600, label='lift cycle time'),
        'working_days': Param('days', 1, 31, label='working days per month'),
        'dry_hire_rate_per_month': Param('currency/month', 0, 1e9, label='dry hire rate per crane per month'),
        'currency': Param('-', label='currency of the rate'),
    },
    outputs={'total_lift_demand_tons': '-', 'crane_capacity_tons': '-', 'utilization_pct': '%', 'shifts_per_day': '-', 'hours_per_shift': '-', 'cycle_time_minutes': '-', 'lifts_per_hour_per_crane': '-', 'daily_tonnage_per_crane': '-', 'cranes_required': '-', 'monthly_rate': 'currency/month', 'total_monthly_cost': 'currency/month', 'currency': '-'},
)
def crane_planning(
    total_lift_demand_tons: float,
    crane_capacity_tons: float,
    utilization_pct: float = 65.0,
    shifts_per_day: int = 2,
    hours_per_shift: float = 9.5,
    cycle_time_minutes: float = 20.0,
    working_days: int = 26,
    dry_hire_rate_per_month: Any = None,
    currency: str = "",
) -> Dict[str, Any]:
    """
    Determine number of cranes needed to service construction lifting demand.

    Formula:
        lifts_per_hour = 60 / cycle_time
        daily_capacity_per_crane = capacity x (util/100) x lifts_per_hour x hours x shifts
        num_cranes = ceil(total_demand / (daily_capacity x working_days))

    Cost: only from the caller's ``dry_hire_rate_per_month`` (the project's
    rate). No rate table is built in; without a rate the crane count is
    returned and the cost lines are omitted with a note.

    Args:
        total_lift_demand_tons: Total tonnage to be lifted over the period
        crane_capacity_tons: Single crane rated capacity (MT)
        utilization_pct: Effective utilization (default 65%)
        shifts_per_day: Number of shifts (default 2)
        hours_per_shift: Hours per shift (default 9.5)
        cycle_time_minutes: Average cycle time per lift (default 20 min)
        working_days: Working days per month (default 26)

    Returns:
        Dict with crane count, monthly cost, and capacity analysis
    """
    if total_lift_demand_tons < 0:
        raise ValueError("total_lift_demand_tons must be >= 0")
    if crane_capacity_tons <= 0 or cycle_time_minutes <= 0 or working_days <= 0:
        raise ValueError("crane_capacity_tons, cycle_time_minutes and working_days must be > 0")
    if hours_per_shift <= 0 or shifts_per_day <= 0:
        raise ValueError("hours_per_shift and shifts_per_day must be > 0")
    lifts_per_hour = 60.0 / cycle_time_minutes
    effective_capacity_per_lift = crane_capacity_tons * (utilization_pct / 100.0)
    daily_lifts_per_crane = lifts_per_hour * hours_per_shift * shifts_per_day
    daily_tonnage_per_crane = effective_capacity_per_lift * daily_lifts_per_crane

    total_working_days = working_days  # Assume 1 month period; scale for longer
    total_capacity_per_crane = daily_tonnage_per_crane * total_working_days

    num_cranes = math.ceil(total_lift_demand_tons / total_capacity_per_crane) if total_capacity_per_crane > 0 else 0

    monthly_rate = None if dry_hire_rate_per_month in (None, "") else float(dry_hire_rate_per_month)
    total_monthly_cost = None if monthly_rate is None else num_cranes * monthly_rate
    cur = f" {currency}" if currency else ""
    notes = [
        f"Each {crane_capacity_tons}T crane handles {daily_tonnage_per_crane:.1f} tons/day",
        f"At {utilization_pct}% utilization, {lifts_per_hour:.1f} lifts/hr, {cycle_time_minutes}min cycle",
    ]
    if monthly_rate is None:
        notes.append(f"Require {num_cranes} crane(s); cost needs the project's dry hire rate per month")
    else:
        notes.append(f"Require {num_cranes} crane(s) @ {monthly_rate:,.0f}{cur}/month each")
        notes.append(f"Total crane cost: {total_monthly_cost:,.0f}{cur}/month")

    return {
        "total_lift_demand_tons": total_lift_demand_tons,
        "crane_capacity_tons": crane_capacity_tons,
        "utilization_pct": utilization_pct,
        "shifts_per_day": shifts_per_day,
        "hours_per_shift": hours_per_shift,
        "cycle_time_minutes": cycle_time_minutes,
        "lifts_per_hour_per_crane": round(lifts_per_hour, 1),
        "daily_tonnage_per_crane": round(daily_tonnage_per_crane, 1),
        "cranes_required": num_cranes,
        "monthly_rate": monthly_rate,
        "total_monthly_cost": total_monthly_cost,
        "currency": currency,
        "notes": notes,
    }
