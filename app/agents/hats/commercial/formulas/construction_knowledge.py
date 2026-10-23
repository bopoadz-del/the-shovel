"""Formulas from app.core.construction_knowledge owned by the commercial hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Dict, Optional


@formula(
    owner='commercial',
    display_name='Payment due on a claim',
    description='Payment due on a claim: certified amount less retention and previous certificates.',
    inputs={
        'claimed_amount': Param('currency', 0, 1e13, label='amount claimed'),
        'certified_amount': Param('currency', 0, 1e13, label='amount certified'),
        'retention_rate': Param('fraction', 0, 1, label='retention'),
        'cumulative_previous_certified': Param('currency', 0, 1e13, label='previously certified amount'),
        'contract_value': Param('currency', 0, 1e13, label='contract value'),
    },
    outputs={'claimed_amount': 'currency', 'certified_amount': 'currency', 'retention_held': 'currency', 'net_payment_due': 'currency', 'cumulative_certified': 'currency', 'percent_complete': '%', 'disputed_amount': 'currency', 'retention_rate_pct': '%'},
)
def calculate_payment(
    claimed_amount: float,
    certified_amount: float,
    retention_rate: float = 0.05,
    cumulative_previous_certified: float = 0.0,
    contract_value: float = 0.0,
) -> Dict:
    """
    Calculate net payment due per the interim payment procedure.
    """
    if claimed_amount < 0 or certified_amount < 0:
        return {"error": "claimed_amount and certified_amount must be >= 0."}
    if not (0.0 <= retention_rate <= 1.0):
        return {"error": "retention_rate must be between 0 and 1 (fraction, not percent)."}
    if cumulative_previous_certified < 0 or contract_value < 0:
        return {"error": "cumulative_previous_certified and contract_value must be >= 0."}
    retention_held = certified_amount * retention_rate
    net_due = certified_amount - retention_held
    cumulative_now = cumulative_previous_certified + certified_amount
    pct_complete = (cumulative_now / contract_value * 100) if contract_value else None

    return {
        "claimed_amount": claimed_amount,
        "certified_amount": certified_amount,
        "retention_held": round(retention_held, 2),
        "net_payment_due": round(net_due, 2),
        "cumulative_certified": round(cumulative_now, 2),
        "percent_complete": round(pct_complete, 1) if pct_complete is not None else None,
        "disputed_amount": round(claimed_amount - certified_amount, 2),
        "retention_rate_pct": retention_rate * 100,
    }

@formula(
    owner='commercial',
    display_name='Earned value measures',
    description='Earned value measures (cost and schedule variance, CPI, SPI, estimate at completion) from PV, EV, AC and BAC.',
    inputs={
        'bac': Param('currency', 0, 1e13, label='budget at completion'),
        'bcwp': Param('currency', 0, 1e13, label='earned value (BCWP)'),
        'bcws': Param('currency', 0, 1e13, label='planned value (BCWS)'),
        'acwp': Param('currency', 0, 1e13, label='actual cost (ACWP)'),
        'pv': Param('currency', 0, 1e13, label='planned value'),
        'ev': Param('currency', 0, 1e13, label='earned value'),
        'ac': Param('currency', 0, 1e13, label='actual cost'),
    },
    outputs={'cost': 'currency (CV, CPI, EAC)', 'schedule': 'currency (SV, SPI)', 'cpi_health': '-'},
)
def calculate_evm(
    bac: Optional[float] = None,       # Budget at Completion (optional — needed for forecasts)
    bcwp: Optional[float] = None,      # Budgeted Cost of Work Performed (Earned Value)
    bcws: Optional[float] = None,      # Budgeted Cost of Work Scheduled (Planned Value)
    acwp: Optional[float] = None,      # Actual Cost of Work Performed
    *,
    pv: Optional[float] = None,        # alias for Planned Value (= BCWS)
    ev: Optional[float] = None,        # alias for Earned Value (= BCWP)
    ac: Optional[float] = None,        # alias for Actual Cost (= ACWP)
) -> Dict:
    """Classic EVM arithmetic. Requires Planned Value, Earned Value, and Actual Cost.

    Accepts either the PMI names (BCWS/BCWP/ACWP) or the PE-sheet aliases
    (PV/EV/AC). BAC is optional: without it, SPI/CPI/SV/CV still compute and
    EAC/ETC/VAC are omitted (honest — no invented budget).

    Default EAC when BAC and CPI>0: ``EAC = BAC / CPI`` (typical cost-performance
    forecast). ETC = EAC − AC; VAC = BAC − EAC.
    """
    # Resolve aliases — PE sheets and chat use PV/EV/AC; the knowledge base
    # historically used BCWS/BCWP/ACWP. Prefer the explicit PMI names when both
    # are supplied so existing callers keep their semantics.
    planned = bcws if bcws is not None else pv
    earned = bcwp if bcwp is not None else ev
    actual = acwp if acwp is not None else ac

    missing = [
        name for name, val in (
            ("PV/BCWS", planned), ("EV/BCWP", earned), ("AC/ACWP", actual),
        )
        if val is None
    ]
    if missing:
        return {
            "error": (
                "EVM requires Planned Value (PV/BCWS), Earned Value (EV/BCWP), "
                f"and Actual Cost (AC/ACWP) — missing: {', '.join(missing)}. "
                "No invented actuals."
            ),
            "required": ["pv|bcws", "ev|bcwp", "ac|acwp"],
            "optional": ["bac"],
        }

    planned_f = float(planned)
    earned_f = float(earned)
    actual_f = float(actual)
    bac_f = float(bac) if bac is not None else None
    if planned_f < 0 or earned_f < 0 or actual_f < 0 or (bac_f is not None and bac_f < 0):
        return {"error": "EVM values (PV, EV, AC, BAC) must be >= 0."}

    # Derived figures come from the UNROUNDED CPI: EAC from the 3dp display
    # value drifted 82 per 211k on the phase-3 hand-check battery (200000/0.947
    # vs 200000/(90000/95000)). Rounding is for display only, never an input.
    cpi_raw = (earned_f / actual_f) if actual_f else None
    cpi = round(cpi_raw, 3) if cpi_raw is not None else None
    spi = round(earned_f / planned_f, 3) if planned_f else None
    # Forecasts need BAC. Default formula: EAC = BAC / CPI when CPI > 0.
    eac = None
    etc = None
    vac = None
    eac_formula = None
    if bac_f is not None and cpi_raw and cpi_raw > 0:
        eac = round(bac_f / cpi_raw, 2)
        etc = round(eac - actual_f, 2)
        vac = round(bac_f - eac, 2)
        eac_formula = "BAC / CPI"
    cv = round(earned_f - actual_f, 2)   # CV = EV − AC  (cost variance)
    sv = round(earned_f - planned_f, 2)  # SV = EV − PV  (schedule variance)

    out: Dict = {
        "BAC": bac_f,
        "BCWP": earned_f, "EV": earned_f,
        "BCWS": planned_f, "PV": planned_f,
        "ACWP": actual_f, "AC": actual_f,
        "CPI": cpi,
        "SPI": spi,
        "EAC": eac,
        "ETC": etc,
        "VAC": vac,
        "CV": cv,
        "SV": sv,
        "formulas": {
            "SPI": "EV / PV",
            "CPI": "EV / AC",
            "SV": "EV − PV",
            "CV": "EV − AC",
            "EAC": eac_formula,
            "ETC": "EAC − AC" if eac is not None else None,
            "VAC": "BAC − EAC" if vac is not None else None,
        },
        "status": {
            "cost": "UNDER BUDGET" if cv >= 0 else "OVER BUDGET",
            "schedule": "AHEAD" if sv >= 0 else "BEHIND",
            "cpi_health": (
                "GOOD" if cpi and cpi >= 1
                else ("WARNING" if cpi and cpi >= 0.9 else "CRITICAL")
            ),
        },
    }
    if bac_f is None:
        out["note"] = (
            "BAC omitted — SPI/CPI/SV/CV computed; EAC/ETC/VAC require BAC."
        )
    return out
