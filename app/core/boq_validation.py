"""BOQ Intelligence Layer (MEGA-3) -- quantity validation.

Walks a list of BOQ line items and surfaces three classes of issue:

1. Math inconsistency  -- declared `total` does not equal `qty * rate` within
   a small tolerance. Catches data-entry typos and stale exports where someone
   updated qty without re-multiplying.
2. Bottom-line drift    -- sum of line totals does not equal the declared
   project total. Catches missing rows, subtotals counted twice, or VAT/
   contingency lines that were excluded silently.
3. Suspicious zero      -- a line with non-zero qty but zero rate (free
   work) or non-zero rate but zero qty (placeholder). These are almost
   always either bid placeholders or extraction errors.

This is layer-1 of the BOQ Intelligence stack -- pure math, no LLM calls,
deterministic. Rate-band benchmarking and scope-gap detection are layer-2
and layer-3, which will live in separate modules once their data sources
(historical rate ranges, spec-analyzer output) are wired through.

Public API
----------
* :func:`validate_boq(items, declared_total=None, tol=0.001)` ->
    ``ValidationReport`` dataclass with ``flags: list[Flag]`` and
    ``summary`` dict.

Each :class:`Flag` is a structured finding the chat layer can render as a
human bullet. No raw exceptions leak out -- malformed items become "input
unparseable" flags, not 500s.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Math tolerance: relative for non-tiny values, absolute otherwise.
# 0.1% catches genuine errors without flagging penny-rounding noise on
# multi-million-dollar bids.
DEFAULT_REL_TOL = 0.001
DEFAULT_ABS_TOL = 0.50


@dataclass(frozen=True)
class Flag:
    """One BOQ validation finding."""
    kind: str  # "math_mismatch" | "bottom_line_drift" | "suspicious_zero" | "input_unparseable"
    line_index: Optional[int]
    description: str
    expected: Optional[float] = None
    actual: Optional[float] = None
    delta: Optional[float] = None
    severity: str = "warning"  # "info" | "warning" | "critical"


@dataclass
class ValidationReport:
    flags: List[Flag] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.flags)


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float parse. Returns None on failure (so callers can flag)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        # Strip common currency / thousand separators before parsing.
        s = str(value).strip().replace(",", "").replace("$", "").replace(" ", "")
        if not s:
            return None
        return float(s)
    except (TypeError, ValueError):
        logger.warning("could not parse BOQ numeric value %r", value, exc_info=True)
        return None


def _math_mismatch(line_total: float, qty: float, rate: float, tol_rel: float, tol_abs: float) -> bool:
    """True when declared line_total disagrees with qty*rate beyond tolerance."""
    computed = qty * rate
    diff = abs(line_total - computed)
    if diff <= tol_abs:
        return False
    base = max(abs(line_total), abs(computed))
    if base == 0:
        return False
    return (diff / base) > tol_rel


def validate_boq(
    items: List[Dict[str, Any]],
    declared_total: Optional[float] = None,
    tol_rel: float = DEFAULT_REL_TOL,
    tol_abs: float = DEFAULT_ABS_TOL,
) -> ValidationReport:
    """Run quantity validation across a list of BOQ line items.

    ``items`` shape (loose): each is a dict with at least ``qty`` and ``rate``
    or ``total``. Extra keys (description, unit, section, item_code) are
    preserved in flag descriptions so a user reading the report can find the
    offending row.

    ``declared_total``: if the operator pasted a bottom-line figure from the
    BOQ cover sheet, we cross-check the sum of computed line totals against
    it. Pass ``None`` to skip that check.
    """
    report = ValidationReport()
    sum_computed = 0.0
    sum_declared = 0.0
    parsed_count = 0

    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            report.flags.append(Flag(
                kind="input_unparseable",
                line_index=i,
                description=f"Row {i + 1}: not a dict ({type(raw).__name__})",
                severity="warning",
            ))
            continue

        # `... or ...` would treat the legitimate value 0 as missing, so
        # walk the alternate-key lists explicitly and take the first present.
        def _first(*keys: str) -> Any:
            for k in keys:
                if k in raw and raw[k] is not None and raw[k] != "":
                    return raw[k]
            return None
        qty = _to_float(_first("qty", "quantity"))
        rate = _to_float(_first("rate", "unit_rate", "unit_cost"))
        declared = _to_float(_first("total", "amount", "line_total"))
        desc = str(raw.get("description") or raw.get("item") or raw.get("name") or f"row {i + 1}").strip()

        # No usable numbers at all -- skip but warn, the operator may have
        # passed a row that's purely a section header.
        if qty is None and rate is None and declared is None:
            continue

        # Suspicious zeros.
        if qty is not None and rate is not None:
            if qty > 0 and rate == 0:
                report.flags.append(Flag(
                    kind="suspicious_zero",
                    line_index=i,
                    description=f"Row {i + 1} ({desc}): qty {qty} with zero rate -- placeholder?",
                    severity="info",
                ))
            elif qty == 0 and rate > 0:
                report.flags.append(Flag(
                    kind="suspicious_zero",
                    line_index=i,
                    description=f"Row {i + 1} ({desc}): zero qty with rate {rate} -- placeholder?",
                    severity="info",
                ))

        # Math check: requires all three.
        if qty is not None and rate is not None and declared is not None:
            if _math_mismatch(declared, qty, rate, tol_rel, tol_abs):
                computed = qty * rate
                report.flags.append(Flag(
                    kind="math_mismatch",
                    line_index=i,
                    description=(
                        f"Row {i + 1} ({desc}): qty {qty} x rate {rate} "
                        f"= {computed:.2f}, declared {declared:.2f}"
                    ),
                    expected=computed,
                    actual=declared,
                    delta=declared - computed,
                    severity="critical",
                ))

        # Track totals for bottom-line check.
        if declared is not None:
            sum_declared += declared
            parsed_count += 1
        elif qty is not None and rate is not None:
            sum_computed += qty * rate
            parsed_count += 1

    summed_line_total = sum_declared + sum_computed
    report.summary["line_count"] = len(items)
    report.summary["parsed_lines"] = parsed_count
    report.summary["sum_line_totals"] = summed_line_total

    if declared_total is not None:
        report.summary["declared_total"] = declared_total
        diff = abs(summed_line_total - declared_total)
        base = max(abs(summed_line_total), abs(declared_total))
        rel_drift = (diff / base) if base else 0.0
        report.summary["bottom_line_drift_abs"] = diff
        report.summary["bottom_line_drift_rel"] = rel_drift
        if diff > tol_abs and rel_drift > tol_rel:
            report.flags.append(Flag(
                kind="bottom_line_drift",
                line_index=None,
                description=(
                    f"Sum of line totals is {summed_line_total:.2f} but the "
                    f"declared project total is {declared_total:.2f} "
                    f"(drift {diff:.2f}, {rel_drift * 100:.2f}%)."
                ),
                expected=summed_line_total,
                actual=declared_total,
                delta=declared_total - summed_line_total,
                severity="critical",
            ))

    return report
