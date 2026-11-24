"""Clash resolution register over the AABB clash list (S-K3 item 6).

The register has one row per detected clash. Each row names the resolution
strategy that the project's own strategy table gives that clash. Nothing in
this module knows a discipline, a strategy name or a depth limit: the table
is the caller's (``clash_strategy_table``), and a clash that no row covers
is listed as unassigned with the reason, never given a strategy by guess.

Table row (a dict):

- ``strategy``: non-empty text, required.
- ``a`` / ``b``: optional selectors, each ``{"category": text}`` or
  ``{"ifc_type": text}``. A missing side matches any element. The two
  selectors match the clash's two elements in either order.
- ``min_overlap_mm`` / ``max_overlap_mm``: optional depth band, the minimum
  inclusive and the maximum exclusive.
- ``responsible``: optional text, carried into the register as written.

When several rows match one clash, the most specific wins: an ``ifc_type``
side counts 2, a ``category`` side 1, a depth band 1. Rows tied at the top
that give different outcomes leave the clash unassigned and name the rows.
A row with a depth band cannot decide a clash that carries no overlap
figure; such a clash is unassigned rather than judged without its depth.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

_ROW_KEYS = {"strategy", "a", "b", "min_overlap_mm", "max_overlap_mm", "responsible"}
_SELECTOR_KEYS = ("category", "ifc_type")
_SELECTOR_WEIGHT = {"category": 1, "ifc_type": 2}


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_selector(index: int, side: str, sel: Any) -> Tuple[str, str]:
    if not isinstance(sel, dict) or len(sel) != 1:
        raise ValueError(
            f"clash_strategy_table row {index}: '{side}' must hold exactly one "
            f"of {', '.join(_SELECTOR_KEYS)}"
        )
    (key, value), = sel.items()
    if key not in _SELECTOR_KEYS:
        raise ValueError(
            f"clash_strategy_table row {index}: '{side}' key '{key}' is not one "
            f"of {', '.join(_SELECTOR_KEYS)}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"clash_strategy_table row {index}: '{side}.{key}' must be non-empty text"
        )
    return key, value.strip().casefold()


def _parse_table(table: Any) -> List[Dict[str, Any]]:
    """Validate the whole table first; one bad row rejects it, named."""
    if not isinstance(table, list):
        raise ValueError("clash_strategy_table must be a list of rows")
    parsed: List[Dict[str, Any]] = []
    for index, row in enumerate(table):
        if not isinstance(row, dict):
            raise ValueError(f"clash_strategy_table row {index}: must be a mapping")
        unknown = sorted(set(row) - _ROW_KEYS)
        if unknown:
            raise ValueError(
                f"clash_strategy_table row {index}: unknown key(s) {', '.join(unknown)}"
            )
        strategy = row.get("strategy")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError(
                f"clash_strategy_table row {index}: 'strategy' must be non-empty text"
            )
        responsible = row.get("responsible")
        if responsible is not None and not isinstance(responsible, str):
            raise ValueError(
                f"clash_strategy_table row {index}: 'responsible' must be text"
            )
        sides = [
            _check_selector(index, side, row[side]) if side in row else None
            for side in ("a", "b")
        ]
        band: Dict[str, float] = {}
        for key in ("min_overlap_mm", "max_overlap_mm"):
            if key in row:
                if not _is_number(row[key]):
                    raise ValueError(
                        f"clash_strategy_table row {index}: '{key}' must be a finite number"
                    )
                band[key] = float(row[key])
        if (
            "min_overlap_mm" in band and "max_overlap_mm" in band
            and band["min_overlap_mm"] >= band["max_overlap_mm"]
        ):
            raise ValueError(
                f"clash_strategy_table row {index}: 'min_overlap_mm' must be below "
                f"'max_overlap_mm'"
            )
        specificity = sum(_SELECTOR_WEIGHT[s[0]] for s in sides if s) + (1 if band else 0)
        parsed.append({
            "index": index,
            "strategy": strategy.strip(),
            "responsible": responsible,
            "sides": sides,
            "band": band,
            "specificity": specificity,
        })
    return parsed


def _clash_side(clash: Dict[str, Any], suffix: str) -> Dict[str, Optional[str]]:
    """One element of a clash. A clash with one shared ``category`` (the
    name-duplicate fallback) has that category on both sides."""
    return {
        "category": clash.get(f"category_{suffix}") or clash.get("category"),
        "ifc_type": clash.get(f"ifc_type_{suffix}"),
    }


def _selector_matches(sel: Optional[Tuple[str, str]], side: Dict[str, Optional[str]]) -> bool:
    if sel is None:
        return True
    key, value = sel
    have = side.get(key)
    return isinstance(have, str) and have.strip().casefold() == value


def _sides_match(row: Dict[str, Any], a: Dict, b: Dict) -> bool:
    sa, sb = row["sides"]
    return (
        (_selector_matches(sa, a) and _selector_matches(sb, b))
        or (_selector_matches(sa, b) and _selector_matches(sb, a))
    )


def _overlap(clash: Dict[str, Any]) -> Optional[float]:
    value = clash.get("overlap_mm")
    return float(value) if _is_number(value) else None


def _in_band(band: Dict[str, float], overlap: float) -> bool:
    if "min_overlap_mm" in band and overlap < band["min_overlap_mm"]:
        return False
    if "max_overlap_mm" in band and overlap >= band["max_overlap_mm"]:
        return False
    return True


def _assign(clash: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    a, b = _clash_side(clash, "a"), _clash_side(clash, "b")
    overlap = _overlap(clash)
    candidates: List[Dict[str, Any]] = []
    needs_depth: List[int] = []
    for row in rows:
        if not _sides_match(row, a, b):
            continue
        if row["band"]:
            if overlap is None:
                needs_depth.append(row["index"])
                continue
            if not _in_band(row["band"], overlap):
                continue
        candidates.append(row)
    if needs_depth:
        return {
            "strategy": None,
            "reason": (
                "a table row for these elements depends on the overlap depth, "
                "which this clash does not carry"
            ),
            "depth_rows": needs_depth,
        }
    if not candidates:
        return {"strategy": None, "reason": "no table row matches this clash"}
    top = max(r["specificity"] for r in candidates)
    best = [r for r in candidates if r["specificity"] == top]
    outcomes = {(r["strategy"], r["responsible"]) for r in best}
    if len(outcomes) > 1:
        return {
            "strategy": None,
            "reason": "equally specific table rows give different strategies",
            "tied_rows": [r["index"] for r in best],
        }
    chosen = best[0]
    return {
        "strategy": chosen["strategy"],
        "responsible": chosen["responsible"],
        "table_row": chosen["index"],
    }


def build_clash_register(
    clashes: List[Dict[str, Any]], strategy_table: Any = None
) -> Dict[str, Any]:
    """Register one row per clash with the strategy the caller's table gives.

    Raises ValueError (naming the row) when the table is malformed; the
    caller decides whether to keep the detection without a register.
    """
    rows = _parse_table(strategy_table) if strategy_table is not None else None
    register: List[Dict[str, Any]] = []
    by_strategy: Dict[str, int] = {}
    used: set = set()
    for position, clash in enumerate(clashes or [], start=1):
        if not isinstance(clash, dict):
            continue
        entry: Dict[str, Any] = {
            "ref": position,
            "element_a": clash.get("element_a"),
            "element_b": clash.get("element_b"),
            "category_a": _clash_side(clash, "a")["category"],
            "category_b": _clash_side(clash, "b")["category"],
            "ifc_type_a": clash.get("ifc_type_a"),
            "ifc_type_b": clash.get("ifc_type_b"),
            "overlap_mm": _overlap(clash),
            # No resolution is recorded at detection time.
            "status": "open",
        }
        if rows is None:
            entry.update({"strategy": None, "reason": "no clash strategy table was given"})
        else:
            entry.update(_assign(clash, rows))
        if entry.get("strategy") is not None:
            by_strategy[entry["strategy"]] = by_strategy.get(entry["strategy"], 0) + 1
            used.add(entry["table_row"])
        register.append(entry)
    return {
        "table_given": rows is not None,
        "clash_rows": len(register),
        "by_strategy": dict(sorted(by_strategy.items(), key=lambda kv: (-kv[1], kv[0]))),
        "unassigned": sum(1 for e in register if e.get("strategy") is None),
        # Rows of the project's table no clash used: a stale or misspelt
        # selector shows up here instead of silently doing nothing.
        "table_rows_unused": (
            [r["index"] for r in rows if r["index"] not in used] if rows is not None else []
        ),
        "rows": register,
    }
