"""Exam D1: calculator entry-point rules, over EVERY registered calculator.

Each calculator reached from chat goes through one door,
``construction_formulas.run_calculation(name, params)``. The order's three
rules are checked at that door for every calculator the registry declares:

  M  every required input asked for when missing: drop one required input
     from a call that runs, and the envelope must not be a success, must
     carry no result, and must name the dropped input (its name or label).
  U  units converted at the door: write one numeric input in another unit of
     the same quantity, and the result must equal the run in the declared
     unit. The other unit and its factor come from THIS file's table of SI
     definitions, never from the product's conversion code.
  D  no default used silently: every optional input with a real default that
     the call did not give must be disclosed in ``result["defaults_used"]``.
     The expected set is read from the function signature here, never from
     the product's disclosure code.

Where a calculator cannot be exercised (a required input that is not a
number, or no in-range call that runs) the sweep reports the stage that
stopped it -- ``baseline`` -- rather than skipping it silently.

Run: ``python -m tests.calc_entry_exam.checker``
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

#: declared unit -> (another unit of the same quantity, value_in_other per
#: one declared unit). SI definitions only; deliberately small and exact.
OTHER_UNIT: dict[str, tuple[str, float]] = {
    "m": ("mm", 1000.0),
    "mm": ("m", 0.001),
    "km": ("m", 1000.0),
    "m2": ("cm2", 10000.0),
    "mm2": ("cm2", 0.01),
    "t": ("kg", 1000.0),
    "kg": ("t", 0.001),
    "kN": ("N", 1000.0),
    "kN/m": ("N/m", 1000.0),
    "MPa": ("kPa", 1000.0),
    "h": ("min", 60.0),
    "min": ("s", 60.0),
}

NUMERIC = ("number", "integer", "series")
FAIL_STATUSES = ("error", "needs_input", "not_run")
_REL = 1e-6


# ---------------------------------------------------------------- pure checks
def _numbers(obj: Any, prefix: str = "") -> dict[str, float]:
    """Every numeric leaf of a result, keyed by path (disclosure rows are
    not results)."""
    out: dict[str, float] = {}
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        if math.isfinite(obj):
            out[prefix] = float(obj)
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("defaults_used", "conversions", "note"):
                continue
            out.update(_numbers(v, f"{prefix}.{k}"))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(_numbers(v, f"{prefix}[{i}]"))
    return out


def check_missing(env: dict[str, Any], name: str, label: str) -> tuple[bool, str]:
    """Rule M on one envelope from a call that lacked input ``name``."""
    if not isinstance(env, dict):
        return False, "no envelope"
    if env.get("status") == "success" or env.get("status") not in FAIL_STATUSES:
        return False, f"ran without {name} (status {env.get('status')!r})"
    if _numbers(env.get("result")):
        return False, f"carries a result without {name}"
    said = " ".join(str(v) for v in (env.get("needs_input"), env.get("missing"),
                                      env.get("required"), env.get("error"),
                                      env.get("reason"))).casefold()
    words = {name.casefold(), name.replace("_", " ").casefold()}
    if label:
        words.add(label.casefold())
    if not any(w and w in said for w in words):
        return False, f"does not name the missing {name}"
    return True, ""


def check_same_result(base: dict[str, Any], other: dict[str, Any], what: str) -> tuple[bool, str]:
    """Rule U: the run with a value in another unit equals the declared-unit run."""
    if not isinstance(other, dict) or other.get("status") != "success":
        status = other.get("status") if isinstance(other, dict) else None
        return False, f"{what}: not run (status {status!r})"
    a, b = _numbers(base.get("result")), _numbers(other.get("result"))
    if not a:
        return False, f"{what}: baseline has no numeric result"
    for key, x in a.items():
        y = b.get(key)
        if y is None:
            return False, f"{what}: result {key} missing"
        if not math.isclose(x, y, rel_tol=_REL, abs_tol=1e-9):
            return False, f"{what}: result {key} {y:g} != {x:g}"
    return True, ""


def check_disclosed(env: dict[str, Any], expected: set[str]) -> tuple[bool, str]:
    """Rule D: every default the call took is named in defaults_used."""
    result = env.get("result") if isinstance(env, dict) else None
    rows = result.get("defaults_used") if isinstance(result, dict) else None
    named = {str(r.get("parameter")) for r in rows or [] if isinstance(r, dict)}
    silent = sorted(expected - named)
    if silent:
        return False, "default used silently: " + ", ".join(silent)
    return True, ""


# ---------------------------------------------------------------- the sweep
def _value(p: Any) -> Any:
    lo, hi = p.lo, p.hi
    if lo is not None and hi is not None:
        v = lo + (hi - lo) * 0.37
    elif lo is not None:
        v = max(lo, 0.0) + 3.7
    elif hi is not None:
        v = min(hi, 3.7)
    else:
        v = 3.7
    if p.kind == "integer":
        v = max(math.ceil(lo) if lo is not None else 1, round(v))
        if hi is not None:
            v = min(v, math.floor(hi))
        return int(v)
    if p.kind == "series":
        return [v, v, v]
    return round(v, 6)


def _write_in(value: Any, unit: str, factor: float) -> Any:
    if isinstance(value, list):
        return [f"{x * factor:.12g} {unit}" for x in value]
    return f"{value * factor:.12g} {unit}"


def _real_defaults(fn: Callable[..., Any], given: set[str]) -> set[str]:
    out = set()
    for key, p in inspect.signature(fn).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) or key in given:
            continue
        if p.default is inspect.Parameter.empty or p.default is None or p.default == "":
            continue
        out.add(key)
    return out


def _run(run: Callable[..., dict], name: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        env = run(name, dict(params))
    except Exception as exc:  # noqa: BLE001 -- a crash is a failed check, not a crashed exam
        return {"status": "crashed", "error": f"{type(exc).__name__}: {exc}"}
    return env if isinstance(env, dict) else {"status": "bad_envelope"}


def sweep(specs: list[Any], parameters: Callable[[Any], dict], run: Callable[..., dict]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        params = parameters(spec)
        required = {k: p for k, p in params.items() if p.required}
        blocked = sorted(k for k, p in required.items() if p.kind not in NUMERIC)
        if blocked:
            rows.append({"calc": spec.name, "rule": "baseline", "passed": False,
                         "why": "non-numeric required input: " + ", ".join(blocked)})
            continue
        call = {k: _value(p) for k, p in required.items()}
        for group in getattr(spec, "required_groups", ()) or ():
            if not any(k in call for k in group):
                first = next((k for k in group if k in params and params[k].kind in NUMERIC), None)
                if first:
                    call[first] = _value(params[first])
        base = _run(run, spec.name, call)
        if base.get("status") != "success":
            # Some calculators need one of a pair of optional inputs (length
            # OR weight): retry with every numeric input given.
            call = {**{k: _value(p) for k, p in params.items() if p.kind in NUMERIC}, **call}
            base = _run(run, spec.name, call)
        if base.get("status") != "success":
            rows.append({"calc": spec.name, "rule": "baseline", "passed": False,
                         "why": f"in-range call not run: {base.get('status')} {str(base.get('error') or '')[:120]}"})
            continue
        rows.append({"calc": spec.name, "rule": "baseline", "passed": True, "why": ""})
        for key, p in required.items():
            if key not in call:
                continue
            env = _run(run, spec.name, {k: v for k, v in call.items() if k != key})
            ok, why = check_missing(env, key, p.label)
            rows.append({"calc": spec.name, "rule": "M", "input": key, "passed": ok, "why": why})
        for key, value in call.items():
            p = params[key]
            if p.unit not in OTHER_UNIT:
                continue
            unit, factor = OTHER_UNIT[p.unit]
            env = _run(run, spec.name, {**call, key: _write_in(value, unit, factor)})
            ok, why = check_same_result(base, env, f"{key} in {unit}")
            rows.append({"calc": spec.name, "rule": "U", "input": key, "passed": ok, "why": why})
        expected = _real_defaults(spec.fn, set(call))
        if expected:
            ok, why = check_disclosed(base, expected)
            rows.append({"calc": spec.name, "rule": "D", "passed": ok, "why": why})
    by_rule: dict[str, list[int]] = {}
    for r in rows:
        k = by_rule.setdefault(r["rule"], [0, 0])
        k[0] += int(r["passed"])
        k[1] += 1
    return {"passed": sum(r["passed"] for r in rows), "total": len(rows),
            "by_rule": by_rule, "rows": rows, "calculators": len(specs)}


def product_sweep() -> dict[str, Any]:
    from app.lib import construction_formulas, formula_registry

    return sweep(formula_registry.all_specs(), formula_registry.parameters,
                 construction_formulas.run_calculation)


def main() -> int:
    report = product_sweep()
    print(f"D1 calculator entry exam: {report['passed']}/{report['total']} "
          f"over {report['calculators']} calculators")
    for rule, (p, t) in sorted(report["by_rule"].items()):
        print(f"  {rule}: {p}/{t}")
    for r in report["rows"]:
        if not r["passed"]:
            print(f"  FAIL {r['rule']} {r['calc']}{'.' + r['input'] if r.get('input') else ''}: {r['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
