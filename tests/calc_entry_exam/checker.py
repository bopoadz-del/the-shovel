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

  C  no unstated constant: a NAMED numeric constant the calculator's own
     body reads as an arithmetic operand on the lines the run executed (a
     local assigned a number, or a module-level number) shapes the result as
     much as a default does, so the envelope must state it -- a disclosure
     row naming it or carrying its value, or the note naming it. Found by
     tracing the run, never from the product's disclosure code. Additive and
     multiplicative identities (0, 1), accumulators and inline coefficients
     of a written formula (the 8 of wL^2/8) are formula structure, not
     constants; tables of rates are D2's audit and are not probed here.

Where a calculator cannot be exercised (a required input that is not a
number, or no in-range call that runs) the sweep reports the stage that
stopped it -- ``baseline`` -- rather than skipping it silently.

Run: ``python -m tests.calc_entry_exam.checker``
"""

from __future__ import annotations

import ast
import inspect
import math
import re
import sys
import textwrap
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


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _literal(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and _is_num(node.value):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _literal(node.operand)
        if inner is not None:
            return -inner if isinstance(node.op, ast.USub) else inner
    return None


def _operand_names(node: ast.AST, literals: list[float] | None = None) -> set[str]:
    """Names read as an arithmetic operand anywhere under ``node``; inline
    number operands are appended to ``literals`` (an exponent is formula
    structure, not a constant, and is left out)."""
    out: set[str] = set()
    for n in ast.walk(node):
        sides: list[ast.AST] = []
        if isinstance(n, ast.BinOp):
            sides = [n.left] if isinstance(n.op, ast.Pow) else [n.left, n.right]
        elif isinstance(n, ast.AugAssign):
            sides = [n.value]
        for s in sides:
            lit = _literal(s)
            if lit is not None and literals is not None:
                literals.append(lit)
            while isinstance(s, ast.UnaryOp):
                s = s.operand
            if isinstance(s, ast.Name):
                out.add(s.id)
    return out


def _structural(v: float) -> bool:
    """Identities and decimal unit scales (0, 1, 10^k) are not constants."""
    if v == 0:
        return True
    k = math.log10(abs(v))
    return abs(k - round(k)) < 1e-12


def named_constants(fn: Callable[..., Any], lines_run: set[int] | None = None) -> dict[str, float]:
    """Named numeric constants ``fn``'s own body reads as arithmetic operands.

    ``lines_run`` (absolute line numbers) restricts the probe to what one run
    executed; None reads the whole body.
    """
    fn = inspect.unwrap(fn)
    code = getattr(fn, "__code__", None)
    try:
        src = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return {}
    tree = ast.parse(src)
    func = tree.body[0] if tree.body else None
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) or code is None:
        return {}
    # getsource starts at the first decorator line, which is co_firstlineno.
    first = code.co_firstlineno - 1  # ast line -> file line offset
    stmts = [s for top in func.body for s in ast.walk(top) if isinstance(s, ast.stmt)]

    def ran(node: ast.AST) -> bool:
        return lines_run is None or (getattr(node, "lineno", 0) + first) in lines_run

    params = set(inspect.signature(fn).parameters)
    operands: set[str] = set()
    inline: list[float] = []
    accumulators: set[str] = set()
    local_names: set[str] = set()
    assigned: dict[str, list[float | None]] = {}  # executed assignments only
    for s in stmts:
        if isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name):
            accumulators.add(s.target.id)
        targets: list[ast.AST] = []
        value: ast.AST | None = None
        if isinstance(s, ast.Assign):
            targets, value = list(s.targets), s.value
        elif isinstance(s, ast.AnnAssign):
            targets, value = [s.target], s.value
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            targets = [s.target]
        for t in targets:
            for n in ast.walk(t):
                if not isinstance(n, ast.Name):
                    continue
                local_names.add(n.id)
                if ran(s):
                    lit = _literal(value) if t is n and value is not None else None
                    assigned.setdefault(n.id, []).append(lit)
        if ran(s):
            # A statement's operands, but not those of a nested block body:
            # walk only the expression fields of this statement.
            for field_name, value in ast.iter_fields(s):
                if field_name in ("body", "orelse", "finalbody", "handlers"):
                    continue
                for v in value if isinstance(value, list) else [value]:
                    if isinstance(v, ast.AST):
                        operands |= _operand_names(v, inline)
    out: dict[str, float] = {}
    for name in sorted(operands - params - accumulators):
        if name in local_names:
            values = assigned.get(name, [])
            if len(values) == 1 and values[0] is not None:
                out[name] = values[0]
            continue
        g = getattr(fn, "__globals__", {}).get(name)
        if _is_num(g):
            out[name] = float(g)
    for v in inline:
        out.setdefault(f"literal {v:g}", v)
    return {k: v for k, v in out.items() if not _structural(v)}


def traced_lines(call: Callable[[], Any], fn: Callable[..., Any]) -> tuple[Any, set[int]]:
    """Run ``call`` and return (its value, the lines of ``fn`` it executed)."""
    code = getattr(inspect.unwrap(fn), "__code__", None)
    lines: set[int] = set()

    def local(frame, event, arg):
        if event == "line":
            lines.add(frame.f_lineno)
        return local

    def tracer(frame, event, arg):
        return local if frame.f_code is code else None

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        value = call()
    finally:
        sys.settrace(previous)
    return value, lines


def _strings(obj: Any) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, (list, tuple)):
        return [s for v in obj for s in _strings(v)]
    return []


def check_stated(env: dict[str, Any], constants: dict[str, float]) -> tuple[bool, str]:
    """Rule C: every named constant the run read is stated in the envelope."""
    result = env.get("result") if isinstance(env, dict) else None
    result = result if isinstance(result, dict) else {}
    rows = [r for key in ("defaults_used", "assumptions", "constants_used")
            for r in (result.get(key) or []) if isinstance(r, dict)]
    names = {str(r.get(k)).casefold() for r in rows for k in ("parameter", "name") if r.get(k)}
    values = [float(r["value"]) for r in rows if _is_num(r.get("value"))]
    note = " ".join(str(x) for x in (env.get("note"), result.get("note")) if x).casefold()
    written = " ".join(_strings(env)).casefold()
    silent = []
    for name, value in sorted(constants.items()):
        bare = name.strip("_").casefold()
        if bare in names or name.casefold() in names:
            continue
        if any(math.isclose(value, v, rel_tol=_REL, abs_tol=1e-12) for v in values):
            continue
        if not name.startswith("literal ") and (bare in note or bare.replace("_", " ") in note):
            continue
        # Written in the envelope's own words (its formula, basis or note).
        if re.search(rf"(?<![\d.]){re.escape(f'{value:g}')}(?![\d])", written):
            continue
        silent.append(name if name.startswith("literal ") else f"{name}={value:g}")
    if silent:
        return False, "constant used unstated: " + ", ".join(silent)
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
        traced, lines = traced_lines(lambda: _run(run, spec.name, call), spec.fn)
        if not lines:
            rows.append({"calc": spec.name, "rule": "C", "passed": False,
                         "why": "the door ran without executing the calculator's body"})
        else:
            consts = named_constants(spec.fn, lines)
            if consts:
                ok, why = check_stated(traced, consts)
                rows.append({"calc": spec.name, "rule": "C", "passed": ok, "why": why})
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
