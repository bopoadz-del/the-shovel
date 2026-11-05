"""Beam statics solver: M(x) derived by double integration, BCs imposed.

P1.2 of the structural-formulas order. The governing moment is a RESULT of
the boundary conditions, never a looked-up coefficient, and the four closed
forms (beam_moment_simple, beam_moment_point_load, beam_moment_fixed_udl,
beam_shear_simple) stay the oracle: CI fails when the solver disagrees with
them on their own cases, and untabulated cases (triangular cantilever,
propped cantilever under UDL) are solved, not looked up.

Statically indeterminate supports are solved through the compatibility
condition -- slope/deflection continuity equations on the integration
constants -- never through an assumed coefficient.

sympy and numpy are declared dependencies (requirements.in); no new
dependency is added here.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import sympy as sp

from app.lib.formula_registry import Param, formula

logger = logging.getLogger(__name__)

x = sp.Symbol("x", nonnegative=True)

SUPPORT_CONDITIONS: tuple[str, ...] = (
    "simply_supported",
    "fixed_fixed",
    "cantilever",
    "propped_cantilever",
    "overhang",
)

# Support-condition classifier: structural words, not project phrasing.
_PROP_RE = re.compile(r"\bpropped\s+cantilever\b", re.IGNORECASE)
_FIXED_BOTH_RE = re.compile(
    r"\bfix(?:ed)?\s+(?:at\s+)?both\s+ends\b|\bboth\s+ends\s+fix(?:ed)?\b"
    r"|\bfixed\s*[-]\s*fixed\b",
    re.IGNORECASE,
)
_OVERHANG_RE = re.compile(r"\boverhang(?:ing)?\b", re.IGNORECASE)
_CANTILEVER_RE = re.compile(r"\bcantilever(?:ed)?\b", re.IGNORECASE)
_SIMPLY_RE = re.compile(r"\bsimply\s+supported\b|\bsimple\s+beam\b", re.IGNORECASE)

# A beam-moment ask: a moment word next to beam/member language.
_MOMENT_RE = re.compile(
    r"\b(?:bending\s+)?moment(?:s)?\b|\bM(?:max)?\b", re.IGNORECASE,
)
_BEAM_WORD_RE = re.compile(
    r"\bbeam\b|\bspan\b|\bmember\b|\bsupport(?:ed)?\b|\bfixed\b|\bcantilever\b",
    re.IGNORECASE,
)
_DISTRIBUTED_RE = re.compile(
    r"\bdistributed\b|\budl\b|\buniform(?:ly)?\b|\bevenly\b", re.IGNORECASE,
)

# Dimensional-form operand resolution: a total force (kN NOT followed by a
# per-length "/") over a length (m NOT joined to "/") is an intensity.
_FORCE_TOTAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s*kN(?!\s*/)", re.IGNORECASE)
_LENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*m\b(?!\s*/)", re.IGNORECASE)

# The closed-form beam-moment family this resolver may bind.
_BEAM_MOMENT_CALCS = frozenset(
    {
        "beam_moment_simple",
        "beam_moment_point_load",
        "beam_moment_fixed_udl",
        "beam_governing_moment",
    }
)

_BOUND_CONDITION_CALC = {
    "simply_supported": "beam_moment_simple",
    "fixed_fixed": "beam_moment_fixed_udl",
    # Solved from the boundary conditions, never a canned coefficient.
    "cantilever": "beam_governing_moment",
    "propped_cantilever": "beam_governing_moment",
    "overhang": "beam_governing_moment",
}


def selected_condition(text: str | None) -> str | None:
    """The support condition a beam ask names, or None when it names none.

    Structural vocabulary only: propped cantilever first (it contains
    "cantilever"), then fixed-both-ends, overhang, cantilever, and the
    simply-supported fallback is NOT inferred -- a beam ask that names no
    condition returns None so the caller does not guess one.
    """
    raw = text or ""
    if _PROP_RE.search(raw):
        return "propped_cantilever"
    if not raw.strip():
        return None
    if _FIXED_BOTH_RE.search(raw):
        return "fixed_fixed"
    if _OVERHANG_RE.search(raw):
        return "overhang"
    if _CANTILEVER_RE.search(raw):
        return "cantilever"
    if _SIMPLY_RE.search(raw):
        return "simply_supported"
    return None


def _num(value: Any) -> sp.Expr:
    return sp.nsimplify(sp.Float(float(value)))


def _load_list(loads: Any) -> list[dict[str, Any]]:
    """Normalise the loads input: a list of load dicts, one dict, or None.

    The calc plumbing flattens a single-element list to its dict; the solver
    accepts both shapes rather than iterating a dict's keys."""
    if isinstance(loads, dict):
        return [loads]
    if loads is None:
        return []
    if isinstance(loads, (list, tuple)):
        return [load for load in loads if isinstance(load, dict)]
    return []


def _intensity(loads: list[dict[str, Any]] | None, span: sp.Expr) -> sp.Expr:
    """The transverse load intensity w(x) from a load list.

    Kinds (Macaulay step functions for anything not full-span):
      udl            w over the whole span
      partial_udl    w over [a, b]
      point          P at a
      triangular     linear ramp, w_max at a falling to 0 at b (defaults 0..L)
      moment         applied moment M at a
    """
    w: sp.Expr = sp.Integer(0)
    for load in _load_list(loads):
        kind = str(load.get("kind") or "")
        if kind == "udl":
            w += _num(load.get("w") or 0)
        elif kind == "point":
            w += _num(load.get("P") or 0) * sp.SingularityFunction(
                x, load.get("a", 0), -1,
            )
        elif kind == "moment":
            w += -_num(load.get("M") or 0) * sp.SingularityFunction(
                x, load.get("a", 0), -2,
            )
        elif kind == "partial_udl":
            a = sp.sympify(load.get("a", 0))
            b = sp.sympify(load.get("b", span))
            w += _num(load.get("w") or 0) * (
                sp.SingularityFunction(x, a, 0) - sp.SingularityFunction(x, b, 0)
            )
        elif kind == "triangular":
            a = sp.sympify(load.get("a", 0))
            b = sp.sympify(load.get("b", span))
            ramp = (b - x) / (b - a)
            w += _num(load.get("w_max") or 0) * ramp * (
                sp.SingularityFunction(x, a, 0) - sp.SingularityFunction(x, b, 0)
            )
        else:
            raise ValueError(f"unknown load kind: {kind}")
    return w


def moment_field(
    span: float,
    loads: list[dict[str, Any]] | None,
    support: str = "simply_supported",
) -> sp.Expr:
    """M(x) derived by double integration with the boundary conditions imposed.

    loads: [{"kind":"udl","w":..}, {"kind":"partial_udl","w":..,"a":..,"b":..},
            {"kind":"point","P":..,"a":..}, {"kind":"triangular","w_max":..},
            {"kind":"moment","M":..,"a":..}]  -- Macaulay step functions.

    support: simply_supported | fixed_fixed | cantilever |
             propped_cantilever | overhang. Indeterminate supports solve the
    compatibility condition on the integration constants -- no assumed
    coefficient. ``overhang`` needs support positions the generic solver
    does not guess, so it errors naming that reason.
    """
    if support == "overhang":
        raise ValueError(
            "overhang requires the support positions; the generic solver does "
            "not guess them -- supply the support locations explicitly",
        )
    L = sp.sympify(span)
    w = _intensity(loads, L)
    C1, C2 = sp.symbols("C1 C2")
    V = -sp.integrate(w, x) + C1
    M = sp.integrate(V, x) + C2

    unknowns: list[sp.Symbol] = [C1, C2]
    eqs: list[sp.Expr] = []
    if support == "simply_supported":
        eqs = [M.subs(x, 0), M.subs(x, L)]
    elif support == "cantilever":
        # Fixed at x=0, free at x=L: no moment and no shear at the free end.
        eqs = [M.subs(x, L), V.subs(x, L)]
    elif support == "fixed_fixed":
        # Fixed at 0 (slope and deflection zero) and fixed at L: the
        # compatibility conditions theta(L)=0 and delta(L)=0.
        C3, C4 = sp.symbols("C3 C4")
        theta = sp.integrate(M, x) + C3
        delta = sp.integrate(theta, x) + C4
        unknowns = [C1, C2, C3, C4]
        eqs = [
            theta.subs(x, 0), delta.subs(x, 0),
            theta.subs(x, L), delta.subs(x, L),
        ]
    elif support == "propped_cantilever":
        # Fixed at 0, pin prop at L: no moment at the prop, no deflection.
        C3, C4 = sp.symbols("C3 C4")
        theta = sp.integrate(M, x) + C3
        delta = sp.integrate(theta, x) + C4
        unknowns = [C1, C2, C3, C4]
        eqs = [
            theta.subs(x, 0), delta.subs(x, 0),
            M.subs(x, L), delta.subs(x, L),
        ]
    else:
        raise ValueError(f"unknown support condition: {support}")

    sol = sp.solve(eqs, unknowns, dict=True)
    if not sol:
        raise ValueError(f"no solution for support={support} with these loads")
    return sp.simplify(M.subs(sol[0]))


def _finite_in_range(value: Any, lo: float, hi: float) -> float | None:
    """The numeric value when it is finite and inside [lo, hi], else None."""
    try:
        v = float(sp.N(value))
    except Exception:  # noqa: BLE001 -- an unevaluable candidate is skipped; the other candidates still govern
        logger.warning(
            "beam solver: candidate point did not evaluate; skipping it",
            exc_info=True,
        )
        return None
    if not (lo <= v <= hi):
        return None
    if not math.isfinite(v):
        return None
    return v


def _critical_points(M: sp.Expr, span: float, loads: list[dict[str, Any]] | None) -> list[float]:
    """Candidate extrema positions: derivative roots, the two ends, and every
    load application point (the kinks of a piecewise-linear moment field)."""
    pts = [0.0, float(span)]
    try:
        roots = sp.solve(sp.diff(M, x), x)
    except Exception:  # noqa: BLE001 -- a piecewise moment field has no closed derivative roots; the ends and load points still hold
        logger.warning(
            "beam solver: derivative roots did not solve symbolically; "
            "using the ends and load points",
            exc_info=True,
        )
        roots = []
    for r in roots:
        v = _finite_in_range(r, 0.0, float(span))
        if v is not None:
            pts.append(v)
    for load in _load_list(loads):
        for key in ("a", "b"):
            if key in load:
                v = _finite_in_range(sp.sympify(load[key]), 0.0, float(span))
                if v is not None:
                    pts.append(v)
    return pts


def governing_moment(
    span: float,
    loads: list[dict[str, Any]] | None,
    support: str = "simply_supported",
) -> dict[str, Any]:
    """The governing (largest magnitude) moment along the beam and where.

    The divisor is a result of the boundary conditions, never an input; the
    expression is returned so the coefficients can be read off by eye.
    """
    if float(span) <= 0:
        raise ValueError("span must be > 0")
    M = moment_field(span, loads, support)
    crit = _critical_points(M, float(span), loads)
    best_m: float | None = None
    best_pos: float = 0.0
    for c in crit:
        val = _finite_in_range(M.subs(x, c), -math.inf, math.inf)
        if val is not None and (best_m is None or abs(val) > abs(best_m)):
            best_m, best_pos = val, c
    if best_m is None:
        raise ValueError("the moment field produced no finite value to govern")
    return {
        "moment_kn_m": round(best_m, 3),
        "position_m": round(best_pos, 3),
        "sense": "hogging" if best_m < 0 else "sagging",
        "expression": sp.sstr(M),
        "support": support,
    }


def reactions(
    span: float,
    loads: list[dict[str, Any]] | None,
    support: str = "simply_supported",
) -> dict[str, Any]:
    """Support reactions from the solved shear field. Absent from the
    platform before this solver -- exposed here, not assumed anywhere."""
    if float(span) <= 0:
        raise ValueError("span must be > 0")
    L = sp.sympify(span)
    w = _intensity(loads, L)
    C1, C2 = sp.symbols("C1 C2")
    V = -sp.integrate(w, x) + C1
    M = sp.integrate(V, x) + C2
    unknowns: list[sp.Symbol] = [C1, C2]
    eqs: list[sp.Expr] = []
    if support == "simply_supported":
        eqs = [M.subs(x, 0), M.subs(x, L)]
    elif support == "cantilever":
        eqs = [M.subs(x, L), V.subs(x, L)]
    elif support == "propped_cantilever":
        C3, C4 = sp.symbols("C3 C4")
        theta = sp.integrate(M, x) + C3
        delta = sp.integrate(theta, x) + C4
        unknowns = [C1, C2, C3, C4]
        eqs = [
            theta.subs(x, 0), delta.subs(x, 0),
            M.subs(x, L), delta.subs(x, L),
        ]
    elif support == "fixed_fixed":
        C3, C4 = sp.symbols("C3 C4")
        theta = sp.integrate(M, x) + C3
        delta = sp.integrate(theta, x) + C4
        unknowns = [C1, C2, C3, C4]
        eqs = [
            theta.subs(x, 0), delta.subs(x, 0),
            theta.subs(x, L), delta.subs(x, L),
        ]
    elif support == "overhang":
        raise ValueError(
            "overhang requires the support positions; the generic solver does "
            "not guess them -- supply the support locations explicitly",
        )
    else:
        raise ValueError(f"unknown support condition: {support}")
    sol = sp.solve(eqs, unknowns, dict=True)
    if not sol:
        raise ValueError(f"no solution for support={support} with these loads")
    V_solved = sp.simplify(V.subs(sol[0]))
    M_solved = sp.simplify(M.subs(sol[0]))
    # Reactions at the beam ends, in the shear-field sign convention.
    shear_left = float(sp.N(V_solved.subs(x, 0)))
    shear_right = float(sp.N(V_solved.subs(x, L)))
    return {
        "reaction_left_kn": round(shear_left, 3),
        "reaction_right_kn": round(-shear_right, 3),
        "moment_left_kn_m": round(float(sp.N(M_solved.subs(x, 0))), 3),
        "moment_right_kn_m": round(float(sp.N(M_solved.subs(x, L))), 3),
        "support": support,
    }


def _ask_text(params: dict[str, Any] | None) -> str:
    """The prose a resolver may read: the ask blob keys of a calc call."""
    out: list[str] = []
    for key in ("text", "formula", "message", "query", "ask"):
        val = (params or {}).get(key)
        if val:
            out.append(str(val))
    return " ".join(out)


def _length_and_force(text: str) -> tuple[float | None, float | None]:
    """(length m, total force kN) read by dimensional form from the ask."""
    span_m: float | None = None
    total_kn: float | None = None
    for m in _LENGTH_RE.finditer(text):
        try:
            span_m = float(m.group(1))
        except ValueError:
            continue
        break
    for m in _FORCE_TOTAL_RE.finditer(text):
        try:
            total_kn = float(m.group(1))
        except ValueError:
            continue
        break
    return span_m, total_kn


def _param_number(params: dict[str, Any], *names: str) -> float | None:
    for name in names:
        try:
            return float(params.get(name))
        except (TypeError, ValueError):
            continue
    return None


def resolve_beam_moment_calc(
    name: str | None,
    params: dict[str, Any] | None,
) -> tuple[str | None, dict[str, Any]]:
    """Bind a beam-moment ask to the calculator its support condition names.

    A recognised condition BINDS the selection: cantilever, propped and
    overhang asks go to the boundary-condition solver; simply-supported and
    fixed-fixed distributed asks keep their closed forms. A classified
    condition never falls through to another condition's coefficient -- when
    the load the binding needs is missing, a ValueError names the reason and
    the caller surfaces it as the error.

    A distributed total over a length (force / length, read by dimensional
    form, never by phrasing) resolves to the intensity the calculators need,
    so "100 kN evenly distributed on 10 m" and any rephrasing of it reach a
    number without an asks-for-inputs turn.
    """
    out = dict(params or {})
    text = _ask_text(out)
    cond = selected_condition(text)
    named = str(name or "").strip() or None
    if cond is None:
        return name, out
    if named is not None and named not in _BEAM_MOMENT_CALCS:
        return name, out
    if not text.strip() or not _MOMENT_RE.search(text):
        return name, out
    if not _BEAM_WORD_RE.search(text):
        return name, out

    span_m = _param_number(out, "span_m", "span", "length_m", "length")
    total_kn = _param_number(
        out, "total_load_kn", "total_kn", "udl_total_kn", "load_kn",
    )
    intensity = _param_number(out, "udl_w_kn_m", "w_kn_m", "w", "intensity_kn_m")
    if span_m is None or total_kn is None:
        text_span, text_total = _length_and_force(text)
        if span_m is None:
            span_m = text_span
        if total_kn is None:
            total_kn = text_total
    if intensity is None and _DISTRIBUTED_RE.search(text) and (
        span_m is not None and total_kn is not None and span_m > 0
    ):
        intensity = total_kn / span_m

    point_kn = _param_number(out, "point_load_kn", "P")
    distributed = intensity is not None
    supplied_loads = out.get("loads")
    if isinstance(supplied_loads, (list, tuple)) and supplied_loads:
        # An explicit load list is already resolved -- respect it and route
        # by the classified condition, never re-derive it from prose.
        out["support"] = cond
        return "beam_governing_moment", out
    if not distributed and point_kn is None:
        raise ValueError(
            f"a {cond} moment ask needs a load: a distributed intensity "
            "(or a total over the span) or a point load",
        )

    # The two supports whose distributed closed forms exist keep them; every
    # other condition -- and any point load -- is solved from the boundary
    # conditions.
    if cond in ("simply_supported", "fixed_fixed") and distributed and point_kn is None:
        bound = _BOUND_CONDITION_CALC[cond]
        out["udl_w_kn_m"] = intensity
        if span_m is not None:
            out["span_m"] = span_m
        return bound, out

    loads: list[dict[str, Any]] = []
    if distributed:
        loads.append({"kind": "udl", "w": intensity})
    if point_kn is not None:
        position = _param_number(out, "distance_from_left_m", "a")
        if position is None:
            raise ValueError(
                f"a point load on a {cond} needs its position along the span "
                "-- supply the load position",
            )
        loads.append({"kind": "point", "P": point_kn, "a": position})
    out["loads"] = loads
    out["support"] = cond
    if span_m is not None:
        out["span_m"] = span_m
    return "beam_governing_moment", out


@formula(
    owner="design",
    display_name="Beam governing bending moment",
    description=(
        "Governing bending moment of a beam under the given loads, derived "
        "from the boundary conditions of the support condition -- solved, "
        "never looked up."
    ),
    inputs={
        "span_m": Param("m", 0, 200, label="span"),
        "support": Param("-", label="support condition"),
        "loads": Param("-", kind="object", label="load list"),
    },
    outputs={
        "moment_kn_m": "kN.m",
        "position_m": "m",
        "sense": "-",
        "support": "-",
        "expression": "-",
    },
)
def beam_governing_moment(
    span_m: float,
    support: str = "simply_supported",
    loads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The governing moment of a beam, solved from its boundary conditions."""
    try:
        out = governing_moment(float(span_m), loads, support)
    except ValueError as exc:
        return {"error": str(exc)}
    return {
        "moment_kn_m": out["moment_kn_m"],
        "position_m": out["position_m"],
        "sense": out["sense"],
        "support": out["support"],
        "expression": out["expression"],
    }


@formula(
    owner="design",
    display_name="Beam support reactions",
    description=(
        "Support reactions of a beam under the given loads, solved from the "
        "boundary conditions of the support condition."
    ),
    inputs={
        "span_m": Param("m", 0, 200, label="span"),
        "support": Param("-", label="support condition"),
        "loads": Param("-", kind="object", label="load list"),
    },
    outputs={
        "reaction_left_kn": "kN",
        "reaction_right_kn": "kN",
        "moment_left_kn_m": "kN.m",
        "moment_right_kn_m": "kN.m",
        "support": "-",
    },
)
def beam_reactions(
    span_m: float,
    support: str = "simply_supported",
    loads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Support reactions of a beam, solved from its boundary conditions."""
    try:
        return reactions(float(span_m), loads, support)
    except ValueError as exc:
        return {"error": str(exc)}
