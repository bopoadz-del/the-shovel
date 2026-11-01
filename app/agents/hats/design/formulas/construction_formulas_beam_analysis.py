"""Formulas from app.lib.construction_formulas_beam_analysis owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula


@formula(
    owner='design',
    display_name='Beam bending moment under a uniform load',
    description='Maximum bending moment of a simply supported beam under a uniformly distributed load.',
    inputs={
        'udl_w_kn_m': Param('kN/m', 0, 100000, label='uniform load'),
        'span_m': Param('m', 0, 200, label='span'),
    },
    outputs={'max_moment_kn_m': 'kN.m'},
)
def beam_moment_simple(udl_w_kn_m: float, span_m: float) -> dict:
    """Simply-supported beam under a uniform load: Mmax = w*L^2/8 at midspan.
    (This is also the "uniform load" case — beam_moment_uniform is the same
    statics; the fixed-end variant is beam_moment_fixed_udl.)"""
    w = float(udl_w_kn_m)
    L = float(span_m)
    if L <= 0:
        return {"error": "span_m must be > 0."}
    if w < 0:
        return {"error": "udl_w_kn_m must be >= 0."}
    m = w * L * L / 8.0
    return {
        "max_moment_kn_m": round(m, 3),
        "location": "midspan",
        "standard": "statics (SS, UDL)",
        "note": f"Mmax = w*L^2/8 = {w}*{L}^2/8 = {m:.3f} kN.m at midspan.",
    }

@formula(
    owner='design',
    display_name='Beam bending moment under a point load',
    description='Maximum bending moment of a simply supported beam under one point load at a given position.',
    inputs={
        'point_load_kn': Param('kN', 0, 1e6, label='point load'),
        'span_m': Param('m', 0, 200, label='span'),
        'distance_from_left_m': Param('m', 0, 200, label='distance of the load from the left support'),
    },
    outputs={'max_moment_kn_m': 'kN.m'},
)
def beam_moment_point_load(
    point_load_kn: float,
    span_m: float,
    distance_from_left_m: float = None,
) -> dict:
    """Simply-supported beam under a single point load. Central: M = P*L/4.
    Off-centre at a from the left support: M = P*a*b/L (b = L-a), under the load."""
    P = float(point_load_kn)
    L = float(span_m)
    if L <= 0:
        return {"error": "span_m must be > 0."}
    if P < 0:
        return {"error": "point_load_kn must be >= 0."}
    if distance_from_left_m is None:
        m = P * L / 4.0
        expr = f"P*L/4 = {P}*{L}/4"
        loc = "midspan"
    else:
        a = float(distance_from_left_m)
        if a < 0 or a > L:
            return {"error": "distance_from_left_m must lie on the span [0, span_m]."}
        b = L - a
        m = P * a * b / L
        expr = f"P*a*b/L = {P}*{a}*{b}/{L}"
        loc = f"under load at {a} m"
    return {
        "max_moment_kn_m": round(m, 3),
        "location": loc,
        "standard": "statics (SS, point load)",
        "note": f"M = {expr} = {m:.3f} kN.m ({loc}).",
    }

@formula(
    owner='design',
    display_name='Beam shear force',
    description='Maximum shear of a simply supported beam under a distributed load and an optional central point load.',
    inputs={
        'udl_w_kn_m': Param('kN/m', 0, 100000, label='uniform load'),
        'span_m': Param('m', 0, 200, label='span'),
        'central_point_load_kn': Param('kN', 0, 1e6, label='central point load'),
    },
    outputs={'max_shear_kn': 'kN'},
    required_groups=(('udl_w_kn_m', 'central_point_load_kn'), ('span_m',)),
)
def beam_shear_simple(
    udl_w_kn_m: float = 0.0,
    span_m: float = 0.0,
    central_point_load_kn: float = 0.0,
) -> dict:
    """Max shear at the support of a simply-supported beam: V = w*L/2 (+ P/2 for
    a central point load)."""
    w = float(udl_w_kn_m)
    L = float(span_m)
    P = float(central_point_load_kn)
    if L < 0 or w < 0 or P < 0:
        return {"error": "span_m, udl_w_kn_m and central_point_load_kn must be >= 0."}
    v = w * L / 2.0 + P / 2.0
    return {
        "max_shear_kn": round(v, 3),
        "location": "support",
        "standard": "statics (SS)",
        "note": f"Vmax = w*L/2 + P/2 = {w}*{L}/2 + {P}/2 = {v:.3f} kN at support.",
    }

@formula(
    owner='design',
    display_name='Fixed-end beam moments under a uniform load',
    description='Support and midspan bending moments of a fixed-ended beam under a uniformly distributed load.',
    inputs={
        'udl_w_kn_m': Param('kN/m', 0, 100000, label='uniform load'),
        'span_m': Param('m', 0, 200, label='span'),
    },
    outputs={'support_moment_kn_m': 'kN.m', 'midspan_moment_kn_m': 'kN.m', 'max_moment_kn_m': 'kN.m'},
)
def beam_moment_fixed_udl(udl_w_kn_m: float, span_m: float) -> dict:
    """Beam fixed at both ends under a uniform load: hogging M = w*L^2/12 at the
    supports (the max), sagging M = w*L^2/24 at midspan."""
    w = float(udl_w_kn_m)
    L = float(span_m)
    if L <= 0:
        return {"error": "span_m must be > 0."}
    if w < 0:
        return {"error": "udl_w_kn_m must be >= 0."}
    m_support = w * L * L / 12.0
    m_mid = w * L * L / 24.0
    return {
        "support_moment_kn_m": round(m_support, 3),
        "midspan_moment_kn_m": round(m_mid, 3),
        "max_moment_kn_m": round(m_support, 3),
        "standard": "statics (fixed-fixed, UDL)",
        "note": (f"Support (hogging) = w*L^2/12 = {w}*{L}^2/12 = {m_support:.3f}; "
                 f"midspan (sagging) = w*L^2/24 = {m_mid:.3f} kN.m."),
    }

ADDITIONAL_CALCULATORS = {
    "beam_moment_simple": beam_moment_simple,
    "beam_moment_point_load": beam_moment_point_load,
    "beam_shear_simple": beam_shear_simple,
    "beam_moment_fixed_udl": beam_moment_fixed_udl,
}
