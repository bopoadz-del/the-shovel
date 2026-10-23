"""Formulas from app.lib.construction_formulas_structural_rc owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math
import re


_ACI = "aci"

_EC = "eurocode"

def _norm_code(code: str) -> str:
    c = (code or "").strip().lower()
    if c in ("ec", "eurocode", "en", "en1992"):
        return _EC
    return _ACI


# ACI 318-19 Table 25.4.2.5. ψg holds for fy up to the stated limit.
_GRADE_FACTOR = ((420.0, 1.0), (550.0, 1.15), (float("inf"), 1.3))


def _grade_factor(fy_mpa: float) -> float:
    fy = float(fy_mpa)
    for limit, factor in _GRADE_FACTOR:
        if fy <= limit:
            return factor
    return _GRADE_FACTOR[-1][1]

@formula(
    owner='design',
    display_name='Concrete beam bending capacity',
    description='Design flexural capacity of a singly reinforced rectangular concrete beam.',
    inputs={
        'steel_area_mm2': Param('mm2', 0, 1e7, label='steel area'),
        'fy_mpa': Param('MPa', 100, 1500, label='reinforcement yield strength', grade='rebar'),
        'width_mm': Param('mm', 50, 10000, label='section width'),
        'eff_depth_mm': Param('mm', 20, 10000, label='effective depth'),
        'fc_mpa': Param('MPa', 5, 150, label='concrete compressive strength', grade='concrete'),
        'code': Param('-', label='design code'),
    },
    outputs={'moment_capacity_kn_m': 'kN.m'},
)
def rc_beam_moment_capacity(
    steel_area_mm2: float,
    fy_mpa: float,
    width_mm: float,
    eff_depth_mm: float,
    fc_mpa: float,
    code: str = _ACI,
) -> dict:
    """Design flexural capacity of a singly-reinforced rectangular RC beam.

    ACI 318-19: a = As*fy/(0.85*fc*b); phi.Mn = 0.9*As*fy*(d - a/2).
    EN 1992-1-1: fcd=fc/1.5, fyd=fy/1.15; x = As*fyd/(0.8*b*fcd); z = d - 0.4x;
    MRd = As*fyd*z.
    """
    cc = _norm_code(code)
    As, b, d = float(steel_area_mm2), float(width_mm), float(eff_depth_mm)
    if cc == _EC:
        fcd, fyd = fc_mpa / 1.5, fy_mpa / 1.15
        x = As * fyd / (0.8 * b * fcd)
        z = d - 0.4 * x
        m_nmm = As * fyd * z
        std = "EN 1992-1-1 §6.1 (gc=1.5, gs=1.15)"
        note = (f"fcd={fcd:.1f}, fyd={fyd:.1f}; x=As*fyd/(0.8*b*fcd)={x:.1f} mm; "
                f"z=d-0.4x={z:.1f} mm; MRd=As*fyd*z={m_nmm/1e6:.2f} kN.m")
        cap = m_nmm / 1e6
    else:
        a = As * fy_mpa / (0.85 * fc_mpa * b)
        m_nmm = 0.9 * As * fy_mpa * (d - a / 2.0)
        std = "ACI 318-19 §22.2 (phi=0.9)"
        note = (f"a=As*fy/(0.85*fc*b)={a:.1f} mm; "
                f"phi.Mn=0.9*As*fy*(d-a/2)={m_nmm/1e6:.2f} kN.m")
        cap = m_nmm / 1e6
    return {
        "code": cc,
        "moment_capacity_kn_m": round(cap, 2),
        "standard": std,
        "note": note,
    }

@formula(
    owner='design',
    display_name='Concrete section shear capacity',
    description='Shear capacity of a concrete section without shear reinforcement.',
    inputs={
        'width_mm': Param('mm', 50, 10000, label='section width'),
        'eff_depth_mm': Param('mm', 20, 10000, label='effective depth'),
        'fc_mpa': Param('MPa', 5, 150, label='concrete compressive strength', grade='concrete'),
        'code': Param('-', label='design code'),
        'rho_l': Param('fraction', 0, 0.1, label='longitudinal reinforcement ratio'),
    },
    outputs={'shear_capacity_kn': 'kN'},
)
def rc_beam_shear_capacity(
    width_mm: float,
    eff_depth_mm: float,
    fc_mpa: float,
    code: str = _ACI,
    rho_l: float = 0.01,
) -> dict:
    """Concrete shear capacity (no shear reinforcement).

    ACI 318-19 Table 22.5.5.1, Av < Av,min: Vc = 0.66*λs*(ρw)^(1/3)*√fc'*bw*d,
    λs = √(2/(1+0.004*d)) ≤ 1 (d in mm), √fc' ≤ 8.3 MPa, and Vc not above
    0.42*√fc'*bw*d. φ = 0.75. The 0.17√fc' equation is the row for members
    that already have Av ≥ Av,min, which this calculator does not.
    EN 1992-1-1 §6.2.2: VRd,c = CRd,c*k*(100*rho_l*fck)^(1/3)*bw*d,
    CRd,c=0.12, k=1+sqrt(200/d)<=2.0, rho_l<=0.02.
    """
    cc = _norm_code(code)
    bw, d = float(width_mm), float(eff_depth_mm)
    if cc == _EC:
        k = min(1.0 + math.sqrt(200.0 / d), 2.0)
        rho = min(rho_l, 0.02)
        vrd_n = 0.12 * k * (100.0 * rho * fc_mpa) ** (1.0 / 3.0) * bw * d
        vmin = 0.035 * (k ** 1.5) * (fc_mpa ** 0.5)  # EN 1992-1-1 Eq. 6.3N
        vrd_n = max(vrd_n, vmin * bw * d)
        std = "EN 1992-1-1 §6.2.2 (CRd,c=0.12)"
        note = (f"k=1+sqrt(200/d)={k:.3f}; VRd,c=0.12*k*(100*rho*fck)^(1/3)*bw*d "
                f"= {vrd_n/1000.0:.2f} kN")
        cap = vrd_n / 1000.0
    else:
        sqrt_fc = min(math.sqrt(fc_mpa), 8.3)
        lam_s = min(1.0, math.sqrt(2.0 / (1.0 + 0.004 * d)))
        rho = max(float(rho_l), 0.0)
        vc = 0.66 * lam_s * (rho ** (1.0 / 3.0)) * sqrt_fc
        vc = min(vc, 0.42 * sqrt_fc)
        vc_n = 0.75 * vc * bw * d
        std = "ACI 318-19 Table 22.5.5.1 (Av < Av,min, phi=0.75)"
        note = (f"phi.Vc=0.75*0.66*lambda_s*(rho)^(1/3)*sqrt(fc)*bw*d "
                f"(lambda_s={lam_s:.3f}) = {vc_n/1000.0:.2f} kN")
        cap = vc_n / 1000.0
    return {
        "code": cc,
        "shear_capacity_kn": round(cap, 2),
        "standard": std,
        "note": note,
    }

# Span/depth (L/ratio) minimum-thickness ratios by support condition.
_ACI_RATIOS = {"simply_supported": 20.0, "one_end_continuous": 24.0,
               "both_ends_continuous": 28.0, "cantilever": 10.0}

_EC_RATIOS = {"simply_supported": 20.0, "one_end_continuous": 26.0,
              "both_ends_continuous": 30.0, "cantilever": 8.0}

@formula(
    owner='design',
    display_name='Minimum slab thickness',
    description='Minimum one-way slab thickness for deflection control, from span and support condition.',
    inputs={
        'span_mm': Param('mm', 100, 200000, label='span'),
        'support_condition': Param('-', label='support condition'),
        'code': Param('-', label='design code'),
        'fy_mpa': Param('MPa', 100, 1500, label='reinforcement yield strength', grade='rebar'),
        'cover_mm': Param('mm', 0, 200, label='nominal cover'),
        'bar_diameter_mm': Param('mm', 0, 60, label='bar diameter'),
    },
    outputs={'min_thickness_mm': 'mm', 'support_condition': '-'},
)
def slab_thickness_min(
    span_mm: float,
    support_condition: str = "simply_supported",
    code: str = _ACI,
    fy_mpa: float = 420.0,
    cover_mm: float = 25.0,
    bar_diameter_mm: float = 0.0,
) -> dict:
    """Minimum one-way slab thickness for deflection control (span/ratio).

    ACI 318-19 Table 7.3.1.1 (fy=420): SS L/20, one-end L/24, both-ends L/28,
    cantilever L/10; x(0.4 + fy/700) for other fy. Those ratios are overall
    thickness.
    EN 1992-1-1 §7.4.2 basic span/depth (effective depth): SS 20, end 26,
    interior 30, cantilever 8. Thickness is that depth plus nominal cover
    plus half the bar diameter. Cover defaults to §4.4.1 XC1 class S4:
    cmin,dur 15 mm + Δcdev 10 mm.
    """
    cc = _norm_code(code)
    L = float(span_mm)
    sc = (support_condition or "simply_supported").strip().lower()
    if cc == _EC:
        ratio = _EC_RATIOS.get(sc, 20.0)
        depth = L / ratio
        t = depth + float(cover_mm) + float(bar_diameter_mm) / 2.0
        std = "EN 1992-1-1 §7.4.2 (basic span/depth) + §4.4.1 cover"
        note = (f"d=L/{ratio:.0f}={depth:.1f} mm; h=d+cover+φ/2 "
                f"= {depth:.1f}+{float(cover_mm):.1f}+{float(bar_diameter_mm)/2:.1f} "
                f"= {t:.1f} mm")
    else:
        ratio = _ACI_RATIOS.get(sc, 20.0)
        fy_mod = 0.4 + fy_mpa / 700.0
        t = (L / ratio) * fy_mod
        std = "ACI 318-19 Table 7.3.1.1"
        note = (f"L/{ratio:.0f} = {L/ratio:.1f} mm; x(0.4+fy/700)={fy_mod:.3f} "
                f"-> {t:.1f} mm")
    return {
        "code": cc,
        "min_thickness_mm": round(t, 1),
        "support_condition": sc,
        "standard": std,
        "note": note,
    }

# Plain-engineer ACI slab asks ("minimum thickness of a one-way solid slab
# spanning 4.8 m") do not contain the registry name slab_thickness_min, and
# "thickness of a … slab" is not the intent-map phrase "slab thickness".
_ONE_WAY_SLAB_RE = re.compile(
    r"\bone[\s-]?way\b(?:\s+\w+){0,6}\s+slab\b",
    re.IGNORECASE,
)

# "spanning 4.8 m" is the original shape. The same span is also written
# "spans 6.0 m", "span length of 4.8 m", "span: 7.2 m", and "8.4 m span".
_SPAN_UNIT = r"mm|m"

_SPAN_NUM = r"\d+(?:\.\d+)?"

_SPAN_RE = re.compile(
    rf"\bspann(?:ing|ed)\s+(?P<num>{_SPAN_NUM})\s*(?P<unit>{_SPAN_UNIT})\b"
    rf"|\bspans\s+(?P<num_spans>{_SPAN_NUM})\s*(?P<unit_spans>{_SPAN_UNIT})\b"
    rf"|\bspan(?:\s+length)?(?:\s+of)?\s*[:=]?\s*"
    rf"(?P<num_span>{_SPAN_NUM})\s*(?P<unit_span>{_SPAN_UNIT})\b"
    rf"|(?P<num_pre>{_SPAN_NUM})\s*(?P<unit_pre>{_SPAN_UNIT})\s+"
    rf"(?:clear\s+)?span\b",
    re.IGNORECASE,
)

_SPAN_GROUP_PAIRS = (
    ("num", "unit"),
    ("num_spans", "unit_spans"),
    ("num_span", "unit_span"),
    ("num_pre", "unit_pre"),
)

_FY_RE = re.compile(
    r"\bfy\s*=?\s*(?P<fy>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_BOTH_ENDS_RE = re.compile(r"both[\s-]+ends?[\s-]+continuous", re.IGNORECASE)

# "one end continuous" and "continuous at one end" are the same support.
_ONE_END_RE = re.compile(
    r"one[\s-]+end[\s-]+continuous"
    r"|continuous(?:\s+at)?[\s-]+one[\s-]+end\b",
    re.IGNORECASE,
)

_EC_ASK_RE = re.compile(r"\beurocode\b|\bEN\s*1992\b", re.IGNORECASE)

_ACI_ASK_RE = re.compile(r"\bACI\b", re.IGNORECASE)

# thickness, "minimum depth" / "minimum-depth", and h_min / "h min".
_THICKNESS_RE = re.compile(
    r"\bthickness\b"
    r"|\bh[_\s]?min\b"
    r"|\bminimum[\s-]+(?:thickness|depth)\b",
    re.IGNORECASE,
)

# A cover or bill-of-quantities ask can share those words and a span.
_NOT_SLAB_THICKNESS_RE = re.compile(
    r"\b(?:boq|bill\s+of\s+quantit(?:y|ies))\b"
    r"|\bcover\b",
    re.IGNORECASE,
)

def _span_mm_from_ask(text: str) -> float | None:
    match = _SPAN_RE.search(text or "")
    if match is None:
        return None
    raw = None
    unit = None
    for num_key, unit_key in _SPAN_GROUP_PAIRS:
        if match.group(num_key):
            raw = match.group(num_key)
            unit = (match.group(unit_key) or "m").lower()
            break
    if raw is None:
        return None
    value = float(raw)
    if unit == "m":
        return value * 1000.0
    return value

def _support_from_ask(text: str) -> str:
    raw = text or ""
    if _BOTH_ENDS_RE.search(raw):
        return "both_ends_continuous"
    # One end wins over the simply-supported fallback, including
    # "continuous at one end, the other end simply supported" (L/24).
    if _ONE_END_RE.search(raw):
        return "one_end_continuous"
    if re.search(r"\bcantilever\b", raw, re.IGNORECASE):
        return "cantilever"
    return "simply_supported"

def looks_like_slab_thickness_min_ask(text: str) -> bool:
    """True when a registered slab_thickness_min call can answer the ask.

    Requires a one-way slab, a thickness word (thickness, minimum depth,
    or h_min), and a span the user supplied. Cover and bill-of-quantities
    asks stay on retrieval, as does a lookup that names neither a span
    nor a one-way slab.
    """
    raw = text or ""
    if not raw.strip():
        return False
    if _NOT_SLAB_THICKNESS_RE.search(raw):
        return False
    if not _ONE_WAY_SLAB_RE.search(raw):
        return False
    if not _THICKNESS_RE.search(raw):
        return False
    return _span_mm_from_ask(raw) is not None

def slab_thickness_params_from_ask(text: str) -> dict:
    """Span, support, fy, and code taken from the ask. Never invents a span."""
    out: dict = {}
    span = _span_mm_from_ask(text)
    if span is None:
        return out
    out["span_mm"] = span
    out["support_condition"] = _support_from_ask(text)
    fy = _FY_RE.search(text or "")
    if fy is not None:
        out["fy_mpa"] = float(fy.group("fy"))
    euro = bool(_EC_ASK_RE.search(text or ""))
    aci = bool(_ACI_ASK_RE.search(text or ""))
    if euro and not aci:
        out["code"] = "eurocode"
    elif aci:
        out["code"] = "aci"
    return out

def _format_mm(value: float) -> str:
    number = float(value)
    if number == int(number):
        return str(int(number))
    return f"{number:.1f}"

def format_slab_thickness_answer(result: dict) -> str:
    """User-facing line. Provenance is the calculator's own standard field."""
    if not isinstance(result, dict):
        return ""
    mm = result.get("min_thickness_mm")
    standard = str(result.get("standard") or "").strip()
    if mm is None or not standard:
        return ""
    note = str(result.get("note") or "").strip()
    line = (
        f"Minimum thickness is {_format_mm(float(mm))} mm. "
        f"Calculator: slab_thickness_min. "
        f"Standard: {standard}."
    )
    if note:
        line = f"{line} {note}"
    return line

def answer_states_slab_thickness_result(ask: str, answer: str) -> bool:
    """True when ``answer`` already states this ask's calculator result."""
    if not looks_like_slab_thickness_min_ask(ask):
        return False
    params = slab_thickness_params_from_ask(ask)
    if "span_mm" not in params:
        return False
    result = slab_thickness_min(
        span_mm=params["span_mm"],
        support_condition=params.get("support_condition", "simply_supported"),
        code=params.get("code", _ACI),
        fy_mpa=float(params.get("fy_mpa", 420.0)),
    )
    standard = str(result.get("standard") or "")
    mm = result.get("min_thickness_mm")
    if mm is None or not standard or standard not in (answer or ""):
        return False
    shown = _format_mm(float(mm))
    return bool(re.search(
        rf"\b{re.escape(shown)}\s*mm\b",
        answer or "",
        re.IGNORECASE,
    ))

@formula(
    owner='design',
    display_name='Reinforcement lap length',
    description='Tension lap-splice length of a deformed reinforcing bar.',
    inputs={
        'bar_diameter_mm': Param('mm', 4, 60, label='bar diameter'),
        'fy_mpa': Param('MPa', 100, 1500, label='reinforcement yield strength', grade='rebar'),
        'fc_mpa': Param('MPa', 5, 150, label='concrete compressive strength', grade='concrete'),
        'code': Param('-', label='design code'),
        'confinement_ratio': Param('ratio', 1, 2.5, label='confinement term'),
    },
    outputs={'lap_length_mm': 'mm'},
)
def rebar_lap_length(
    bar_diameter_mm: float,
    fy_mpa: float,
    fc_mpa: float,
    code: str = _ACI,
    confinement_ratio: float = 1.5,
) -> dict:
    """Tension lap-splice length for a deformed bar.

    ACI 318-19 §25.4.2: ld = (fy*ψt*ψe*ψs*ψg/(1.1*λ*sqrt(fc)*((cb+Ktr)/db)))*db;
    ψs = 0.8 for No. 19 and smaller, else 1.0. Table 25.4.2.5: ψg = 1.0 for
    fy ≤ 420 MPa, 1.15 above that up to 550 MPa, and 1.3 above 550 MPa.
    Other ψ and λ stay 1 when the ask does not state top-bar, coating, or
    lightweight concrete.
    Class B lap = 1.3*ld. ``confinement_ratio`` = (cb+Ktr)/db, capped 2.5.
    EN 1992-1-1 §8.4/8.7: lb,rqd = (db/4)*(fyd/fbd), fbd=2.25*η1*η2*fctd,
    η1 = 1 (good bond unless stated otherwise), η2 = 1 for db ≤ 32 mm and
    (132-db)/100 above that, fctd=0.7*0.3*fck^(2/3)/1.5;
    lap l0 = 1.5*lb,rqd (>50% lapped).
    """
    cc = _norm_code(code)
    db = float(bar_diameter_mm)
    if cc == _EC:
        fyd = fy_mpa / 1.15
        fctm = 0.3 * fc_mpa ** (2.0 / 3.0)
        fctd = 0.7 * fctm / 1.5
        eta2 = 1.0 if db <= 32.0 else (132.0 - db) / 100.0
        fbd = 2.25 * eta2 * fctd
        lb_rqd = (db / 4.0) * (fyd / fbd)
        lap = 1.5 * lb_rqd
        std = "EN 1992-1-1 §8.4/8.7"
        note = (f"fbd=2.25*eta2*fctd={fbd:.3f} MPa (eta2={eta2:.3f}); "
                f"lb,rqd=(db/4)*(fyd/fbd)={lb_rqd:.1f} mm; l0=1.5*lb,rqd={lap:.1f} mm")
    else:
        conf = min(confinement_ratio, 2.5)
        psi_s = 0.8 if db <= 19.0 else 1.0
        psi_g = _grade_factor(fy_mpa)
        ld = (fy_mpa * psi_s * psi_g / (1.1 * math.sqrt(fc_mpa) * conf)) * db
        lap = 1.3 * ld
        std = "ACI 318-19 §25.4.2 (Class B lap)"
        note = (f"ld=(fy*ψs*ψg/(1.1*sqrt(fc)*{conf}))*db={ld:.1f} mm "
                f"(ψs={psi_s}, ψg={psi_g}); lap=1.3*ld={lap:.1f} mm")
    return {
        "code": cc,
        "lap_length_mm": round(lap, 1),
        "standard": std,
        "note": note,
    }

ADDITIONAL_CALCULATORS = {
    "rc_beam_moment_capacity": rc_beam_moment_capacity,
    "rc_beam_shear_capacity": rc_beam_shear_capacity,
    "slab_thickness_min": slab_thickness_min,
    "rebar_lap_length": rebar_lap_length,
}
