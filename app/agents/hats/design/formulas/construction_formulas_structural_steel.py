"""Formulas from app.lib.construction_formulas_structural_steel owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
import re
from app.lib.formula_registry import Param, formula
import math
from typing import Optional


_ACI = "aci"

_EC = "eurocode"

# EN 1993-1-1 Table 3.1 fu for t ≤ 40 mm, and EN 1993-1-8 Table 4.1 βw.
# Keyed by the yield number in the grade label (S275 → 275).
_FILLET_PARENT_METAL = {
    235: (360.0, 0.80),
    275: (430.0, 0.85),
    355: (510.0, 0.90),
    420: (520.0, 1.00),
    460: (540.0, 1.00),
}


def _fillet_parent_metal(value):
    """(fu MPa, βw) for a structural-steel grade label, or None if it is not tabulated."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text.startswith("S"):
        text = text[1:]
    digits = []
    for ch in text:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return None
    return _FILLET_PARENT_METAL.get(int("".join(digits)))

# Every way the Eurocode steel family is written: EC, EC3, Eurocode 3,
# EN 1993-1-8, BS EN 1993 ... Anything else is the AISC/ACI family.
_EC_CODE_RE = re.compile(r"^(?:bs\s*)?(?:ec\s*\d*|eurocode(?:\s*\d+)?|en(?:\s*1993\S*)?)\b", re.IGNORECASE)


def _norm_code(code: str) -> str:
    c = (code or "").strip()
    if _EC_CODE_RE.match(c):
        return _EC
    return _ACI  # default AISC/ACI family

@formula(
    owner='design',
    display_name='Steel member tension capacity',
    description='Design tensile strength of a steel member: the lesser of gross-section yielding and net-section rupture.',
    inputs={
        'gross_area_mm2': Param('mm2', 1, 1e8, label='gross area'),
        'net_area_mm2': Param('mm2', 1, 1e8, label='net area'),
        'fy_mpa': Param('MPa', 100, 1500, label='steel yield strength', grade='steel'),
        'fu_mpa': Param('MPa', 200, 2000, label='steel ultimate strength'),
        'code': Param('-', label='design code'),
    },
    outputs={'capacity_kn': 'kN', 'governing_limit_state': '-', 'yield_capacity_kn': 'kN', 'rupture_capacity_kn': 'kN'},
)
def steel_tension_capacity(
    gross_area_mm2: float,
    net_area_mm2: Optional[float] = None,
    fy_mpa: float = 345.0,
    fu_mpa: float = 450.0,
    code: str = _ACI,
) -> dict:
    """Design tensile strength of a steel member — the lesser of gross-section
    yielding and net-section rupture.

    AISC 360-16 §D2 (LRFD): yield phi.Pn = 0.90*Fy*Ag; rupture phi.Pn = 0.75*Fu*Ae.
    EN 1993-1-1 §6.2.3: Npl,Rd = A*fy/gM0 (gM0=1.0); Nu,Rd = 0.9*Anet*fu/gM2 (gM2=1.25).
    Governing capacity = min(yield, rupture).
    """
    cc = _norm_code(code)
    net = float(net_area_mm2) if net_area_mm2 is not None else float(gross_area_mm2)
    ag = float(gross_area_mm2)
    if cc == _EC:
        yield_n = ag * fy_mpa / 1.0
        rupture_n = 0.9 * net * fu_mpa / 1.25
        std = "EN 1993-1-1 §6.2.3 (gM0=1.0, gM2=1.25)"
        y_expr = f"A*fy/gM0 = {ag}*{fy_mpa}/1.0"
        r_expr = f"0.9*Anet*fu/gM2 = 0.9*{net}*{fu_mpa}/1.25"
    else:
        yield_n = 0.90 * fy_mpa * ag
        rupture_n = 0.75 * fu_mpa * net
        std = "AISC 360-16 §D2 (LRFD, phi_y=0.90, phi_r=0.75)"
        y_expr = f"0.90*Fy*Ag = 0.90*{fy_mpa}*{ag}"
        r_expr = f"0.75*Fu*Ae = 0.75*{fu_mpa}*{net}"
    yield_kn = yield_n / 1000.0
    rupture_kn = rupture_n / 1000.0
    governs = "yield" if yield_kn <= rupture_kn else "rupture"
    cap_kn = min(yield_kn, rupture_kn)
    return {
        "code": cc,
        "capacity_kn": round(cap_kn, 2),
        "governing_limit_state": governs,
        "yield_capacity_kn": round(yield_kn, 2),
        "rupture_capacity_kn": round(rupture_kn, 2),
        "standard": std,
        "note": (
            f"Yield: {y_expr} = {yield_kn:.1f} kN. "
            f"Rupture: {r_expr} = {rupture_kn:.1f} kN. "
            f"Governs: {governs} -> {cap_kn:.1f} kN."
        ),
    }

# ISO 898-1 coarse-pitch tensile stress areas, mm². A nominal size is this
# area; a caller who already has an area passes that number.
_ISO_TENSILE_STRESS_AREA_MM2 = {
    "M10": 58.0, "M12": 84.3, "M14": 115.0, "M16": 157.0, "M18": 192.0,
    "M20": 245.0, "M22": 303.0, "M24": 353.0, "M27": 459.0, "M30": 561.0,
    "M33": 694.0, "M36": 817.0, "M39": 976.0, "M42": 1120.0, "M45": 1300.0,
    "M48": 1470.0, "M52": 1760.0, "M56": 2030.0, "M60": 2360.0, "M64": 2680.0,
}
# ISO 898-1 nominal tensile strength, and EN 1993-1-8 Table 3.4 αv.
_BOLT_FUB_MPA = {
    "4.6": 400.0, "4.8": 400.0, "5.6": 500.0, "5.8": 500.0,
    "6.8": 600.0, "8.8": 800.0, "10.9": 1000.0, "12.9": 1200.0,
}
_BOLT_ALPHA_V = {
    "4.6": 0.6, "5.6": 0.6, "8.8": 0.6,
    "4.8": 0.5, "5.8": 0.5, "6.8": 0.5, "10.9": 0.5, "12.9": 0.5,
}


def _bolt_tensile_area_mm2(value):
    """Tensile stress area (mm²) from a number or a nominal metric designation."""
    if isinstance(value, str):
        token = "".join(value.split()).upper()
        if token in _ISO_TENSILE_STRESS_AREA_MM2:
            return _ISO_TENSILE_STRESS_AREA_MM2[token], None
        try:
            area = float(token)
        except ValueError:
            return None, (
                f"Bolt size {value} is not a tensile stress area or a "
                "nominal metric size."
            )
    else:
        try:
            area = float(value)
        except (TypeError, ValueError):
            return None, (
                "bolt_area_mm2 must be a tensile stress area or a "
                "nominal metric size."
            )
    return area, None


def _bolt_class_key(value):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return f"{float(value):.1f}"
    text = str(value).strip().lower()
    try:
        return f"{float(text):.1f}"
    except ValueError:
        return text


@formula(
    owner='design',
    display_name='Bolt shear capacity',
    description='Design shear capacity of a bolt over its shear planes, to the selected code. A nominal metric size is the tensile stress area.',
    inputs={
        'bolt_area_mm2': Param('mm2', 1, 20000, label='bolt area'),
        'shear_strength_mpa': Param('MPa', 10, 2000, label='bolt shear strength'),
        'n_shear_planes': Param('count', 1, 4, label='number of shear planes'),
        'code': Param('-', label='design code'),
        'alpha_v': Param('ratio', 0, 1, label='shear coefficient'),
        'grade': Param('-', label='bolt property class'),
    },
    outputs={'capacity_kn': 'kN', 'n_shear_planes': '-'},
)
def bolt_shear_capacity(
    bolt_area_mm2: str,
    shear_strength_mpa: float = 372.0,
    n_shear_planes: int = 1,
    code: str = _ACI,
    alpha_v: float = 0.6,
    grade: str = None,
) -> dict:
    """Design shear strength of ONE bolt (all shear planes).

    AISC 360-16 §J3.6 (LRFD): phi.Rn = 0.75 * Fnv * Ab * n_planes, Fnv = nominal
    shear strength (e.g. A325-N ~372 MPa, A325-X ~469 MPa).
    EN 1993-1-8 Table 3.4: Fv,Rd = alpha_v * fub * A * n_planes / gM2 (gM2=1.25),
    alpha_v = 0.6 (classes 4.6/5.6/8.8) or 0.5 (4.8/5.8/6.8/10.9). Here
    ``shear_strength_mpa`` carries fub for the Eurocode branch.
    """
    cc = _norm_code(code)
    ab, area_error = _bolt_tensile_area_mm2(bolt_area_mm2)
    if area_error:
        return {"error": area_error}
    n = int(n_shear_planes)
    if ab <= 0:
        return {"error": "bolt_area_mm2 must be > 0."}
    if n <= 0:
        return {"error": "n_shear_planes must be >= 1."}
    fub = float(shear_strength_mpa)
    alpha = float(alpha_v)
    if fub <= 0:
        return {"error": "shear_strength_mpa must be > 0."}
    bolt_class = None
    if grade not in (None, "") and cc == _EC:
        key = _bolt_class_key(grade)
        if key not in _BOLT_FUB_MPA:
            return {"error": f"Bolt property class {grade} is not tabulated."}
        bolt_class = key
        if shear_strength_mpa == 372.0:
            fub = _BOLT_FUB_MPA[key]
        if alpha_v == 0.6:
            alpha = _BOLT_ALPHA_V[key]
    elif cc == _EC and alpha_v == 0.6:
        # No class given: the shear coefficient still comes from the
        # standard's class table, read through the given fub when that fub
        # belongs to classes sharing one coefficient (1000 MPa is 10.9,
        # alpha_v 0.5). An fub two classes share with different coefficients
        # (400: 4.6 and 4.8) keeps the 0.6 and says the class was not given.
        classes = sorted(k for k, v in _BOLT_FUB_MPA.items() if abs(v - fub) < 1e-9)
        alphas = {_BOLT_ALPHA_V[k] for k in classes}
        if len(alphas) == 1:
            alpha = alphas.pop()
            bolt_class = " / ".join(classes)
    if cc == _EC:
        cap_n = alpha * fub * ab * n / 1.25
        std = "EN 1993-1-8 Table 3.4 (gM2=1.25)"
        expr = f"alpha_v*fub*A*n/gM2 = {alpha}*{fub}*{ab}*{n}/1.25"
    else:
        cap_n = 0.75 * fub * ab * n
        std = "AISC 360-16 §J3.6 (LRFD, phi=0.75)"
        expr = f"0.75*Fnv*Ab*n = 0.75*{fub}*{ab}*{n}"
    cap_kn = cap_n / 1000.0
    return {
        "code": cc,
        "capacity_kn": round(cap_kn, 2),
        "n_shear_planes": n,
        "standard": std,
        **({"bolt_class": bolt_class, "alpha_v": alpha} if cc == _EC else {}),
        "note": (f"{expr} = {cap_n:.0f} N = {cap_kn:.2f} kN (one bolt)."
                 + (f" Property class {bolt_class}, alpha_v {alpha} (EN 1993-1-8 Table 3.4)." if cc == _EC and bolt_class
                    else (" The property class was not given; alpha_v 0.6 is used." if cc == _EC else ""))),
    }

@formula(
    owner='design',
    display_name='Fillet weld capacity',
    description='Design strength of a fillet weld over its length.',
    inputs={
        'leg_size_mm': Param('mm', 2, 100, label='weld leg size'),
        'length_mm': Param('mm', 1, 100000, label='weld length'),
        'electrode_strength_mpa': Param('MPa', 200, 1500, label='electrode strength'),
        'code': Param('-', label='design code'),
        'beta_w': Param('ratio', 0.5, 1.5, label='weld correlation factor'),
        'steel_grade': Param('-', label='steel grade', grade='steel'),
    },
    outputs={'capacity_kn': 'kN', 'capacity_per_mm_kn': 'kN', 'effective_throat_mm': 'mm'},
)
def weld_capacity(
    leg_size_mm: float,
    length_mm: float,
    electrode_strength_mpa: float = None,
    code: str = _ACI,
    beta_w: float = None,
    steel_grade: str = None,
) -> dict:
    """Design strength of a fillet weld over ``length_mm``.

    Effective throat = 0.707 * leg (equal-leg fillet).
    AISC 360-16 §J2.4 (LRFD): phi.Rn = 0.75 * (0.60*FEXX) * throat * L.
    EN 1993-1-8 §4.5.3.3 (simplified): Fw,Rd = fvw,d * throat * L,
    fvw,d = fu / (sqrt(3) * beta_w * gM2), gM2=1.25, beta_w correlation factor
    (0.8 S235 / 0.85 S275 / 0.9 S355).
    """
    cc = _norm_code(code)
    throat = 0.707 * float(leg_size_mm)
    L = float(length_mm)
    fu = electrode_strength_mpa
    beta = beta_w
    if cc == _EC and steel_grade not in (None, ""):
        parent = _fillet_parent_metal(steel_grade)
        if parent is None and (fu is None or beta is None):
            return {"error": (
                f"Steel grade {steel_grade} has no tabulated fillet-weld fu and "
                "beta_w. Pass electrode strength and the correlation factor."
            )}
        if parent is not None:
            if fu is None:
                fu = parent[0]
            if beta is None:
                beta = parent[1]
    if fu is None:
        fu = 490.0
    if beta is None:
        beta = 0.9
    if cc == _EC:
        fvw_d = fu / (math.sqrt(3.0) * beta * 1.25)
        per_mm = fvw_d * throat
        std = "EN 1993-1-8 §4.5.3.3 (gM2=1.25)"
        expr = (f"fvw,d = fu/(sqrt3*beta_w*gM2) = {fu}/"
                f"(1.732*{beta}*1.25) = {fvw_d:.1f} MPa; x throat {throat:.3f}")
    else:
        per_mm = 0.75 * 0.60 * fu * throat
        std = "AISC 360-16 §J2.4 (LRFD, phi=0.75)"
        expr = (f"0.75*0.60*FEXX*throat = 0.75*0.60*{fu}*"
                f"{throat:.3f}")
    total_n = per_mm * L
    return {
        "code": cc,
        "capacity_kn": round(total_n / 1000.0, 2),
        "capacity_per_mm_kn": round(per_mm / 1000.0, 4),
        "effective_throat_mm": round(throat, 3),
        "standard": std,
        "note": (
            f"Throat = 0.707*{leg_size_mm} = {throat:.3f} mm. "
            f"{expr} = {per_mm:.1f} N/mm. x L={L} mm = {total_n/1000.0:.2f} kN."
        ),
    }

ADDITIONAL_CALCULATORS = {
    "steel_tension_capacity": steel_tension_capacity,
    "bolt_shear_capacity": bolt_shear_capacity,
    "weld_capacity": weld_capacity,
}
