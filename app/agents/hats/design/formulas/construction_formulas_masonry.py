"""Formulas from app.lib.construction_formulas_masonry owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math


_ACI = "aci"

_EC = "eurocode"

def _norm_code(code: str) -> str:
    c = (code or "").strip().lower()
    if c in ("ec", "eurocode", "en", "en1996", "ec6"):
        return _EC
    return _ACI

@formula(
    owner='design',
    display_name='Masonry wall capacity',
    description='Axial compressive capacity of a masonry wall, allowing for slenderness.',
    inputs={
        'masonry_strength_mpa': Param('MPa', 0.5, 100, label='masonry compressive strength'),
        'net_area_mm2': Param('mm2', 1, 1e8, label='net area'),
        'height_mm': Param('mm', 50, 50000, label='wall height'),
        'thickness_mm': Param('mm', 25, 5000, label='wall thickness'),
        'code': Param('-', label='design code'),
        'radius_gyration_mm': Param('mm', 1, 10000, label='radius of gyration'),
        'steel_area_mm2': Param('mm2', 0, 1e7, label='steel area'),
        'steel_stress_mpa': Param('MPa', 0, 1500, label='steel stress'),
        'gamma_m': Param('ratio', 1, 5, label='partial safety factor for masonry'),
        'eccentricity_mm': Param('mm', 0, 10000, label='eccentricity'),
        'ke': Param('ratio', 100, 2000, label='masonry modulus factor'),
        'eff_height_mm': Param('mm', 50, 50000, label='effective height'),
        'eff_thickness_mm': Param('mm', 25, 5000, label='effective thickness'),
    },
    outputs={'capacity_kn': 'kN', 'slenderness_reduction_R': '-', 'h_over_r': '-', 'phi_reduction': '-', 'slenderness_lambda_c': 'degC'},
)
def masonry_wall_capacity(
    masonry_strength_mpa: float,
    net_area_mm2: float,
    height_mm: float,
    thickness_mm: float,
    code: str = _ACI,
    radius_gyration_mm: float = None,
    steel_area_mm2: float = 0.0,
    steel_stress_mpa: float = 0.0,
    gamma_m: float = 2.7,
    eccentricity_mm: float = None,
    ke: float = 1000.0,
    eff_height_mm: float = None,
    eff_thickness_mm: float = None,
) -> dict:
    """Axial compressive capacity of a masonry wall including slenderness.

    TMS 402: Pa = (0.25*f'm*An + 0.65*Ast*Fs) * R, where
    R = 1 - (h/(140 r))^2 for h/r <= 99, else (70 r/h)^2.
    EN 1996-1-1: NRd = Phi_m * A * (fk/gM), Phi_m = A1*exp(-u^2/2) (Annex G),
    A1 = 1 - 2*emk/t, u = (lam_c - 0.063)/(0.73 - 1.17*emk/t),
    lam_c = (hef/tef)*sqrt(fk/E), E = KE*fk.
    """
    cc = _norm_code(code)
    fm = float(masonry_strength_mpa)
    An = float(net_area_mm2)
    h = float(height_mm)
    t = float(thickness_mm)
    r = float(radius_gyration_mm) if radius_gyration_mm else t / math.sqrt(12.0)

    if cc == _EC:
        fd = fm / gamma_m
        hef = float(eff_height_mm) if eff_height_mm else h
        tef = float(eff_thickness_mm) if eff_thickness_mm else t
        e = float(eccentricity_mm) if eccentricity_mm is not None else 0.05 * t
        emk_t = e / t
        E = ke * fm
        lam_c = (hef / tef) * math.sqrt(fm / E)
        A1 = 1.0 - 2.0 * emk_t
        u = (lam_c - 0.063) / (0.73 - 1.17 * emk_t)
        phi_m = A1 * math.exp(-(u * u) / 2.0)
        nrd = phi_m * An * fd
        return {
            "code": cc,
            "capacity_kn": round(nrd / 1000.0, 1),
            "phi_reduction": round(phi_m, 4),
            "slenderness_lambda_c": round(lam_c, 4),
            "standard": "EN 1996-1-1 §6.1 / Annex G",
            "note": (f"fd=fk/gM={fd:.3f}; lam_c=(hef/tef)*sqrt(fk/E)={lam_c:.3f}; "
                     f"A1=1-2emk/t={A1:.3f}; u={u:.3f}; Phi_m=A1*exp(-u^2/2)={phi_m:.4f}; "
                     f"NRd=Phi_m*A*fd={nrd/1000.0:.1f} kN."),
        }

    # TMS 402
    h_over_r = h / r
    if h_over_r <= 99.0:
        R = 1.0 - (h / (140.0 * r)) ** 2
        r_expr = f"1-(h/140r)^2 = 1-({h}/(140*{r:.1f}))^2"
    else:
        R = (70.0 * r / h) ** 2
        r_expr = f"(70r/h)^2 = (70*{r:.1f}/{h})^2"
    axial = 0.25 * fm * An + 0.65 * float(steel_area_mm2) * float(steel_stress_mpa)
    pa = axial * R
    return {
        "code": cc,
        "capacity_kn": round(pa / 1000.0, 1),
        "slenderness_reduction_R": round(R, 4),
        "h_over_r": round(h_over_r, 2),
        "standard": "TMS 402 §8.2/§8.3",
        "note": (f"Axial = 0.25*f'm*An + 0.65*Ast*Fs = {axial/1000.0:.1f} kN; "
                 f"R = {r_expr} = {R:.4f}; Pa = axial*R = {pa/1000.0:.1f} kN "
                 f"(h/r={h_over_r:.1f})."),
    }

ADDITIONAL_CALCULATORS = {
    "masonry_wall_capacity": masonry_wall_capacity,
}
