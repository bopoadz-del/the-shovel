"""Formulas from app.lib.construction_formulas_columns owned by the design hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math


_ACI = "aci"

_EC = "eurocode"

def _norm_code(code: str) -> str:
    c = (code or "").strip().lower()
    if c in ("ec", "eurocode", "en", "en1992"):
        return _EC
    return _ACI

@formula(
    owner='design',
    display_name='Concrete column axial capacity',
    description='Axial load capacity of a reinforced concrete column, to the selected code. Slenderness is assessed when unsupported length and radius of gyration are both given.',
    inputs={
        'gross_area_mm2': Param('mm2', 1, 1e8, label='gross area'),
        'steel_area_mm2': Param('mm2', 0, 1e7, label='steel area'),
        'fc_mpa': Param('MPa', 5, 150, label='concrete compressive strength', grade='concrete'),
        'fy_mpa': Param('MPa', 100, 1500, label='reinforcement yield strength', grade='rebar'),
        'unsupported_length_mm': Param('mm', 100, 100000, label='unsupported length'),
        'radius_gyration_mm': Param('mm', 1, 10000, label='radius of gyration'),
        'code': Param('-', label='design code'),
        'tie_type': Param('-', label='tie type'),
        'k_factor': Param('ratio', 0.5, 2.5, label='effective length factor'),
        'applied_load_kn': Param('kN', 0, 1e7, label='applied load'),
        'cm': Param('ratio', 0.2, 1, label='moment correction factor'),
        'slenderness_limit': Param('ratio', 1, 200, label='slenderness limit'),
    },
    outputs={'axial_capacity_kn': 'kN', 'slenderness_ratio': '-', 'slenderness_limit': '-', 'is_slender': '-', 'euler_critical_load_kn': 'kN', 'moment_magnifier_delta_ns': '-'},
)
def column_axial_capacity(
    gross_area_mm2: float,
    steel_area_mm2: float,
    fc_mpa: float,
    fy_mpa: float,
    unsupported_length_mm: float = None,
    radius_gyration_mm: float = None,
    code: str = _ACI,
    tie_type: str = "tied",
    k_factor: float = 1.0,
    applied_load_kn: float = None,
    cm: float = 1.0,
    slenderness_limit: float = 22.0,
) -> dict:
    """RC column axial capacity, with slenderness only when both geometry inputs are given.

    ACI 318-19 §22.4 / §6.6.4: P0 = 0.85*fc*(Ag-Ast) + fy*Ast;
    Pn,max = alpha*P0 (alpha=0.80 tied / 0.85 spiral); phi.Pn = phi*Pn,max
    (phi=0.65 tied / 0.75 spiral). Slenderness ratio = k*lu/r; Euler
    Pc = pi^2*EI/(k*lu)^2, EI = 0.4*Ec*Ig, Ec=4700*sqrt(fc), Ig=Ag*r^2;
    magnifier delta_ns = Cm/(1 - Pu/(0.75*Pc)) >= 1.0.
    EN 1992-1-1 §6.1/§5.8: NRd = fcd*(Ag-As) + fyd*As; slenderness lambda = l0/i;
    lambda_lim = 20*A*B*C/sqrt(n), n = NEd/(Ac*fcd).
    """
    cc = _norm_code(code)
    Ag, Ast = float(gross_area_mm2), float(steel_area_mm2)
    if Ag <= 0 or Ast < 0 or Ast >= Ag:
        return {"error": "gross_area_mm2 must be > steel_area_mm2 >= 0."}
    if fc_mpa <= 0 or fy_mpa <= 0:
        return {"error": "fc_mpa and fy_mpa must be > 0."}
    assess = unsupported_length_mm is not None and radius_gyration_mm is not None
    if assess and (unsupported_length_mm <= 0 or radius_gyration_mm <= 0):
        return {"error": "unsupported_length_mm and radius_gyration_mm must be > 0."}
    Ac = Ag - Ast
    out = {"code": cc}
    slenderness = None
    klu = r = None
    if assess:
        lu, r, k = float(unsupported_length_mm), float(radius_gyration_mm), float(k_factor)
        klu = k * lu
        slenderness = klu / r
        out["slenderness_ratio"] = round(slenderness, 2)

    if cc == _EC:
        fcd, fyd = fc_mpa / 1.5, fy_mpa / 1.15
        nrd_n = fcd * Ac + fyd * Ast
        out["axial_capacity_kn"] = round(nrd_n / 1000.0, 1)
        out["standard"] = "EN 1992-1-1 §6.1/§5.8"
        note = (f"NRd = fcd*Ac + fyd*As = {fcd:.1f}*{Ac:.0f} + {fyd:.1f}*{Ast:.0f} "
                f"= {nrd_n/1000.0:.1f} kN")
        if assess:
            note += f"; lambda = l0/i = {klu:.0f}/{r} = {slenderness:.2f}"
            if applied_load_kn is not None:
                n = (float(applied_load_kn) * 1000.0) / (Ac * fcd)
                lam_lim = 20.0 * 0.7 * 1.1 * 0.7 / math.sqrt(n) if n > 0 else float("inf")
                out["slenderness_limit"] = round(lam_lim, 2)
                out["is_slender"] = slenderness > lam_lim
                note += f"; lambda_lim = 20*A*B*C/sqrt(n) = {lam_lim:.2f} ({'slender' if slenderness>lam_lim else 'short'})"
            else:
                out["is_slender"] = slenderness > slenderness_limit
        out["note"] = note + "."
        return out

    # ACI
    alpha = 0.85 if tie_type.strip().lower() == "spiral" else 0.80
    phi = 0.75 if tie_type.strip().lower() == "spiral" else 0.65
    p0 = 0.85 * fc_mpa * Ac + fy_mpa * Ast
    pn_max = alpha * p0
    phi_pn = phi * pn_max
    out["axial_capacity_kn"] = round(phi_pn / 1000.0, 1)
    out["standard"] = "ACI 318-19 §22.4/§6.6.4"
    note = (f"P0 = 0.85*fc*Ac + fy*Ast = {p0/1000.0:.0f} kN; Pn,max = {alpha}*P0; "
            f"phi.Pn = {phi}*Pn,max = {phi_pn/1000.0:.1f} kN")
    if assess:
        ec = 4700.0 * math.sqrt(fc_mpa)
        ig = Ag * r * r
        ei = 0.4 * ec * ig
        pc = math.pi ** 2 * ei / (klu ** 2)
        out["euler_critical_load_kn"] = round(pc / 1000.0, 1)
        out["is_slender"] = slenderness > slenderness_limit
        note += (f". slenderness = k*lu/r = {klu:.0f}/{r} = {slenderness:.2f} "
                 f"({'slender' if out['is_slender'] else 'short'} vs {slenderness_limit}); "
                 f"Pc = pi^2*EI/(k*lu)^2 = {pc/1000.0:.0f} kN")
        if applied_load_kn is not None:
            pu = float(applied_load_kn) * 1000.0
            denom = 1.0 - pu / (0.75 * pc)
            delta = max(cm / denom, 1.0) if denom > 0 else float("inf")
            out["moment_magnifier_delta_ns"] = round(delta, 3)
            note += f"; delta_ns = Cm/(1 - Pu/(0.75*Pc)) = {delta:.3f}"
    out["note"] = note + "."
    return out

ADDITIONAL_CALCULATORS = {
    "column_axial_capacity": column_axial_capacity,
}
