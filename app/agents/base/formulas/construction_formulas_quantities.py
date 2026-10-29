"""Formulas from app.lib.construction_formulas_quantities owned by the base package (every hat).

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import logging
import math


logger = logging.getLogger(__name__)

def _concrete_quantity(quantity: float) -> float | None:
    """Element count. None when the value is not a positive number."""
    if quantity in (None, ""):
        return 1.0
    try:
        qty = float(quantity)
    except (TypeError, ValueError):
        logger.debug("concrete quantity %r is not numeric", quantity)
        return None
    if qty <= 0:
        return None
    return qty

@formula(
    owner='base',
    display_name='Concrete volume',
    description='Concrete volume of a rectangular element, a cylinder or a trapezoidal section, with an optional waste factor and element count.',
    inputs={
        'length_m': Param('m', 0, 10000, label='length'),
        'width_m': Param('m', 0, 10000, label='width'),
        'thickness_m': Param('m', 0, 50, label='thickness'),
        'shape': Param('-', label='shape'),
        'diameter_m': Param('m', 0, 100, label='diameter'),
        'height_m': Param('m', 0, 1000, label='height'),
        'top_width_m': Param('m', 0, 10000, label='top width'),
        'bottom_width_m': Param('m', 0, 10000, label='bottom width'),
        'depth_m': Param('m', 0, 500, label='depth'),
        'waste_factor': Param('fraction', 0, 1, label='waste allowance'),
        'quantity': Param('nr', 1, 100000, label='number of elements'),
    },
    outputs={'shape': '-', 'quantity': '-', 'volume_m3': 'm3', 'net_volume_m3': 'm3', 'volume_with_waste_m3': 'm3', 'value': 'currency', 'waste_factor': '-'},
)
def concrete_volume(
    length_m: float = 0.0,
    width_m: float = 0.0,
    thickness_m: float = 0.0,
    shape: str = "rectangular",
    diameter_m: float = 0.0,
    height_m: float = 0.0,
    top_width_m: float = 0.0,
    bottom_width_m: float = 0.0,
    depth_m: float = 0.0,
    waste_factor: float = 0.05,
    quantity: float = 1,
) -> dict:
    """Concrete volume for a rectangular slab/element, a cylinder (column/pile),
    or a trapezoidal section (channel/footing), plus a waste allowance.

    rectangular: L*W*T. cylinder: pi*(D/2)^2*H.
    trapezoidal: ((top+bottom)/2 * depth) * length.
    ``quantity`` is how many such elements (24 pile caps). It used to be
    dropped on bind, so a follow-up priced one cap.

    Headline ``volume_m3`` is the with-waste figure (net volume × the
    documented waste factor), not the net figure.
    """
    if min(float(length_m), float(width_m), float(thickness_m),
           float(diameter_m), float(height_m), float(top_width_m),
           float(bottom_width_m), float(depth_m)) < 0:
        return {"error": "concrete_volume dimensions must be >= 0."}
    qty = _concrete_quantity(quantity)
    if qty is None:
        return {"error": "concrete_volume quantity must be > 0."}
    waste_factor = float(waste_factor)
    if waste_factor < 0:
        return {"error": "waste_factor must be >= 0."}
    s = (shape or "rectangular").strip().lower()
    if s == "rectangular" and not float(length_m):
        # No shape named: the dimensions given say which one it is.
        if float(diameter_m) and float(height_m):
            s = "cylinder"
        elif (float(top_width_m) or float(bottom_width_m)) and float(depth_m):
            s = "trapezoidal"
    if s == "cylinder":
        unit = math.pi * (diameter_m / 2.0) ** 2 * height_m
        expr = f"pi*({diameter_m}/2)^2*{height_m}"
    elif s == "trapezoidal":
        area = (top_width_m + bottom_width_m) / 2.0 * depth_m
        unit = area * length_m
        expr = f"(({top_width_m}+{bottom_width_m})/2*{depth_m})*{length_m}"
    else:
        s = "rectangular"
        unit = length_m * width_m * thickness_m
        expr = f"{length_m}*{width_m}*{thickness_m}"
    if qty != 1:
        expr = f"{qty:g}*({expr})"
    net = unit * qty
    with_waste = net * (1.0 + waste_factor)
    headline = round(with_waste, 3)
    net_r = round(net, 3)
    return {
        "shape": s,
        "quantity": qty,
        "volume_m3": headline,
        "net_volume_m3": net_r,
        "volume_with_waste_m3": headline,
        "value": headline,
        "waste_factor": waste_factor,
        "standard": "geometry",
        "note": (f"Net = {expr} = {net:.3f} m3; "
                 f"+{waste_factor*100:.0f}% waste = {with_waste:.3f} m3."),
    }
