"""Formulas from app.lib.construction_formulas_planning owned by the quantities hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Optional


@formula(
    owner='quantities',
    display_name='Material consumption',
    description='Material required for a quantity of work from the output per unit and an optional waste factor.',
    inputs={
        'quantity_of_work': Param('count', 0, 1e12, label='quantity of work'),
        'output_per_unit': Param('count', 0, 1e12, label='output per unit'),
        'waste_factor': Param('ratio', 0, 1, label='waste allowance'),
        'waste_percent': Param('%', 0, 100, label='waste allowance'),
    },
    outputs={'material_required': '-', 'base_without_waste': '-', 'waste_factor': '-'},
)
def material_consumption(
    quantity_of_work: Optional[float] = None,
    output_per_unit: Optional[float] = None,
    waste_factor: Optional[float] = None,
    waste_percent: Optional[float] = None,
) -> dict:
    """Material Required = Quantity of Work / Output per Unit (× waste).

    Requires quantity_of_work and output_per_unit. Waste is optional: pass
    ``waste_factor`` (e.g. 1.05) or ``waste_percent`` (e.g. 5 → factor 1.05).
    Refuses invented consumption rates.
    """
    if quantity_of_work is None or output_per_unit is None:
        return {
            "error": (
                "material_consumption requires quantity_of_work and "
                "output_per_unit — refuse without inputs."
            ),
            "required": ["quantity_of_work", "output_per_unit"],
            "optional": ["waste_factor", "waste_percent"],
        }
    qty = float(quantity_of_work)
    out_u = float(output_per_unit)
    if out_u <= 0:
        return {"error": "output_per_unit must be > 0.", "required": ["output_per_unit"]}
    base = qty / out_u
    factor = None
    if waste_factor is not None:
        factor = float(waste_factor)
    elif waste_percent is not None:
        factor = 1.0 + float(waste_percent) / 100.0
    material = base * factor if factor is not None else base
    formulas = ["Material Required = Quantity of Work / Output per Unit"]
    if factor is not None:
        formulas.append("Waste Factor = 1 + %Waste/100; Material × Waste Factor")
    return {
        "material_required": round(material, 6),
        "base_without_waste": round(base, 6),
        "waste_factor": factor,
        "formulas_used": formulas,
        "standard": "PE formula sheet (Material Consumption)",
        "note": "; ".join(formulas),
    }

@formula(
    owner='quantities',
    display_name='Concrete mix material quantities',
    description='Material volumes for a concrete volume from cement, sand and aggregate proportions.',
    inputs={
        'wet_volume': Param('m3', 0, 1e7, label='wet concrete volume'),
        'cement_parts': Param('ratio', 0, 20, label='cement parts'),
        'sand_parts': Param('ratio', 0, 20, label='sand parts'),
        'aggregate_parts': Param('ratio', 0, 20, label='aggregate parts'),
        'dry_volume_factor': Param('ratio', 1, 2, label='dry volume factor'),
        'waste_factor': Param('fraction', 0, 1, label='waste allowance'),
    },
    outputs={'wet_volume': '-', 'dry_volume': '-', 'dry_volume_factor': '-', 'proportions': '-', 'cement_volume': '-', 'sand_volume': '-', 'aggregate_volume': '-'},
)
def concrete_mix_proportions(
    wet_volume: Optional[float] = None,
    cement_parts: Optional[float] = None,
    sand_parts: Optional[float] = None,
    aggregate_parts: Optional[float] = None,
    dry_volume_factor: float = 1.54,
    waste_factor: Optional[float] = None,
) -> dict:
    """Concrete mix by caller-supplied cement:sand:aggregate proportions.

    Dry Volume = Wet Volume × dry_volume_factor (default 1.54 from PE sheets).
    ``waste_factor`` is a fraction from 0 to 1, so dry volume is increased by
    that share: ×(1 + waste_factor). Each constituent = Dry Volume × (parts /
    sum of parts). Refuses when wet volume or any mix part is missing — does
    not invent a design mix.
    """
    missing = [
        n for n, v in (
            ("wet_volume", wet_volume),
            ("cement_parts", cement_parts),
            ("sand_parts", sand_parts),
            ("aggregate_parts", aggregate_parts),
        )
        if v is None
    ]
    if missing:
        return {
            "error": (
                "concrete_mix_proportions requires wet_volume and "
                "cement_parts:sand_parts:aggregate_parts — refuse invented mixes."
            ),
            "required": ["wet_volume", "cement_parts", "sand_parts", "aggregate_parts"],
            "optional": ["dry_volume_factor", "waste_factor"],
            "missing": missing,
        }
    wet = float(wet_volume)
    c = float(cement_parts)
    s = float(sand_parts)
    a = float(aggregate_parts)
    if wet <= 0 or c < 0 or s < 0 or a < 0 or (c + s + a) <= 0:
        return {
            "error": "wet_volume must be > 0 and mix parts must sum to > 0.",
        }
    dry = wet * float(dry_volume_factor)
    if waste_factor is not None:
        dry = dry * (1.0 + float(waste_factor))
    total_parts = c + s + a
    cement_vol = dry * (c / total_parts)
    sand_vol = dry * (s / total_parts)
    agg_vol = dry * (a / total_parts)
    formulas = [
        f"Dry Volume = Wet Volume × {dry_volume_factor}",
        "Constituent = Dry Volume × (parts / total parts)",
    ]
    if waste_factor is not None:
        formulas.append(f"Dry Volume × (1 + waste_factor) ({waste_factor})")
    return {
        "wet_volume": wet,
        "dry_volume": round(dry, 6),
        "dry_volume_factor": float(dry_volume_factor),
        "proportions": f"{c:g}:{s:g}:{a:g}",
        "cement_volume": round(cement_vol, 6),
        "sand_volume": round(sand_vol, 6),
        "aggregate_volume": round(agg_vol, 6),
        "units": "same volume unit as wet_volume",
        "formulas_used": formulas,
        "standard": "PE formula sheet (Concrete Mix — Manual)",
        "note": (
            f"1:{s/c if c else '?'}:{a/c if c else '?'} mix on wet={wet}; "
            f"dry={dry:.4g}; cement={cement_vol:.4g}, sand={sand_vol:.4g}, "
            f"agg={agg_vol:.4g}."
        ),
    }
