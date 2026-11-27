"""Formulas from app.lib.construction_formulas_quantities owned by the quantities hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
import math
from decimal import Decimal


_STEEL_DENSITY = 7850.0  # kg/m^3

_WEIGHT_TO_LENGTH_MODES = {
    "weight_to_length",
    "weight2length",
    "to_length",
    "metres_run",
    "meters_run",
    "metre_run",
    "meter_run",
}

def _rebar_mass_kg(
    total_weight_kg: float,
    total_mass_kg: float,
    total_mass_t: float,
) -> float:
    for raw, scale in (
        (total_weight_kg, 1.0),
        (total_mass_kg, 1.0),
        (total_mass_t, 1000.0),
    ):
        if raw in (None, "", 0, 0.0):
            continue
        try:
            value = float(raw) * scale
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0

@formula(
    owner='quantities',
    display_name='Reinforcement weight',
    description='Mass of reinforcing bars from diameter and length, or the length from a mass.',
    inputs={
        'bar_diameter_mm': Param('mm', 4, 60, label='bar diameter'),
        'total_length_m': Param('m', 0, 1e8, label='total length'),
        'quantity': Param('count', 1, 1e7, label='number of bars'),
        'density_kg_m3': Param('kg/m3', 1000, 20000, label='steel density'),
        'total_weight_kg': Param('kg', 0, 1e8, label='total weight'),
        'total_mass_kg': Param('kg', 0, 1e8, label='total mass'),
        'total_mass_t': Param('t', 0, 100000, label='total mass in tonnes'),
        'mode': Param('-', label='calculation mode'),
    },
    outputs={'unit_mass_kg_m': 'kg/m', 'total_mass_kg': 'kg', 'total_mass_t': 't', 'total_length_m': 'm', 'quantity': '-', 'metres_run': '-', 'mode': '-'},
)
def rebar_weight(
    bar_diameter_mm: float,
    total_length_m: float = 0.0,
    quantity: int = 1,
    density_kg_m3: float = _STEEL_DENSITY,
    total_weight_kg: float = 0.0,
    total_mass_kg: float = 0.0,
    total_mass_t: float = 0.0,
    mode: str = "",
) -> dict:
    """Mass of reinforcement bars, or metres run from a given mass.

    Unit mass = (pi/4)*d^2 * density (kg/m), reproducing the standard
    bar-mass table (d=16 -> 1.578 kg/m at 7850 kg/m3).

    Live A2-2: ``mode=weight_to_length`` + ``total_weight_kg`` (12 t of
    Y16) returns metres run. ``total_length_m`` stays required for the
    original length → mass path.
    """
    d = float(bar_diameter_mm)
    area_m2 = math.pi / 4.0 * (d / 1000.0) ** 2
    unit_mass = area_m2 * density_kg_m3  # kg/m
    qty = int(quantity) if quantity not in (None, "") else 1
    if qty <= 0:
        qty = 1
    try:
        length = float(total_length_m or 0.0)
    except (TypeError, ValueError):
        length = 0.0
    mass_kg = _rebar_mass_kg(total_weight_kg, total_mass_kg, total_mass_t)
    want_length = str(mode or "").strip().lower() in _WEIGHT_TO_LENGTH_MODES
    if not want_length and mass_kg > 0 and length <= 0:
        want_length = True

    if want_length:
        if unit_mass <= 0:
            return {
                "error": (
                    "rebar_weight weight_to_length needs bar_diameter_mm > 0 "
                    "to compute unit mass."
                ),
            }
        if mass_kg <= 0:
            return {
                "error": (
                    "rebar_weight needs bar_diameter_mm (mm) and either "
                    "total_length_m (m) or total_weight_kg (kg) with "
                    "mode=weight_to_length."
                ),
                "required": ["bar_diameter_mm", "total_weight_kg"],
            }
        metres = mass_kg / unit_mass / qty
        return {
            "unit_mass_kg_m": round(unit_mass, 4),
            "total_mass_kg": round(mass_kg, 2),
            "total_mass_t": round(mass_kg / 1000.0, 4),
            "total_length_m": round(metres, 2),
            "metres_run": round(metres, 2),
            "quantity": qty,
            "mode": "weight_to_length",
            "standard": "BS 8666 / bar-mass relation",
            "note": (
                f"Unit mass = (pi/4)*({d}/1000)^2*{density_kg_m3:.0f} = "
                f"{unit_mass:.4f} kg/m; {mass_kg:g} kg / {unit_mass:.4f} "
                f"kg/m = {metres:.2f} m."
            ),
        }

    if length <= 0:
        return {
            "error": (
                "rebar_weight needs bar_diameter_mm (mm) and either "
                "total_length_m (m) or total_weight_kg (kg) with "
                "mode=weight_to_length."
            ),
            "required": ["bar_diameter_mm", "total_length_m"],
        }

    total = unit_mass * length * qty
    return {
        "unit_mass_kg_m": round(unit_mass, 4),
        "total_mass_kg": round(total, 2),
        "total_mass_t": round(total / 1000.0, 4),
        "total_length_m": round(length, 4),
        "quantity": qty,
        "standard": "BS 8666 / bar-mass relation",
        "note": (f"Unit mass = (pi/4)*({d}/1000)^2*{density_kg_m3:.0f} = "
                 f"{unit_mass:.4f} kg/m; x {length} m x {qty} = "
                 f"{total:.2f} kg."),
    }

@formula(
    owner='quantities',
    display_name='Reinforcement mass by area',
    description='Reinforcement mass for a slab or wall area from bar diameter and spacing.',
    inputs={
        'area_m2': Param('m2', 0, 1e8, label='area'),
        'spacing_mm': Param('mm', 10, 2000, label='bar spacing'),
        'bar_diameter_mm': Param('mm', 4, 60, label='bar diameter'),
        'both_ways': Param('-', label='both ways'),
        'density_kg_m3': Param('kg/m3', 1000, 20000, label='steel density'),
    },
    outputs={'bars_per_m': 'm', 'total_bar_length_m': 'm', 'total_mass_kg': 'kg', 'both_ways': '-'},
)
def rebar_by_area(
    area_m2: float,
    spacing_mm: float,
    bar_diameter_mm: float,
    both_ways: bool = False,
    density_kg_m3: float = _STEEL_DENSITY,
) -> dict:
    """Reinforcement mass for a slab/wall area from a bar spacing. Bars per metre
    = 1000/spacing; length per m2 ~= bars/m (one way) x area; both_ways doubles;
    bar mass per metre = pi/4 x d^2 x density."""
    d = float(bar_diameter_mm)
    unit_mass = math.pi / 4.0 * (d / 1000.0) ** 2 * density_kg_m3  # kg/m
    bars_per_m = 1000.0 / float(spacing_mm)
    length_per_m2 = bars_per_m  # 1 m run of bar per bar-line across a 1 m width
    ways = 2 if both_ways else 1
    total_length = length_per_m2 * float(area_m2) * ways
    total_mass = total_length * unit_mass
    return {
        "bars_per_m": round(bars_per_m, 3),
        "total_bar_length_m": round(total_length, 2),
        "total_mass_kg": round(total_mass, 2),
        "both_ways": both_ways,
        "standard": "geometry / bar-mass relation",
        "note": (f"1000/{spacing_mm} = {bars_per_m:.3f} bars/m x {area_m2} m2 x "
                 f"{ways} way(s) = {total_length:.2f} m; unit mass = "
                 f"(pi/4)*({d}/1000)^2*{density_kg_m3:g} = {unit_mass:.4f} kg/m; "
                 f"{total_length:.2f} m x {unit_mass:.4f} kg/m = {total_mass:.2f} kg."),
    }

def _floored(difference: float, reported: float) -> str:
    """The working for a quantity floored at zero: it states the floor it
    applied, so the printed line never reads ``a - b = 0`` for a negative."""
    if difference < 0:
        return f"{difference:.2f}, below zero: {reported:.2f}"
    return f"{reported:.2f}"

@formula(
    owner='quantities',
    display_name='Interior finishes take-off',
    description='Interior finish quantities (wall, ceiling, skirting) from a room take-off and the floor-to-ceiling height.',
    inputs={
        'floor_area_m2': Param('m2', 0, 1e8, label='floor area'),
        'perimeter_m': Param('m', 0, 100000, label='perimeter'),
        'room_count': Param('count', 1, 100000, label='number of rooms'),
        'floor_to_ceiling_m': Param('m', 0, 20, label='floor-to-ceiling height'),
        'door_width_m': Param('m', 0, 10, label='door width'),
        'door_height_m': Param('m', 0, 10, label='door height'),
        'doors_per_room': Param('count', 0, 50, label='doors per room'),
        'shared_wall_fraction': Param('fraction', 0, 1, label='shared-wall fraction'),
        'window_deduction_m2': Param('m2', 0, 1e7, label='window deduction'),
    },
    outputs={'floor_screed_m2': 'm2', 'floor_tiling_m2': 'm2', 'ceiling_finish_m2': 'm2', 'skirting_m': 'm', 'wall_paint_m2': 'm2', 'blockwork_m2': 'm2', 'gross_wall_face_m2': 'm2', 'door_deduction_m2': 'm2'},
)
def interior_finishes_takeoff(
    floor_area_m2: float,
    perimeter_m: float,
    room_count: int,
    floor_to_ceiling_m: float,
    door_width_m: float = 0.0,
    door_height_m: float = 0.0,
    doors_per_room: float = 0.0,
    shared_wall_fraction: float = 0.5,
    window_deduction_m2: float = 0.0,
) -> dict:
    """Interior finishes quantities from a room take-off -- PROCESS, not facts.

    Every project-specific value is a PARAMETER: the floor-to-ceiling height,
    door schedule, and window deductions change per project and per floor, so
    none of them has a baked-in value here. ``floor_to_ceiling_m`` is required
    -- calling without it is a refusal, not a guess. ``shared_wall_fraction``
    models internal walls being shared between two rooms (each wall face is
    counted once per room in the summed perimeters, so the wall ELEMENT area
    is roughly half the summed face area); it is an explicit modelling
    parameter, stated in the basis, adjustable per project.

    Outputs: screed/tiling/ceiling areas (= floor area), skirting length
    (perimeter minus door widths), wall paint area (perimeter x height minus
    door and window deductions), and blockwork element area (shared-wall
    fraction of the net wall face area). The ``basis`` echoes every input so
    the take-off is auditable line by line.
    """
    if floor_to_ceiling_m <= 0:
        return {
            "status": "error",
            "error": "floor_to_ceiling_m is required (> 0): the height varies "
                     "per project and per floor and is never assumed here. "
                     "Take it from the section drawing or project facts.",
        }
    fa = float(floor_area_m2)
    pm = float(perimeter_m)
    rooms = int(room_count)
    h = float(floor_to_ceiling_m)
    doors_total = rooms * float(doors_per_room)
    door_area = float(door_width_m) * float(door_height_m)
    door_area_total = doors_total * door_area
    door_width_total = doors_total * float(door_width_m)

    gross_wall_face = pm * h
    paint_difference = gross_wall_face - door_area_total - float(window_deduction_m2)
    net_wall_paint = max(paint_difference, 0.0)
    blockwork = net_wall_paint * float(shared_wall_fraction)
    skirting_difference = pm - door_width_total
    skirting = max(skirting_difference, 0.0)

    return {
        "floor_screed_m2": round(fa, 2),
        "floor_tiling_m2": round(fa, 2),
        "ceiling_finish_m2": round(fa, 2),
        "skirting_m": round(skirting, 2),
        "wall_paint_m2": round(net_wall_paint, 2),
        "blockwork_m2": round(blockwork, 2),
        "gross_wall_face_m2": round(gross_wall_face, 2),
        "door_deduction_m2": round(door_area_total, 2),
        "standard": "geometry (parameterised process)",
        "basis": [
            f"floor area {fa} m2; perimeter {pm} m; rooms {rooms}",
            f"floor-to-ceiling {h} m (project input, not a default)",
            f"doors: {doors_per_room}/room x {rooms} rooms, "
            f"{door_width_m} x {door_height_m} m = {door_area_total:.2f} m2 deducted",
            f"window deduction {window_deduction_m2} m2 (project input)",
            f"shared wall fraction {shared_wall_fraction} (modelling parameter)",
            "wall paint = perimeter x height - doors - windows; "
            "blockwork = wall face x shared fraction; "
            "skirting = perimeter - door widths",
            f"wall paint = {pm} x {h} - {door_area_total:.2f} - "
            f"{window_deduction_m2} = {_floored(paint_difference, net_wall_paint)} m2",
            f"blockwork = {net_wall_paint:.2f} x {shared_wall_fraction} = "
            f"{blockwork:.2f} m2",
            f"skirting = {pm} - {door_width_total:.2f} = "
            f"{_floored(skirting_difference, skirting)} m",
        ],
    }

@formula(
    owner='quantities',
    display_name='Estimate line cost split',
    description='Material, labour and plant split of one estimate line.',
    inputs={
        'quantity': Param('count', 0, 1e12, label='quantity'),
        'daily_output': Param('count/d', 0, 1e9, label='daily output'),
        'day_rate': Param('currency/d', 0, 1e9, label='day rate'),
        'material_rate_per_unit': Param('currency/count', 0, 1e9, label='material rate per unit'),
        'plant_fraction_of_labour': Param('ratio', 0, 10, label='plant cost as a fraction of labour'),
    },
    outputs={'quantity': '-', 'crew_days': 'days', 'labour_cost': 'currency', 'plant_cost': 'currency', 'material_cost': 'currency', 'total_cost': 'currency', 'total_is_partial': '-'},
)
def resource_line_cost(
    quantity: float,
    daily_output: float,
    day_rate: float,
    material_rate_per_unit: float | None = None,
    plant_fraction_of_labour: float = 0.0,
) -> dict:
    """Material / labour / plant split for one estimate line -- PROCESS only.

    The productivity (daily output per tradesman) and the day rate vary per
    market, project, and even floor: both are required inputs, never
    defaults. Labour = quantity / daily_output x day_rate. Material rate is
    optional; when absent the material cell is reported as
    NO SOURCED RATE rather than a number, mirroring the estimating rule that
    a missing rate is a refusal, not an invention. Plant is modelled as a
    declared fraction of labour (small plant and scaffold allowance).
    """
    if daily_output <= 0 or day_rate <= 0:
        return {
            "status": "error",
            "error": "daily_output and day_rate are required (> 0): "
                     "productivity and rates vary per project and are never "
                     "assumed here. Supply them from project facts, the "
                     "knowledge base, or the operator.",
        }
    q = float(quantity)
    crew_days = q / float(daily_output)
    labour = crew_days * float(day_rate)
    plant = labour * float(plant_fraction_of_labour)
    has_material = material_rate_per_unit is not None and float(material_rate_per_unit) >= 0
    material = q * float(material_rate_per_unit) if has_material else None
    total = labour + plant + (material or 0.0)
    return {
        "quantity": q,
        "crew_days": round(crew_days, 2),
        "labour_cost": round(labour, 2),
        "plant_cost": round(plant, 2),
        "material_cost": round(material, 2) if has_material else "NO SOURCED RATE",
        "total_cost": round(total, 2),
        "total_is_partial": not has_material,
        "standard": "resource build-up (parameterised process)",
        "note": (f"labour = {q}/{daily_output} x {day_rate} = {labour:.2f}; "
                 f"plant = {plant_fraction_of_labour} x labour = {plant:.2f}; "
                 + (f"material = {q} x {material_rate_per_unit} = {material:.2f}."
                    if has_material else "material: NO SOURCED RATE (no rate supplied).")),
    }


def _qty(x: float) -> str:
    """A working figure written out in full: thousands separators, no
    exponent, and every digit up to ten significant figures, so the figures
    printed in a step multiply to the result printed after it. ``:.10g``
    already drops trailing zeros and a bare point, so nothing is stripped."""
    return format(Decimal(f"{x:.10g}"), ",f")


def _as_shown(x: float) -> float:
    """The figure a reader sees for ``x`` (``_qty``), as a number: each step
    of the working computes from the figures printed before it, so the
    printed steps reconcile however many digits the inputs carry."""
    return float(Decimal(f"{x:.10g}"))


@formula(
    owner='quantities',
    display_name='Resource cost line',
    description=('Cost of one resource on an estimate line: quantity of work x '
                 'consumption of the resource per unit of work x waste factor x the '
                 'cited price per resource unit. Waste and markup are applied only '
                 'when they are supplied; no default markup is ever added.'),
    inputs={
        'quantity': Param('count', 0, 1e12, label='quantity of work'),
        'consumption_per_unit': Param('count/count', 0, 1e9,
                                      label='resource consumed per unit of work'),
        'price_per_resource_unit': Param('currency/count', 0, 1e9,
                                         label='cited price per resource unit'),
        'waste_percent': Param('%', 0, 100, label='waste allowance'),
        'markup_percent': Param('%', 0, 100, label='markup on the resource cost'),
    },
    outputs={'resource_quantity': '-', 'resource_cost': 'currency',
             'markup_amount': 'currency', 'line_total': 'currency', 'working': '-'},
)
def resource_cost_line(
    quantity: float,
    consumption_per_unit: float,
    price_per_resource_unit: float,
    waste_percent: float | None = None,
    markup_percent: float | None = None,
) -> dict:
    """Resource-based cost line: qty x consumption x waste x cited price.

    The quantity, the consumption rate and the price are project facts (a
    measured quantity, a norm or the document's own build-up, a cited
    quotation or rate card), so none has a default and a missing one is
    named. Waste and markup are optional: when absent, none is applied and
    the working says so -- a markup is never assumed.
    """
    supplied = {"quantity": quantity, "consumption_per_unit": consumption_per_unit,
                "price_per_resource_unit": price_per_resource_unit}
    missing = [k for k, v in supplied.items() if v is None]
    if missing:
        return {"error": ", ".join(missing) + " required; no default is assumed.",
                "missing": missing}
    # ``+ 0.0`` turns a negative zero into zero, so no figure or result is
    # ever written "-0".
    q = float(quantity) + 0.0
    rate = float(consumption_per_unit) + 0.0
    price = float(price_per_resource_unit) + 0.0
    if waste_percent is not None:
        waste_percent = float(waste_percent) + 0.0
    if markup_percent is not None:
        markup_percent = float(markup_percent) + 0.0
    optional = [v for v in (waste_percent, markup_percent) if v is not None]
    if min([q, rate, price] + optional) < 0:
        return {"error": "quantity, consumption_per_unit, price_per_resource_unit, "
                         "waste_percent and markup_percent must be >= 0."}
    net = _as_shown(q * rate)
    working = [(f"Net resource = quantity {_qty(q)} x consumption {_qty(rate)} "
                f"= {_qty(net)}")]
    if waste_percent is None:
        factor = None
        gross = net
        working.append("No waste allowance supplied: none applied.")
    else:
        # The factor is written exactly as the percent gives it (1 + w/100
        # in decimal), so the printed net x printed factor is the printed
        # gross.
        exact = 1 + Decimal(repr(waste_percent)) / 100
        factor = float(exact)
        gross = _as_shown(float(Decimal(repr(net)) * exact))
        working.append(f"With waste {_qty(waste_percent)}% = {_qty(net)} x "
                       f"{exact:f} = {_qty(gross)}")
    cost = gross * price
    working.append(f"Resource cost = {_qty(gross)} x price {_qty(price)} = {cost:,.2f}")
    if markup_percent is None:
        markup = None
        total = cost
        working.append("No markup supplied: none applied (no default markup).")
    else:
        markup = cost * float(markup_percent) / 100.0
        total = cost + markup
        working.append(f"Markup {_qty(float(markup_percent))}% = {markup:,.2f}; "
                       f"line total = {cost:,.2f} + {markup:,.2f} = {total:,.2f}")
    return {
        "resource_quantity_net": round(net, 6),
        "waste_factor": factor,
        "resource_quantity": round(gross, 6),
        "resource_cost": round(cost, 2),
        "markup_amount": round(markup, 2) if markup is not None else None,
        "line_total": round(total, 2),
        "standard": "resource-based estimating (quantity x consumption x waste x price)",
        "working": working,
    }
