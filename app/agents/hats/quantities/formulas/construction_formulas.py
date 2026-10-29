"""Formulas from app.lib.construction_formulas owned by the quantities hat.

Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).
"""
from __future__ import annotations
from app.lib.formula_registry import Param, formula
from typing import Any, Dict


@formula(
    owner='quantities',
    display_name='Crane hire cost',
    description='Crane cost over a hire period including operator and rigging crew.',
    inputs={
        'num_cranes': Param('count', 1, 100, label='number of cranes'),
        'crane_capacity_tons': Param('t', 0, 10000, label='crane capacity'),
        'duration_months': Param('month', 1, 240, label='duration'),
        'dry_hire_rate_per_month': Param('currency/month', 0, 1e9, label='dry hire rate per crane per month'),
        'include_operator': Param('-', label='include operator'),
        'include_riggers': Param('-', label='include riggers'),
        'operator_rate_per_month': Param('currency/month', 0, 1e9, label='operator cost per crane per month'),
        'riggers_rate_per_month': Param('currency/month', 0, 1e9, label='rigging crew cost per crane per month'),
        'remote_area_factor': Param('ratio', 1, 3, label='remote area factor'),
        'currency': Param('-', label='currency of the rates'),
    },
    outputs={'num_cranes': '-', 'crane_capacity_tons': '-', 'duration_months': '-', 'dry_hire_per_month': 'currency/month', 'operator_per_month': 'currency/month', 'riggers_per_month': 'currency/month', 'remote_area_factor': '-', 'per_crane_monthly': 'currency/month', 'total_project_cost': 'currency', 'mobilization': 'currency', 'demobilization': 'currency', 'grand_total': 'currency', 'currency': '-'},
)
def crane_cost_estimate(
    num_cranes: int,
    crane_capacity_tons: float,
    duration_months: int,
    dry_hire_rate_per_month: float,
    include_operator: bool = True,
    include_riggers: bool = True,
    operator_rate_per_month: Any = None,
    riggers_rate_per_month: Any = None,
    remote_area_factor: float = 1.0,
    currency: str = "",
) -> Dict[str, Any]:
    """
    Crane hire cost over a period. Every rate is the PROJECT's rate, supplied
    by the caller (priced BOQ / quotation / a cited, dated rate item) -- no
    rate or currency is built in. Output keys are currency-neutral; the
    ``currency`` input is echoed back.

    Mobilisation is 50% of one month's dry hire per crane, demobilisation
    75% of mobilisation (method ratios, not prices).
    """
    if num_cranes < 0 or duration_months < 0:
        raise ValueError("num_cranes and duration_months must be >= 0")
    if crane_capacity_tons <= 0:
        raise ValueError("crane_capacity_tons must be > 0")
    if remote_area_factor < 0:
        raise ValueError("remote_area_factor must be >= 0")
    if dry_hire_rate_per_month < 0:
        raise ValueError("dry_hire_rate_per_month must be >= 0")
    missing = []
    if include_operator and operator_rate_per_month in (None, ""):
        missing.append("operator_rate_per_month")
    if include_riggers and riggers_rate_per_month in (None, ""):
        missing.append("riggers_rate_per_month")
    if missing:
        return {"error": "crew included but its rate was not given", "needs_input": missing}
    dry_hire_monthly = float(dry_hire_rate_per_month)
    operator_monthly = float(operator_rate_per_month) if include_operator else 0.0
    riggers_monthly = float(riggers_rate_per_month) if include_riggers else 0.0

    per_crane_monthly = (dry_hire_monthly + operator_monthly + riggers_monthly) * remote_area_factor
    total_project_cost = per_crane_monthly * num_cranes * duration_months
    mobilization = num_cranes * dry_hire_monthly * 0.5
    demobilization = mobilization * 0.75

    return {
        "num_cranes": num_cranes,
        "crane_capacity_tons": crane_capacity_tons,
        "duration_months": duration_months,
        "dry_hire_per_month": dry_hire_monthly,
        "operator_per_month": operator_monthly,
        "riggers_per_month": riggers_monthly,
        "remote_area_factor": remote_area_factor,
        "per_crane_monthly": round(per_crane_monthly, 0),
        "total_project_cost": round(total_project_cost, 0),
        "mobilization": round(mobilization, 0),
        "demobilization": round(demobilization, 0),
        "grand_total": round(total_project_cost + mobilization + demobilization, 0),
        "currency": currency,
    }

@formula(
    owner='quantities',
    display_name='Concrete cost build-up',
    description='Cost build-up of concrete per cubic metre from materials, labour, plant and overheads.',
    inputs={
        'quantity_m3': Param('m3', 0, 1e7, label='quantity'),
        'cement_kg_m3': Param('kg/m3', 0, 1000, label='cement content'),
        'cement_price_per_t': Param('currency/t', 0, 1e6, label='cement price'),
        'aggregate_price_per_t': Param('currency/t', 0, 1e6, label='aggregate price'),
        'water_price_per_m3': Param('currency/m3', 0, 100000, label='water price'),
        'microsilica_kg_m3': Param('kg/m3', 0, 200, label='microsilica content'),
        'microsilica_price_per_kg': Param('currency/kg', 0, 100000, label='microsilica price'),
        'plasticizer_lit_m3': Param('l/m3', 0, 100, label='plasticizer dose per m3'),
        'plasticizer_price_per_lit': Param('currency/l', 0, 100000, label='plasticizer price'),
        'plant_cost_per_m3': Param('currency/m3', 0, 100000, label='plant cost'),
        'labour_cost_per_m3': Param('currency/m3', 0, 100000, label='labour cost'),
        'erection_per_m3': Param('currency/m3', 0, 100000, label='erection cost'),
        'power_per_m3': Param('currency/m3', 0, 100000, label='power cost'),
        'indirect_per_m3': Param('currency/m3', 0, 100000, label='indirect cost'),
        'indirect_pct': Param('fraction', 0, 1, label='indirect cost share'),
        'markup_pct': Param('fraction', 0, 1, label='markup'),
        'waste_pct': Param('fraction', 0, 1, label='waste allowance'),
        'currency': Param('-', label='currency of the rates'),
    },
    outputs={'material_cost_per_m3': 'currency/m3', 'direct_cost_per_m3': 'currency/m3', 'selling_price_per_m3': 'currency/m3', 'total_project_value': 'currency'},
    main_output='selling_price_per_m3',
)
def cost_buildup_concrete(
    quantity_m3: float,
    cement_kg_m3: float = 400,
    *,
    cement_price_per_t: float,
    aggregate_price_per_t: float,
    water_price_per_m3: float,
    microsilica_kg_m3: float = 12,
    microsilica_price_per_kg: float,
    plasticizer_lit_m3: float = 5,
    plasticizer_price_per_lit: float,
    plant_cost_per_m3: float,
    labour_cost_per_m3: float,
    erection_per_m3: float,
    power_per_m3: float,
    indirect_per_m3: float,
    indirect_pct: float = 0.18,
    markup_pct: float = 0.15,
    waste_pct: float = 0.03,
    currency: str = "",
) -> Dict[str, Any]:
    """Full concrete cost build-up per m3, from the project's own rates
    (no built-in prices; every price is a required input).

    Waste is a material allowance (batching loss), same as
    ``cost_buildup_rebar``. Plant, labour, erection, power and the
    indirect per-m3 line are not wasted. The indirect percent is applied
    once, on that direct cost. Selling price is

        direct × (1 + indirect_pct) / (1 − markup_pct)

    Documented batch masses (not caller inputs): combined aggregate
    1839 kg/m3, mixing water 160 L/m3 (= 0.16 m3).
    """
    if quantity_m3 < 0:
        raise ValueError("quantity_m3 must be >= 0")
    cement_cost = (cement_kg_m3 / 1000) * cement_price_per_t
    aggregate_cost = (1839 / 1000) * aggregate_price_per_t  # ~1839 kg/m3 combined
    water_cost = (160 / 1000) * water_price_per_m3  # 160 L/m3 = 0.16 m3
    material_cost = (
        cement_cost
        + aggregate_cost
        + water_cost
        + (microsilica_kg_m3 * microsilica_price_per_kg)
        + (plasticizer_lit_m3 * plasticizer_price_per_lit)
    )
    wasted_material = material_cost * (1 + waste_pct)
    direct = (
        wasted_material
        + plant_cost_per_m3
        + labour_cost_per_m3
        + erection_per_m3
        + power_per_m3
        + indirect_per_m3
    )
    selling = direct * (1 + indirect_pct) / (1 - markup_pct)
    selling_r = round(selling, 2)
    total_r = round(selling_r * quantity_m3, 0)
    _cu = f" {currency}" if currency else ""
    return {
        "material_cost_per_m3": round(material_cost, 2),
        "direct_cost_per_m3": round(direct, 2),
        "selling_price_per_m3": selling_r,
        "total_project_value": total_r,
        "note": (
            f"Material {material_cost:.2f}{_cu}/m3 × (1+{waste_pct:g} waste) "
            f"= {wasted_material:.2f}; direct {direct:.2f}{_cu}/m3 "
            f"(plant, labour, erection, power and the indirect line are "
            f"not wasted); selling {selling_r:.2f}{_cu}/m3 "
            f"= direct × (1+{indirect_pct:g}) / (1-{markup_pct:g}); "
            f"total {total_r:.0f}{_cu} for {quantity_m3:g} m3."
        ),
        "currency": currency,
    }

@formula(
    owner='quantities',
    display_name='Reinforcement cost build-up',
    description='Cost build-up of reinforcement per tonne from material, fabrication, fixing and overheads.',
    inputs={
        'quantity_kg': Param('kg', 0, 1e8, label='quantity'),
        'material_price_per_t': Param('currency/t', 0, 1e6, label='material price'),
        'labour_mhr_t': Param('man-hour/t', 0, 10000, label='labour hours per tonne'),
        'labour_rate_per_hr': Param('currency/h', 0, 100000, label='labour rate'),
        'crane_hr_t': Param('h/t', 0, 1000, label='crane hours per tonne'),
        'crane_rate_per_hr': Param('currency/h', 0, 1e6, label='crane rate'),
        'waste_pct': Param('fraction', 0, 1, label='waste allowance'),
        'indirect_pct': Param('fraction', 0, 1, label='indirect cost share'),
        'markup_pct': Param('fraction', 0, 1, label='markup'),
        'currency': Param('-', label='currency of the rates'),
    },
    outputs={'material_per_t': 'currency/t', 'labour_per_t': 'currency/t', 'equipment_per_t': 'currency/t', 'selling_price_per_t': 'currency/t', 'total_project_value': 'currency'},
    main_output='selling_price_per_t',
)
def cost_buildup_rebar(
    quantity_kg: float,
    *,
    material_price_per_t: float,
    labour_mhr_t: float = 90,
    labour_rate_per_hr: float,
    crane_hr_t: float = 2,
    crane_rate_per_hr: float,
    waste_pct: float = 0.10,
    indirect_pct: float = 0.18,
    markup_pct: float = 0.15,
    currency: str = "",
) -> Dict[str, float]:
    """Rebar cost build-up per tonne, from the project's own rates (no built-in prices)."""
    if quantity_kg <= 0:
        raise ValueError("quantity_kg must be > 0")
    qty_t = quantity_kg / 1000
    material = qty_t * material_price_per_t * (1 + waste_pct)
    labour = qty_t * labour_mhr_t * labour_rate_per_hr
    equipment = qty_t * crane_hr_t * crane_rate_per_hr
    direct = material + labour + equipment
    with_markup = direct * (1 + indirect_pct) / (1 - markup_pct)
    sell_t = round(with_markup / qty_t, 0) if qty_t > 0 else 0
    return {
        "material_per_t": round(material / qty_t, 0) if qty_t > 0 else 0,
        "labour_per_t": round(labour / qty_t, 0) if qty_t > 0 else 0,
        "equipment_per_t": round(equipment / qty_t, 0) if qty_t > 0 else 0,
        "selling_price_per_t": sell_t,
        "total_project_value": round(sell_t * qty_t, 0) if qty_t > 0 else 0,
        "currency": currency,
    }

@formula(
    owner='quantities',
    display_name='Formwork cost build-up',
    description='Cost build-up of formwork per square metre from materials, reuses, labour and overheads.',
    inputs={
        'area_m2': Param('m2', 0, 1e8, label='area'),
        'shuttering_supply_per_m2': Param('currency/m2', 0, 100000, label='shuttering supply cost'),
        'scaffolding_per_m2_day': Param('currency/m2/day', 0, 10000, label='scaffolding cost'),
        'cycle_days': Param('days', 1, 365, label='formwork cycle'),
        'labour_mhr_m2': Param('man-hour/m2', 0, 1000, label='labour hours per m2'),
        'labour_rate_per_hr': Param('currency/h', 0, 100000, label='labour rate'),
        'crane_rate_per_hr': Param('currency/h', 0, 1e6, label='crane rate'),
        'crane_output_m2_hr': Param('m2/h', 0, 100000, label='crane output'),
        'indirect_pct': Param('fraction', 0, 1, label='indirect cost share'),
        'markup_pct': Param('fraction', 0, 1, label='markup'),
        'currency': Param('-', label='currency of the rates'),
    },
    outputs={'shuttering_per_m2': 'currency/m2', 'scaffolding_per_m2': 'currency/m2', 'material_per_m2': 'currency/m2', 'labour_per_m2': 'currency/m2', 'selling_price_per_m2': 'currency/m2', 'total_for_area': 'currency'},
    main_output='selling_price_per_m2',
)
def cost_buildup_formwork(
    area_m2: float,
    *,
    shuttering_supply_per_m2: float,
    scaffolding_per_m2_day: float,
    cycle_days: int = 10,
    labour_mhr_m2: float = 9,
    labour_rate_per_hr: float,
    crane_rate_per_hr: float,
    crane_output_m2_hr: float = 30,
    indirect_pct: float = 0.18,
    markup_pct: float = 0.15,
    currency: str = "",
) -> Dict[str, float]:
    """Formwork cost build-up per m2, from the project's own rates (no built-in prices)."""
    if area_m2 < 0:
        raise ValueError("area_m2 must be >= 0")
    if crane_output_m2_hr <= 0:
        raise ValueError("crane_output_m2_hr must be > 0")
    shuttering = shuttering_supply_per_m2 / 6  # 6 uses
    scaffolding = scaffolding_per_m2_day * cycle_days
    material = shuttering + scaffolding
    labour = labour_mhr_m2 * labour_rate_per_hr
    equipment = crane_rate_per_hr / crane_output_m2_hr
    direct = material + labour + equipment
    with_markup = direct * (1 + indirect_pct) / (1 - markup_pct)
    return {
        "shuttering_per_m2": round(shuttering, 2),
        "scaffolding_per_m2": round(scaffolding, 2),
        "material_per_m2": round(material, 2),
        "labour_per_m2": round(labour, 2),
        "selling_price_per_m2": round(with_markup, 2),
        "total_for_area": round(with_markup * area_m2, 0),
        "currency": currency,
    }

@formula(
    owner='quantities',
    display_name='Mobilisation cost',
    description='Mobilisation cost for site set-up from its component items, at the project\'s own rates.',
    inputs={
        'num_personnel': Param('count', 1, 100000, label='number of personnel'),
        'duration_months': Param('month', 1, 240, label='duration'),
        'camp_type': Param('-', label='camp type'),
        'include_offices': Param('-', label='include offices'),
        'include_camp': Param('-', label='include camp'),
        'include_transport': Param('-', label='include transport'),
        'include_safety_medical': Param('-', label='include safety and medical'),
        'remote_area_factor': Param('ratio', 1, 3, label='remote area factor'),
        'office_rate_per_person_month': Param('currency/month', 0, 1e7, label='office cost per person per month'),
        'camp_rate_per_person_month': Param('currency/month', 0, 1e7, label='camp cost per person per month'),
        'transport_rate_per_person_month': Param('currency/month', 0, 1e7, label='transport cost per person per month'),
        'safety_medical_rate_per_person_month': Param('currency/month', 0, 1e7, label='safety and medical cost per person per month'),
        'it_rate_per_person_month': Param('currency/month', 0, 1e7, label='IT cost per person per month'),
        'office_setup_cost': Param('currency', 0, 1e10, label='office set-up cost'),
        'camp_setup_cost': Param('currency', 0, 1e10, label='camp set-up cost'),
        'safety_setup_cost': Param('currency', 0, 1e10, label='safety set-up cost'),
        'warehouse_equipment_cost': Param('currency', 0, 1e10, label='warehouse equipment cost'),
        'demobilization_cost': Param('currency', 0, 1e10, label='demobilisation cost'),
        'currency': Param('-', label='currency of the rates'),
    },
    outputs={'num_personnel': '-', 'duration_months': '-', 'camp_type': '-', 'remote_area_factor': '-', 'breakdown': '-', 'recurring_monthly': 'currency', 'total_recurring': 'currency', 'total_fixed': 'currency', 'grand_total': 'currency', 'currency': '-'},
    main_output='grand_total',
)
def mobilization_cost_estimate(
    num_personnel: int,
    duration_months: int,
    camp_type: str = "",
    include_offices: bool = True,
    include_camp: bool = True,
    include_transport: bool = True,
    include_safety_medical: bool = True,
    remote_area_factor: float = 1.0,
    office_rate_per_person_month: Any = None,
    camp_rate_per_person_month: Any = None,
    transport_rate_per_person_month: Any = None,
    safety_medical_rate_per_person_month: Any = None,
    it_rate_per_person_month: Any = None,
    office_setup_cost: Any = None,
    camp_setup_cost: Any = None,
    safety_setup_cost: Any = None,
    warehouse_equipment_cost: Any = None,
    demobilization_cost: Any = None,
    currency: str = "",
) -> Dict[str, Any]:
    """
    Mobilisation cost for site set-up: offices, camp, transport,
    safety/medical, IT, warehouse equipment, demobilisation.

    Every rate and lump sum is the PROJECT's (priced BOQ / quotation),
    a required input for each included item -- no price is built in.
    ``camp_type`` is a label only. ``remote_area_factor`` multiplies all
    lines (1.0 unless the caller states one).
    """
    def _v(x):
        return 0.0 if x in (None, "") else float(x)

    need = {
        "office_rate_per_person_month": include_offices,
        "office_setup_cost": include_offices,
        "camp_rate_per_person_month": include_camp,
        "camp_setup_cost": include_camp,
        "transport_rate_per_person_month": include_transport,
        "safety_medical_rate_per_person_month": include_safety_medical,
        "safety_setup_cost": include_safety_medical,
        "it_rate_per_person_month": True,
        "warehouse_equipment_cost": True,
        "demobilization_cost": True,
    }
    given = dict(locals())
    missing = [k for k, on in need.items() if on and given.get(k) in (None, "")]
    if missing:
        return {"error": "project rates not given", "needs_input": missing}

    office_pp = _v(office_rate_per_person_month) if include_offices else 0.0
    camp_pp = _v(camp_rate_per_person_month) if include_camp else 0.0
    transport_pp = _v(transport_rate_per_person_month) if include_transport else 0.0
    safety_pp = _v(safety_medical_rate_per_person_month) if include_safety_medical else 0.0

    total_monthly = (office_pp + camp_pp + transport_pp + safety_pp) * num_personnel * remote_area_factor

    office_setup = _v(office_setup_cost) if include_offices else 0.0
    camp_setup = _v(camp_setup_cost) if include_camp else 0.0
    safety_setup = _v(safety_setup_cost) if include_safety_medical else 0.0
    demobilization = _v(demobilization_cost)
    it_costs = num_personnel * _v(it_rate_per_person_month) * duration_months
    warehouse_equip = _v(warehouse_equipment_cost)

    recurring = total_monthly * duration_months
    fixed = (office_setup + camp_setup + demobilization + safety_setup +
             it_costs + warehouse_equip) * remote_area_factor
    f = remote_area_factor
    return {
        "num_personnel": num_personnel,
        "duration_months": duration_months,
        "camp_type": camp_type,
        "remote_area_factor": remote_area_factor,
        "breakdown": {
            "offices": round((office_setup + office_pp * num_personnel * duration_months) * f, 0),
            "camp": round((camp_setup + camp_pp * num_personnel * duration_months) * f, 0),
            "transport": round(transport_pp * num_personnel * duration_months * f, 0),
            "safety_medical": round((safety_setup + safety_pp * num_personnel * duration_months) * f, 0),
            "it": round(it_costs * f, 0),
            "warehouse_equipment": round(warehouse_equip * f, 0),
            "demobilization": round(demobilization * f, 0),
        },
        "recurring_monthly": round(total_monthly, 0),
        "total_recurring": round(recurring, 0),
        "total_fixed": round(fixed, 0),
        "grand_total": round(recurring + fixed, 0),
        "currency": currency,
    }

