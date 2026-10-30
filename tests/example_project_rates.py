"""Example PROJECT rates for tests only.

The cost / mobilisation / supervision / programme calculators carry no
built-in rates; a project supplies them. These are the figures the old
in-code defaults used, kept here so the hand-derived arithmetic in the
existing tests still checks the maths -- they are test inputs, not
product data.
"""
CONCRETE_RATES = {
    "cement_price_per_t": 190, "aggregate_price_per_t": 29, "water_price_per_m3": 10,
    "microsilica_price_per_kg": 1.5, "plasticizer_price_per_lit": 2.4,
    "plant_cost_per_m3": 39.2, "labour_cost_per_m3": 8.0, "erection_per_m3": 3.0,
    "power_per_m3": 2.5, "indirect_per_m3": 4.0, "currency": "SAR",
}
REBAR_RATES = {"material_price_per_t": 2600, "labour_rate_per_hr": 4.1,
               "crane_rate_per_hr": 134.6, "currency": "SAR"}
FORMWORK_RATES = {"shuttering_supply_per_m2": 55, "scaffolding_per_m2_day": 2.15,
                  "labour_rate_per_hr": 4.1, "crane_rate_per_hr": 134.6, "currency": "SAR"}
MOBILIZATION_RATES = {
    "office_rate_per_person_month": 350, "camp_rate_per_person_month": 2800,
    "transport_rate_per_person_month": 220, "safety_medical_rate_per_person_month": 150,
    "it_rate_per_person_month": 85, "office_setup_cost": 450000, "camp_setup_cost": 60000, "remote_area_factor": 1.35,
    "safety_setup_cost": 40000, "warehouse_equipment_cost": 35000,
    "demobilization_cost": 280000, "currency": "SAR",
}
SUPERVISION_RATIOS = {
    "concrete_m3_per_supervisor": 35000, "steel_t_per_supervisor": 2500,
    "dia_inch_per_supervisor": 17500, "cable_km_per_supervisor": 65,
    "area_m2_per_supervisor": 20000, "supervisors_per_hse_officer": 8,
    "staff_per_document_controller": 15,
}
ELECTRICAL_STAGE_RATES = {"stage_output_rates": {
    "1st_fix_conduit_boxes": 50, "2nd_fix_equipment_panels": 75, "cabling_testing": 100,
    "final_test_temp_power": 100, "authority_inspection": 50, "final_fix_fittings": 75,
    "commissioning": 30}}
