"""A pressure converts by the platform's one conversion table.

Lane 4 (order 1310Z item 6): the PT grout calculator wrote its own factors
(x 145.038, rounded to a whole psi) instead of the platform converter: 0.5
N/mm2 showed as 73 psi where the converter gives 72.5. Both directions
agree with the converter.
"""
from app.lib import construction_formulas as cf


def test_the_grout_pressure_is_in_the_converters_psi_and_kgf():
    r = cf.run_calculation("grout_pressure_calc", {"tendon_duct_diameter_mm": 60})["result"]
    assert r["pressure_n_mm2"] == 0.5
    assert r["pressure_psi"] == 72.5
    assert r["pressure_kg_cm2"] == 5.1


def test_mpa_and_psi_convert_both_ways_by_the_same_table():
    psi = cf._convert_units(0.5, "MPa", "psi")["value_out"]
    back = cf._convert_units(psi, "psi", "MPa")["value_out"]
    assert round(psi, 1) == 72.5 and abs(back - 0.5) < 1e-9
