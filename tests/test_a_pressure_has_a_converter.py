"""Pressure and stress had no conversion table, so every one was done by hand.

``pe_unit_convert`` carried four families -- m/ft, m2/ft2, m3/ft3, day/hour --
and nothing for pressure. Construction asks for pressure conversions constantly
(concrete grades in MPa and psi, bearing capacity in kg/cm2, a spec in N/mm2),
and each one fell back to arithmetic in the answer text with no tool behind it.

Live SET5:

    E15  "Convert 40 MPa to psi."          -- one run returned nothing at all
    U3   "Convert 250 kg/cm2 to MPa."      -- 24.5, and 25.0 if you divide by 10
    U4   "Express 0.5 N/mm2 in kPa."       -- 500

U3 and U4 passed on hand arithmetic; E15 did not. The same shape as SET5 E3,
where the only registered cantilever calculator was the UDL one: when the tool
for a question does not exist, the answer is a guess that usually looks right.

kg/cm2 is the trap worth naming. It is 0.0980665 MPa, not 0.1 -- dividing by
ten is a 2% error that reads as a rounding difference.
"""
import pytest

from app.lib import construction_formulas as cf

CALC = "pe_unit_convert"


def _out(value, src, dst):
    env = cf.run_calculation(CALC, {"value": value, "from_unit": src, "to_unit": dst})
    assert env["status"] == "success", env
    r = env["result"]
    assert "error" not in r, r
    return r


# ── the three live cases ───────────────────────────────────────────────────

def test_e15_mpa_to_psi():
    assert _out(40, "MPa", "psi")["value_out"] == pytest.approx(5801.5, abs=1.0)


def test_u3_kg_per_cm2_to_mpa():
    r = _out(250, "kg/cm2", "MPa")
    assert r["value_out"] == pytest.approx(24.52, abs=0.02)
    assert r["value_out"] != pytest.approx(25.0, abs=0.02), (
        "dividing by 10 instead of converting is the live trap")


def test_u4_n_per_mm2_to_kpa():
    assert _out(0.5, "N/mm2", "kPa")["value_out"] == pytest.approx(500.0, abs=0.01)


# ── the spellings a question actually uses ─────────────────────────────────

@pytest.mark.parametrize("spelling", [
    "MPa", "mpa", "N/mm2", "N/mm²", "n/mm2", "MN/m2",
])
def test_one_megapascal_by_any_name(spelling):
    assert _out(1, spelling, "kPa")["value_out"] == pytest.approx(1000.0, abs=0.01)


@pytest.mark.parametrize("src,expect", [
    ("bar", 100.0), ("kg/cm2", 98.0665), ("psi", 6.894757),
    ("kN/m2", 1.0), ("Pa", 0.001), ("GPa", 1_000_000.0),
])
def test_the_family_converts_to_kpa(src, expect):
    assert _out(1, src, "kPa")["value_out"] == pytest.approx(expect, rel=1e-6)


def test_ksi_is_a_thousand_psi():
    assert _out(1, "ksi", "psi")["value_out"] == pytest.approx(1000.0, abs=0.01)


# ── it stays a conversion, not a reinterpretation ──────────────────────────

def test_a_round_trip_returns_the_same_number():
    there = _out(40, "MPa", "psi")["value_out"]
    assert _out(there, "psi", "MPa")["value_out"] == pytest.approx(40.0, abs=1e-6)


def test_the_dimension_is_named_so_the_answer_can_cite_it():
    assert _out(40, "MPa", "psi")["dimension"] == "pressure"


def test_a_pressure_cannot_be_converted_to_a_length():
    env = cf.run_calculation(CALC, {"value": 40, "from_unit": "MPa", "to_unit": "m"})
    assert "error" in env.get("result", env), env


def test_the_four_original_families_are_untouched():
    assert _out(1, "m", "ft")["value_out"] == pytest.approx(3.28084, abs=1e-4)
    assert _out(1, "day", "hour")["value_out"] == pytest.approx(24.0, abs=1e-9)
    assert _out(1, "m3", "ft3")["value_out"] == pytest.approx(35.3147, abs=1e-3)


def test_the_error_message_lists_pressure_now():
    env = cf.run_calculation(CALC, {"value": 1, "from_unit": "furlong", "to_unit": "m"})
    assert "psi" in str(env).lower() or "pressure" in str(env).lower()
