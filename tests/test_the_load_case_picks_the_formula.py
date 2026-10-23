"""A point load is not a distributed load, and the tool must be able to say so.

Live SET5 E3, failed on both runs: a 4 m cantilever with a **30 kN point load
at the tip**, E = 200 GPa, I = 3.0e-4 m4. The answer was **16.0 mm**. That is
wL^4/8EI -- the cantilever *UDL* formula -- with the point load's 30 substituted
for w. The right answer is PL^3/3EI = 10.67 mm.

The cause was not a mis-selection the model could have avoided: until this
change ``beam_deflection_cantilever_udl`` was the only cantilever deflection
calculator in the registry, so the nearest available tool WAS the wrong one.
The arithmetic is then internally consistent and the working reads correctly,
which is why the derivation check in
``test_the_working_and_the_result_must_agree.py`` cannot catch this shape --
every number shown is right for the formula that was used.

The second half of this file is the unit trap sitting underneath: the question
states I in **m4**, the formulas take **mm4**, and nothing converted between
them. Passing 3.0e-4 straight through returns a deflection 1e12 times too
large. A figure that wrong must be refused, not printed.
"""
import pytest

from app.lib import construction_formulas as cf

CANTILEVER_P = "beam_deflection_cantilever_point_load"
SS_P = "beam_deflection_ss_point_load_midspan"
CANTILEVER_W = "beam_deflection_cantilever_udl"
SS_W = "beam_deflection_ss_udl"

# E = 200 GPa = 200,000 MPa; I = 3.0e-4 m4 = 3.0e8 mm4.
E3 = {"p_kn": 30.0, "span_m": 4.0, "ec_mpa": 200_000.0, "i_mm4": 3.0e8}


def _result(calc, params):
    """The deflection in mm. A bare float comes back wrapped as {"value": x}."""
    out = cf.run_calculation(calc, params)
    assert out["status"] == "success", out
    result = out["result"]
    return result["value"] if isinstance(result, dict) else result


def _error(calc, params):
    out = cf.run_calculation(calc, params)
    assert out["status"] != "success", out
    return out


# ── the live defect ────────────────────────────────────────────────────────

def test_the_cantilever_tip_point_load_case():
    assert _result(CANTILEVER_P, E3) == pytest.approx(10.67, abs=0.01)


def test_the_udl_formula_is_what_produced_the_wrong_answer():
    # Documents the defect: same numbers, distributed formula, 16.0 mm. This
    # tool is still right for its own load case -- it was simply the only one.
    wrong = _result(CANTILEVER_W, {"w_kn_m": 30.0, "span_m": 4.0,
                                   "ec_mpa": 200_000.0, "i_mm4": 3.0e8})
    assert wrong == pytest.approx(16.0, abs=0.01)
    assert wrong != pytest.approx(_result(CANTILEVER_P, E3), abs=0.5), (
        "the two load cases must not be interchangeable")


def test_the_point_load_formula_is_registered_so_it_can_be_chosen():
    assert CANTILEVER_P in cf.CALCULATORS
    assert SS_P in cf.CALCULATORS


# ── the simply supported sibling, so the pair is complete ──────────────────

def test_the_simply_supported_midspan_point_load():
    # PL^3/48EI: 50 kN at midspan of 8 m, E 200 GPa, I 2.0e8 mm4.
    d = _result(SS_P, {"p_kn": 50.0, "span_m": 8.0,
                       "ec_mpa": 200_000.0, "i_mm4": 2.0e8})
    assert d == pytest.approx(50_000 * 8000**3 / (48 * 200_000 * 2.0e8), abs=0.01)


def test_a_point_load_deflects_less_than_the_same_number_as_udl():
    # 5wL^4/384EI vs PL^3/48EI -- a sanity relation that holds for any span,
    # and would fail immediately if a formula were transcribed wrongly.
    common = {"span_m": 6.0, "ec_mpa": 200_000.0, "i_mm4": 1.5e8}
    point = _result(SS_P, {"p_kn": 30.0, **common})
    udl = _result(SS_W, {"w_kn_m": 30.0, **common})
    assert point < udl


# ── m4 stated, mm4 expected: refuse, never print ───────────────────────────

@pytest.mark.parametrize("calc,load", [
    (CANTILEVER_P, {"p_kn": 30.0}),
    (SS_P, {"p_kn": 30.0}),
    (CANTILEVER_W, {"w_kn_m": 30.0}),
    (SS_W, {"w_kn_m": 30.0}),
])
def test_a_second_moment_in_m4_is_refused_not_silently_used(calc, load):
    out = _error(calc, {"span_m": 4.0, "ec_mpa": 200_000.0,
                        "i_mm4": 3.0e-4, **load})
    assert "m4" in str(out).lower() or "mm4" in str(out).lower(), (
        "the error must name the unit so the caller can fix the input")


def test_the_refusal_says_how_to_fix_the_input():
    # The caller has to be able to recover from this without guessing, or it
    # falls back to hand arithmetic -- which is where 16.0 mm came from.
    out = _error(CANTILEVER_P, {**E3, "i_mm4": 3.0e-4})
    assert "1e12" in str(out)


def test_i_mm4_stays_a_required_parameter():
    # Not given a default: the sweep and the tool schema both read "no
    # default" as "required", and a second moment the model may omit is a
    # deflection computed from a guess.
    import inspect
    for fn in (cf.beam_deflection_cantilever_point_load,
               cf.beam_deflection_ss_point_load_midspan,
               cf.beam_deflection_cantilever_udl,
               cf.beam_deflection_ss_udl):
        assert inspect.signature(fn).parameters["i_mm4"].default is inspect.Parameter.empty


# ── the ordinary guards ────────────────────────────────────────────────────

def test_zero_load_deflects_nothing():
    assert _result(CANTILEVER_P, {**E3, "p_kn": 0.0}) == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [{"ec_mpa": 0.0}, {"i_mm4": 0.0}, {"p_kn": -1.0}])
def test_impossible_inputs_are_errors(bad):
    _error(CANTILEVER_P, {**E3, **bad})
