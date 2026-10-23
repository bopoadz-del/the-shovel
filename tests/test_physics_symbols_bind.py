"""The model writes E and I, the calculators take ec_mpa and i_mm4.

Live on 9e6fe98, both cases answered with a tool error instead of a figure:

    T2   "Unknown argument(s) for beam_deflection_cantilever_udl: E, I.
          These were not applied. Expected: ... ec_mpa (MPa), i_mm4"
    T12  "Unknown argument(s) for modulus_of_elasticity_concrete: c"

Both had been passing. #740 made an unbound key a hard error rather than a
silent drop -- which is right, a silently dropped input is how a wrong figure
gets computed -- but it turned two long-standing binding gaps into visible
failures. `E` and `I` are what an engineer writes and what the model therefore
sends; `c` is the model shortening `code`.

Renaming is not enough, and this is the whole difficulty: `E` is quoted in
GPa (200) while ``ec_mpa`` wants MPa (200 000), and `I` is quoted in m4
(2.0e-4) while ``i_mm4`` wants mm4 (2.0e8). A rename alone would compute a
deflection a thousand times wrong with no error at all -- worse than the tool
error it replaced.

The unit is inferred from magnitude, and only where the two ranges cannot
overlap in engineering practice: a modulus of elasticity below 1000 is GPa
(concrete 20-40, steel 200; nothing real sits between 1000 and 20 000 MPa),
and a second moment of area below 1.0 is m4 (a 10 mm square bar is 833 mm4).
Outside those ranges the value is taken as already canonical, and the existing
m4 guard still refuses anything it cannot make sense of.
"""
import pytest

from app.lib import construction_formulas as cf

CANTILEVER = "beam_deflection_cantilever_udl"
MODULUS = "modulus_of_elasticity_concrete"


def _run(name, params):
    return cf.run_calculation(name, params)


def _value(name, params):
    out = _run(name, params)
    assert out["status"] == "success", out
    r = out["result"]
    return r["value"] if isinstance(r, dict) else r


# ── the two live failures ──────────────────────────────────────────────────

def test_t2_physics_symbols_bind_and_convert():
    # 10 kN/m over 3 m, E = 200 GPa, I = 2.0e-4 m4 -> 2.53 mm.
    assert _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                               "E": 200, "I": 2.0e-4}) == pytest.approx(2.53, abs=0.01)


def test_t12_shortened_code_binds():
    r = _run(MODULUS, {"fck_n_mm2": 35, "c": "aci"})
    assert r["status"] == "success", r
    assert r["result"]["value"] == pytest.approx(27806.0, abs=1.0)
    assert r["result"]["unit"] == "MPa"


def test_the_canonical_names_are_unchanged():
    assert _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                               "ec_mpa": 200_000, "i_mm4": 2.0e8}) == pytest.approx(2.53, abs=0.01)
    assert _run(MODULUS, {"fck_n_mm2": 35, "code": "aci"})["status"] == "success"


# ── the unit inference, which is where a wrong figure would come from ──────

def test_a_modulus_already_in_mpa_is_not_scaled_again():
    # E = 200000 is MPa already; scaling would give 2.53e-3 mm.
    assert _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                               "E": 200_000, "I": 2.0e8}) == pytest.approx(2.53, abs=0.01)


def test_both_spellings_agree():
    symbols = _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "E": 200, "I": 2.0e-4})
    canonical = _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                                    "ec_mpa": 200_000, "i_mm4": 2.0e8})
    assert symbols == pytest.approx(canonical, abs=1e-9)


def test_a_canonical_key_wins_over_a_symbol():
    # If the model sends both, the explicit one is the instruction.
    out = _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                              "ec_mpa": 200_000, "i_mm4": 2.0e8, "E": 999, "I": 999})
    assert out == pytest.approx(2.53, abs=0.01)


def test_a_second_moment_that_is_neither_unit_is_still_refused():
    # 1e-30 converts to 1e-18 mm4 -- no section. The m4 guard must still fire.
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "E": 200, "I": 1e-30})
    assert out["status"] != "success"


# ── an unknown key is still an error; this does not weaken that ────────────

def test_a_genuinely_unknown_argument_still_errors():
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                            "ec_mpa": 200_000, "i_mm4": 2.0e8, "bogus_param": 1})
    assert out["status"] != "success"
    assert "bogus_param" in str(out)


def test_symbols_do_not_leak_into_calculators_without_those_params():
    # `modulus_of_rupture` has no ec_mpa/i_mm4; E must not bind to anything.
    out = _run("modulus_of_rupture", {"fck_n_mm2": 30, "E": 200})
    assert out["status"] != "success"
    assert "E" in str(out.get("unknown", [])) or "E" in str(out)


# ── second attempt: live still failed on 533f08c ────────────────────────────
#
# The runtime log for the live T2 probe showed construction_calc dispatched
# twelve times, each returning the unknown-argument envelope, until
# MAX_TOOL_ITERATIONS forced a no-tools retry. The alias above was hooked
# where ``fn`` is looked up from the ORIGINAL name; the live tool resolves the
# calculator later, so at hook time ``fn`` was None and the alias was skipped.
# Local calls passed the resolved name directly, which is why they passed.
#
# The alias now lives in the binder itself, so it holds for every entry point.

def test_the_binder_itself_binds_the_symbols():
    fn = cf.CALCULATORS["beam_deflection_cantilever_udl"]
    bound = cf.bind_calculation_params(fn, {"w_kn_m": 10, "span_m": 3, "E": 200, "I": 2.0e-4})
    bound = bound[0] if isinstance(bound, tuple) else bound
    assert bound["ec_mpa"] == pytest.approx(200_000)
    assert bound["i_mm4"] == pytest.approx(2.0e8)
    assert "E" not in bound and "I" not in bound


# ── unit-bearing strings: the model writes "200 GPa", not 200 ──────────────

@pytest.mark.parametrize("e,i", [
    ("200 GPa", "2.0e-4 m4"), ("200GPa", "2.0e-4 m^4"), ("200 gpa", "2.0e-4 m4"),
    ("200000 MPa", "2.0e8 mm4"), ("200 GPa", "200000000 mm^4"),
])
def test_unit_bearing_strings_convert_to_the_right_figure(e, i):
    assert _value(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "E": e, "I": i}) == pytest.approx(2.53, abs=0.01)


def test_a_string_with_an_unknown_unit_is_refused_not_guessed():
    # 253,125,000,000 mm was the figure a pass-through produced. Never again.
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "E": "200 psi", "I": "2.0e-4 m4"})
    assert out["status"] != "success"
    assert "psi" in str(out).lower() or "unit" in str(out).lower()


# ── the code the model actually writes ─────────────────────────────────────

@pytest.mark.parametrize("code", ["ACI 318-19", "ACI318-19", "aci 318", "ACI 318-19 (SI)", "ACI"])
def test_the_aci_form_is_recognised_however_it_is_spelled(code):
    r = _run(MODULUS, {"fck_n_mm2": 35, "code": code})
    assert r["status"] == "success", r
    assert r["result"]["unit"] == "MPa", "the kg/cm2 form under an ACI label is a wrong figure"
    assert r["result"]["value"] == pytest.approx(27806.0, abs=1.0)
    rr = _run("modulus_of_rupture", {"fck_n_mm2": 30, "code": code})
    assert rr["result"]["value"] == pytest.approx(3.396, abs=0.005)


def test_a_non_aci_code_still_gets_the_default_form():
    r = _run(MODULUS, {"fck_n_mm2": 35, "code": "BS 8110"})
    assert r["result"]["unit"] == "kg/cm2"


# ── third attempt: live still 0/6 on 7dc1133 — the ask TEXT re-injects E/I ───
#
# #741 and #743 both assumed the model sends E/I in the params. It does not:
# the live tool_call sent canonical keys (ec_mpa, i_mm4) and STILL got
# "Unknown argument(s): E, I". The ask text rides along in ``text`` (and
# ``query``), and run_calculation runs two extractors over it AFTER the symbol
# alias -- _extract_calc_kwargs_from_ask and extract_calculation_params_from_text
# both pull "E = 200" and "I = 2.0e-4" back out of the sentence as bare keys,
# and _partition_bound_params then reports them unknown. The local tests above
# attached no text, which is why every earlier fix was green and live failed.
#
# The alias must therefore run again AFTER both extractors. It is idempotent:
# once ec_mpa is set (from the canonical key or the first pass) a re-extracted
# E is dropped, not converted, so canonical still wins.

ASK = ("What is the tip deflection of a 3 m cantilever carrying 10 kN/m, "
       "E = 200 GPa, I = 2.0e-4 m4?")


@pytest.mark.parametrize("carrier", ["text", "query", "message", "formula"])
def test_canonical_params_survive_the_ask_text_riding_along(carrier):
    # THE live failure: perfect canonical keys, plus the sentence in `carrier`.
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                            "ec_mpa": 200_000, "i_mm4": 2.0e8, carrier: ASK})
    assert out["status"] == "success", out
    val = out["result"]["value"] if isinstance(out["result"], dict) else out["result"]
    assert val == pytest.approx(2.53, abs=0.01)


@pytest.mark.parametrize("carrier", ["text", "query"])
def test_symbol_params_survive_the_ask_text_riding_along(carrier):
    # Symbols in the params AND the same sentence in the carrier: still 2.53.
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3,
                            "E": 200, "I": 2.0e-4, carrier: ASK})
    assert out["status"] == "success", out
    val = out["result"]["value"] if isinstance(out["result"], dict) else out["result"]
    assert val == pytest.approx(2.53, abs=0.01)


def test_the_ask_text_alone_binds_when_no_numeric_params_given():
    # Only the sentence: the extractor pulls E/I and the alias must convert them.
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "text": ASK})
    assert out["status"] == "success", out
    val = out["result"]["value"] if isinstance(out["result"], dict) else out["result"]
    assert val == pytest.approx(2.53, abs=0.01)


def test_a_shortened_code_survives_ask_text_the_c_class():
    # The c/code member of _SYMBOL_DEST, same path: canonical code + ask text.
    out = _run(MODULUS, {"fck_n_mm2": 35, "code": "ACI 318-19",
                         "text": "modulus of elasticity of 35 MPa concrete to ACI 318"})
    assert out["status"] == "success", out
    assert out["result"]["unit"] == "MPa"
    assert out["result"]["value"] == pytest.approx(27806.0, abs=1.0)


def test_an_unknown_key_still_errors_even_with_ask_text_present():
    # The re-alias must not swallow a genuinely unknown argument (#740 stands).
    out = _run(CANTILEVER, {"w_kn_m": 10, "span_m": 3, "ec_mpa": 200_000,
                            "i_mm4": 2.0e8, "bogus_param": 1, "text": ASK})
    assert out["status"] != "success"
    assert "bogus_param" in str(out)
