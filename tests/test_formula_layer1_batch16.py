"""Formula Layer 1, batch 16 (exam 5, B1): the missing-input kind for the three
registered calculators that had a typical and a boundary case but no M case.

How the gap was found: every ``tests/test_formula_layer1_batch*.py`` was run
with a recorder on ``run_calculation`` that noted the kind letter of the test
that called each calculator (``test_<id>_t``/``_u``/``_m``/``_b``).  Of the 101
registered calculators, 98 had T, M and B; these three had no M:

  * ``concrete_curing_time``     -- three inputs, each with a stated default;
  * ``concrete_mix_slip_form``   -- declares no inputs;
  * ``guardrail_top_rail_height`` -- declares no inputs.

What "missing input" means for each:

  * An input with a stated default is not asked for: the calculator runs on
    the default and must SAY it did (the note calls it an assumed default and
    ``defaults_used`` names the parameter), and must not list an input the
    caller did give.
  * A calculator that declares no inputs has nothing to be missing: it runs
    with none and returns the values its own docstring documents.
  * A figure given to a calculator that declares no inputs is not silently
    taken as if it applied: the guardrail calculator refuses it and says it
    was not applied.

Expected values are worked by hand from each function's own formula or the
standard its docstring cites, in the comment beside the row; none was copied
from the product's output.  All inputs are invented.
"""
from __future__ import annotations

import pytest

from app.lib.construction_formulas import run_calculation

_REL = 1e-3   # plan tolerance: 0.1 %


def _ok(name: str, params: dict) -> dict:
    env = run_calculation(name, params)
    assert env.get("status") == "success", env
    return env["result"]


# ---------------------------------------------------------------- F52 curing time
# t = a*frac / (1 - b*frac); stated defaults frac=0.70, a=4.0, b=0.85.
F52 = "concrete_curing_time"


@pytest.mark.parametrize("given, omitted, days", [
    # frac omitted -> 0.70: 2.3*0.7 / (1 - 0.92*0.7) = 1.61 / 0.356 = 4.52247
    ({"gain_a": 2.3, "gain_b": 0.92}, "target_strength_fraction", 4.52247),
    # b omitted -> 0.85: 4.0*0.9 / (1 - 0.85*0.9) = 3.6 / 0.235 = 15.31915
    ({"target_strength_fraction": 0.9, "gain_a": 4.0}, "gain_b", 15.31915),
])
def test_f52_m_omitted_input_runs_on_its_stated_default_and_says_so(given, omitted, days):
    r = _ok(F52, given)
    assert r["days_to_target"] == pytest.approx(days, rel=_REL)
    used = {d.get("parameter") for d in (r.get("defaults_used") or [])}
    assert omitted in used, r
    # an input the caller gave is never reported as a default
    assert not (used & set(given)), r
    assert "assumed default" in str(r.get("note") or ""), r


# ------------------------------------------------- zero-input documented values
@pytest.mark.parametrize("name, key, expected", [
    # OSHA 29 CFR 1926.502(b): top rail 42 in; 42 x 25.4 = 1066.8 -> 1067 mm
    ("guardrail_top_rail_height", "top_rail_height_mm", 1067),
    # docstring: slip-forming mix 1:2:2.6, W/C = 0.42
    ("concrete_mix_slip_form", "w_c_ratio", 0.42),
])
def test_zero_input_m_runs_with_no_inputs_on_its_documented_values(name, key, expected):
    r = _ok(name, {})
    assert r[key] == pytest.approx(expected, rel=_REL)


@pytest.mark.parametrize("stray", [
    {"top_rail_height_in": 30},
    {"mid_rail_height_mm": 400},
])
def test_guardrail_m_a_figure_it_takes_no_input_for_is_refused_not_applied(stray):
    env = run_calculation("guardrail_top_rail_height", stray)
    assert env.get("status") != "success", env
    assert (env.get("result") or {}).get("top_rail_height_mm") is None, env
    text = str(env.get("error") or "")
    assert next(iter(stray)) in text and "not applied" in text, env
