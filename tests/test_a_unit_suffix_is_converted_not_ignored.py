"""A value handed over in metres must not be read as millimetres.

Live SET4 T20 on 3403cb5. The agent called the slab calculator correctly --

    construction_calc slab_thickness_min {"span_m": 4.8,
                                          "support_condition": "simply_supported"}

-- and the binder mapped ``span_m`` onto the ``span_mm`` parameter WITHOUT
converting. 4.8 metres became 4.8 millimetres, the calculator returned
``min_thickness_mm: 0.2`` with **status success**, and the answer reported
"200 mm" (0.2 read back as metres) while its own working said "L/20 = 4800/20"
-- which is 240. Every visible part of that answer was right except the number.

A silently wrong unit is worse than a rejected call: nothing in the answer
tells the reader to check.
"""
import pytest

from app.lib import construction_formulas as cf

SLAB = "slab_thickness_min"


def _thickness(params):
    out = cf.run_calculation(SLAB, params)
    assert out["status"] == "success", out
    return out["result"]["min_thickness_mm"]


# ── the live defect ────────────────────────────────────────────────────────

def test_a_span_given_in_metres_is_converted():
    assert _thickness({"span_m": 4.8, "support_condition": "simply_supported"}) == pytest.approx(240.0)


def test_the_same_span_in_millimetres_is_unchanged():
    assert _thickness({"span_mm": 4800, "support_condition": "simply_supported"}) == pytest.approx(240.0)


def test_both_spellings_agree():
    assert _thickness({"span_m": 6.0}) == _thickness({"span_mm": 6000})


# ── the binder's contract ──────────────────────────────────────────────────

def test_the_binder_scales_metres_to_millimetres():
    bound = cf.bind_calculation_params(cf.CALCULATORS[SLAB], {"span_m": 4.8})
    assert bound == {"span_mm": pytest.approx(4800.0)}


def test_an_exact_parameter_name_is_never_rescaled():
    bound = cf.bind_calculation_params(cf.CALCULATORS[SLAB], {"span_mm": 4800})
    assert bound["span_mm"] == pytest.approx(4800)


def test_a_different_quantity_is_never_converted():
    # Same units, different stems: depth_m must not become span_mm. The bind
    # may drop it, but it must never arrive as a scaled span.
    bound = cf.bind_calculation_params(cf.CALCULATORS[SLAB], {"depth_m": 4.8})
    assert bound.get("span_mm") != pytest.approx(4800.0)


def test_a_non_numeric_value_is_left_alone():
    bound = cf.bind_calculation_params(cf.CALCULATORS[SLAB], {"span_m": "not a number"})
    assert bound.get("span_mm") == "not a number"


def test_an_unmapped_unit_fails_loudly_rather_than_silently():
    # No calculator takes a _cm parameter, so this must not quietly become a
    # span: a named error is the right outcome.
    out = cf.run_calculation(SLAB, {"span_cm": 480})
    assert out["status"] == "error"
    assert "span_mm" in out["error"]


def test_the_tonne_rule_still_works():
    bound = cf.bind_calculation_params(cf.CALCULATORS["cost_buildup_rebar"], {"quantity_t": 2})
    assert bound.get("quantity_kg") == pytest.approx(2000.0)


def test_the_conversion_only_fires_for_the_same_quantity():
    # The guard that keeps this from becoming "any _m onto any _mm".
    assert cf._length_unit_factor("span_m", "span_mm") == pytest.approx(1000.0)
    assert cf._length_unit_factor("span_mm", "span_m") == pytest.approx(0.001)
    assert cf._length_unit_factor("depth_m", "span_mm") is None
    assert cf._length_unit_factor("span_mm", "span_mm") is None
    assert cf._length_unit_factor("span_m", "load_kn") is None
    assert cf._length_unit_factor("span", "span_mm") is None
