"""Reworded one-way slab minimum-thickness asks must still predispatch.

The detector accepts the original wording ("minimum thickness" +
"spanning N m" + "one end continuous") and misses the same ask when
the span, the thickness word, or the support clause is rephrased.
A missed ask is not predispatched, and the turn has nothing to answer
from. This file locks the class: every phrasing of the same shape,
not one captured sentence.

Synthetic asks only. No project documents. No live client names.
"""
from __future__ import annotations

import pytest

from app.agents.runtime import Agent, _predispatch_formula_calc
from app.lib.construction_formulas_structural_rc import (
    looks_like_slab_thickness_min_ask,
    slab_thickness_min,
    slab_thickness_params_from_ask,
)


# Original wordings. These already predispatch; they are the control
# the rewordings are measured against.
ORIGINAL_T20 = (
    "What is the ACI minimum thickness of a simply supported one-way "
    "solid slab spanning 4.8 m?"
)
ORIGINAL_U5 = (
    "To ACI 318-19, minimum thickness of a simply supported one-way "
    "solid slab spanning 8.4 m, fy 420, normal weight — answer in mm."
)
ORIGINAL_E13 = (
    "To ACI 318-19, what is the minimum thickness of a non-prestressed, "
    "normal-weight, one-way solid slab with one end continuous, spanning "
    "6.0 m, fy = 420 MPa, not supporting partitions liable to deflection "
    "damage?"
)

# (ask, span_mm, support, thickness_mm at fy 420, ACI)
# Span gap: "span length (of) N", "spans N", "N unit span", "span: N".
# Thickness gap: "minimum depth", "minimum-depth", "h_min", "h min".
# Support gap: "continuous at one end" must beat "simply supported".
REWORDED = (
    pytest.param(
        "What is the ACI minimum thickness of a simply supported one-way "
        "solid slab with a span length of 4.8 m?",
        4800.0, "simply_supported", 240.0,
        id="span-length-of-4.8",
    ),
    pytest.param(
        "To ACI 318-19, minimum thickness of a simply supported one-way "
        "solid slab, span length 6.0 m, fy 420, normal weight.",
        6000.0, "simply_supported", 300.0,
        id="span-length-6.0",
    ),
    pytest.param(
        "What is the minimum thickness of a simply supported one-way "
        "solid slab with a span length of 5.2 m?",
        5200.0, "simply_supported", 260.0,
        id="span-length-of-5.2",
    ),
    pytest.param(
        "What is the ACI minimum thickness of a simply supported one-way "
        "solid slab over a 4.8 m span?",
        4800.0, "simply_supported", 240.0,
        id="over-a-4.8-m-span",
    ),
    pytest.param(
        "To ACI 318-19, minimum thickness of a simply supported one-way "
        "solid slab, 8.4 m span, fy 420, normal weight.",
        8400.0, "simply_supported", 420.0,
        id="8.4-m-span",
    ),
    pytest.param(
        "What is the minimum thickness of a simply supported one-way "
        "solid slab on a 7.0 m clear span?",
        7000.0, "simply_supported", 350.0,
        id="7.0-m-clear-span",
    ),
    pytest.param(
        "What is the ACI minimum thickness when a simply supported "
        "one-way solid slab spans 6.0 m?",
        6000.0, "simply_supported", 300.0,
        id="slab-spans-6.0",
    ),
    pytest.param(
        "What is the minimum thickness when a simply supported one-way "
        "solid slab spans 7.2 m?",
        7200.0, "simply_supported", 360.0,
        id="slab-spans-7.2",
    ),
    pytest.param(
        "What is the ACI minimum thickness of a simply supported one-way "
        "solid slab with span: 7.2 m?",
        7200.0, "simply_supported", 360.0,
        id="span-colon-7.2",
    ),
    pytest.param(
        "What is the minimum thickness of a simply supported one-way "
        "solid slab with a span length of 4800 mm?",
        4800.0, "simply_supported", 240.0,
        id="span-length-4800-mm",
    ),
    pytest.param(
        "What is the ACI minimum depth of a simply supported one-way "
        "solid slab spanning 7.0 m?",
        7000.0, "simply_supported", 350.0,
        id="minimum-depth-7.0",
    ),
    pytest.param(
        "To ACI 318-19, h_min of a simply supported one-way solid slab "
        "spanning 5.2 m, fy 420.",
        5200.0, "simply_supported", 260.0,
        id="h-min-5.2",
    ),
    pytest.param(
        "What h min is required for a simply supported one-way solid slab "
        "spanning 7.2 m?",
        7200.0, "simply_supported", 360.0,
        id="h-space-min-7.2",
    ),
    pytest.param(
        "What is the minimum-depth of a simply supported one-way solid "
        "slab spanning 8.4 m?",
        8400.0, "simply_supported", 420.0,
        id="minimum-depth-hyphen-8.4",
    ),
    pytest.param(
        "To ACI 318-19, what is the minimum thickness of a one-way solid "
        "slab continuous at one end, spanning 6.0 m, fy = 420 MPa?",
        6000.0, "one_end_continuous", 250.0,
        id="continuous-at-one-end-6.0",
    ),
    pytest.param(
        "To ACI 318-19, what is the minimum thickness of a one-way solid "
        "slab continuous at one end, the other end simply supported, "
        "spanning 6.0 m, fy = 420 MPa?",
        6000.0, "one_end_continuous", 250.0,
        id="one-end-beats-simply-supported-6.0",
    ),
    pytest.param(
        "What is the minimum thickness of a normal-weight one-way solid "
        "slab that is continuous at one end, spanning 7.2 m?",
        7200.0, "one_end_continuous", 300.0,
        id="continuous-at-one-end-7.2",
    ),
    pytest.param(
        "What is the ACI minimum thickness of a one-way solid slab "
        "continuous at one-end, with a span length of 4.8 m?",
        4800.0, "one_end_continuous", 200.0,
        id="continuous-at-one-end-hyphen-span-length-4.8",
    ),
    pytest.param(
        "What is the minimum thickness of a one-way solid slab continuous "
        "one end, spanning 5.2 m?",
        5200.0, "one_end_continuous", 216.7,
        id="continuous-one-end-5.2",
    ),
)

NEGATIVES = (
    pytest.param(
        "slab thickness in the BOQ item",
        id="boq-item-no-span",
    ),
    pytest.param(
        "minimum depth of cover for the pile cap",
        id="cover-pile-cap",
    ),
    pytest.param(
        "What is the minimum depth of cover for the one-way solid slab "
        "spanning 4.8 m?",
        id="cover-on-a-one-way-slab",
    ),
    pytest.param(
        "Give the one-way solid slab thickness in the BOQ item spanning "
        "6.0 m.",
        id="boq-item-with-span",
    ),
    pytest.param(
        "What is the nominal cover to the reinforcement in the pile cap?",
        id="nominal-cover",
    ),
)

# Reworded T20 / U5 / E13. Truth is ACI 318 Table 7.3.1.1 at fy 420
# (modifier 1.0): L/20, L/20, and L/24.
E2E = (
    pytest.param(
        "What is the ACI minimum thickness of a simply supported one-way "
        "solid slab with a span length of 4.8 m?",
        240.0, "simply_supported",
        id="T20-span-length",
    ),
    pytest.param(
        "To ACI 318-19, minimum thickness of a simply supported one-way "
        "solid slab, 8.4 m span, fy 420, normal weight.",
        420.0, "simply_supported",
        id="U5-metre-span",
    ),
    pytest.param(
        "To ACI 318-19, what is the minimum thickness of a one-way solid "
        "slab continuous at one end, the other end simply supported, "
        "spanning 6.0 m, fy = 420 MPa?",
        250.0, "one_end_continuous",
        id="E13-one-end-not-l20",
    ),
)

_FIXTURE_PID = "proj_c2_slab_reword_fixture"


def _parsed(ask: str) -> dict:
    return slab_thickness_params_from_ask(ask)


@pytest.mark.parametrize("ask,span_mm,support,thickness_mm", (
    pytest.param(ORIGINAL_T20, 4800.0, "simply_supported", 240.0, id="original-T20"),
    pytest.param(ORIGINAL_U5, 8400.0, "simply_supported", 420.0, id="original-U5"),
    pytest.param(ORIGINAL_E13, 6000.0, "one_end_continuous", 250.0, id="original-E13"),
))
def test_original_wording_still_detects(ask, span_mm, support, thickness_mm):
    assert looks_like_slab_thickness_min_ask(ask) is True
    params = _parsed(ask)
    assert params.get("span_mm") == pytest.approx(span_mm)
    assert params.get("support_condition") == support
    result = slab_thickness_min(
        span_mm=params["span_mm"],
        support_condition=params["support_condition"],
        code=params.get("code", "aci"),
        fy_mpa=float(params.get("fy_mpa", 420.0)),
    )
    assert result["min_thickness_mm"] == pytest.approx(thickness_mm)


@pytest.mark.parametrize("ask,span_mm,support,thickness_mm", REWORDED)
def test_reworded_ask_is_detected(ask, span_mm, support, thickness_mm):
    """Detection, span, and support for every reworded shape in the class."""
    detected = looks_like_slab_thickness_min_ask(ask)
    params = _parsed(ask)
    assert detected is True, (ask, params)
    assert params.get("span_mm") == pytest.approx(span_mm), params
    assert params.get("support_condition") == support, params
    result = slab_thickness_min(
        span_mm=params["span_mm"],
        support_condition=params["support_condition"],
        code=params.get("code", "aci"),
        fy_mpa=float(params.get("fy_mpa", 420.0)),
    )
    assert result["min_thickness_mm"] == pytest.approx(thickness_mm), result
    if support == "one_end_continuous":
        simply = slab_thickness_min(
            span_mm=params["span_mm"],
            support_condition="simply_supported",
            code=params.get("code", "aci"),
            fy_mpa=float(params.get("fy_mpa", 420.0)),
        )
        assert result["min_thickness_mm"] != pytest.approx(
            simply["min_thickness_mm"]
        )


@pytest.mark.parametrize("ask", NEGATIVES)
def test_cover_and_boq_asks_are_not_slab_thickness(ask):
    assert looks_like_slab_thickness_min_ask(ask) is False


@pytest.mark.parametrize("ask,thickness_mm,support", E2E)
async def test_predispatch_uses_slab_thickness_min(ask, thickness_mm, support):
    agent = Agent(
        name="project-assistant",
        description="t",
        system_prompt="t",
        allowed_blocks=["construction"],
    )
    msgs = [{"role": "user", "content": ask}]
    rec = await _predispatch_formula_calc(
        agent, msgs, _FIXTURE_PID, operator_text=ask,
    )
    assert rec is not None, ask
    assert rec["name"] == "construction_calc"
    result = rec.get("result") or {}
    assert result.get("calculation") == "slab_thickness_min", result
    inner = result.get("result") if isinstance(result.get("result"), dict) else {}
    assert inner.get("min_thickness_mm") == pytest.approx(thickness_mm), result
    assert inner.get("support_condition") == support, result
    assert inner.get("standard") == "ACI 318-19 Table 7.3.1.1", result
