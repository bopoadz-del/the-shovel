"""A figure the operator asked to multiply is grounded; an invented one is not.

SET4 on live 5caac10 (22 Sep 2026):
  * M3 -- "8.5% of the Accepted Contract Amount" was REFUSED. The contract
    amount is grounded (it is in the contract) and 0.085 is grounded (the
    operator typed it), but their product was not: the pairwise closure runs
    over the CONTEXT figures before the operator's are merged in, so a
    percentage of a document figure could never ground.
  * T6 -- "add 5% waste and price it at SAR 390 per cubic metre" dropped the
    pricing step: same shape, with a tool-computed volume and a typed rate.

The gate exists to refuse a rate nobody supplied. It must not refuse
arithmetic the operator asked for. Synthetic figures throughout.
"""
import json

from app.agents import runtime as rt

ACA = 3096184.35
VOLUME = 8.064
RATE = 390.0


def _grounded(rag="", messages=None):
    return rt._cg_grounded_numbers(rag, messages or [])


def _contract_turn():
    rag = (f"[doc_id=c1 chunk=3] Contract Data: the Accepted Contract Amount is "
           f"SAR {ACA:,.2f}. Performance security 8.5 per cent.")
    return _grounded(rag, [{"role": "user", "content": "What is 8.5% of the Accepted Contract Amount in SAR?"}])


def _priced_takeoff_turn():
    return _grounded("", [
        {"role": "user", "content": "How much concrete for 18 pad footings, each 3.2 m by 3.2 m by 0.75 m deep?"},
        {"role": "tool", "content": json.dumps({"status": "success", "result": {"volume_m3": VOLUME}})},
        {"role": "user", "content": "Add 5% waste and price it at SAR 390 per cubic metre."},
    ])


# ── what the operator asked for grounds ────────────────────────────────────

def test_a_percentage_of_a_contract_figure_grounds():
    assert rt._cg_is_grounded(round(ACA * 0.085, 2), _contract_turn())


def test_a_tool_figure_at_the_operators_rate_grounds():
    grounded = _priced_takeoff_turn()
    assert rt._cg_is_grounded(round(VOLUME * RATE, 2), grounded)
    assert rt._cg_is_grounded(round(VOLUME * 1.05 * RATE, 2), grounded)


# ── an invented figure still fails ─────────────────────────────────────────

def test_the_tolerance_scales_with_the_factor():
    # The gate's tolerance is 0.5% OF THE ANSWER. Dividing the answer by 0.085
    # to compare against the contract figure must widen the window by the same
    # factor, or a correctly rounded answer (here 8.50 SAR inside tolerance)
    # reads as ungrounded.
    value = round(ACA * 0.085 * 1.0025, 2)  # 0.25% out: inside the 0.5% tolerance
    assert abs(value - ACA * 0.085) <= max(0.5, value * 0.005)
    # ...but 0.25% of the CONTRACT figure is ~7,700, far outside a window that
    # forgot to divide by 0.085.
    assert abs(value / 0.085 - ACA) > max(0.5, value * 0.005)
    assert rt._cg_is_grounded(value, _contract_turn())


def test_an_invented_total_is_still_refused():
    assert not rt._cg_is_grounded(987654.32, _contract_turn())


def test_a_rate_nobody_supplied_is_still_refused():
    # The operator typed 390, not 512.
    assert not rt._cg_is_grounded(round(VOLUME * 512.0, 2), _priced_takeoff_turn())


def test_a_percentage_the_operator_never_typed_is_still_refused():
    # 8.5% was typed; 23% was not, and 23% of the ACA is nobody's figure.
    assert not rt._cg_is_grounded(round(ACA * 0.23, 2), _contract_turn())


# ── the factor list is operator-only, deterministic and capped ─────────────

def test_factors_come_from_the_operator_not_the_platform():
    tool_bubble = f"{rt._TOOL_RESULT_PREFIX}boq_processor): " + json.dumps(
        {"line_items": [{"unit_cost": 512.0, "total_cost": 4096.0}]})
    grounded = _grounded("", [{"role": "user", "content": "Add 5% waste."},
                              {"role": "user", "content": tool_bubble}])
    assert 512.0 not in grounded.operator_factors
    assert 0.05 in grounded.operator_factors


def test_the_factor_list_is_capped():
    typed = " ".join(f"add {i}% then" for i in range(1, 61))  # 120 factors
    grounded = _grounded("", [{"role": "user", "content": typed}])
    assert len(grounded.operator_factors) <= rt._CG_OPERATOR_FACTOR_MAX
    assert grounded.operator_factors == tuple(sorted(grounded.operator_factors))


def test_the_check_allocates_nothing_on_a_large_base():
    # 4,000 context figures: the pair check is a binary search per factor.
    rag = " ".join(f"SAR {1000 + i * 7}.00" for i in range(4000))
    grounded = _grounded(rag, [{"role": "user", "content": "Add 8.5%."}])
    import time
    started = time.perf_counter()
    for probe in (1234.0, 98765.0, 4321.0):
        rt._cg_is_grounded(probe, grounded)
    assert time.perf_counter() - started < 1.0
