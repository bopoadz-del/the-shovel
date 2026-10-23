"""The cost gate's user closure is seeded by the OPERATOR, not the platform.

The runtime replays tool output into the conversation as user-role bubbles
("PLATFORM PRE-DISPATCH: ...", "Tool result (boq_processor): ...") and folds
retrieved chunks into the last user turn. Seeding the user-arithmetic closure
from every user-role message therefore treated a whole priced BOQ -- every
quantity, rate and amount a block returned -- as figures the operator had
typed. Two consequences, one test group each below:

  * the gate grounded numbers it exists to refuse (its own rule is "a rate
    the user never typed still fails the gate");
  * the closure was seeded with hundreds of figures and grew past 2 GiB,
    which OOM-killed the live web instance on 21 Sep 2026 (#692).

Synthetic figures throughout.
"""
import json

from app.agents import runtime as rt

RATE = 137.0
QTY = 400.0
INVENTED_TOTAL = QTY * RATE  # 54,800 -- only the tool ever said 137.00


def _tool_bubble():
    # Same three figures on every row, so no unrelated pair of context
    # numbers can land near the multi-hop value the test checks for.
    rows = [{"description": f"Synthetic item {i}", "quantity": QTY,
             "unit": "m", "unit_cost": RATE, "total_cost": INVENTED_TOTAL}
            for i in range(40)]
    return (f"{rt._TOOL_RESULT_PREFIX}boq_processor): "
            + json.dumps({"status": "success", "line_items": rows}))


def _predispatch_bubble():
    return (f"{rt._PREDISPATCH_PREFIX} boq_processor has ALREADY been run from "
            f"the uploaded bill: quantity {QTY}, rate {RATE}, amount {INVENTED_TOTAL}.")


def _grounded(messages):
    return rt._cg_grounded_numbers("", messages)


# ── platform bubbles are not operator figures ──────────────────────────────

def test_a_replayed_tool_result_grounds_only_what_it_contains():
    # Tool output is legitimate grounding, and the operator may ask for a
    # percentage of it (SET4 T6) -- what must NOT happen is a rate nobody
    # supplied riding along on the replayed figures.
    messages = [{"role": "user", "content": "Add 10% to that."},
                {"role": "user", "content": _tool_bubble()}]
    grounded = _grounded(messages)
    assert rt._cg_is_grounded(INVENTED_TOTAL, grounded), "pairwise over context stands"
    assert rt._cg_is_grounded(INVENTED_TOTAL * 1.10, grounded), "the operator asked for +10%"
    assert not rt._cg_is_grounded(QTY * 512.0, grounded), "512 is nobody's rate"


def test_a_predispatch_bubble_does_not_supply_operator_multipliers():
    # The platform's bubble states a rate of 137. A later answer priced at a
    # rate the OPERATOR never typed must still fail the gate.
    messages = [{"role": "user", "content": "And with 10% more?"},
                {"role": "user", "content": _predispatch_bubble()}]
    grounded = _grounded(messages)
    assert RATE not in grounded.operator_factors
    assert not rt._cg_is_grounded(QTY * 512.0, grounded)


def test_platform_bubbles_are_recognised_by_both_prefixes():
    assert rt._cg_is_platform_bubble(_tool_bubble())
    assert rt._cg_is_platform_bubble(_predispatch_bubble())
    assert rt._cg_is_platform_bubble("  " + _predispatch_bubble())
    assert not rt._cg_is_platform_bubble("Price 500 m at SAR 95 per metre.")


def test_the_closure_is_not_seeded_with_hundreds_of_tool_figures(monkeypatch):
    seen = {}

    def spy(text, seeded):
        seen["n"] = len(seeded)
        return set(seeded)

    monkeypatch.setattr(rt, "_cg_user_arithmetic_closure", spy)
    _grounded([{"role": "user", "content": "Add 10% to that."},
               {"role": "user", "content": _tool_bubble()}])
    assert seen.get("n", 0) <= 5, f"{seen.get('n')} figures seeded from a tool bubble"


# ── what the operator typed still grounds ──────────────────────────────────

def test_the_operators_own_arithmetic_is_still_grounded():
    ask = ("How much concrete for 18 pad footings, each 3.2 m by 3.2 m by 0.75 m deep? "
           "Add 5% waste, price at SAR 390 per cubic metre.")
    grounded = _grounded([{"role": "user", "content": ask},
                          {"role": "user", "content": _tool_bubble()}])
    priced = 18 * 3.2 * 3.2 * 0.75 * 1.05 * 390
    assert rt._cg_is_grounded(priced, grounded), priced


def test_a_rag_folded_operator_bubble_still_grounds_the_typed_figures():
    folded = ("[doc_id=d1 chunk=0] Synthetic bill excerpt: item 1.1 excavation "
              f"m3 900 21.00 18,900.00\n{rt._RAG_FOLD_END}"
              "\n\nQUESTION: Price 500 m of pipe at SAR 95 per metre.")
    assert rt._cg_is_grounded(500 * 95, _grounded([{"role": "user", "content": folded}]))


def test_a_folded_chunks_figures_do_not_seed_the_closure(monkeypatch):
    # The excerpt folded into the user bubble is retrieved text, not typed
    # figures. Seeding the closure from it is how a retrieved BOQ page put
    # hundreds of figures into an O(n^2) expansion.
    rows = " ".join(f"item {i} m3 {900 + i * 3} {21 + i}.00 {18900 + i * 37}.00" for i in range(60))
    folded = (f"[doc_id=d1 chunk=0] Synthetic bill excerpt: {rows}\n{rt._RAG_FOLD_END}"
              "\n\nQUESTION: Price 500 m of pipe at SAR 95 per metre.")
    seen = {}

    def spy(text, seeded):
        seen["n"] = len(seeded)
        return set(seeded)

    monkeypatch.setattr(rt, "_cg_user_arithmetic_closure", spy)
    _grounded([{"role": "user", "content": folded}])
    assert seen.get("n", 0) <= 5, f"{seen.get('n')} figures seeded from a folded chunk"
