"""One answer with a money figure must not cost the server 2 GB.

Live, 2026-09-21 (UAE 17:01-17:13): the 2 GB web instance was OOM-killed five
times in twelve minutes by ONE tester asking calculator questions, and five
times on 2026-09-20. Every kill followed a `construction_calc` turn that
reached its final LLM call and never logged STREAMING-FINAL: it died in
post-processing.

`_cg_grounded_numbers` collected every number in the retrieved chunks and then
stored EVERY pairwise sum, difference, product and quotient. Retrieved BOQ
tables carry thousands of numbers. Measured on this code before the fix:

    numbers   set entries     peak memory   time (event loop blocked)
       300        218 thousand     16 MB        3 s
       900        1.7 million     131 MB       24 s
     1,800        6.3 million     523 MB       95 s
     3,750       24 million     2,094 MB      414 s

It only fired when the model produced a properly costed answer, which is why
the free fallback models never triggered it and DeepSeek did.

The pairwise rule is kept -- qty x rate IS the QS's basic operation -- but it
is now CHECKED per figure against a sorted list (n log n), never materialised.
"""
import random
import time
import tracemalloc

import pytest

from app.agents.runtime import (
    _CG_REFUSAL,
    _cg_grounded_numbers,
    _cg_is_grounded,
    _cost_grounding_gate,
)

USER = {"role": "user", "content": "Add 5% waste and price it at SAR 390 per cubic metre."}
TOOL = {"role": "tool", "content": '{"status":"success","result":{"volume_m3":138.24}}'}


def _boq_context(rows_per_chunk, chunks=5, seed=7):
    """Synthetic priced bill: qty, rate, amount per row. Returns (context, rows)."""
    rng = random.Random(seed)
    rows, blocks = [], []
    for k in range(chunks):
        lines = ["Item Description Unit Qty Rate (SAR) Amount (SAR) -- rate per unit"]
        for i in range(rows_per_chunk):
            q = round(rng.uniform(1, 50000), 2)
            r = round(rng.uniform(5, 2500), 2)
            rows.append((q, r))
            lines.append(f"Z{500 + i}.{i % 9} Synthetic item m {q:,.2f} {r:,.2f} {q * r:,.2f}")
        blocks.append(f"[doc_id=d{k} chunk={k} score=0.9] " + "\n".join(lines) + "\n")
    return "".join(blocks), rows


def test_a_retrieved_bill_of_thousands_of_numbers_costs_megabytes_not_gigabytes():
    ctx, _ = _boq_context(250)                      # ~3,750 numbers: the 2 GB case
    tracemalloc.start()
    started = time.perf_counter()
    grounded = _cg_grounded_numbers(ctx, [USER, TOOL])
    ok = _cg_is_grounded(390.0, grounded)
    elapsed = time.perf_counter() - started
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert ok
    assert peak < 60e6, f"peak {peak / 1e6:.0f} MB"
    assert elapsed < 10, f"{elapsed:.1f}s with the event loop blocked"


def _narrow_band_context(n=400):
    """More numbers than the eager limit, ALL between 1,000 and 1,200 -- so a
    sum (~2,200), a difference (<200), a product (~1.2e6) or a quotient (~1.0)
    of two of them is nowhere near any number that is actually in the text.
    Only the pairwise rule can ground those."""
    rng = random.Random(11)
    values = sorted({round(rng.uniform(1000, 1200), 2) for _ in range(n)})
    body = " ".join(f"{v:,.2f}" for v in values)
    # One huge number: 5e8 / ~1,100 = ~450,000-500,000 is not a sum (<2,400 or
    # >5e8), a difference (<200 or ~5e8), a product (>=1e6) or a number in the text.
    return (f"[doc_id=d0 chunk=0 score=0.9] Rate (SAR) per unit: {body} 500,000,000.00\n",
            values)


def test_the_pairwise_rule_still_grounds_what_it_grounded_before():
    ctx, values = _narrow_band_context()
    grounded = _cg_grounded_numbers(ctx, [])
    assert getattr(grounded, "lazy_base", ()), "this context must take the lazy path"
    a, b = values[5], values[-7]
    for what, v in (("sum", a + b), ("difference", b - a),
                    ("product", a * b), ("quotient", 5.0e8 / a)):
        assert not any(abs(v - g) <= max(0.5, v * 0.005) for g in grounded), (
            f"{what} must not be grounded by a direct hit, or this test proves nothing")
        assert _cg_is_grounded(v, grounded), what
    assert not _cg_is_grounded(5.0e9, grounded), "far beyond any pair"
    assert not _cg_is_grounded(5000.0, grounded), "between the sums and the products"


def test_a_fabricated_figure_is_still_refused_against_a_large_bill():
    ctx, _ = _boq_context(250)
    grounded = _cg_grounded_numbers(ctx, [USER, TOOL])
    # Beyond anything two table numbers can make: the largest amount is
    # 50,000 x 2,500 = 1.25e8, so the largest pairwise product is ~1.6e16.
    assert not _cg_is_grounded(9.9e17, grounded)


@pytest.mark.parametrize("rows", [2, 250])
def test_the_gate_end_to_end_small_and_large(rows):
    ctx, table = _boq_context(rows)
    q, r = table[0]
    rag = {"role": "system", "content": ctx}
    good = f"The amount is SAR {q * r:,.2f}."
    bad = "The amount is SAR 990,000,000,000,000,000.00."      # 9.9e17: unreachable
    assert _cost_grounding_gate(good, rag, [USER, TOOL]) == good
    assert _cost_grounding_gate(bad, rag, [USER, TOOL]) == _CG_REFUSAL


def test_a_small_context_behaves_exactly_as_before():
    """The eager closure is kept where it is cheap, so every existing
    expectation about `x in grounded` holds."""
    msgs = [{"role": "user", "content": "Extend 100 square metres at the instructed 250 riyals."}]
    grounded = _cg_grounded_numbers("", msgs)
    assert 25000.0 in grounded
