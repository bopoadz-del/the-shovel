"""The cost gate's user-arithmetic closure must not eat the server.

Live 8e2f736, 21 Sep 2026 20:51 UTC: a priced-BOQ item question, then "+10%".
The RSS watchdog logged the main thread in
_cost_grounding_gate -> _cg_grounded_numbers -> _cg_user_arithmetic_closure
-> _times at 1.2, 1.5 and 1.8 GB; Render OOM-killed the 2 GiB instance
seconds later. The closure multiplies every pairwise product of the user
figures by every count, every percent factor and every money figure --
O(n^2 x counts x money) entries, unbounded. Synthetic figures only.
"""
import time
import tracemalloc

from app.agents import runtime as rt


def _figure_heavy_user_text(n=80):
    items = [f"item {i}: {100 + i * 7} m at SAR {50 + i * 3}.25" for i in range(n)]
    return "; ".join(items) + ". Add 5% waste and 10% contingency."


def _seed(text):
    return {v for tok in rt._CG_NUM_RE.findall(text) if (v := rt._cg_to_number(tok)) is not None}


def test_a_figure_heavy_user_text_stays_small_and_fast():
    text = _figure_heavy_user_text()
    seed = _seed(text)
    assert len(seed) > 100
    tracemalloc.start()
    started = time.perf_counter()
    out = rt._cg_user_arithmetic_closure(text, seed)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert elapsed < 10, f"closure took {elapsed:.1f}s"
    assert peak < 150e6, f"closure peaked at {peak / 1e6:.0f} MB"
    assert len(out) <= 2 * rt._CG_USER_CLOSURE_MAX


def _bounded(text):
    started = time.perf_counter()
    out = rt._cg_user_arithmetic_closure(text, _seed(text))
    assert time.perf_counter() - started < 10
    assert len(out) <= 2 * rt._CG_USER_CLOSURE_MAX, len(out)


def test_many_percentages_stay_bounded():
    figures = " ".join(f"{3 + i * 1.5} m" for i in range(60))
    percents = " ".join(f"add {p}% " for p in range(1, 41))
    _bounded(f"Dimensions {figures}. {percents}")


def test_hundreds_of_figures_stay_bounded():
    # Unevenly spaced so pairwise sums/products do not collapse into duplicates.
    _bounded(" ".join(f"{1000 + i * i * 0.37 + i * 1.13:.2f}" for i in range(1500)))


def test_a_normal_ask_is_grounded_exactly_as_before():
    # 18 footings 3.2 x 3.2 x 0.75, +5% waste, SAR 390/m3, +10% contingency.
    text = ("How much concrete for 18 pad footings, each 3.2 m by 3.2 m by 0.75 m deep? "
            "Add 5% waste, price at SAR 390 per cubic metre and add 10% contingency.")
    out = rt._cg_user_arithmetic_closure(text, _seed(text))
    volume = 18 * 3.2 * 3.2 * 0.75
    for value in (volume, volume * 1.05, volume * 1.05 * 390, volume * 1.05 * 390 * 1.10):
        assert any(abs(value - g) <= max(0.5, value * 0.005) for g in out), value


def test_an_invented_rate_is_still_not_grounded():
    text = "Price 500 m of pipe at SAR 95 per metre and add 10% contingency."
    out = rt._cg_user_arithmetic_closure(text, _seed(text))
    assert not any(abs(500 * 173.0 - g) <= 250 for g in out)
