"""LIVE LLM end-to-end test — Reasoning Engine Plan 4.

Off by default — even with a cloud key in .env, this test stays skipped
unless LIVE_LLM_TESTS=1 is explicitly set. Two-key gate prevents a routine
`pytest` run from silently burning the LLM credit (the key in .env is for ad-hoc
chat use; arming live tests is an explicit opt-in).
"""

import os

import pytest

from app.blocks.formula_executor_v2 import FormulaExecutorV2Block

pytestmark = pytest.mark.skipif(
    not ((os.getenv("KIMI_API_KEY") or os.getenv("GROQ_API_KEY")) and os.getenv("LIVE_LLM_TESTS") == "1"),
    reason="live LLM tests off — set LIVE_LLM_TESTS=1 (plus KIMI_API_KEY or GROQ_API_KEY) to arm",
)


@pytest.mark.asyncio
async def test_live_codegen_simple_arithmetic():
    block = FormulaExecutorV2Block()
    out = await block.process({
        "task": "concrete volume of a slab: length x width x thickness",
        "variables": {"length_m": 10, "width_m": 8, "thickness_m": 0.2},
    })
    assert out["status"] == "success"
    assert abs(out["result"] - 16.0) < 1e-6


@pytest.mark.asyncio
async def test_live_codegen_uses_pm_library():
    block = FormulaExecutorV2Block()
    out = await block.process({
        "task": (
            "Given activities A(3 days) -> B(5 days) -> C(2 days) in series, "
            "use app.lib.pm_computations.compute_cpm to get the project "
            "duration. Set result to the integer duration."
        ),
        "variables": {},
    })
    assert out["status"] == "success"
    assert out["result"] == 10
