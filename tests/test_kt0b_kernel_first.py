"""Kernel first: formula_executor_v2 never writes its own formula when a kernel covers the ask."""
from __future__ import annotations

import pytest

from app.blocks.formula_executor_v2 import FormulaExecutorV2Block, kernel_covering


class _Mock(FormulaExecutorV2Block):
    def __init__(self, scripted, **kw):
        super().__init__(**kw)
        self._scripted = list(scripted)
        self.llm_calls = 0

    async def _call_llm(self, prompt: str) -> str:
        self.llm_calls += 1
        return self._scripted.pop(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("task,kernel", [
    ("Calculate concrete volume for a 10x8m slab, 200mm thick", "concrete_volume"),
    ("beam deflection cantilever point load 5 kN at 3 m", "beam_deflection_cantilever_point_load"),
])
async def test_kernel_covered_ask_is_refused_without_codegen(task, kernel):
    block = _Mock(["result = 1"])
    out = await block.process({"task": task, "variables": {"a": 1}})
    assert out["status"] == "error"
    assert out["kernel"] == kernel
    assert out["use_action"] == "construction_calc"
    assert kernel in out["error"]
    assert block.llm_calls == 0, "the model wrote a formula a kernel covers"


@pytest.mark.asyncio
@pytest.mark.parametrize("task,variables,code,want", [
    ("rectangle area", {"length_m": 10, "width_m": 8}, "result = length_m * width_m", 80),
    ("multiply length by width", {"length_m": 3, "width_m": 4}, "result = length_m * width_m", 12),
])
async def test_uncovered_ask_codegens_and_is_labelled_improvised(task, variables, code, want):
    block = _Mock([code])
    out = await block.process({"task": task, "variables": variables})
    assert out["status"] == "success"
    assert out["result"] == want
    assert out["provenance"] == "improvised"
    assert out["certified"] is False
    assert block.llm_calls == 1


@pytest.mark.parametrize("task", ["", "compute area of the rectangle"])
def test_kernel_covering_misses_return_none(task):
    assert kernel_covering(task) is None


def test_nudge_offers_construction_calc_first():
    from app.agents.runtime import _EMPTY_ROUTER_NUDGE as n
    assert n.index("construction_calc") < n.index("formula_executor_v2")
    assert "improvised" in n


# ── the construction container's formula_execute action ──────────────────

def _container(block):
    from app.containers.construction import ConstructionContainer

    c = ConstructionContainer()
    c._resolve_block = lambda name: block if name == "formula_executor_v2" else None
    return c


@pytest.mark.asyncio
@pytest.mark.parametrize("task,kernel", [
    ("Calculate concrete volume for a 10x8m slab, 200mm thick", "concrete_volume"),
    ("excavation volume 20 x 10 x 3 m", "excavation_volume"),
])
async def test_container_runs_the_kernel_not_codegen(task, kernel):
    block = _Mock(["result = 1"])
    out = await _container(block).formula_execute({"task": task}, {})
    assert block.llm_calls == 0, "codegen ran for an ask a kernel covers"
    assert out["kernel"] == kernel
    assert out["routed_from"] == "formula_execute"
    assert out.get("calculation") == kernel


@pytest.mark.asyncio
@pytest.mark.parametrize("task,variables,code,want", [
    ("rectangle area", {"length_m": 10, "width_m": 8}, "result = length_m * width_m", 80),
    ("multiply length by width", {"length_m": 3, "width_m": 4}, "result = length_m * width_m", 12),
])
async def test_container_codegen_is_labelled_improvised(task, variables, code, want):
    block = _Mock([code])
    out = await _container(block).formula_execute({"task": task, "variables": variables}, {})
    assert block.llm_calls == 1
    assert out["result"] == want
    assert out["provenance"] == "improvised"
    assert out["certified"] is False
