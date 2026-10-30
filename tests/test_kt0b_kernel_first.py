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
    assert out["provenance"] == "kernel"
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


# Fail CLOSED: a registry miss (K2) or an import failure (K3) refuses codegen.
@pytest.mark.asyncio
@pytest.mark.parametrize("named,refused", [
    ("not_a_registered_kernel", True),   # text names a calculator the registry lacks
    (None, False),                       # text names nothing -> improvised codegen
])
async def test_registry_miss_refuses_codegen(monkeypatch, named, refused):
    import app.lib.construction_formulas as cf
    monkeypatch.setattr(cf, "calculator_name_from_text", lambda _t: named)
    block = _Mock(["result = 2"])
    out = await block.process({"task": "some sum", "variables": {"a": 1}})
    if refused:
        assert out["status"] == "error" and out["provenance"] == "refused"
        assert out["certified"] is False
        assert "not_a_registered_kernel" in out["error"]
        assert block.llm_calls == 0
    else:
        assert out["status"] == "success" and out["provenance"] == "improvised"
        assert block.llm_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [True, False])
async def test_import_failure_refuses_codegen(monkeypatch, broken):
    import sys
    if broken:
        monkeypatch.setitem(sys.modules, "app.lib.construction_formulas", None)
    block = _Mock(["result = length_m * width_m"])
    out = await block.process({"task": "rectangle area", "variables": {"length_m": 2, "width_m": 3}})
    if broken:
        assert out["status"] == "error" and out["provenance"] == "refused"
        assert "unavailable" in out["error"]
        assert block.llm_calls == 0
    else:
        assert out["status"] == "success" and out["result"] == 6
        assert block.llm_calls == 1


# C1: a cache hit re-runs improvised code and is labelled so.
@pytest.mark.asyncio
@pytest.mark.parametrize("variables,want", [({"a": 2}, 4), ({"a": 5}, 10)])
async def test_cache_hit_is_labelled_improvised(variables, want):
    from types import SimpleNamespace
    from app.blocks.formula_executor_v2 import _cache_key
    task = "double the value"
    session = SimpleNamespace(code_cache={_cache_key(task, variables): "result = a * 2"})
    block = _Mock([])
    out = await block.process({"task": task, "variables": variables, "session": session})
    assert out["cache_hit"] is True and out["result"] == want
    assert out["provenance"] == "improvised"
    assert out["certified"] is False
    assert block.llm_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"task": "   "}, "  \n "])
async def test_an_empty_task_is_refused_before_any_model_call(payload):
    block = _Mock(["result = 1"])
    out = await block.process(payload)
    assert out["status"] == "error"
    assert "No task description provided" in out["error"]
    assert block.llm_calls == 0


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


# N4-N6: every place the container reads the ask from.
CONCRETE = "Calculate concrete volume for a 10x8m slab, 200mm thick"


@pytest.mark.asyncio
@pytest.mark.parametrize("input_data,params", [
    ({"formula_description": CONCRETE}, {}),          # N4
    ({"text": CONCRETE}, {}),                         # N4
    ({"input": CONCRETE}, {}),                        # N4
    ({}, {"task": CONCRETE}),                         # N5
    ({}, {"text": CONCRETE}),                         # N5
    (CONCRETE, {}),                                   # N6
])
async def test_container_reads_the_ask_from_every_field(input_data, params):
    block = _Mock(["result = 1"])
    out = await _container(block).formula_execute(input_data, params)
    assert block.llm_calls == 0
    assert out["kernel"] == "concrete_volume"
    assert out["routed_from"] == "formula_execute"  # the container ran it, not the block


@pytest.mark.asyncio
@pytest.mark.parametrize("broken", [True, False])
async def test_container_refuses_when_the_kernel_check_cannot_run(monkeypatch, broken):
    import sys
    if broken:
        monkeypatch.setitem(sys.modules, "app.lib.construction_formulas", None)
    block = _Mock(["result = length_m * width_m"])
    out = await _container(block).formula_execute(
        {"task": "rectangle area", "variables": {"length_m": 2, "width_m": 3}}, {},
    )
    if broken:
        assert out["provenance"] == "refused" and out["status"] == "error"
        assert out["routed_from"] == "formula_execute"
        assert block.llm_calls == 0
    else:
        assert out["provenance"] == "improvised" and out["result"] == 6


# :2213 the ask reaches construction_calc as params["text"]; :2199 params=None is an empty dict.
@pytest.mark.asyncio
@pytest.mark.parametrize("task,kernel", [
    ("Calculate concrete volume for a 10x8m slab, 200mm thick", "concrete_volume"),
    ("excavation volume 20 x 10 x 3 m", "excavation_volume"),
])
async def test_container_hands_the_ask_to_the_kernel_as_text(task, kernel):
    import app.containers.construction as construction

    seen = {}
    c = _container(_Mock(["result = 1"]))

    async def _calc(data, params):
        seen.update(params)
        return {"status": "success"}

    c.construction_calc = _calc
    out = await construction.ConstructionContainer.formula_execute(c, {"task": task}, None)
    assert seen["text"] == task
    assert seen["calculation"] == kernel
    assert out["routed_from"] == "formula_execute"
