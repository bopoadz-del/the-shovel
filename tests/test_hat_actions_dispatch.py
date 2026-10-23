"""Every declared hat action dispatches to REAL machinery -- gap closed.

KNOWN_INCOMPLETE carried this for three days: the hat manifests declared 35
plain action names of which 32 resolved to nothing -- no block, no container
route, no synthetic tool. Closure (2026-08-15) was two honest moves:

  * 25 names alias to the machinery they always meant -- 11 registry
    calculators, 12 construction-container routes, 3 search renames. The
    result carries ``aliased_to`` so a trace shows the true implementation.
  * 7 names with NO real backing were REMOVED from the manifests rather
    than stubbed (document_export, generate_brief, get_learning_summary,
    goods_receipt_logger, delivery_tracker, logistics_scheduler,
    lookahead_generator).

The dormancy fence (test_hats_are_dormant) now pins the gap at ZERO; this
file proves the aliases actually run.
"""
from __future__ import annotations

import json

import pytest

import app.agents.runtime as rt


def _call(agent, name, args):
    return rt.Agent.__dict__["_run_tool_call"](
        agent, {"function": {"name": name, "arguments": json.dumps(args)}},
    )


@pytest.fixture
def agent():
    return rt.Agent(name="qa-fence", description="", system_prompt="x",
                    allowed_blocks=["construction"])


@pytest.mark.asyncio
async def test_a_calculator_alias_runs_the_real_calculator(agent):
    r = await _call(agent, "concrete_maturity_calculator", {"params": {
        "temperature_history_c": [20, 22, 25, 24],
        "time_intervals_hours": [12, 12, 12, 12]}})
    assert r["ok"], r
    assert r["result"]["aliased_to"] == "construction_calc:concrete_maturity_strength"
    assert r["result"]["result"]["maturity_index_c_hrs"] == pytest.approx(1572.0)


@pytest.mark.asyncio
async def test_float_analysis_reproduces_the_cpm_calculator(agent):
    r = await _call(agent, "float_analysis", {"params": {
        "early_start": 0, "early_finish": 10, "late_start": 5, "late_finish": 15}})
    assert r["ok"], r
    assert r["result"]["result"]["total_float"] == pytest.approx(5.0)
    assert r["result"]["result"]["is_critical"] is False


@pytest.mark.asyncio
async def test_a_container_alias_reaches_the_container_route(agent, monkeypatch):
    calls = {}

    class FakeBlock:
        async def process(self, input_data, params):
            calls["action"] = params.get("action")
            return {"status": "success", "certified": 123}

    import app.dependencies as deps
    monkeypatch.setattr(deps, "get_block_instance", lambda name: FakeBlock())
    r = await _call(agent, "interim_certificate_generator",
                    {"params": {"claimed_amount": 100}})
    assert r["ok"], r
    assert calls["action"] == "payment_certificate"
    assert r["result"]["aliased_to"] == "construction:payment_certificate"


@pytest.mark.asyncio
async def test_container_aliases_respect_the_allowed_blocks_gate(monkeypatch):
    bare = rt.Agent(name="bare", description="", system_prompt="x",
                    allowed_blocks=[])
    r = await _call(bare, "schedule_analysis", {})
    assert r["ok"] is False
    assert "allowed_blocks" in r["result"]["error"]


def test_every_calc_alias_target_is_a_real_calculator():
    from app.agents.formulas import resolve_binding
    for name, calc in rt._HAT_CALC_ALIASES.items():
        assert resolve_binding(calc) is not None, f"{name} -> {calc} unresolvable"


def test_every_container_alias_target_is_a_real_route():
    import re
    from pathlib import Path
    src = Path("app/containers/construction/__init__.py").read_text(encoding="utf-8")
    start = src.index("handlers = {")
    block = src[start:src.index("\n        }", start)]
    routes = set(re.findall(r'"([a-z0-9_]+)":\s*self\.', block))
    for name, route in rt._HAT_CONTAINER_ALIASES.items():
        assert route in routes, f"{name} -> {route} is not a container route"


def test_the_removed_names_are_gone_from_every_manifest():
    import glob
    removed = {"document_export", "generate_brief", "get_learning_summary",
               "goods_receipt_logger", "delivery_tracker",
               "logistics_scheduler", "lookahead_generator"}
    for p in glob.glob("app/agents/manifests/*.json"):
        actions = set(json.load(open(p)).get("allowed_actions") or [])
        leak = actions & removed
        assert not leak, f"{p} still declares {sorted(leak)}"
