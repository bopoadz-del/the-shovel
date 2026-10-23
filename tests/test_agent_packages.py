"""F-DRIVER Phase A: tools live in their owner's package, hats never import each
other, and a new hat is a new package with no core edit."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pytest

from app.agents.core import tool_registry as reg

ROOT = Path(__file__).resolve().parent.parent
HATS = ROOT / "app" / "agents" / "hats"


def test_every_tool_has_one_known_owner_and_a_handler():
    reg.load()
    for owner in reg.OWNERS:
        for name in reg.tools_of(owner):
            spec = reg.get(name)
            assert spec.owner == owner and callable(spec.handler)


def test_tools_live_in_their_owners_package():
    reg.load()
    for owner in reg.OWNERS:
        for name in reg.tools_of(owner):
            mod = reg.get(name).handler.__module__
            package = "app.agents.base" if owner == "base" else f"app.agents.hats.{owner}"
            # A module directly in the owner's package (tools, blocks, knowledge, ...).
            assert mod.rsplit(".", 1)[0] == package, (name, mod)


def test_no_hat_imports_another_hat():
    offenders = []
    for hat_dir in sorted(p for p in HATS.iterdir() if p.is_dir() and (p / "__init__.py").exists()):
        for py in hat_dir.rglob("*.py"):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for m in mods:
                    if m.startswith("app.agents.hats.") and not m.startswith(f"app.agents.hats.{hat_dir.name}"):
                        offenders.append(f"{py.relative_to(ROOT)} imports {m}")
    assert offenders == []


def test_a_new_hat_package_registers_its_tools_with_no_core_edit(tmp_path, monkeypatch):
    """A synthetic hat dropped into app/agents/hats is discovered and its tool runs."""
    import app.agents.hats as hats_pkg

    pkg = tmp_path / "synthetic_hat_zeta"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('"""synthetic test hat."""\n', encoding="utf-8")
    (pkg / "tools.py").write_text(
        "from app.agents.core.tool_registry import ToolCall, tool\n\n"
        "@tool('zeta_echo', owner='qaqc', display_name='Zeta echo')\n"
        "async def handle(call: ToolCall) -> dict:\n"
        "    return {'name': call.name, 'ok': True, 'result': {'echo': call.args.get('x')}}\n",
        encoding="utf-8")
    monkeypatch.setattr(hats_pkg, "__path__", list(hats_pkg.__path__) + [str(tmp_path)])
    monkeypatch.setattr(reg, "_loaded", False)
    try:
        assert "app.agents.hats.synthetic_hat_zeta.tools" in reg.tool_modules()
        spec = reg.get("zeta_echo")
        assert spec is not None
        import asyncio
        out = asyncio.run(spec.handler(reg.ToolCall(agent=None, name="zeta_echo", args={"x": 7})))
        assert out["result"]["echo"] == 7
    finally:
        reg._TOOLS.pop("zeta_echo", None)
        for m in [m for m in sys.modules if m.startswith("app.agents.hats.synthetic_hat_zeta")]:
            del sys.modules[m]


def test_a_tool_cannot_be_registered_twice_or_by_an_unknown_owner():
    with pytest.raises(ValueError):
        reg.tool("x", owner="marketing", display_name="X")
    with pytest.raises(ValueError):
        reg.tool("x", owner="base", display_name=" ")

    async def a(call):
        return {}

    async def b(call):
        return {}
    reg.tool("dup_probe", owner="base", display_name="Probe")(a)
    try:
        with pytest.raises(ValueError):
            reg.tool("dup_probe", owner="base", display_name="Probe")(b)
    finally:
        reg._TOOLS.pop("dup_probe", None)


def test_the_agent_dispatches_registered_tools_through_the_registry(monkeypatch):
    """_run_tool_call answers a registered tool from its package handler."""
    import asyncio

    from app.agents import runtime as rt

    seen = {}

    async def fake(call):
        seen["call"] = call
        return {"name": call.name, "ok": True, "result": {"from": "package"}}
    spec = reg.get("remember_fact")
    monkeypatch.setitem(reg._TOOLS, "remember_fact", reg.ToolSpec("remember_fact", spec.owner, fake))
    rt.load_agents()
    agent = rt.AGENT_REGISTRY["project-assistant"]
    out = asyncio.run(agent._run_tool_call(
        {"function": {"name": "remember_fact", "arguments": '{"key": "k", "value": "v"}'}},
        project_id="synthetic-p", conversation_id="c1", user_message="note this"))
    assert out["result"] == {"from": "package"}
    call = seen["call"]
    assert call.agent is agent and call.args == {"key": "k", "value": "v"}
    assert call.project_id == "synthetic-p" and call.conversation_id == "c1"


def test_tool_packages_are_loaded_when_the_runtime_is_imported():
    """No tool package is imported on a turn's first tool call (that import
    ran on the event loop and stalled it ~1 s under 15 users)."""
    import subprocess
    import sys

    code = ("import app.agents.runtime, app.agents.core.tool_registry as r, sys; "
            "assert r._loaded; "
            "assert 'app.agents.base.tools' in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


def test_block_adapters_live_in_their_owner_package():
    from app.agents.core import tool_registry as r

    assert r.block_args_of("base") == ["construction"]
    assert r.block_args_of("qaqc") == ["validation_pipeline"]
    assert "app.agents.base.blocks" in r.tool_modules()
    assert "app.agents.hats.qaqc.blocks" in r.tool_modules()


def test_validation_adapter_takes_the_claim_from_the_user_when_no_value():
    import asyncio

    from app.agents.core import tool_registry as r

    call = r.ToolCall(agent=None, name="validation_pipeline", args={"unit": "m"},
                      user_message="a 40 m span on a 50 mm beam is fine")
    bi, bp = asyncio.run(r.get_block_args("validation_pipeline").adapt(call, {"value": None}, {"k": 1}))
    assert bi == {"value": None, "k": 1, "unit": "m", "claim": "a 40 m span on a 50 mm beam is fine"}
    assert bp == {"k": 1}


def test_construction_adapter_folds_top_level_figures_into_params():
    import asyncio

    from app.agents.core import tool_registry as r

    call = r.ToolCall(agent=None, name="construction", args={"action": "status", "area": 12},
                      user_message=None, project_id=None)
    bi, bp = asyncio.run(r.get_block_args("construction").adapt(call, None, {"area": None}))
    assert bp == {"area": 12, "action": "status"}
    assert bi == {"area": 12, "action": "status"}
