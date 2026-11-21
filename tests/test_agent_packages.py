"""F-DRIVER Phase A: tools live in their owner's package, hats never import each
other, and a new hat is a new package with no core edit."""
from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import shutil
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


# The directory the interpreter writes bytecode into (PEP 3147). It never holds
# source, and other test workers create and delete it while this walk runs.
_BYTECODE_DIR = Path(importlib.util.cache_from_source(str(HATS / "x.py"))).parent.name


def _vanished_dir_only(err: OSError) -> None:
    """os.walk error hook: a directory deleted between listing and entering it
    holds nothing to check; any other error (permissions, I/O) still fails."""
    if err.filename is not None and not os.path.lexists(err.filename):
        return
    raise err


def _hat_sources(hat_dir: Path):
    """Every .py file under ``hat_dir``, in a stable order, without entering the
    bytecode cache and without failing on a directory that vanishes mid-walk."""
    for here, dirs, files in os.walk(hat_dir, onerror=_vanished_dir_only):
        dirs[:] = sorted(d for d in dirs if d != _BYTECODE_DIR)
        for name in sorted(files):
            if name.endswith(".py"):
                yield Path(here) / name


def _cross_hat_imports(hats_root: Path, base: Path) -> list[str]:
    offenders = []
    for hat_dir in sorted(p for p in hats_root.iterdir() if p.is_dir() and (p / "__init__.py").exists()):
        for py in _hat_sources(hat_dir):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                mods = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    mods = [node.module]
                elif isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                for m in mods:
                    if m.startswith("app.agents.hats.") and not m.startswith(f"app.agents.hats.{hat_dir.name}"):
                        offenders.append(f"{py.relative_to(base).as_posix()} imports {m}")
    return offenders


def test_no_hat_imports_another_hat():
    assert _cross_hat_imports(HATS, ROOT) == []


def _plant_hats(root: Path) -> Path:
    """Two synthetic hats; hat ``alpha`` has a nested sub-package."""
    hats = root / "hats"
    for rel, body in {
        "alpha/__init__.py": "",
        "alpha/own.py": "from app.agents.hats.alpha.own import x\n",
        "alpha/deep/__init__.py": "",
        "alpha/deep/leaf.py": "import os\n",
        "beta/__init__.py": "",
        "beta/tools.py": "from app.agents.core import tool_registry\n",
    }.items():
        f = hats / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")
    return hats


def test_a_planted_cross_hat_import_is_caught_at_any_depth(tmp_path):
    hats = _plant_hats(tmp_path)
    assert _cross_hat_imports(hats, tmp_path) == []
    (hats / "alpha" / "deep" / "leaf.py").write_text("from app.agents.hats.beta.tools import h\n", encoding="utf-8")
    (hats / "beta" / "tools.py").write_text("import app.agents.hats.alpha.own\n", encoding="utf-8")
    assert _cross_hat_imports(hats, tmp_path) == [
        "hats/alpha/deep/leaf.py imports app.agents.hats.beta.tools",
        "hats/beta/tools.py imports app.agents.hats.alpha.own",
    ]


def test_the_walk_never_enters_the_bytecode_cache(tmp_path):
    hats = _plant_hats(tmp_path)
    cache = hats / "alpha" / _BYTECODE_DIR
    cache.mkdir()
    # A .py inside the cache would be read if the walk entered it.
    (cache / "stray.py").write_text("from app.agents.hats.beta import t\n", encoding="utf-8")
    assert _cross_hat_imports(hats, tmp_path) == []
    files = [p.relative_to(hats).as_posix() for p in _hat_sources(hats / "alpha")]
    assert files == ["alpha/__init__.py", "alpha/own.py", "alpha/deep/__init__.py", "alpha/deep/leaf.py"]


@pytest.mark.parametrize("victim", ["deep", _BYTECODE_DIR])
def test_the_walk_survives_a_directory_removed_mid_walk(tmp_path, victim):
    hats = _plant_hats(tmp_path)
    (hats / "alpha" / _BYTECODE_DIR).mkdir()
    (hats / "alpha" / _BYTECODE_DIR / "own.cpython-311.pyc").write_bytes(b"\0")
    seen = []
    for py in _hat_sources(hats / "alpha"):
        seen.append(py.relative_to(hats).as_posix())
        if len(seen) == 1:
            # Listed by the walk already, deleted before it is entered.
            shutil.rmtree(hats / "alpha" / victim)
    expected = ["alpha/__init__.py", "alpha/own.py"]
    if victim != "deep":
        expected += ["alpha/deep/__init__.py", "alpha/deep/leaf.py"]
    assert seen == expected


def test_a_walk_error_other_than_a_vanished_directory_still_fails(tmp_path):
    present = tmp_path / "present"
    present.mkdir()
    with pytest.raises(PermissionError):
        _vanished_dir_only(PermissionError(13, "denied", str(present)))
    _vanished_dir_only(FileNotFoundError(2, "gone", str(tmp_path / "gone")))


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
