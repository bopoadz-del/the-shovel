"""F-DRIVER Phase A: the generic core (app/agents/core) has no file over
1,500 lines, and the Agent's turn methods live there."""
from __future__ import annotations

from pathlib import Path

CORE = Path(__file__).resolve().parents[1] / "app" / "agents" / "core"
LIMIT = 1500


def test_no_core_file_is_over_the_limit():
    sizes = {p.name: p.read_text(encoding="utf-8").count("\n") for p in CORE.glob("*.py")}
    assert sizes and {n: s for n, s in sizes.items() if s > LIMIT} == {}


def test_the_turn_methods_are_defined_in_the_core():
    from app.agents.runtime import Agent

    where = {name: getattr(Agent, name).__module__ for name in (
        "chat", "_chat_impl", "chat_stream", "_chat_stream_impl", "_call_llm",
        "_stream_synthesis", "_run_tool_call", "tool_definitions", "_build_messages")}
    assert all(m.startswith("app.agents.core.") for m in where.values()), where


def test_no_core_module_imports_the_runtime_at_import_time():
    """Core modules read runtime names at call time only, so the runtime can
    bind core functions as Agent methods without an import cycle."""
    import ast

    offenders = []
    for path in CORE.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.agents.runtime"):
                offenders.append(path.name)
            elif isinstance(node, ast.Import) and any(a.name.startswith("app.agents.runtime") for a in node.names):
                offenders.append(path.name)
    assert offenders == []
