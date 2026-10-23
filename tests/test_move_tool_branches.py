"""scripts/move_tool_branches.py on a synthetic dispatcher module."""
from __future__ import annotations

import ast
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("move_tool_branches", ROOT / "scripts" / "move_tool_branches.py")
mover = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mover  # dataclasses look the module up
spec.loader.exec_module(mover)

SOURCE = '''
import json

def _helper(x):
    return x * 2


class Agent:
    async def _run_tool_call(self, tool_call, api_key, project_id, conversation_id,
                             _depth, _call_stack, user_message, history):
        name = tool_call["function"]["name"]
        args = {}
        if name == "search":
            name = "search_project_documents"

        # ── widget_count ────────
        # Counts widgets in the project.
        # ----------------------------------
        if name == "widget_count":
            n = _helper(len(args))
            return {"count": n, "agent": self.name, "project": project_id}

        # widget_echo
        if name == "widget_echo":
            return {"echo": user_message}

        if name == "falls_through":
            args["seen"] = True

        if name == "nobody_owns_this":
            return {}

        return {"error": "unknown tool"}
'''

OWNERS = {"widget_count": "planning", "widget_echo": "base", "falls_through": "base"}


def test_plan_moves_only_returning_owned_branches():
    p = mover.plan(SOURCE, OWNERS, source_module="synthetic.runtime")
    assert [n for n, _ in p.moved] == ["widget_count", "widget_echo"]
    kept = {n: why for n, _line, why in p.kept}
    assert kept == {"search": "falls through", "falls_through": "falls through",
                    "nobody_owns_this": "no owner"}
    assert p.problems == []


def test_generated_handler_binds_call_fields_and_imports_helpers():
    p = mover.plan(SOURCE, OWNERS, source_module="synthetic.runtime")
    code = mover.module_text("planning", p.handlers["planning"])
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef))
    assert fn.name == "handle_widget_count"
    assert ast.unparse(fn.decorator_list[0]) == "tool('widget_count', owner='planning')"
    # Docstring from the comment above the branch: no box-drawing, no ruler.
    assert ast.get_docstring(fn) == "widget_count Counts widgets in the project"
    src = ast.unparse(fn)
    assert "agent = call.agent" in src and "project_id = call.project_id" in src
    assert "agent = agent" not in src
    assert "from synthetic.runtime import _helper" in src
    assert "self" not in src
    assert not any("─" <= ch <= "╿" for ch in code)


def test_a_comment_that_only_names_the_tool_becomes_a_plain_docstring():
    p = mover.plan(SOURCE, OWNERS, source_module="synthetic.runtime")
    fn = next(n for n in ast.parse(mover.module_text("base", p.handlers["base"])).body
              if isinstance(n, ast.AsyncFunctionDef))
    assert ast.get_docstring(fn) == "The ``widget_echo`` tool."


def test_removal_leaves_the_rest_and_adds_the_dispatch_once():
    p = mover.plan(SOURCE, OWNERS, source_module="synthetic.runtime")
    out = mover.remove_branches(SOURCE, p)
    assert 'name == "widget_count"' not in out and 'name == "widget_echo"' not in out
    assert "Counts widgets" not in out
    assert 'name == "falls_through"' in out and 'name == "nobody_owns_this"' in out
    assert out.count("_tool_registry.get(name)") == 1
    again = mover.remove_branches(out, mover.plan(out, OWNERS, source_module="synthetic.runtime"))
    assert again.count("_tool_registry.get(name)") == 1  # idempotent


def test_generated_modules_carry_no_box_drawing():
    agents = ROOT / "app" / "agents"
    generated = list((agents / "base").glob("tools.py")) + list((agents / "hats").glob("*/tools.py"))
    assert generated
    for path in generated:  # the mover's output: base and hat tool modules
        text = path.read_text(encoding="utf-8")
        assert not any("─" <= ch <= "╿" for ch in text), path
