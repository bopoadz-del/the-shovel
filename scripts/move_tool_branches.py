"""Move self-contained ``if name == "<tool>":`` branches of a dispatcher method
into per-owner tool modules registered with app.agents.core.tool_registry
(F-DRIVER Phase A).

    python scripts/move_tool_branches.py --source app/agents/runtime.py \
        --out app/agents [--owners owners.json] [--apply]

Without --apply it prints the plan and writes nothing. With --apply it writes
``<out>/base/tools.py`` and ``<out>/hats/<owner>/tools.py`` and removes the
moved branches from the source, adding the registry dispatch once.

A branch moves only when it always returns (so removing it cannot change
what runs after it) and its owner is given. Every name the branch reads from
the dispatcher's locals is bound from the ``ToolCall``; every module-level
name it reads is imported from the source module at call time. Comments
directly above a branch become the handler's docstring, with box-drawing
rulers and markers removed.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import os
import re
import sys
import textwrap
import tokenize
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

#: Default owners for the tools moved in slice 1 (one owner per tool).
OWNERS: Dict[str, str] = {
    "delegate_to_agent": "base", "search_project_documents": "base", "list_project_documents": "base",
    "fetch_document": "base", "remember_fact": "base", "construction_calc": "base",
    "generate_wbs": "planning", "look_ahead": "planning", "resource_histogram": "planning",
    "cash_flow_forecast": "commercial", "payment_certificate": "commercial", "evm_calculate": "commercial",
    "commissioning_checklist": "qaqc", "validation_pipeline": "qaqc",
    "procurement_list_generator": "procurement", "rfi_generator": "contracts",
}

#: Dispatcher locals and the ToolCall expression each is bound from.
CALL_ATTR: Dict[str, str] = {
    "self": "call.agent", "tool_call": "call.tool_call", "api_key": "call.api_key",
    "project_id": "call.project_id", "conversation_id": "call.conversation_id",
    "_depth": "call.depth", "_call_stack": "call.call_stack", "user_message": "call.user_message",
    "history": "call.history", "args": "call.args", "name": "call.name",
    "fn": '(call.tool_call.get("function") or {})',
    "raw_args": '((call.tool_call.get("function") or {}).get("arguments") or "{}")',
}

# Box-drawing block (U+2500-U+257F) and the ASCII rulers people draw with.
_BOX = re.compile(r"[─-╿]+")
_RULER = re.compile(r"^[\s\-=_*#~]*$")

DISPATCH = '''
        # Tools owned by a package (app.agents.base, app.agents.hats.<hat>)
        # answer from their own module; see app.agents.core.tool_registry.
        from app.agents.core import tool_registry as _tool_registry

        _spec = _tool_registry.get(name)
        if _spec is not None:
            return await _spec.handler(_tool_registry.ToolCall(
                agent=self, name=name, args=args, tool_call=tool_call, api_key=api_key,
                project_id=project_id, conversation_id=conversation_id, depth=_depth,
                call_stack=_call_stack, user_message=user_message, history=history,
            ))
'''


@dataclass
class Plan:
    moved: List[Tuple[str, ast.If]] = field(default_factory=list)
    kept: List[Tuple[str, int, str]] = field(default_factory=list)
    problems: List[Tuple[str, List[str]]] = field(default_factory=list)
    handlers: Dict[str, List[str]] = field(default_factory=dict)


def always_returns(stmts: List[ast.stmt]) -> bool:
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return always_returns(last.body) and always_returns(last.orelse)
    if isinstance(last, ast.Try):
        body_ok = always_returns(last.body + last.orelse) if not last.finalbody else True
        return body_ok and all(always_returns(h.body) for h in last.handlers)
    if isinstance(last, (ast.With, ast.AsyncWith)):
        return always_returns(last.body)
    return False


def branch_name(node: ast.stmt) -> Optional[str]:
    if (isinstance(node, ast.If) and not node.orelse and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "name"
            and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq)
            and isinstance(node.test.comparators[0], ast.Constant)):
        return node.test.comparators[0].value
    return None


def module_level_names(tree: ast.Module) -> set:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def free_names(stmts: List[ast.stmt]) -> Tuple[set, set]:
    loaded, stored = set(), set()
    for s in stmts:
        for n in ast.walk(s):
            if isinstance(n, ast.Name):
                (loaded if isinstance(n.ctx, ast.Load) else stored).add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                stored.update(a.arg for a in n.args.args + n.args.kwonlyargs)
                if not isinstance(n, ast.Lambda):
                    stored.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                stored.update((a.asname or a.name).split(".")[0] for a in n.names)
            elif isinstance(n, ast.ExceptHandler) and n.name:
                stored.add(n.name)
            elif isinstance(n, ast.comprehension):
                stored.update(t.id for t in ast.walk(n.target) if isinstance(t, ast.Name))
    return loaded, stored


def rename_self(code: str) -> str:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.NAME and tok.string == "self":
            tok = tok._replace(string="agent")
        out.append(tok)
    return tokenize.untokenize(out)


def clean_comment(text: str) -> str:
    """A comment line as docstring text: no box-drawing, no rulers."""
    text = _BOX.sub(" ", text)
    text = text.replace("synthetic tool:", " ")
    text = re.sub(r"\s+", " ", text).strip(" .")
    return "" if _RULER.match(text) else text


def find_method(tree: ast.Module, cls: str, method: str):
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in klass.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method)


def plan(src: str, owners: Dict[str, str], *, cls: str = "Agent", method: str = "_run_tool_call",
         source_module: str = "app.agents.runtime") -> Plan:
    lines = src.splitlines(keepends=True)
    tree = ast.parse(src)
    module_names = module_level_names(tree)
    fnode = find_method(tree, cls, method)
    p = Plan()
    for stmt in fnode.body:
        nm = branch_name(stmt)
        if nm is None:
            continue
        if nm in owners and always_returns(stmt.body):
            p.moved.append((nm, stmt))
        else:
            p.kept.append((nm, stmt.lineno, "falls through" if not always_returns(stmt.body) else "no owner"))

    for nm, stmt in p.moved:
        body_src = "".join(lines[stmt.body[0].lineno - 1: stmt.end_lineno])
        i, comments = stmt.lineno - 2, []
        while i >= 0 and lines[i].strip().startswith("#"):
            comments.insert(0, lines[i].strip().lstrip("#").strip())
            i -= 1
        loaded, stored = free_names(stmt.body)
        locals_used = sorted(n for n in loaded if n in CALL_ATTR)
        globals_used = sorted(n for n in loaded - stored if n not in CALL_ATTR and not hasattr(builtins, n))
        missing = [g for g in globals_used if g not in module_names]
        if missing:
            p.problems.append((nm, missing))
        body = rename_self(textwrap.dedent(body_src))
        bind = "\n".join(f"{('agent' if n == 'self' else n)} = {CALL_ATTR[n]}" for n in locals_used)
        imp = ""
        if globals_used:
            imp = (f"from {source_module} import (  # noqa: F401 -- runtime helpers, imported at call time\n"
                   + "".join(f"    {g},\n" for g in globals_used) + ")\n")
        doc = " ".join(t for t in (clean_comment(c) for c in comments) if t)
        if not doc or doc == nm:  # a comment that only names the tool says nothing
            doc = f"The ``{nm}`` tool."
        code = (f'@tool("{nm}", owner="{owners[nm]}")\nasync def handle_{nm}(call: ToolCall) -> dict:\n'
                f'    """{doc}"""\n' + textwrap.indent(imp + bind + ("\n" if bind else "") + body, "    "))
        p.handlers.setdefault(owners[nm], []).append(code)
    return p


def module_text(owner: str, handlers: List[str]) -> str:
    who = "the base package (every hat)" if owner == "base" else f"the {owner} hat"
    header = (f'"""Tools owned by {who}.\n\n'
              "Generated by scripts/move_tool_branches.py from Agent._run_tool_call\n"
              "(F-DRIVER Phase A); each registers itself with app.agents.core.tool_registry.\n"
              '"""\n'
              "from __future__ import annotations\n\n"
              "from app.agents.core.tool_registry import ToolCall, tool\n\n\n")
    return header + "\n\n".join(h.rstrip("\n") + "\n" for h in handlers)


def remove_branches(src: str, p: Plan) -> str:
    lines = src.splitlines(keepends=True)
    for _nm, stmt in sorted(p.moved, key=lambda x: -x[1].lineno):
        start = stmt.lineno - 1
        while start - 1 >= 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        end = stmt.end_lineno
        while end < len(lines) and lines[end].strip() == "":
            end += 1
        del lines[start:end]
    out = "".join(lines)
    anchor = '            name = "search_project_documents"\n'
    if "_tool_registry.get(name)" not in out and out.count(anchor) == 1:
        out = out.replace(anchor, anchor + DISPATCH, 1)
    ast.parse(out)
    return out


def write_modules(out_dir: str, p: Plan) -> List[str]:
    written = []
    for owner, hs in p.handlers.items():
        pkg = os.path.join(out_dir, "base" if owner == "base" else os.path.join("hats", owner))
        os.makedirs(pkg, exist_ok=True)
        for d, label in ((os.path.join(out_dir, "hats"), "hats"), (pkg, owner)):
            os.makedirs(d, exist_ok=True)
            init = os.path.join(d, "__init__.py")
            if not os.path.exists(init):
                with open(init, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(f'"""{label} package (F-DRIVER Phase A)."""\n')
        path = os.path.join(pkg, "tools.py")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(module_text(owner, hs))
        written.append(path)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", default="app/agents/runtime.py")
    ap.add_argument("--out", default="app/agents")
    ap.add_argument("--owners", help="JSON file {tool: owner}; default: the slice-1 table")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--keep-source", action="store_true", help="write modules only; leave the source")
    a = ap.parse_args(argv)
    owners = OWNERS if not a.owners else json.load(open(a.owners, encoding="utf-8"))
    src = open(a.source, encoding="utf-8").read()
    p = plan(src, owners)
    print("MOVE:", [n for n, _ in p.moved])
    print("KEEP:", p.kept)
    print("problems:", p.problems)
    if p.problems:
        return 1
    if a.apply:
        for path in write_modules(a.out, p):
            print("wrote", path)
        if not a.keep_source:
            with open(a.source, "w", encoding="utf-8", newline="") as fh:
                fh.write(remove_branches(src, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
