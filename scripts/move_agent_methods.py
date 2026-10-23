"""Move methods of a class into core modules (F-DRIVER Phase A, core split).

    python scripts/move_agent_methods.py [--apply]

Each listed method of ``Agent`` in ``app/agents/runtime.py`` becomes a module
function of the same name in its target module, with its body unchanged (the
first parameter stays ``self``). The class keeps the name bound to that
function (``_chat_impl = _core_loop._chat_impl``), so it is still a method:
calls, ``Agent.x`` patches and subclass overrides behave as before.

The module-level runtime names a moved function reads -- found with
``symtable``, nested scopes included -- are imported from
``app.agents.runtime`` at the top of the function, at call time. So the
target module never imports runtime at import time (no cycle), and a test
that patches ``app.agents.runtime.<name>`` still reaches the moved code.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import os
import symtable
import sys
import textwrap
from typing import Dict, List, Optional, Set

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, "app", "agents", "runtime.py")
SOURCE_MODULE = "app.agents.runtime"
CLASS = "Agent"

#: method -> core module (app.agents.core.<module>)
TARGETS: Dict[str, str] = {
    "chat": "loop", "_chat_impl": "loop", "chat_stream": "loop",
    "unavailable_reason": "loop", "_fetch_named_missing_input": "context",
    "_build_messages": "context",
    "_chat_stream_impl": "streaming",
    "_call_llm": "llm", "_stream_synthesis": "llm",
    "tool_definitions": "tools", "_run_tool_call": "tools",
}

DOCS: Dict[str, str] = {
    "loop": "The agent's turn: the non-streaming loop and the streaming entry point.",
    "context": "What a turn's model call is given: the messages and named inputs.",
    "streaming": "The streaming turn: retrieval, model calls, tools and the end event.",
    "llm": "Model calls: one call with tools, and the streamed synthesis.",
    "tools": "The tools a turn offers and the dispatch of one tool call.",
}


def module_level_names(tree: ast.Module) -> Set[str]:
    names: Set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, (ast.If, ast.Try)):
            for sub in ast.walk(node):
                if isinstance(sub, (ast.Import, ast.ImportFrom)):
                    names.update((a.asname or a.name).split(".")[0] for a in sub.names)
                elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(sub.name)
                elif isinstance(sub, ast.Assign):
                    names.update(x.id for t in sub.targets for x in ast.walk(t) if isinstance(x, ast.Name))
    return names


def globals_read(table: symtable.SymbolTable) -> Set[str]:
    """Names this scope or any nested scope reads as (implicit) globals."""
    out = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        out |= globals_read(child)
    return out


def method_tables(src: str) -> Dict[str, symtable.SymbolTable]:
    top = symtable.symtable(src, SOURCE, "exec")
    cls = next(c for c in top.get_children() if c.get_name() == CLASS and c.get_type() == "class")
    return {c.get_name(): c for c in cls.get_children() if c.get_type() == "function"}


def wrap_names(names: List[str], indent: str, width: int = 96) -> str:
    lines, cur = [], indent
    for n in names:
        piece = n + ", "
        if len(cur) + len(piece) > width and cur.strip():
            lines.append(cur.rstrip())
            cur = indent
        cur += piece
    if cur.strip():
        lines.append(cur.rstrip())
    return "\n".join(lines)


def build(src: str, targets: Dict[str, str]) -> tuple:
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    module_names = module_level_names(tree)
    tables = method_tables(src)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == CLASS)
    methods = {m.name: m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [m for m in targets if m not in methods]
    if missing:
        raise SystemExit(f"not methods of {CLASS}: {missing}")

    modules: Dict[str, List[str]] = {}
    for name, mod in targets.items():
        node = methods[name]
        if node.decorator_list:
            raise SystemExit(f"{name} has decorators; not moved")
        start = node.lineno - 1
        while start > 0 and lines[start - 1].strip().startswith("#"):
            start -= 1
        text = textwrap.dedent("".join(lines[start:node.end_lineno]))
        needed = sorted((globals_read(tables[name]) & module_names) - set(dir(builtins)))
        if needed:
            body_first = node.body[0]
            has_doc = isinstance(body_first, ast.Expr) and isinstance(body_first.value, ast.Constant) \
                and isinstance(body_first.value.value, str)
            anchor_node = node.body[1] if has_doc and len(node.body) > 1 else body_first
            # Insert the call-time import before the first statement after the docstring.
            rel = anchor_node.lineno - start - 1
            # include comment lines directly above that statement in the insertion point
            tlines = text.splitlines(keepends=True)
            while rel > 0 and tlines[rel - 1].strip().startswith("#"):
                rel -= 1
            ind = " " * (len(tlines[rel]) - len(tlines[rel].lstrip()))
            imp = (f"{ind}from {SOURCE_MODULE} import (  # noqa: F401 -- read at call time\n"
                   + wrap_names(needed, ind + "    ") + "\n" + f"{ind})\n")
            text = "".join(tlines[:rel]) + imp + "".join(tlines[rel:])
        modules.setdefault(mod, []).append(text.rstrip("\n") + "\n")

    # The class: each moved method is replaced by a binding to its function.
    out = lines[:]
    for name in sorted(targets, key=lambda n: -methods[n].lineno):
        node = methods[name]
        start = node.lineno - 1
        while start > 0 and out[start - 1].strip().startswith("#"):
            start -= 1
        ind = " " * node.col_offset
        out[start:node.end_lineno] = [f"{ind}{name} = _core_{targets[name]}.{name}\n"]
    new_src = "".join(out)
    imports = "".join(f"from app.agents.core import {m} as _core_{m}  # noqa: E402\n"
                      for m in sorted(set(targets.values())))
    anchor = f"\nclass {CLASS}"
    i = new_src.index(anchor)
    # place the core imports just above the class (after any decorator lines)
    j = new_src.rfind("\n\n", 0, i)
    new_src = new_src[:j + 2] + imports + "\n" + new_src[j + 2:]
    texts = {}
    annotation_names = _annotation_names(methods, targets)
    for mod, funcs in modules.items():
        head = (f'"""{DOCS.get(mod, mod)}\n\n'
                "Moved from the Agent class in app/agents/runtime.py by\n"
                "scripts/move_agent_methods.py (F-DRIVER Phase A). Each function is still\n"
                "an Agent method (the class binds it); runtime names are read at call\n"
                'time, so this module does not import runtime when it is imported.\n"""\n'
                "from __future__ import annotations\n")
        typing_names, runtime_names = annotation_names.get(mod, (set(), set()))
        stdlib = sorted(n for n in runtime_names if n in sys.stdlib_module_names)
        runtime_names = runtime_names - set(stdlib)
        if not runtime_names:
            typing_names = typing_names - {"TYPE_CHECKING"}
        if stdlib:
            head += "\n" + "".join(f"import {n}  # noqa: F401 -- annotations\n" for n in stdlib)
        if typing_names:
            head += "\nfrom typing import " + ", ".join(sorted(typing_names)) + "\n"
        if runtime_names:
            head += ("\nif TYPE_CHECKING:  # annotations only; runtime names are read at call time\n"
                     "    from app.agents.runtime import " + ", ".join(sorted(runtime_names)) + "\n")
        texts[mod] = head + "\n\n" + "\n\n".join(funcs)
    return texts, new_src


def _annotation_names(methods: Dict[str, ast.AST], targets: Dict[str, str]) -> Dict[str, tuple]:
    """Per module: (typing names, runtime names) its moved functions' annotations use.

    Annotations are strings here (``from __future__ import annotations``), so
    these imports serve readers and checkers: typing names at module level,
    runtime names only under ``TYPE_CHECKING`` (no import cycle)."""
    import typing

    out: Dict[str, tuple] = {}
    for name, mod in targets.items():
        node = methods[name]
        args = node.args
        params = args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]
        anns = [a.annotation for a in params if a.annotation is not None]
        if node.returns is not None:
            anns.append(node.returns)
        anns += [n.annotation for n in ast.walk(node) if isinstance(n, ast.AnnAssign)]
        used = {x.id for a in anns for x in ast.walk(a) if isinstance(x, ast.Name)}
        typing_names, runtime_names = out.setdefault(mod, (set(), set()))
        for u in used - set(dir(builtins)):
            (typing_names if hasattr(typing, u) else runtime_names).add(u)
    for typing_names, runtime_names in out.values():
        if runtime_names:
            typing_names.add("TYPE_CHECKING")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    src = open(SOURCE, encoding="utf-8").read()
    texts, new_src = build(src, TARGETS)
    for mod, text in texts.items():
        ast.parse(text)
        print(f"app/agents/core/{mod}.py: {text.count(chr(10))} lines")
    ast.parse(new_src)
    print(f"app/agents/runtime.py: {src.count(chr(10))} -> {new_src.count(chr(10))} lines")
    if a.apply:
        for mod, text in texts.items():
            with open(os.path.join(REPO, "app", "agents", "core", f"{mod}.py"), "w",
                      encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        with open(SOURCE, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
