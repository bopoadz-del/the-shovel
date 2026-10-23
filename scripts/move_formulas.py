"""Move declared formulas into their owner's package (F-DRIVER Phase A).

    python scripts/move_formulas.py app/lib/construction_formulas_qc.py [...] [--apply]

For each module, every ``@formula(owner=...)`` function moves to
``app/agents/base/formulas/<module>.py`` (owner ``base``) or
``app/agents/hats/<owner>/formulas/<module>.py``, together with every
module-level name only that owner's formulas use. A name used by formulas of
two or more owners goes to ``app/agents/base/formulas/<module>_shared.py``.
Names no formula uses (the calculator engine, for one) stay in the source
module -- unless every formula in it has one owner, when the whole module
moves. The source module then imports every moved name back, so existing
imports of it keep working.

Statements move unchanged and in their original order; each new module gets
only the imports its code uses. Nothing is moved when a moved name would
have to import a name that stays (that would be a cycle).
"""
from __future__ import annotations

import argparse
import ast
import collections
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = "_shared"


def formula_owner(node: ast.AST) -> Optional[str]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    for d in node.decorator_list:
        if isinstance(d, ast.Call) and getattr(d.func, "id", getattr(d.func, "attr", "")) == "formula":
            for k in d.keywords:
                if k.arg == "owner" and isinstance(k.value, ast.Constant):
                    return k.value.value
    return None


def defined_names(node: ast.stmt) -> List[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [x.id for t in targets for x in ast.walk(t) if isinstance(x, ast.Name)]
    return []


def import_names(node: ast.stmt) -> List[str]:
    if isinstance(node, ast.Import):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.asname or a.name for a in node.names]
    return []


def loaded_names(node: ast.AST) -> Set[str]:
    return {x.id for x in ast.walk(node) if isinstance(x, ast.Name)}


@dataclass
class Split:
    module: str                                   # dotted source module
    owners_of: Dict[str, str] = field(default_factory=dict)        # name -> owner | SHARED | ""
    parts: Dict[str, List[ast.stmt]] = field(default_factory=dict)  # owner -> statements
    kept: List[ast.stmt] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)


def target_module(owner: str, source_leaf: str) -> str:
    if owner == SHARED:
        return f"app.agents.base.formulas.{source_leaf}{SHARED}"
    if owner == "base":
        return f"app.agents.base.formulas.{source_leaf}"
    return f"app.agents.hats.{owner}.formulas.{source_leaf}"


def split(src: str, module: str) -> Split:
    tree = ast.parse(src)
    sp = Split(module)
    defs: Dict[str, ast.stmt] = {}
    for node in tree.body:
        for n in defined_names(node):
            defs[n] = node
    uses = {n: {u for u in loaded_names(node) if u in defs and u != n} for n, node in defs.items()}
    formulas = {n: o for n, node in defs.items() if (o := formula_owner(node))}
    need: Dict[str, Set[str]] = collections.defaultdict(set)
    for f, o in formulas.items():
        stack, seen = [f], set()
        while stack:
            k = stack.pop()
            if k in seen:
                continue
            seen.add(k)
            need[k].add(o)
            stack.extend(uses[k])
    single = len(set(formulas.values())) == 1
    for n in defs:
        os_ = need.get(n, set())
        if len(os_) == 1:
            sp.owners_of[n] = next(iter(os_))
        elif len(os_) > 1:
            sp.owners_of[n] = SHARED
        else:
            sp.owners_of[n] = next(iter(formulas.values())) if single and formulas else ""

    for node in tree.body:
        names = defined_names(node)
        if not names:
            if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)) \
                    and not isinstance(node, (ast.Import, ast.ImportFrom)):
                sp.problems.append(f"line {node.lineno}: top-level statement that defines nothing")
            continue
        owners = {sp.owners_of[n] for n in names}
        if len(owners) > 1:
            sp.problems.append(f"line {node.lineno}: one statement defines names of owners {sorted(owners)}")
            continue
        owner = owners.pop()
        if owner:
            sp.parts.setdefault(owner, []).append(node)
        else:
            sp.kept.append(node)

    # A moved statement must not read a name that stays (the stayer imports the mover).
    for owner, stmts in sp.parts.items():
        for node in stmts:
            for u in loaded_names(node):
                if u in defs and sp.owners_of[u] == "":
                    sp.problems.append(f"{defined_names(node)[0]} ({owner}) reads {u}, which stays")
                if u in defs and owner == SHARED and sp.owners_of[u] not in (SHARED,):
                    sp.problems.append(f"shared {defined_names(node)[0]} reads {u} owned by {sp.owners_of[u]}")
    return sp


def _imports_for(tree: ast.Module, used: Set[str], src: str) -> List[str]:
    out = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            out.append(ast.get_source_segment(src, node))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            keep = [a for a in node.names if (a.asname or a.name).split(".")[0] in used]
            if not keep:
                continue
            if isinstance(node, ast.Import):
                out.append("import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep))
            else:
                mod = "." * node.level + (node.module or "")
                names = ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)
                out.append(f"from {mod} import {names}")
    return out


def render(src: str, sp: Split) -> Tuple[Dict[str, str], str]:
    """(new module dotted name -> text, new text of the source module)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    leaf = sp.module.rsplit(".", 1)[-1]

    def seg(node: ast.stmt) -> str:
        start = node.lineno - 1
        if getattr(node, "decorator_list", None):
            start = min(d.lineno for d in node.decorator_list) - 1
        # comment lines directly above travel with the statement
        while start > 0 and lines[start - 1].lstrip().startswith("#"):
            start -= 1
        return "".join(lines[start:node.end_lineno])

    out: Dict[str, str] = {}
    for owner, stmts in sp.parts.items():
        used = set().union(*(loaded_names(n) for n in stmts))
        defined_here = {n for s in stmts for n in defined_names(s)}
        imports = _imports_for(tree, used - defined_here, src)
        shared = sorted(u for u in used - defined_here if sp.owners_of.get(u) == SHARED)
        if shared and owner != SHARED:
            imports.append(f"from {target_module(SHARED, leaf)} import (  # noqa: F401\n"
                           + "".join(f"    {n},\n" for n in shared) + ")")
        who = {"base": "the base package (every hat)", SHARED: "more than one owner"}.get(owner, f"the {owner} hat")
        head = (f'"""Formulas from {sp.module} owned by {who}.\n\n'
                "Moved unchanged by scripts/move_formulas.py (F-DRIVER Phase A).\n"
                '"""\n')
        future = [i for i in imports if "__future__" in i]
        rest = [i for i in imports if "__future__" not in i]
        body = "\n\n".join(seg(n).rstrip("\n") for n in stmts)
        out[target_module(owner, leaf)] = head + "\n".join(future + rest) + "\n\n\n" + body + "\n"

    # The source keeps what stays and imports every moved name back.
    moved = sorted(n for n, o in sp.owners_of.items() if o)
    keep_text = []
    first = True
    reexports = []
    for owner in sorted(sp.parts):
        names = sorted(n for s in sp.parts[owner] for n in defined_names(s))
        reexports.append(f"from {target_module(owner, leaf)} import (  # noqa: F401 -- moved\n"
                         + "".join(f"    {n},\n" for n in names) + ")")
    docstring = ast.get_docstring(tree, clean=False)
    body_nodes = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    kept_at = {(n.lineno, n.col_offset) for n in sp.kept}  # sp's nodes are from its own parse
    for node in body_nodes:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_text.append(seg(node).rstrip("\n"))
        elif (node.lineno, node.col_offset) in kept_at:
            if first and reexports:
                keep_text.append(_REEXPORT_NOTE + "\n" + "\n".join(reexports))
                first = False
            keep_text.append(seg(node).rstrip("\n"))
    if first and reexports:
        keep_text.append(_REEXPORT_NOTE + "\n" + "\n".join(reexports))
    head = f'"""{docstring}"""\n' if docstring is not None else ""
    if not sp.kept:
        head = (f'"""{sp.module}: its formulas moved to their owners\' packages (F-DRIVER\n'
                'Phase A). Kept so existing imports keep working; removed with the\n'
                'legacy paths at F-DRIVER step 16."""\n')
        keep_text = [t for t in keep_text if t.startswith(("from __future__",)) or t.startswith(_REEXPORT_NOTE)]
    return out, head + "\n\n".join(keep_text) + "\n"


_REEXPORT_NOTE = ("# Formulas (and the helpers only they use) live in their owner's package;\n"
                  "# imported back so existing imports of this module keep working.")


def module_path(dotted: str) -> str:
    return os.path.join(REPO, *dotted.split(".")) + ".py"


def ensure_packages(dotted: str) -> None:
    parts = dotted.split(".")[:-1]
    for i in range(2, len(parts) + 1):
        d = os.path.join(REPO, *parts[:i])
        os.makedirs(d, exist_ok=True)
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            with open(init, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(f'"""{parts[i - 1]} package (F-DRIVER Phase A)."""\n')


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    rc = 0
    for path in a.paths:
        rel = os.path.relpath(os.path.abspath(path), REPO)
        dotted = rel[:-3].replace(os.sep, ".").replace("/", ".")
        src = open(path, encoding="utf-8").read()
        sp = split(src, dotted)
        counts = collections.Counter(o or "(stays)" for o in sp.owners_of.values())
        print(f"{dotted}: {dict(counts)}")
        if sp.problems:
            rc = 1
            for p in sp.problems:
                print("   problem:", p)
            continue
        if a.apply:
            new, rest = render(src, sp)
            for mod, text in new.items():
                ensure_packages(mod)
                with open(module_path(mod), "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(text)
                print("   wrote", mod)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(rest)
    return rc


if __name__ == "__main__":
    sys.exit(main())
