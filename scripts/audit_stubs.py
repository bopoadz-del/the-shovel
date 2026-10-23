#!/usr/bin/env python3
"""Prints every reachable hollow function file:line name. Exit 1 if any remain
(not in docs/KNOWN_INCOMPLETE.md, not a decorated abstractmethod, not under tests/)."""
import ast, os, sys

def load_known():
    p = "docs/KNOWN_INCOMPLETE.md"
    if not os.path.exists(p): return set()
    out = set()
    for line in open(p, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if line.startswith("- ") and "::" in line:
            out.add(line[2:].split("  ")[0].strip())
    return out

def is_abstract(node):
    for d in node.decorator_list:
        n = d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
        if n == "abstractmethod": return True
    return False

def hollow(node):
    body = [x for x in node.body if not (isinstance(x, ast.Expr)
            and isinstance(x.value, ast.Constant)
            and isinstance(x.value.value, str))]
    if not body: return True
    if len(body) == 1:
        b = body[0]
        if isinstance(b, ast.Pass): return True
        if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) \
           and b.value.value is Ellipsis: return True
        if isinstance(b, ast.Raise) and isinstance(getattr(b, "exc", None), ast.Call) \
           and isinstance(b.exc.func, ast.Name) and b.exc.func.id == "NotImplementedError":
            return True
        if isinstance(b, ast.Return) and (b.value is None
           or (isinstance(b.value, ast.Constant) and b.value.value in (None, "", 0))
           or (isinstance(b.value, ast.Dict) and not b.value.keys)
           or (isinstance(b.value, ast.List) and not b.value.elts)):
            return True
    return False

def main():
    known = load_known(); reachable = []
    for root, dirs, files in os.walk("."):
        # DEVIATION FROM THE SPEC, required for the count to mean anything on
        # this checkout: `.claude/worktrees` and `.worktrees` are LOCAL git
        # worktrees -- 8 stale copies of the tree. Walking them counted every
        # stub 9 times and reported TOTAL: 552 instead of the real figure.
        #
        # `scripts/` is deliberately NOT excluded. Dev tooling can hide a
        # hollow function just as well as shipped code can, and excluding it
        # would drop 16 real hits from the count. They are registered in
        # docs/KNOWN_INCOMPLETE.md with their reasons instead, so they stay visible.
        #
        # Everything else is the spec's logic, untouched.
        dirs[:] = [d for d in dirs if d not in
                   {".git","__pycache__",".venv","node_modules","generated",
                    ".worktrees",".claude","dist","build",".pytest_cache"}]
        for f in files:
            if not f.endswith(".py"): continue
            rel = os.path.relpath(os.path.join(root, f), ".")
            try: tree = ast.parse(open(os.path.join(root,f), encoding="utf-8", errors="ignore").read())
            except SyntaxError: continue
            # DEVIATION: normalise separators. On Windows os.path.relpath yields
            # `tests\foo.py`, so the spec's `rel.startswith("tests/")` never
            # matched and every test helper was reported as a shipped stub.
            rel = rel.replace(os.sep, "/")
            if rel.startswith("tests/") or "/tests/" in rel: continue
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not hollow(n) or is_abstract(n): continue
                    if f"{rel} :: {n.name}" in known: continue
                    reachable.append(f"{rel}:{n.lineno} {n.name}")
    if reachable:
        sys.stdout.write("REACHABLE HOLLOW (implement or register):\n")
        for t in reachable: sys.stdout.write("  " + t + "\n")
        sys.stdout.write(f"TOTAL: {len(reachable)}\n"); sys.exit(1)
    sys.stdout.write("NO REACHABLE HOLLOW FUNCTIONS.\n")

if __name__ == "__main__":
    main()
