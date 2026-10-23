"""Fail when a function in app/ is never executed by the test suite.

Line coverage hides this. A module can sit at 80% while a specific function
has NEVER run, because coverage reports lines, not functions. The 2026-08-15
sweep found 15 such functions behind 76% line coverage and 3500 passing tests.

Usage (after a coverage run that produced coverage.json):

    pytest tests/ --cov=app --cov-report=json:coverage.json
    python scripts/audit_dead_functions.py coverage.json

Exit 0 when every function in app/ executed at least once, or is declared in
ALLOWED_DEAD below with a reason. Exit 1 otherwise, naming the offenders.

The allowlist is deliberately a source file, not a count: "we accept 7 dead
functions" rots into "we accept whatever number it is today". Each entry has
to say WHY, and a function that becomes reachable must have its line deleted.
"""
from __future__ import annotations

import ast
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fully.qualified.path::function -> why it cannot be executed by the suite.
# Adding an entry is a deliberate act; the reason is reviewed like any code.
ALLOWED_DEAD = {
    "app/core/doc_index.py::_read_stream":
        "legacy .doc reader; needs a real OLE2 compound binary, which olefile "
        "can read but not synthesise, and a binary .doc fixture is a liability",
    "app/routers/admin.py::_list":
        "nested in the Drive folder scanner; needs a live Google Drive token",
    "app/routers/admin.py::_walk":
        "nested in the Drive folder scanner; needs a live Google Drive token",
    "app/routers/mcp.py::_list_tools":
        "only runs inside a live MCP SSE session driven by the mcp transport",
    "app/routers/mcp.py::_call_tool":
        "only runs inside a live MCP SSE session driven by the mcp transport",
    "app/routers/mcp.py::mount_message_endpoint":
        "defined only in the else-branch taken when mcp[server] is ABSENT; the "
        "package is installed, so this branch is unreachable by construction",
    "app/routers/mcp.py::mcp_sse_unavailable":
        "defined only in the else-branch taken when mcp[server] is ABSENT",
}


def dead_functions(coverage_json: str):
    data = json.load(open(coverage_json, encoding="utf-8"))
    out = []
    for relpath, info in (data.get("files") or {}).items():
        norm = relpath.replace("\\", "/")
        if not norm.startswith("app/"):
            continue
        src = os.path.join(REPO, relpath)
        if not os.path.exists(src):
            continue
        try:
            tree = ast.parse(open(src, encoding="utf-8").read())
        except SyntaxError:
            continue
        executed = set(info.get("executed_lines") or [])
        missing = set(info.get("missing_lines") or [])
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = set(range(node.lineno, getattr(node, "end_lineno", node.lineno) + 1))
            ran, miss = len(span & executed), len(span & missing)
            if ran == 0 and miss > 0:
                out.append(f"{norm}::{node.name}")
    return sorted(out)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: audit_dead_functions.py <coverage.json>", file=sys.stderr)
        return 2
    dead = dead_functions(sys.argv[1])
    undeclared = [d for d in dead if d not in ALLOWED_DEAD]
    stale = [d for d in ALLOWED_DEAD if d not in dead]

    print(f"functions never executed : {len(dead)}")
    print(f"declared + justified     : {len(dead) - len(undeclared)}")
    print(f"UNDECLARED               : {len(undeclared)}")

    if stale:
        # Not a failure: a function becoming reachable is good news. But the
        # allowlist must not accumulate entries that no longer describe reality.
        print("\nallowlist entries that are now COVERED (delete these lines):")
        for s in stale:
            print(f"  {s}")

    if undeclared:
        print("\nFUNCTIONS WITH ZERO COVERAGE AND NO DECLARED REASON:")
        for d in undeclared:
            print(f"  {d}")
        print("\nWrite a test, or add it to ALLOWED_DEAD with the reason it "
              "cannot be executed.")
        return 1

    print("\nNO UNDECLARED DEAD FUNCTIONS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
