#!/usr/bin/env python3
"""CI-M -- mutation gate + two-input test lint (ORDERS_29OCT_1710Z-B).

Gate 1 (mutants): for every function CHANGED in ``base...HEAD`` under the
source roots, apply the standard mutants one at a time --

* ``constant_return``   the body becomes ``return None``
* ``drop_rule``         the first ``if`` is removed (its branch never runs)
* ``swallow_exception`` the first ``raise`` becomes ``pass``
* ``bypass_dispatcher`` the first dispatch-like call (``execute`` /
                        ``dispatch*`` / ``route*`` / ``handle*``) is replaced
                        by its first argument

-- and run the tests that reference the function's module. A mutant the
suite does not kill FAILS the PR. A changed function no test references is
a FAIL too (nothing can kill its mutants). No allowlists.

Gate 2 (lint): every NEW test function in the diff that asserts a computed
value (``assert ... == ...`` / ``!=``) must be ``pytest.mark.parametrize``d
with >= 2 distinct inputs.

Stdlib only. Writes a markdown summary to ``--summary`` (posted on the PR
by the workflow) and exits 1 on any red.
"""

from __future__ import annotations

import argparse
import ast
import copy
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

MUTANTS = ("constant_return", "drop_rule", "swallow_exception", "bypass_dispatcher")
DISPATCH_RE = re.compile(r"^(execute|dispatch\w*|route\w*|handle\w*|call_block|run_block)$")
TEST_TIMEOUT_S = 600


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def changed_lines(base: str, repo: Path, roots: Sequence[str]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    diff = git("diff", "--unified=0", f"{base}...HEAD", "--", *roots, cwd=repo)
    cur: str | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            p = line[4:]
            cur = p[2:] if p.startswith("b/") and p.endswith(".py") else None
        elif cur and line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start, n = int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1)
                out.setdefault(cur, set()).update(range(start, start + max(n, 1)))
    return out


def functions_touching(tree: ast.AST, lines: set[int]) -> list[ast.FunctionDef]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            span = set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
            inner = {
                l for c in ast.walk(node) if c is not node and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
                for l in range(c.lineno, (c.end_lineno or c.lineno) + 1)
            }
            if (span - inner) & lines:
                found.append(node)
    return found


def _is_generator(fn: ast.AST) -> bool:
    return any(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(fn))


def _is_trivial(fn: ast.FunctionDef) -> bool:
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(getattr(s, "value", None), ast.Constant))]
    return not body or all(isinstance(s, ast.Pass) for s in body)


def mutate(fn: ast.FunctionDef, kind: str) -> bool:
    """Mutate ``fn`` in place. False when the mutant does not apply."""
    doc = fn.body[:1] if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) else []
    if kind == "constant_return":
        if _is_generator(fn) or _is_trivial(fn):
            return False
        fn.body = doc + [ast.Return(value=ast.Constant(value=None))]
        return True
    for parent in ast.walk(fn):
        for field in ("body", "orelse", "finalbody"):
            stmts = getattr(parent, field, None)
            if not isinstance(stmts, list):
                continue
            for i, s in enumerate(stmts):
                if kind == "drop_rule" and isinstance(s, ast.If):
                    stmts[i : i + 1] = s.orelse or [ast.Pass()]
                    return True
                if kind == "swallow_exception" and isinstance(s, ast.Raise):
                    stmts[i] = ast.Pass()
                    return True
    if kind == "bypass_dispatcher":
        for node in ast.walk(fn):
            for name, val in ast.iter_fields(node):
                if isinstance(val, ast.Call) and val.args:
                    f = val.func
                    fname = f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")
                    if DISPATCH_RE.match(fname or ""):
                        setattr(node, name, val.args[0])
                        return True
    return False


def module_tokens(path: str, src_roots: Sequence[str], repo: Path | None = None) -> list[str]:
    p = Path(path)
    prefix: list[str] = []
    for root in src_roots:
        try:
            rel = p.relative_to(root)
            # The import name starts at the root's own package: backend/app/x.py -> app.x
            # ...only when the root is itself a package (src/ layouts are not).
            if ((repo or Path.cwd()) / root / "__init__.py").is_file():
                prefix = [Path(root).name]
            break
        except ValueError:
            continue
    else:
        rel = p
    parts = prefix + list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts)] if parts else []


def tests_for(path: str, fn_name: str, repo: Path, test_roots: Sequence[str], src_roots: Sequence[str]) -> list[str]:
    toks = module_tokens(path, src_roots, repo)
    hits = []
    for root in test_roots:
        for t in sorted((repo / root).rglob("test_*.py")):
            text = t.read_text("utf-8", "replace")
            if any(tok and tok in text for tok in toks) and re.search(rf"\b{re.escape(fn_name)}\b", text) or any(tok and re.search(rf"import {re.escape(tok)}\b|from {re.escape(tok)} import", text) for tok in toks):
                hits.append(str(t.relative_to(repo)))
    return list(dict.fromkeys(hits))


def is_test_file(path: str) -> bool:
    """Test modules are the killers, never the mutated."""
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


#: The only green outcome. Survived, uncovered and timeout are all red.
GREEN_OUTCOMES = frozenset({"killed"})


def is_red(outcome: str) -> bool:
    return outcome not in GREEN_OUTCOMES


def exit_code(results: Sequence[MutantResult], lint: Sequence[str]) -> int:
    """Non-zero on any red mutant result OR any lint finding."""
    return 1 if (any(is_red(r.outcome) for r in results) or lint) else 0


@dataclass
class MutantResult:
    path: str
    function: str
    mutant: str
    outcome: str  # killed | survived | uncovered | timeout


def run_tests(repo: Path, tests: Sequence[str], workdir: str, pytest_args: Sequence[str]) -> str:
    """Outcome of the suite against a mutant: ``survived`` (suite passed),
    ``killed`` (suite failed) or ``timeout`` -- a hang proves nothing, so it
    is reported as its own red outcome, never counted as a kill."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *pytest_args, *tests],
            check=False, cwd=repo / workdir, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        print(f"mutation_gate: suite timed out after {exc.timeout}s: {list(tests)}", file=sys.stderr)
        return "timeout"
    return "survived" if proc.returncode == 0 else "killed"


def mutation_gate(repo: Path, base: str, src_roots: Sequence[str], test_roots: Sequence[str], workdir: str, pytest_args: Sequence[str]) -> list[MutantResult]:
    results: list[MutantResult] = []
    for path, lines in sorted(changed_lines(base, repo, src_roots).items()):
        file = repo / path
        if not file.is_file() or is_test_file(path):
            continue
        original = file.read_text("utf-8")
        tree = ast.parse(original)
        targets = [f.name for f in functions_touching(tree, lines)]
        for fname in dict.fromkeys(targets):
            wd_rel = [str(Path(t).relative_to(workdir)) if workdir != "." else t for t in tests_for(path, fname, repo, test_roots, src_roots)]
            for kind in MUTANTS:
                mtree = copy.deepcopy(tree)
                fn = next(f for f in functions_touching(mtree, lines) if f.name == fname)
                if not mutate(fn, kind):
                    continue
                if not wd_rel:
                    results.append(MutantResult(path, fname, kind, "uncovered"))
                    continue
                try:
                    file.write_text(ast.unparse(ast.fix_missing_locations(mtree)) + "\n", "utf-8")
                    outcome = run_tests(repo, wd_rel, workdir, pytest_args)
                finally:
                    file.write_text(original, "utf-8")
                results.append(MutantResult(path, fname, kind, outcome))
    return results


# --- Gate 2: two-input lint --------------------------------------------------

def _param_cases(fn: ast.FunctionDef) -> int:
    best = 0
    for d in fn.decorator_list:
        if isinstance(d, ast.Call) and getattr(d.func, "attr", "") == "parametrize" and len(d.args) >= 2:
            vals = d.args[1]
            if isinstance(vals, (ast.List, ast.Tuple)):
                distinct = {ast.dump(e) for e in vals.elts}
                best = max(best, len(distinct))
            else:
                best = max(best, 2)  # a computed case list: counted by pytest, trusted as >1 only if non-literal
    return best


def _asserts_value(fn: ast.FunctionDef) -> bool:
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Assert)
            and isinstance(n.test, ast.Compare)
            and any(isinstance(op, (ast.Eq, ast.NotEq)) for op in n.test.ops)
        ):
            return True
    return False


def _test_fns(src: str) -> dict[str, ast.FunctionDef]:
    # A test file that does not parse is not linted around: the SyntaxError
    # propagates and the gate goes red with it.
    tree = ast.parse(src)
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test"):
            out[n.name] = n
    return out


def lint_two_inputs(repo: Path, base: str, test_roots: Sequence[str]) -> list[str]:
    bad = []
    names = git("diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD", "--", *test_roots, cwd=repo).split()
    for path in names:
        if not re.search(r"(^|/)test_[^/]*\.py$", path):
            continue
        try:
            old = git("show", f"{base}:{path}", cwd=repo)
        except subprocess.CalledProcessError:
            old = ""
        before = set(_test_fns(old))
        for name, fn in _test_fns((repo / path).read_text("utf-8")).items():
            if name in before or not _asserts_value(fn):
                continue
            if _param_cases(fn) < 2:
                bad.append(f"{path}::{name}")
    return bad


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--src", action="append", required=True, help="source root(s) to mutate")
    ap.add_argument("--tests", action="append", required=True, help="test root(s)")
    ap.add_argument("--workdir", default=".", help="directory pytest runs from")
    ap.add_argument("--summary", default="")
    ap.add_argument("--pytest-arg", action="append", default=[])
    a = ap.parse_args(argv)
    repo = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()).strip())
    results = mutation_gate(repo, a.base, a.src, a.tests, a.workdir, a.pytest_arg)
    lint = lint_two_inputs(repo, a.base, a.tests)
    red = [r for r in results if is_red(r.outcome)]
    lines = ["## CI-M mutation gate", "", f"Mutants run: {len(results)}; killed: {len(results) - len(red)}; red: {len(red)}", ""]
    lines += [f"- {r.outcome.upper()}: `{r.path}::{r.function}` mutant `{r.mutant}`" for r in red]
    lines += ["", "## Two-input test lint", "", f"New single-input value tests: {len(lint)}"]
    lines += [f"- FAIL: `{t}` (needs parametrize with >= 2 distinct inputs)" for t in lint]
    text = "\n".join(lines) + "\n"
    print(text)
    if a.summary:
        Path(a.summary).write_text(text, "utf-8")
    return exit_code(results, lint)


if __name__ == "__main__":
    sys.exit(main())
