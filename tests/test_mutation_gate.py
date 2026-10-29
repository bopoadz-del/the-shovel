"""CI-M: the mutation gate reddens on a surviving mutant and an untested
function, and the lint reddens on a new single-input value test."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

GATE = next(p / "scripts" / "mutation_gate.py" for p in Path(__file__).resolve().parents if (p / "scripts" / "mutation_gate.py").is_file())
spec = importlib.util.spec_from_file_location("mutation_gate", GATE)
mg = importlib.util.module_from_spec(spec)
sys.modules["mutation_gate"] = mg
spec.loader.exec_module(mg)

FN = (
    "def fee(amount, rate):\n"
    "    if amount < 0:\n"
    "        raise ValueError('negative')\n"
    "    return execute(amount) * rate\n\n"
    "def execute(x):\n"
    "    return x + 1\n"
)
STRONG = (
    "import pytest\nfrom pkg.calc import fee\n\n"
    "@pytest.mark.parametrize('a,r,want', [(10, 2, 22), (3, 3, 12)])\n"
    "def test_fee(a, r, want):\n    assert fee(a, r) == want\n\n"
    "def test_negative():\n    with pytest.raises(ValueError):\n        fee(-1, 2)\n"
)
WEAK = "from pkg.calc import fee\n\ndef test_fee_runs():\n    fee(10, 2)\n"


def _repo(tmp: Path, test_src: str) -> Path:
    (tmp / "src" / "pkg").mkdir(parents=True)
    (tmp / "tests").mkdir()
    (tmp / "src" / "pkg" / "__init__.py").write_text("")
    (tmp / "src" / "pkg" / "calc.py").write_text("def fee(amount, rate):\n    return amount\n")
    (tmp / "tests" / "conftest.py").write_text("import sys, pathlib\nsys.path.insert(0, str(pathlib.Path(__file__).parents[1] / 'src'))\n")
    run = lambda *a: subprocess.run(["git", *a], cwd=tmp, check=True, capture_output=True)
    run("init", "-q", "-b", "main")
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    (tmp / "src" / "pkg" / "calc.py").write_text(FN)
    (tmp / "tests" / "test_calc.py").write_text(test_src)
    run("add", "-A")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "head")
    return tmp


@pytest.mark.parametrize("tests,red", [(STRONG, set()), (WEAK, set(mg.MUTANTS) - {"bypass_dispatcher"})])
def test_mutants_killed_or_survive(tmp_path, tests, red):
    repo = _repo(tmp_path, tests)
    res = mg.mutation_gate(repo, "HEAD~1", ["src"], ["tests"], ".", [])
    survived = {r.mutant for r in res if r.function == "fee" and r.outcome != "killed"}
    assert {r.mutant for r in res if r.function == "fee"} == set(mg.MUTANTS)
    assert survived >= red and (not red or survived)
    if not red:
        assert survived == set()


@pytest.mark.parametrize("tests", [STRONG, WEAK])
def test_two_input_lint_passes_parametrized_or_valueless(tmp_path, tests):
    # STRONG: value test parametrized, raises-test asserts no value; WEAK asserts nothing.
    repo = _repo(tmp_path, tests)
    assert mg.lint_two_inputs(repo, "HEAD~1", ["tests"]) == []


@pytest.mark.parametrize("cases,ok", [("[(1, 1)]", False), ("[(1, 1), (2, 2)]", True)])
def test_lint_counts_distinct_cases(tmp_path, cases, ok):
    repo = _repo(tmp_path, STRONG)
    (repo / "tests" / "test_more.py").write_text(
        f"import pytest\n@pytest.mark.parametrize('a,b', {cases})\ndef test_eq(a, b):\n    assert a == b\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "more"], cwd=repo, check=True)
    assert (mg.lint_two_inputs(repo, "HEAD~2", ["tests"]) == []) is ok


@pytest.mark.parametrize("name", ["test_one", "test_two"])
def test_lint_reddens_unparametrized_value_test(tmp_path, name):
    repo = _repo(tmp_path, STRONG + f"\ndef {name}():\n    assert fee(1, 1) == 2\n")
    assert mg.lint_two_inputs(repo, "HEAD~1", ["tests"]) == [f"tests/test_calc.py::{name}"]


@pytest.mark.parametrize("tests", [STRONG, WEAK])
def test_untested_function_is_red(tmp_path, tests):
    repo = _repo(tmp_path, tests)
    (repo / "src" / "pkg" / "other.py").write_text("def lonely(x):\n    if x:\n        return 1\n    return 2\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "o"], cwd=repo, check=True)
    res = mg.mutation_gate(repo, "HEAD~2", ["src"], ["tests"], ".", [])
    assert any(r.function == "lonely" and r.outcome == "uncovered" for r in res)


@pytest.mark.parametrize("root,want", [("src", "pkg.calc"), ("src/pkg", "pkg.calc")])
def test_module_token_follows_package_root(tmp_path, root, want):
    repo = _repo(tmp_path, STRONG)
    assert mg.module_tokens("src/pkg/calc.py", [root], repo) == [want]


# --- G6 repair: timeout, SyntaxError, lint filters, mutate guards, main() --
import ast as _ast


def _commit(repo, msg="c"):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg], cwd=repo, check=True)


@pytest.mark.parametrize("timeout", [1, 600])
def test_timeout_is_never_a_kill(tmp_path, monkeypatch, timeout):
    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=timeout)

    monkeypatch.setattr(mg.subprocess, "run", hang)
    out = mg.run_tests(tmp_path, ["tests/test_x.py"], ".", [])
    assert out == "timeout" and out != "killed" and mg.is_red(out)


@pytest.mark.parametrize("rc,want", [(0, "survived"), (1, "killed"), (5, "killed")])
def test_run_tests_maps_returncode(tmp_path, monkeypatch, rc, want):
    monkeypatch.setattr(mg.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, rc, "", ""))
    assert mg.run_tests(tmp_path, ["t"], ".", []) == want


@pytest.mark.parametrize("outcome,red", [("killed", False), ("survived", True), ("uncovered", True), ("timeout", True)])
def test_every_non_kill_is_red(outcome, red):
    assert mg.is_red(outcome) is red


@pytest.mark.parametrize("src", ["def test_x(:\n", "def test_y()\n    pass\n"])
def test_syntax_error_is_not_swallowed(src):
    with pytest.raises(SyntaxError):
        mg._test_fns(src)


@pytest.mark.parametrize("src,names", [
    ("def test_a():\n    pass\ndef helper():\n    pass\n", ["test_a"]),
    ("async def test_b():\n    pass\nclass T:\n    def test_c(self):\n        pass\n", ["test_b", "test_c"]),
])
def test_test_fns_finds_test_functions_only(src, names):
    assert sorted(mg._test_fns(src)) == names


# Lint filters L1-L5
@pytest.mark.parametrize("fname", ["helpers.py", "test_notes.txt"])
def test_lint_l1_only_test_modules(tmp_path, fname):
    repo = _repo(tmp_path, STRONG)
    (repo / "tests" / fname).write_text("def test_z():\n    assert 1 == 1\n")
    _commit(repo)
    assert mg.lint_two_inputs(repo, "HEAD~2", ["tests"]) == []


@pytest.mark.parametrize("extra", ["\ndef test_old():\n    assert fee(1, 1) == 2\n", "\ndef test_old2():\n    assert fee(2, 1) != 9\n"])
def test_lint_l2_existing_tests_not_relinted(tmp_path, extra):
    repo = _repo(tmp_path, STRONG + extra)
    (repo / "tests" / "test_calc.py").write_text(STRONG + extra + "\n# touch\n")
    _commit(repo)
    assert mg.lint_two_inputs(repo, "HEAD~1", ["tests"]) == []


@pytest.mark.parametrize("op", ["==", "!="])
def test_lint_l3_catches_eq_and_neq(tmp_path, op):
    repo = _repo(tmp_path, STRONG + f"\ndef test_cmp():\n    assert fee(1, 1) {op} 3\n")
    assert mg.lint_two_inputs(repo, "HEAD~1", ["tests"]) == ["tests/test_calc.py::test_cmp"]


@pytest.mark.parametrize("op,rhs", [("<", "3"), ("is", "None"), ("in", "[2]")])
def test_lint_l4_ignores_non_value_compares(tmp_path, op, rhs):
    repo = _repo(tmp_path, STRONG + f"\ndef test_cmp():\n    assert fee(1, 1) {op} {rhs}\n")
    assert mg.lint_two_inputs(repo, "HEAD~1", ["tests"]) == []


@pytest.mark.parametrize("deco,n", [
    ("@pytest.mark.parametrize('a', [1, 1])", 1),
    ("@pytest.mark.parametrize('a', [1, 2, 3])", 3),
    ("@pytest.mark.parametrize('a', CASES)", 2),
    ("@pytest.mark.parametrize('a')", 0),
    ("@pytest.mark.skip('x')", 0),
])
def test_lint_l5_counts_distinct_parametrize_cases(deco, n):
    fn = _ast.parse(f"{deco}\ndef test_x(a):\n    assert a == a\n").body[0]
    assert mg._param_cases(fn) == n


# Mutate guards G1, G2, D1, N1, X1, W1, R1
def _fn(src):
    return _ast.parse(src).body[0]


@pytest.mark.parametrize("src,applies", [
    ("def f(x):\n    return x + 1\n", True),
    ("def f(x):\n    yield x\n", False),            # G1: generators keep their shape
    ("def f(x):\n    '''doc'''\n    pass\n", False),  # G2: trivial body has nothing to mutate
    ("def f(x):\n    '''doc'''\n", False),
])
def test_constant_return_guards(src, applies):
    fn = _fn(src)
    assert mg.mutate(fn, "constant_return") is applies
    if applies:
        assert _ast.unparse(fn.body[-1]) == "return None"


@pytest.mark.parametrize("src,applies,after", [
    ("def f(x):\n    if x:\n        return 1\n    return 2\n", True, "pass"),           # D1
    ("def f(x):\n    if x:\n        return 1\n    else:\n        return 3\n", True, "return 3"),
    ("def f(x):\n    return 2\n", False, "return 2"),
])
def test_drop_rule(src, applies, after):
    fn = _fn(src)
    assert mg.mutate(fn, "drop_rule") is applies
    assert _ast.unparse(fn.body[0]) == after


@pytest.mark.parametrize("src,applies", [
    ("def f(x):\n    raise ValueError(x)\n", True),
    ("def f(x):\n    for i in x:\n        raise KeyError(i)\n", True),
    ("def f(x):\n    return x\n", False),   # N1: no raise, no mutant
])
def test_swallow_exception(src, applies):
    fn = _fn(src)
    assert mg.mutate(fn, "swallow_exception") is applies
    assert not any(isinstance(n, _ast.Raise) for n in _ast.walk(fn)) or not applies


@pytest.mark.parametrize("call,applies", [
    ("execute(x)", True), ("self.dispatch_job(x)", True), ("route_to(x, 1)", True), ("handle(x)", True),
    ("run_block(x)", True), ("call_block(x)", True),
    ("executor(x)", False),     # X1: anchored name match only
    ("dispatch()", False),      # W1: no argument to bypass to
    ("compute(x)", False),      # R1: not a dispatcher
])
def test_bypass_dispatcher_guards(call, applies):
    fn = _fn(f"def f(x):\n    return {call}\n")
    assert mg.mutate(fn, "bypass_dispatcher") is applies
    if applies:
        assert _ast.unparse(fn.body[0]) == "return x"


@pytest.mark.parametrize("name,is_test", [("test_a.py", True), ("a_test.py", True), ("conftest.py", True), ("a.py", False)])
def test_is_test_file(name, is_test):
    assert mg.is_test_file(f"src/pkg/{name}") is is_test


@pytest.mark.parametrize("results,lint,code", [
    ([], [], 0), (["killed"], [], 0), (["killed"], ["t::x"], 1), (["survived"], [], 1), (["timeout"], [], 1), (["uncovered"], [], 1),
])
def test_exit_code(results, lint, code):
    rs = [mg.MutantResult("p.py", "f", "constant_return", o) for o in results]
    assert mg.exit_code(rs, lint) == code


@pytest.mark.parametrize("extra,code", [("", 0), ("\ndef test_one():\n    assert fee(1, 1) == 3\n", 1)])
def test_main_exits_nonzero_when_lint_red(tmp_path, monkeypatch, extra, code):
    repo = _repo(tmp_path, STRONG + extra)
    monkeypatch.chdir(repo)
    summary = tmp_path / "s.md"
    assert mg.main(["--base", "HEAD~1", "--src", "src", "--tests", "tests", "--summary", str(summary)]) == code
    assert ("FAIL: `tests/test_calc.py::test_one`" in summary.read_text()) is bool(code)
