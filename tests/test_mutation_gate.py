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
