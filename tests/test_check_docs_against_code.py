"""G1 docs against code: a name a contributor doc puts in backticks must exist.

``scripts/check_docs_against_code.py`` reads the inline code spans of a doc
and tests each path, source file name and symbol against the tracked tree.
Every test below builds its own small git repo, so the decisions are proven
on inputs the real docs never contained. The last test runs the check on the
repo's own agent docs.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "check_docs_against_code", _REPO / "scripts" / "check_docs_against_code.py"
)
cdac = importlib.util.module_from_spec(_SPEC)
sys.modules["check_docs_against_code"] = cdac
_SPEC.loader.exec_module(cdac)

_CODE = {
    "pkg/runner.py": "def _start_the_run():\n    return WARN_AFTER_S\n\nclass StopSignal: ...\n",
    "pkg/sub/__init__.py": "",
    "tools/do_thing.sh": "echo $TIMEOUT_LIMIT\n",
    "ci/check.yml": "name: check\n",
    ".gitignore": "out/\nstate_*.json\n",
    "NOTES.md": "`_only_in_a_doc` is named here and nowhere in code.\n",
}


def _repo(tmp_path: Path, files: dict[str, str] | None = None) -> Path:
    root = tmp_path / "repo"
    for rel, body in (files or _CODE).items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _stale(root: Path, doc_text: str) -> list[tuple[str, str]]:
    (root / "DOC.md").write_text(doc_text, encoding="utf-8")
    subprocess.run(["git", "add", "DOC.md"], cwd=root, check=True)
    return [(c.kind, c.token) for c in cdac.run(root, ["DOC.md"])]


# ── repo paths ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "token",
    ["pkg/runner.py", "pkg/sub", "pkg/sub/", "pkg/runner", "tools/do_thing.sh"],
)
def test_a_path_the_tree_holds_resolves(tmp_path, token):
    """Mutation killed: a path check that only accepts exact files (no dir,
    no module without ``.py``, no package)."""
    assert _stale(_repo(tmp_path), f"see `{token}`\n") == []


@pytest.mark.parametrize("token", ["pkg/gone.py", "tools/moved/do_thing.sh"])
def test_a_path_the_tree_lacks_is_stale(tmp_path, token):
    """Mutation killed: a check that accepts every path."""
    assert _stale(_repo(tmp_path), f"see `{token}`\n") == [("path", token)]


@pytest.mark.parametrize("token", ["out/build.bin", "pkg/state_run.json"])
def test_a_missing_path_git_ignores_is_a_runtime_artifact(tmp_path, token):
    """Mutation killed: dropping the ignore filter flags build outputs."""
    assert _stale(_repo(tmp_path), f"written to `{token}`\n") == []


@pytest.mark.parametrize("token", ["VENDOR/model-v1.5", "AED/m2"])
def test_a_slash_token_not_rooted_in_the_tree_is_not_a_path(tmp_path, token):
    """Mutation killed: treating every ``/`` token as a repo path."""
    assert _stale(_repo(tmp_path), f"use `{token}`\n") == []


# ── source file names ─────────────────────────────────────────────────────


@pytest.mark.parametrize("token", ["runner.py", "check.yml"])
def test_a_source_file_name_the_tree_holds_resolves(tmp_path, token):
    assert _stale(_repo(tmp_path), f"run `{token}`\n") == []


@pytest.mark.parametrize("token", ["renamed.py", "lint.yml"])
def test_a_source_file_name_the_tree_lacks_is_stale(tmp_path, token):
    """Mutation killed: a file-name check that never fails."""
    assert _stale(_repo(tmp_path), f"run `{token}`\n") == [("file", token)]


# ── symbols ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "token", ["_start_the_run", "WARN_AFTER_S", "StopSignal", "TIMEOUT_LIMIT=30"]
)
def test_a_symbol_the_code_holds_resolves(tmp_path, token):
    """Includes the name on the left of ``NAME=value``."""
    assert _stale(_repo(tmp_path), f"set `{token}`\n") == []


@pytest.mark.parametrize(
    ("token", "name"),
    [
        ("_start_a_run", "_start_a_run"),
        ("StopFlag", "StopFlag"),
        ("WARN_AFTER_MS=900", "WARN_AFTER_MS"),
    ],
)
def test_a_symbol_the_code_lacks_is_stale(tmp_path, token, name):
    """Mutation killed: a symbol check that accepts any word."""
    assert _stale(_repo(tmp_path), f"set `{token}`\n") == [("symbol", name)]


@pytest.mark.parametrize("doc", ["`_only_in_a_doc`\n", "`OnlyInADoc`\n"])
def test_a_doc_never_confirms_its_own_symbol(tmp_path, doc):
    """Mutation killed: reading markdown into the word set, which lets a doc
    (or another doc that repeats it) vouch for a name the code dropped."""
    root = _repo(tmp_path)
    (root / "MORE.md").write_text("OnlyInADoc again\n", encoding="utf-8")
    subprocess.run(["git", "add", "MORE.md"], cwd=root, check=True)
    assert len(_stale(root, doc)) == 1


# ── what is not a claim ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "token",
    ["*_RESCUE", "$DATA_DIR/x.db", "pytest tests/ -q", "main", ".txt", "https://x.y/z"],
)
def test_patterns_commands_and_plain_words_make_no_claim(tmp_path, token):
    root = _repo(tmp_path)
    assert cdac.classify(token, cdac.Tree.read(root, cdac.tracked_files(root))) is None


@pytest.mark.parametrize(
    ("doc", "line"), [("ok `pkg/runner.py`\n`pkg/gone.py`\n", 2), ("`gone.py`\n", 1)]
)
def test_a_stale_reference_names_its_doc_line(tmp_path, doc, line):
    root = _repo(tmp_path)
    (root / "DOC.md").write_text(doc, encoding="utf-8")
    subprocess.run(["git", "add", "DOC.md"], cwd=root, check=True)
    (claim,) = cdac.run(root, ["DOC.md"])
    assert (claim.doc, claim.line) == ("DOC.md", line)


@pytest.mark.parametrize(("doc", "code"), [("`pkg/runner.py`\n", 0), ("`pkg/x.py`\n", 1)])
def test_main_exit_code(tmp_path, capsys, doc, code):
    root = _repo(tmp_path)
    (root / "DOC.md").write_text(doc, encoding="utf-8")
    subprocess.run(["git", "add", "DOC.md"], cwd=root, check=True)
    assert cdac.main(["--root", str(root), "DOC.md"]) == code
    out = capsys.readouterr().out
    assert ("pkg/x.py" in out) is bool(code)


# ── the repo's own agent docs ─────────────────────────────────────────────


@pytest.mark.parametrize("doc", ["AGENTS.md", "docs/ORDER-no-hardwiring.md"])
def test_the_agent_docs_name_only_what_the_tree_holds(doc):
    """Every path, file name and symbol the agent docs (AGENTS.md and the
    standing order it sends every agent to) put in backticks exists in this
    checkout. A doc that is gone fails here rather than passing empty."""
    assert (_REPO / doc).is_file(), doc
    bad = cdac.run(_REPO, [doc])
    assert bad == [], "\n".join(f"{c.doc}:{c.line}: {c.kind} `{c.token}`" for c in bad)
