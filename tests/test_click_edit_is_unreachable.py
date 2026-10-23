"""The mechanical half of an adjudicated advisory.

PYSEC-2026-2132 is command injection in ``click.edit()``, the helper that
shells out to $EDITOR for interactive CLI apps. It affects click <= 8.3.2
and this repository pins 8.1.8.

It is not fixed, and cannot be: the fix is click 8.3.3, while
``gtts==2.5.4`` requires ``click<8.2,>=7.1``. 2.5.4 is the latest gtts, so
no version pair satisfies both. ``dependency-audit.yml`` therefore carries
``--ignore-vuln PYSEC-2026-2132``.

That suppression rests on one claim: **nothing here reaches
``click.edit()``**. click is transitive only (uvicorn / typer / gtts bring
it), and no module imports it.

A claim written in a YAML comment goes stale silently. This is the twin
that does not: add a ``click.edit()`` call, or import click directly, and
the adjudication becomes false and CI says so.

Delete this file only together with the ignore.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("app", "scripts")
SKIP_DIRS = {
    "__pycache__", ".venv", "node_modules", ".git", "build", "dist",
    ".pytest_cache", ".sweep",
}


def _python_files():
    for top in SCANNED:
        for path in (ROOT / top).rglob("*.py"):
            if SKIP_DIRS & set(path.parts):
                continue
            yield path


def _trees():
    for path in _python_files():
        try:
            yield path, ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except SyntaxError:
            continue


def test_no_module_imports_click():
    """click is transitive. A direct import is the first step to using it."""
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name == "click" or a.name.startswith("click.") for a in node.names):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "") == "click" or (node.module or "").startswith("click."):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not offenders, (
        "click is imported directly, which the PYSEC-2026-2132 adjudication in "
        ".github/workflows/dependency-audit.yml assumes never happens: "
        + ", ".join(sorted(offenders))
    )


def test_nothing_calls_click_edit():
    """The specific function the advisory is about."""
    offenders = []
    for path, tree in _trees():
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            named_edit = (
                isinstance(fn, ast.Attribute)
                and fn.attr == "edit"
                and isinstance(fn.value, ast.Name)
                and fn.value.id == "click"
            )
            if named_edit:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert not offenders, (
        "click.edit() is reachable; PYSEC-2026-2132 is no longer adjudicated "
        "away. Either remove the call or remove --ignore-vuln PYSEC-2026-2132 "
        "from dependency-audit.yml: " + ", ".join(sorted(offenders))
    )


def test_the_ignore_and_this_twin_reference_each_other():
    """Neither half may drift away from the other unnoticed."""
    workflow = (ROOT / ".github/workflows/dependency-audit.yml").read_text(encoding="utf-8")
    if "PYSEC-2026-2132" not in workflow:
        pytest.skip("the ignore is gone — click was upgraded; delete this file too")
    assert "test_click_edit_is_unreachable" in workflow, (
        "the adjudication must name the twin that keeps it true"
    )
    assert "gtts" in workflow, (
        "the adjudication must name WHY the fix is unavailable, not just that "
        "it is ignored"
    )


def test_the_gtts_ceiling_is_still_the_reason():
    """When gtts lifts its cap, this stops being unfixable — say so loudly."""
    reqs = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "gtts==" in reqs, (
        "gtts is gone from requirements.txt; the click ceiling it imposed is "
        "gone with it, so PYSEC-2026-2132 is now fixable — bump click to "
        ">=8.3.3 and drop the ignore"
    )
