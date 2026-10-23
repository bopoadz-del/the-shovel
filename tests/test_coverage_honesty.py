"""Coverage honesty line + forbidden complete-search phrases on a partial index.

The 2,935 / 6,206 pair is the historical coverage fixture, not a live Neon
count. Mutants: UNPRODUCED (mutmut is not in this environment).
"""

from __future__ import annotations

import ast
from pathlib import Path

from app.agents.runtime import _postprocess_answer
from app.core.rag.coverage_honesty import (
    FORBIDDEN_ABSENCE_PHRASES,
    apply_coverage_honesty,
    format_coverage_line,
    format_not_found,
    is_partial,
    rewrite_forbidden_absence_claims,
)

# Historical coverage fixture — do not treat as a live client figure.
FIXTURE_INDEXED = 2935
FIXTURE_TOTAL = 6206

REPO = Path(__file__).resolve().parents[1]
# Answer-producing modules the AST guard walks. The rewriter itself may
# mention the forbidden phrases so it can replace them.
ANSWER_ROOTS = (
    REPO / "app" / "agents",
    REPO / "app" / "core" / "rag",
)
ALLOWED_TO_MENTION = {
    REPO / "app" / "core" / "rag" / "coverage_honesty.py",
}


def test_fixture_is_the_historical_pair_not_a_live_count():
    assert FIXTURE_INDEXED == 2935
    assert FIXTURE_TOTAL == 6206
    assert is_partial(FIXTURE_INDEXED, FIXTURE_TOTAL)


def test_coverage_line_uses_the_counts_it_was_given():
    line = format_coverage_line(FIXTURE_INDEXED, FIXTURE_TOTAL)
    assert line == "2935 of 6206 project documents indexed"


def test_partial_index_rewrites_forbidden_absence_claims():
    raw = (
        "Clause 8.8 does not exist in this volume. "
        "There is no such clause for delay damages."
    )
    out = rewrite_forbidden_absence_claims(raw, FIXTURE_INDEXED, FIXTURE_TOTAL)
    assert "does not exist" not in out.lower()
    assert "no such clause" not in out.lower()
    assert format_not_found(FIXTURE_INDEXED) in out
    assert "not found in the 2935 indexed" in out


def test_full_index_leaves_those_phrases_alone():
    raw = "That annex does not exist."
    assert rewrite_forbidden_absence_claims(raw, 10, 10) == raw


def test_apply_stamps_the_line_and_rewrites_on_the_fixture():
    out = apply_coverage_honesty(
        "The clause does not exist.",
        coverage=(FIXTURE_INDEXED, FIXTURE_TOTAL),
    )
    assert "2935 of 6206 project documents indexed" in out
    assert "does not exist" not in out.lower()
    assert "not found in the 2935 indexed" in out


def test_postprocess_wires_coverage_honesty_on_a_fixture_project():
    out = _postprocess_answer(
        "There is no such clause in the Particular Conditions.",
        None,
        [],
        coverage=(FIXTURE_INDEXED, FIXTURE_TOTAL),
    )
    assert "2935 of 6206 project documents indexed" in out
    assert "no such clause" not in out.lower()
    assert "not found in the 2935 indexed" in out


def test_postprocess_is_wired_to_the_helper():
    import inspect
    from app.agents import runtime

    src = inspect.getsource(runtime._postprocess_answer)
    assert "apply_coverage_honesty" in src


def _docstring_constant_ids(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            ids.add(id(first.value))
    return ids


def test_ast_guard_forbids_unguarded_absence_phrases():
    """Non-docstring string literals in answer-producing modules may not
    claim a complete search. The rewriter is the only allowed home."""
    offenders: list[str] = []
    for root in ANSWER_ROOTS:
        for path in root.rglob("*.py"):
            if path.resolve() in {p.resolve() for p in ALLOWED_TO_MENTION}:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            skip = _docstring_constant_ids(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if id(node) in skip:
                    continue
                low = node.value.lower()
                for phrase in FORBIDDEN_ABSENCE_PHRASES:
                    if phrase in low:
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}:{phrase}")
    assert offenders == [], (
        "complete-search phrases in answer-producing modules must go through "
        f"coverage_honesty rewrite: {offenders}"
    )
