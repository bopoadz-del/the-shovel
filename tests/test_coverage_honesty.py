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
    answer_corpus_id,
    apply_coverage_honesty,
    format_coverage_line,
    format_not_found,
    finalise_answer,
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
import pytest


@pytest.mark.parametrize("indexed,total", [(0, 118), (0, 0)])
def test_zero_coverage_never_claims_a_search(indexed, total):
    out = finalise_answer(
        "I searched the project's documents and found no IT load "
        "specification.", indexed, total)
    assert "I searched the project's documents" not in out
    assert "did not search" in out


@pytest.mark.parametrize("indexed,total", [(2, 118), (117, 118)])
def test_partial_coverage_states_the_numbers(indexed, total):
    out = finalise_answer("... found no X.", indexed, total)
    assert f"{indexed} of {total} project documents indexed" in out


def test_void_coverage_drops_every_search_claim_sentence():
    raw = ("I searched the documents and found nothing. "
           "The results are all general reference material.")
    out = finalise_answer(raw, 0, 5)
    assert "searched" not in out.lower()
    assert "did not search" in out


def test_total_zero_is_not_a_free_pass():
    out = finalise_answer("No such clause in the contract.", 0, 0)
    assert "did not search" in out


@pytest.mark.parametrize("session_pid,turn_pid,expected",
    [("session_pid", "turn_pid", "turn_pid"), ("session_pid", None, "session_pid")])
def test_the_answer_names_the_corpus_it_searched(session_pid, turn_pid, expected):
    # the turn's own project wins; the session project is only a fallback
    assert answer_corpus_id(session_pid, turn_pid) == expected


@pytest.mark.parametrize("text,indexed,total",
    [("Clause 8.8 does not exist in this volume.", 10, 10),
     ("No delay-damages clause here.", 5, 5)])
def test_full_index_keeps_the_answer_unchanged(text, indexed, total):
    assert finalise_answer(text, indexed, total) == text


@pytest.mark.parametrize("raw",
    ["The clause does not exist.", "Nothing was found anywhere."])
def test_apply_with_nothing_returns_unchanged(raw):
    assert apply_coverage_honesty(raw) == raw


def test_apply_with_project_id_uses_live_coverage(monkeypatch):
    from app.core.rag import coverage_honesty as ch
    monkeypatch.setattr(ch, "live_coverage", lambda pid: (3, 9))
    out = apply_coverage_honesty("The clause does not exist.", project_id="demo_proj")
    assert "3 of 9 project documents indexed" in out
    assert "does not exist" not in out.lower()


def test_apply_with_void_coverage_never_claims_a_search():
    out = apply_coverage_honesty(
        "I searched the project's documents and found nothing.", coverage=(0, 118))
    assert "searched" not in out.lower()
    assert "did not search" in out
