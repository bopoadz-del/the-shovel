"""No silent exception handlers in app/ — enforced in the suite itself.

The lint workflow blocks on `ruff check app --select S110` staying at 0,
but that job ALSO runs an advisory ruff step whose thousands of legacy
findings render the check permanently red — so a genuine blocking failure
is indistinguishable from the standing noise at a glance. It was: a
`try/except/pass` shipped in the dense-CAD page-size probe (F26b, PR #357)
sat red on main for a day while every observer, including this one, read
the red X as "just advisory ruff".

Stated here as a test, where it cannot be mistaken for style debt, and
implemented with `ast` so it needs no ruff install and runs on every CI
leg. The class is not cosmetic: per the workflow's own history, swallowed
handlers hid a helper that persisted nothing, and left DECRYPTED client
documents on disk after a failed cleanup.
"""
from __future__ import annotations

import ast
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"


_BROAD = {"Exception", "BaseException"}


def _is_broad(handler: ast.ExceptHandler) -> bool:
    """Bare `except:` or a handler catching Exception/BaseException.

    Matches ruff's S110 default (`check-typed-exception = false`): a
    narrow `except OSError: pass` on a best-effort cleanup is a deliberate
    decision, not a swallowed bug, and the gate permits it. Enforcing
    something stricter here than CI enforces would fail honest code.
    """
    t = handler.type
    if t is None:
        return True
    names = t.elts if isinstance(t, ast.Tuple) else [t]
    for n in names:
        if isinstance(n, ast.Name) and n.id in _BROAD:
            return True
        if isinstance(n, ast.Attribute) and n.attr in _BROAD:
            return True
    return False


def _silent_handlers(tree: ast.AST) -> list[int]:
    """Line numbers of broad `except ...: pass` handlers (ruff's S110 shape)."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and _is_broad(node):
            body = [n for n in node.body
                    if not (isinstance(n, ast.Expr)
                            and isinstance(n.value, ast.Constant)
                            and isinstance(n.value.value, str))]  # drop docstrings
            if len(body) == 1 and isinstance(body[0], ast.Pass):
                out.append(node.lineno)
    return out


def test_app_has_no_silent_exception_handlers():
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # not this test's job to police syntax
            continue
        for lineno in _silent_handlers(tree):
            offenders.append(f"{path.relative_to(APP.parent)}:{lineno}")
    assert not offenders, (
        "silent `except: pass` handlers in app/ (baseline is 0 — log the "
        "exception or name what is absorbed and why):\n  "
        + "\n  ".join(offenders)
    )


def test_the_detector_actually_detects():
    """A guard that cannot fail is not a guard."""
    tree = ast.parse("try:\n    x()\nexcept Exception:\n    pass\n")
    assert _silent_handlers(tree) == [3]
    tree = ast.parse("try:\n    x()\nexcept:\n    pass\n")
    assert _silent_handlers(tree) == [3]
    tree = ast.parse("try:\n    x()\nexcept Exception:\n    log('boom')\n")
    assert _silent_handlers(tree) == []
    # narrow, deliberate best-effort cleanup — permitted, as in CI
    tree = ast.parse("try:\n    x()\nexcept OSError:\n    pass\n")
    assert _silent_handlers(tree) == []
