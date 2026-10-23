"""The silent-return twin: an except whose whole body is an empty value.

``scan_exception_pass`` already fails the build on ``except Exception: pass``.
The larger and quieter version is the one that returns instead:

    except Exception:
        return None

The caller then cannot tell "there is nothing there" from "the lookup
failed" — same outage, two meanings, one answer. Measured on this tree the
day the scanner was written: **124 sites**, 58 of them under ``app/core``.

The twin does not pretend those are fine. It baselines them by enclosing
function -- ``path::qualified_function`` plus how many sites that function
may hold -- and fails on the next one, so the count can only fall. An edit
above a baselined handler moves its line, not its function, so it does not
redden the gate.

Any exception TYPE counts, unlike the pass-twin which is scoped to
``Exception``: swallowing ``OSError`` into ``return None`` is the same
degradation. One statement only — a handler that logs first has made the
degradation visible, which is the whole ask, and is not flagged.
"""
from __future__ import annotations

import ast

import scripts.scan_exception_pass as scanner
from scripts.scan_exception_pass import (
    RETURN_ALLOWLIST,
    all_return_sites,
    exception_return_lines,
    scan_returns,
    stale_entries,
)


def _lines(src: str) -> list[int]:
    return exception_return_lines(ast.parse(src))


def test_it_flags_every_empty_shape():
    src = (
        "def a():\n    try:\n        x()\n    except Exception:\n        return None\n"
        "def b():\n    try:\n        x()\n    except ValueError:\n        return {}\n"
        "def c():\n    try:\n        x()\n    except (OSError, KeyError) as e:\n        return ''\n"
        "def d():\n    try:\n        x()\n    except Exception:\n        return []\n"
        "def e():\n    try:\n        x()\n    except Exception:\n        return\n"
        "def f():\n    try:\n        x()\n    except Exception:\n        return 0\n"
    )
    assert _lines(src) == [4, 9, 14, 19, 24, 29]


def test_a_narrow_exception_type_counts_too():
    """OSError swallowed into None is the same lie as Exception swallowed."""
    src = "def a():\n    try:\n        x()\n    except OSError:\n        return None\n"
    assert _lines(src) == [4]


def test_a_logged_handler_is_not_flagged():
    """Visibly degraded is the ask. Logging first satisfies it."""
    src = (
        "def a():\n    try:\n        x()\n    except Exception:\n"
        "        log.warning('x failed', exc_info=True)\n        return None\n"
    )
    assert _lines(src) == []


def test_a_typed_outcome_is_not_flagged():
    src = (
        "def a():\n    try:\n        x()\n    except Exception as exc:\n"
        "        return {'ok': False, 'error': str(exc)}\n"
    )
    assert _lines(src) == []


def test_a_reraise_is_not_flagged():
    src = "def a():\n    try:\n        x()\n    except Exception:\n        raise\n"
    assert _lines(src) == []


def test_a_docstring_does_not_hide_the_return():
    """A handler cannot escape by carrying a docstring above the return."""
    src = (
        "def a():\n    try:\n        x()\n    except Exception:\n"
        '        """why this is fine"""\n        return None\n'
    )
    assert _lines(src) == [4]


def test_the_repo_has_no_new_sites():
    """The gate itself. A new empty-return handler fails the build."""
    assert scan_returns() == [], (
        "new silent empty-return handler(s); log what failed or raise a typed "
        "outcome — do not add them to RETURN_ALLOWLIST"
    )


def test_the_baseline_is_only_ever_smaller():
    """Every allowlisted key must still be a real site.

    A stale allowance is how a baseline quietly grows room: a handler is
    fixed (or its function renamed), the entry keeps its old count, and the
    next real site in that function slips through under an old reason.
    """
    stale = stale_entries(RETURN_ALLOWLIST, all_return_sites())
    assert not stale, (
        "RETURN_ALLOWLIST allows more sites than the tree holds — lower the "
        "count or delete the entry: " + ", ".join(stale)
    )


def test_the_baseline_is_not_a_place_to_add_things():
    """A ceiling, stated. Raise it only by deleting this assertion on purpose."""
    total = sum(entry["count"] for entry in RETURN_ALLOWLIST.values())
    assert total <= 70, (
        f"the baseline grew to {total} sites; it was 124 on "
        "2026-09-10 and is meant to shrink"
    )


def test_every_entry_names_a_function_a_count_and_a_reason():
    for key, entry in RETURN_ALLOWLIST.items():
        assert "::" in key, f"RETURN_ALLOWLIST[{key!r}] is not keyed path::function"
        assert isinstance(entry["count"], int) and entry["count"] > 0, key
        assert isinstance(entry["reason"], str) and entry["reason"].strip(), key


# ── keyed by function, not line ───────────────────────────────────────────

_ONE = "def f():\n    try:\n        x()\n    except Exception:\n        return None\n"
_TWO = _ONE + "    try:\n        y()\n    except OSError:\n        return {}\n"


def _allow(key: str, count: int = 1) -> dict:
    return {key: {"count": count, "reason": "fixture"}}


def test_unrelated_lines_above_a_baselined_handler_do_not_fail(tmp_path, monkeypatch):
    """Mutation killed: keying RETURN_ALLOWLIST by ``path:line`` again."""
    mod = tmp_path / "mod.py"
    mod.write_text(_ONE, encoding="utf-8")
    monkeypatch.setattr(scanner, "RETURN_ALLOWLIST", _allow("mod.py::f"))
    assert scan_returns(tmp_path) == []
    mod.write_text("import os\n\n\n# unrelated\nX = 1\n\n" + _ONE, encoding="utf-8")
    assert scan_returns(tmp_path) == []


def test_a_new_site_beyond_the_functions_count_fails(tmp_path, monkeypatch):
    """Mutation killed: a function key that admits any number of sites."""
    (tmp_path / "mod.py").write_text(_TWO, encoding="utf-8")
    monkeypatch.setattr(scanner, "RETURN_ALLOWLIST", _allow("mod.py::f"))
    findings = scan_returns(tmp_path)
    assert findings, "a second empty return in a 1-count function slipped through"
    assert all(f.startswith("mod.py:") and "mod.py::f" in f for f in findings), findings
    monkeypatch.setattr(scanner, "RETURN_ALLOWLIST", _allow("mod.py::f", count=2))
    assert scan_returns(tmp_path) == []


def test_a_new_site_in_an_unlisted_function_fails(tmp_path, monkeypatch):
    (tmp_path / "mod.py").write_text(
        _ONE + "def g():\n    try:\n        x()\n    except Exception:\n        return []\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(scanner, "RETURN_ALLOWLIST", _allow("mod.py::f"))
    assert scan_returns(tmp_path) == ["mod.py:9"]


def test_an_allowance_larger_than_the_function_is_reported_stale(tmp_path):
    (tmp_path / "mod.py").write_text(_ONE, encoding="utf-8")
    live = all_return_sites(tmp_path)
    assert live == {"mod.py::f": 1}
    allow = {**_allow("mod.py::f", count=2), **_allow("mod.py::gone")}
    stale = stale_entries(allow, live)
    assert len(stale) == 2, stale
    assert any("mod.py::f" in s for s in stale), stale
    assert any("mod.py::gone" in s for s in stale), stale
    assert stale_entries(_allow("mod.py::f"), live) == []
