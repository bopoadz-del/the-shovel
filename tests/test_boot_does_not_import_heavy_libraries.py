"""Importing the app must not drag in the heavy ML stack.

2026-08-11: the Render deploy failed because startup never opened a port inside
the platform's scan window, so the container was restarted on top of itself and
the two overlapping processes were killed (status 137). Measured cause:
`import app.main` alone took 10.7s and 282 MB, because app/blocks/__init__.py
imports every block eagerly and three of them imported heavy libraries at module
scope:

  * app/blocks/vector_search.py + zvec.py -> sklearn  (~3.2s, pulls scipy+pandas)
    (both blocks have since been retired; sklearn is still used by
    app/core/learning/router.py, which imports it lazily)
  * app/blocks/_knowledge.py              -> sympy    (~1.0s)
  * app/blocks/safety_world_detector.py   -> ultralytics (~265 MB, pulls torch+cv2)

None of them is used at import time, and the safety detector is disabled
entirely on that host (SAFETY_WORLD_WEIGHTS unset), so the cost bought nothing.
Deferring them cut import to 5.0s and 177 MB.

This is a REGRESSION FENCE, not a style rule. The libraries are still
dependencies and still used — the requirement is only that they load on first
use rather than at boot. A new `import torch` at the top of any block would
silently undo the fix, and the symptom would be a failed deploy, not a failed
test. So the test asserts on a subprocess's real sys.modules after importing
the app, which is the same thing the container does.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Cost measured 2026-08-11 on this tree; see the module docstring.
HEAVY = ("torch", "sklearn", "scipy", "pandas", "ultralytics", "cv2", "sympy")


@pytest.fixture(scope="module")
def modules_after_importing_app(tmp_path_factory) -> set[str]:
    """Import app.main in a CLEAN interpreter and report top-level modules.

    A subprocess is essential: by the time this test runs, the pytest session
    has already imported plenty of these libraries for other tests, so checking
    this process's sys.modules would pass no matter what the app does.

    Module-scoped so the (multi-second) app import runs ONCE for all the
    parametrised cases rather than once per library.
    """
    # A throwaway DB under tmp_path, never the repo's data/ directory: leftover
    # files there have previously contaminated later runs of the RAG suite.
    db_path = tmp_path_factory.mktemp("boot_imports") / "boot.db"
    code = textwrap.dedent(
        """
        import sys, json
        import app.main  # noqa: F401
        print(json.dumps(sorted({m.split(".")[0] for m in sys.modules})))
        """
    )
    # Inherit the real environment (PATH, SYSTEMROOT, venv) and override only
    # the database, so this measures the app rather than a crippled interpreter.
    env = dict(os.environ)
    env["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=REPO_ROOT,
        env=env,
    )
    if proc.returncode != 0:
        pytest.fail(f"importing app.main failed:\n{proc.stderr[-3000:]}")
    out = proc.stdout.strip()
    if not out:
        pytest.fail(f"subprocess produced no output.\nstderr:\n{proc.stderr[-3000:]}")
    return set(json.loads(out.splitlines()[-1]))


@pytest.mark.parametrize("library", HEAVY)
def test_boot_does_not_import(library: str, modules_after_importing_app: set) -> None:
    assert library not in modules_after_importing_app, (
        f"`import app.main` pulled in {library!r} at boot. Every block in "
        "app/blocks/__init__.py is imported eagerly, so a module-scope import "
        f"of {library} is paid on every start, including deployments where the "
        "feature is switched off. Move it inside the function that uses it — "
        "see app/core/learning/router.py:385 for the pattern."
    )
