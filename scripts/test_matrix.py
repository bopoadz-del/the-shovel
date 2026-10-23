"""Run the test suite exactly the way CI does — every matrix profile.

WHY THIS EXISTS
---------------
2026-08-11: a PR was reported locally as "3302 passed, 0 failed" and CI failed
anyway, on `tests/blocks/test_chaining.py::test_standardized_response_format`.
Nothing was flaky. The local run had simply exercised ONE of CI's two profiles:
that test is skipped under the virgin profile and only runs under
production-like. "Local green" and "CI green" did not mean the same thing.

This script closes that gap. It PARSES `.github/workflows/test.yml` and runs
each matrix entry with the same environment CI uses, rather than hardcoding a
copy of the matrix here — a hardcoded copy would drift the first time someone
edits the workflow, which is the same class of bug all over again.

USAGE
-----
    python scripts/test_matrix.py                 # all profiles, whole suite
    python scripts/test_matrix.py -k chaining     # extra args go to pytest
    python scripts/test_matrix.py --profile virgin

Exit code is non-zero if ANY profile fails, so it is usable as a pre-push gate.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"


def _load_workflow() -> dict:
    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml ships with the test deps
        sys.exit("pyyaml is required: pip install pyyaml")
    if not WORKFLOW.is_file():
        sys.exit(f"cannot find {WORKFLOW} — has the workflow moved?")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _matrix_and_env(workflow: dict) -> tuple[list[dict], dict]:
    """Pull the matrix entries and the 'Run tests' step env out of the workflow."""
    try:
        job = workflow["jobs"]["test"]
        include = job["strategy"]["matrix"]["include"]
    except (KeyError, TypeError):
        sys.exit("could not read jobs.test.strategy.matrix.include from test.yml")

    # Match the step that RUNS pytest, not merely one that mentions it. The
    # first version of this matched `"pytest" in step.run`, which hit the
    # "Install Python deps" step (it pip-installs pytest). That step has no
    # env, so the runner silently set NOTHING and every profile reported a
    # meaningless pass — the exact failure this script exists to prevent.
    candidates = [
        s for s in job.get("steps", [])
        if isinstance(s, dict) and "pytest tests/" in str(s.get("run", ""))
    ]
    if not candidates:
        sys.exit("could not find the step that runs `pytest tests/` in test.yml")
    step_env = candidates[0].get("env", {}) or {}
    if not step_env:
        sys.exit("the pytest step in test.yml declares no env — refusing to run a "
                 "matrix that would set nothing (that is a false-green machine)")
    return include, step_env


def _resolve(value, profile: dict, data_dir: Path) -> str:
    """Expand the ${{ ... }} expressions the workflow uses in its env block."""
    text = str(value)
    if "${{" not in text:
        return text
    for key, val in profile.items():
        text = text.replace(f"${{{{ matrix.{key} }}}}", str(val))
    text = text.replace("${{ runner.temp }}", str(data_dir))
    if "${{" in text:  # an expression we do not understand — do not guess
        raise SystemExit(f"unhandled workflow expression in env value: {value!r}")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="run only this profile (e.g. virgin)")
    # parse_known_args, not REMAINDER: pytest flags like `-k foo` start with a
    # dash and argparse would otherwise reject them as unknown options.
    args, passthrough = parser.parse_known_args()

    include, step_env = _matrix_and_env(_load_workflow())
    profiles = [p for p in include
                if args.profile is None or p.get("profile") == args.profile]
    if not profiles:
        sys.exit(f"no profile named {args.profile!r}; found: "
                 f"{[p.get('profile') for p in include]}")

    extra = [a for a in passthrough if a != "--"]
    results: list[tuple[str, int]] = []

    for profile in profiles:
        name = profile.get("profile", "?")
        with tempfile.TemporaryDirectory(prefix=f"fork-{name}-") as tmp:
            env = dict(os.environ)
            for key, value in step_env.items():
                env[key] = _resolve(value, profile, Path(tmp))
            shown = {k: env[k] for k in step_env}
            print(f"\n{'=' * 70}\nPROFILE: {name}\n  {shown}\n{'=' * 70}", flush=True)

            cmd = [sys.executable, "-m", "pytest", "tests/", "-q", "--timeout=300", *extra]
            code = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
            results.append((name, code))

    print(f"\n{'=' * 70}\nMATRIX RESULT")
    for name, code in results:
        print(f"  {name:<20} {'PASS' if code == 0 else f'FAIL (exit {code})'}")
    failed = [n for n, c in results if c != 0]
    if failed:
        print(f"\n{len(failed)} profile(s) failed: {', '.join(failed)}")
        print("CI runs every profile — a green run on only one of them proves nothing.")
        return 1
    print("\nAll profiles passed. This is what CI green means.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
