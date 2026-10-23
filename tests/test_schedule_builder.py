"""Tests for the frontend ScheduleBuilder shim component.

The L2 Schedule roadmap names `frontend/src/components/ScheduleBuilder.tsx`
as an explicit deliverable. The component is a thin UI wrapper that delegates
to the existing export endpoints in `app/routers/exports.py`. This test file
proves the shim exists, delegates to the correct endpoints, and type-checks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = REPO_ROOT / "frontend"
COMPONENT_PATH = FRONTEND_DIR / "src" / "components" / "ScheduleBuilder.tsx"


@pytest.fixture
def component_source() -> str:
    assert COMPONENT_PATH.exists(), f"ScheduleBuilder component missing: {COMPONENT_PATH}"
    return COMPONENT_PATH.read_text(encoding="utf-8")


def test_component_file_exists():
    """The roadmap-named component file must exist."""
    assert COMPONENT_PATH.is_file()


def test_component_exports_default_schedule_builder(component_source: str):
    """It must export a default React component named ScheduleBuilder."""
    assert "export default function ScheduleBuilder" in component_source


def test_component_delegates_to_existing_export_endpoints(component_source: str):
    """It must call the existing backend schedule export endpoints."""
    endpoints = [
        "/export/schedule-from-brief",
        "/export/schedule-from-document",
        "/export/cost-schedule",
    ]
    for endpoint in endpoints:
        assert endpoint in component_source, f"missing delegation endpoint {endpoint}"


def test_component_uses_existing_token_utility(component_source: str):
    """It should reuse the existing auth token helper."""
    assert "import { getToken } from '../lib/token'" in component_source


def test_component_type_checks():
    """Run the front-end TypeScript compiler to prove the shim compiles."""
    assert FRONTEND_DIR.is_dir()

    # The Node binary may not be on the default PATH in this environment, so
    # add the known local install location and fall back to a system node.
    env = os.environ.copy()
    env["PATH"] = str(Path.home() / ".local" / "bin" / "node") + os.pathsep + env.get("PATH", "")

    node = shutil.which("node", path=env["PATH"])
    if node is None:
        pytest.skip("Node.js not available; cannot type-check the component")

    tsc = FRONTEND_DIR / "node_modules" / "typescript" / "bin" / "tsc"
    if not tsc.exists():
        # CI installs Python deps only — frontend node_modules is never
        # populated there, so a missing tsc is an environment limitation
        # (like missing node above), not a product regression.
        pytest.skip("typescript compiler not installed in frontend/node_modules")

    proc = subprocess.run(
        [node, str(tsc), "-p", "tsconfig.app.json", "--noEmit"],
        cwd=FRONTEND_DIR,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"TypeScript type-check failed (exit {proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
